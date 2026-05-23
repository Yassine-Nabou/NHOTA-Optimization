# ============================================================================
# NHOTA - Nonmonotone Higher-Order Taylor Approximation
# Logistic Regression with NONCONVEX Regularization on Real Dataset
# ============================================================================
# Dataset: Dynamic from LIBSVM (small, real-world binary classification)
# Regularizer: h(x) = lambda * sum_i x_i^2/(1+x_i^2)  [nonconvex]
# ============================================================================
# Methods:
#   - Baseline PG (p=1, monotone)
#   - NHOTA p=1, u=0.5
#   - NHOTA p=2, u=0.5 and u=1.0
#   - NHOTA p=3, u=0.5 and u=1.0
# ============================================================================
# ALL methods use the same acceptance condition:
#   f(x_{k+1}) <= reference - (M/(p+1)!) * ||x_{k+1} - x_k||^{p+1}
# ============================================================================

using LinearAlgebra
using Statistics
using Random
using Printf
using Plots
gr()

# ============================================================================
# GLOBAL CONFIGURATION: Change your dataset link here!
# ============================================================================
const DATASET_URL = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/gisette_scale.bz2"
# a1a: n=1605 samples, d=119 features
# gisette_scale.bz2
# duke.bz2
#
# leu.bz2
#
# ============================================================================
# PART 1: Load LIBSVM Dataset (Supports both text and .bz2 files)
# ============================================================================

using CodecBzip2  

function download_dataset()
    println("Downloading from: $DATASET_URL")
    datafile = download(DATASET_URL)

    labels = Float64[]
    rows = Int[]
    cols = Int[]
    vals = Float64[]

    row_idx = 0
    max_col = 0

    is_compressed = endswith(DATASET_URL, ".bz2")
    file_stream = is_compressed ? Bzip2DecompressorStream(open(datafile)) : open(datafile)      

    try
        for line in eachline(file_stream)
            row_idx += 1
            parts = split(strip(line))
            if isempty(parts)
                continue
            end
            
            label = parse(Float64, parts[1])
            push!(labels, label > 0 ? 1.0 : -1.0)

            for part in parts[2:end]
                if ':' in part
                    idx_str, val_str = split(part, ':')
                    idx = parse(Int, idx_str)
                    val = parse(Float64, val_str)
                    push!(rows, row_idx)
                    push!(cols, idx)
                    push!(vals, val)
                    max_col = max(max_col, idx)
                end
            end
        end
    finally
        close(file_stream)  
        rm(datafile)        
    end

    n = row_idx
    d = max_col
    A = zeros(n, d)
    for (r, c, v) in zip(rows, cols, vals)
        A[r, c] = v
    end

    for j in 1:d
        col_norm = norm(A[:, j])
        if col_norm > 0
            A[:, j] ./= col_norm
        end
    end

    println("Loaded $(basename(DATASET_URL)): n=$n samples, d=$d features")
    return A, labels
end

# ============================================================================
# PART 2: Logistic Loss with Nonconvex Regularization
# ============================================================================

function logistic_loss(x::Vector{Float64}, A::Matrix{Float64}, b::Vector{Float64})
    n = size(A, 1)
    z = b .* (A * x)
    loss = 0.0
    for i in 1:n
        if z[i] >= 0
            loss += log(1.0 + exp(-z[i]))
        else
            loss += -z[i] + log(1.0 + exp(z[i]))
        end
    end
    return loss / n
end

function gradient_logistic(x::Vector{Float64}, A::Matrix{Float64}, b::Vector{Float64})
    n = size(A, 1)
    z = b .* (A * x)
    sig = zeros(n)
    for i in 1:n
        if z[i] >= 0
            sig[i] = 1.0 / (1.0 + exp(z[i]))
        else
            ez = exp(z[i])
            sig[i] = ez / (1.0 + ez)
        end
    end
    return -(A' * (b .* sig)) / n
end

function hessian_logistic(x::Vector{Float64}, A::Matrix{Float64}, b::Vector{Float64})
    n, d = size(A)
    z = b .* (A * x)
    w = zeros(n)
    for i in 1:n
        if abs(z[i]) < 500
            ez = exp(z[i])
            w[i] = ez / ((1.0 + ez)^2)
        else
            w[i] = 0.0
        end
    end
    weighted_A = sqrt.(w) .* A
    return (weighted_A' * weighted_A) / n
end

function h_nonconvex(x::Vector{Float64}, lambda::Float64)
    s = 0.0
    for xi in x
        s += xi^2 / (1.0 + xi^2)
    end
    return lambda * s
end

function grad_h_nonconvex(x::Vector{Float64}, lambda::Float64)
    grad = zeros(length(x))
    for i in eachindex(x)
        xi = x[i]
        grad[i] = lambda * 2.0 * xi / ((1.0 + xi^2)^2)
    end
    return grad
end

function hess_h_nonconvex(x::Vector{Float64}, lambda::Float64)
    d = length(x)
    H = zeros(d, d)
    for i in 1:d
        xi = x[i]
        xi2 = xi^2
        H[i, i] = lambda * 2.0 * (1.0 - 3.0 * xi2) / ((1.0 + xi2)^3)
    end
    return H
end

function third_deriv_action(x::Vector{Float64}, s::Vector{Float64}, lambda::Float64)
    d = length(x)
    result = zeros(d)
    for i in 1:d
        xi = x[i]
        si = s[i]
        xi2 = xi^2
        result[i] = lambda * 24.0 * xi * (xi2 - 1.0) * si^2 / ((1.0 + xi2)^4)
    end
    return result
end

function full_objective(x::Vector{Float64}, A::Matrix{Float64}, b::Vector{Float64}, lambda::Float64)
    return logistic_loss(x, A, b) + h_nonconvex(x, lambda)
end

function gradient_F(x::Vector{Float64}, A::Matrix{Float64}, b::Vector{Float64}, lambda::Float64)
    return gradient_logistic(x, A, b) + grad_h_nonconvex(x, lambda)
end

function hessian_F(x::Vector{Float64}, A::Matrix{Float64}, b::Vector{Float64}, lambda::Float64)
    return hessian_logistic(x, A, b) + hess_h_nonconvex(x, lambda)
end

function compute_stationarity_measure(x::Vector{Float64}, A::Matrix{Float64}, b::Vector{Float64}, lambda::Float64)
    return norm(gradient_F(x, A, b, lambda))
end

# Matrix-free Hessian action closure
function make_hessian_action(x::Vector{Float64}, 
                             A::Matrix{Float64}, 
                             b::Vector{Float64}, 
                             lambda::Float64)
    n, d = size(A)
    
    z = b .* (A * x)
    w = zeros(n)
    for i in 1:n
        if abs(z[i]) < 500  
            ez = exp(z[i])
            w[i] = ez / ((1.0 + ez)^2)  
        else
            w[i] = 0.0
        end
    end
    
    reg_diag = zeros(d)
    for i in 1:d
        xi = x[i]
        xi2 = xi^2
        reg_diag[i] = lambda * 2.0 * (1.0 - 3.0*xi2) / ((1.0 + xi2)^3)
    end
    
    return function(v::Vector{Float64})
        Av = A * v
        weighted_Av = w .* Av
        logistic_hv = (A' * weighted_Av) / n
        reg_hv = reg_diag .* v
        return logistic_hv .+ reg_hv
    end
end

# ============================================================================
# PART 3: Subproblem Solvers
# ============================================================================

function solve_subproblem_p1(xk::Vector{Float64}, grad_F::Vector{Float64}, lambda::Float64, M::Float64)
    return xk .- grad_F ./ M
end

# ===== IMPLEMENTED: Matrix-free CG for Exact p=2 Subproblem =====
function solve_subproblem_p2(xk::Vector{Float64},
                             grad_F::Vector{Float64},
                             lambda::Float64,
                             M::Float64,
                             A::Matrix{Float64},
                             b::Vector{Float64};
                             max_cg_iter::Int=30,
                             cg_tol::Float64=1e-5)
    d = length(xk)
    gnorm = max(norm(grad_F), 1e-12)
    reg = sqrt(gnorm)

    # Instantiate the efficient linear action closure
    H_F_action = make_hessian_action(xk, A, b, lambda)

    # Solve: (H_F + reg*I) * s = -grad_F
    target = -grad_F
    s = zeros(d)
    r = copy(target)
    p_cg = copy(r)
    r_norm_sq = dot(r, r)
    stop_tol = (cg_tol * max(1.0, gnorm))^2

    for iter in 1:max_cg_iter
        if r_norm_sq < stop_tol
            break
        end
        
        # Apply matrix-free system operator
        Ap = H_F_action(p_cg) .+ reg .* p_cg
        
        alpha_cg = r_norm_sq / max(dot(p_cg, Ap), 1e-15)
        s .+= alpha_cg .* p_cg
        r .-= alpha_cg .* Ap
        
        next_r_norm_sq = dot(r, r)
        beta_cg = next_r_norm_sq / max(r_norm_sq, 1e-15)
        p_cg .= r .+ beta_cg .* p_cg
        r_norm_sq = next_r_norm_sq
    end

    return xk + s
end

# ============================================================================
# solve_subproblem_p3: Implemented with Matrix-free Inexact Newton-CG
# ============================================================================
function solve_subproblem_p3(xk::Vector{Float64},
                             grad_F::Vector{Float64},
                             H_F_action::Function,  
                             lambda::Float64,
                             M::Float64,
                             A::Matrix{Float64},    
                             b::Vector{Float64};    
                             max_inner::Int=15,
                             tol_inner::Float64=1e-6,
                             max_cg_iter::Int=20)

    d = length(xk)
    gnorm = max(norm(grad_F), 1e-12)
    reg = gnorm^(2/3)
    
    function Hv(v::Vector{Float64})
        return H_F_action(v) .+ reg .* v
    end
    
    # Accurate warm start initialization
    s = -grad_F ./ (gnorm + reg)
    
    for inner in 1:max_inner
        tensor_reg = third_deriv_action(xk, s, lambda)  
        tensor_term = 0.5 .* tensor_reg
        quartic_term = (M / 6.0) * norm(s)^2 .* s
        
        # Non-linear model gradient residual
        residual = grad_F .+ Hv(s) .+ tensor_term .+ quartic_term
        res_norm = norm(residual)
        
        if res_norm < tol_inner * max(1.0, gnorm)
            break
        end
        
        # --- IMPLEMENTED: Matrix-free full Jacobian Action Operator ---
        function Jv_action(v::Vector{Float64})
            Hv_term = Hv(v)
            tensor_deriv = zeros(d)
            for i in 1:d
                xi, si, vi = xk[i], s[i], v[i]
                denom = (1.0 + xi^2)^4
                if denom > 1e-300
                    tensor_deriv[i] = lambda * 12.0 * xi * (xi^2 - 1.0) * si * vi / denom
                end
            end
            quartic_deriv = (M / 6.0) * (2.0 * dot(s, v) .* s .+ norm(s)^2 .* v)
            return Hv_term .+ tensor_deriv .+ quartic_deriv
        end
        
        # --- IMPLEMENTED: Conjugate Gradient Loop to replace diagonal step ---
        delta = zeros(d)
        r_cg = -residual
        p_cg = copy(r_cg)
        r_norm_sq = dot(r_cg, r_cg)
        cg_target = (tol_inner * res_norm)^2
        
        for cg_step in 1:max_cg_iter
            if r_norm_sq < cg_target
                break
            end
            Jp = Jv_action(p_cg)
            alpha_cg = r_norm_sq / max(dot(p_cg, Jp), 1e-15)
            delta .+= alpha_cg .* p_cg
            r_cg .-= alpha_cg .* Jp
            
            next_r_norm_sq = dot(r_cg, r_cg)
            beta_cg = next_r_norm_sq / max(r_norm_sq, 1e-15)
            p_cg .= r_cg .+ beta_cg .* p_cg
            r_norm_sq = next_r_norm_sq
        end
        
        if any(!isfinite, delta)
            delta = -residual ./ (abs.(residual) .+ 1e-8)
        end
        
        alpha = 0.5
        s_new = s .+ alpha .* delta
        
        for backtrack in 1:8
            step = s_new .- s
            model_decrease = dot(residual, step)
            if model_decrease < -1e-12 * norm(step)^2 || backtrack == 8
                break
            end
            alpha *= 0.5
            s_new = s .+ alpha .* delta
        end
        
        if norm(s_new .- s) < tol_inner * max(norm(s), 1e-10)
            s = s_new
            break
        end
        s = s_new
    end
    
    return xk .+ s
end

# ============================================================================
# PART 4: NHOTA Algorithm
# ============================================================================

function NHOTA(x0::Vector{Float64}, A::Matrix{Float64}, b::Vector{Float64},
               lambda::Float64, p::Int, u_val::Float64;
               M0::Float64=1.0, M_min::Float64=1e-4, max_iter::Int=100, epsilon::Float64=1e-4)

    d = length(x0)
    xk = copy(x0)
    Mk = M0
    Rk = full_objective(xk, A, b, lambda)

    history = Dict(
        "obj_values" => Float64[],
        "ref_values" => Float64[],
        "stationarity" => Float64[],
        "iter_times" => Float64[],
        "M_values" => Float64[],
        "inner_iters" => Int[],
        "iterations" => 0,
        "final_x" => copy(x0)
    )

    f_x0 = full_objective(xk, A, b, lambda)
    stat_x0 = compute_stationarity_measure(xk, A, b, lambda)
    push!(history["obj_values"], f_x0)
    push!(history["ref_values"], Rk)
    push!(history["stationarity"], stat_x0)
    push!(history["iter_times"], 0.0)   
    push!(history["M_values"], Mk)
    push!(history["inner_iters"], 0)

    total_time = 0.0

    for k in 1:max_iter
        iter_start = time()

        i = 0
        x_found = false
        x_next = copy(xk)

        grad = gradient_F(xk, A, b, lambda)
        
        # p=2 and p=3 are now both fully matrix-free operators
        H_action = nothing
        if p == 3
            H_action = make_hessian_action(xk, A, b, lambda)
        end

        while !x_found && i < 50
            Mi = 2.0^i * Mk

            if p == 1
                x_next = solve_subproblem_p1(xk, grad, lambda, Mi)
            elseif p == 2
                x_next = solve_subproblem_p2(xk, grad, lambda, Mi, A, b)
            elseif p == 3
                x_next = solve_subproblem_p3(xk, grad, H_action, lambda, Mi, A, b;
                                             max_inner=50, tol_inner=1e-8)
            else
                error("p must be 1, 2, or 3")
            end

            f_next = full_objective(x_next, A, b, lambda)
            diff_x = x_next .- xk
            norm_diff = norm(diff_x)
            regularization = Mi / factorial(p + 1) * norm_diff^(p + 1)

            if f_next <= Rk - regularization
                x_found = true
                break
            end

            i += 1
        end

        if !x_found
            println("WARNING: Inner loop failed at iter $k, using p=1 fallback")
            x_next = solve_subproblem_p1(xk, grad, lambda, max(Mk, 1.0))
        end

        xk = copy(x_next)
        Mk = max(Mk / 2.0, M_min)

        f_xk = full_objective(xk, A, b, lambda)
        Rk = (1.0 - u_val) * Rk + u_val * f_xk

        iter_time = time() - iter_start
        total_time += iter_time

        stat_measure = compute_stationarity_measure(xk, A, b, lambda)

        push!(history["obj_values"], f_xk)
        push!(history["ref_values"], Rk)
        push!(history["stationarity"], stat_measure)
        push!(history["iter_times"], total_time)
        push!(history["M_values"], Mk)
        push!(history["inner_iters"], i)

        history["iterations"] = k
        history["final_x"] = copy(xk)

        if k % 50 == 0 || k <= 5
            @printf("p=%d, u=%.2f, iter=%3d: f=%.6e, S_f=%.6e, M=%.4f, inner=%d\n",
                    p, u_val, k, f_xk, stat_measure, Mk, i)
        end

        if stat_measure < epsilon
            @printf("p=%d, u=%.2f: CONVERGED at iter=%d, S_f=%.6e\n", p, u_val, k, stat_measure)
            break
        end
    end

    return history
end

# ============================================================================
# PART 5: Baseline - Monotone PG
# ============================================================================

function proximal_gradient_baseline(x0::Vector{Float64}, A::Matrix{Float64}, b::Vector{Float64},
                                    lambda::Float64; 
                                    M0::Float64=1.0, M_min::Float64=1e-4,
                                    max_iter::Int=500, epsilon::Float64=1e-6)

    d = length(x0)
    xk = copy(x0)
    Mk = M0
    f_prev = full_objective(xk, A, b, lambda)

    history = Dict(
        "obj_values" => Float64[],
        "stationarity" => Float64[],
        "iter_times" => Float64[],
        "M_values" => Float64[],
        "inner_iters" => Int[],
        "iterations" => 0,
        "final_x" => copy(x0)
    )
    f_x0 = full_objective(xk, A, b, lambda)
    stat_x0 = compute_stationarity_measure(xk, A, b, lambda)
    push!(history["obj_values"], f_x0)
    push!(history["stationarity"], stat_x0)
    push!(history["iter_times"], 0.0)   
    push!(history["M_values"], Mk)
    push!(history["inner_iters"], 0)

    total_time = 0.0

    for k in 1:max_iter
        iter_start = time()

        i = 0
        x_found = false
        x_next = copy(xk)

        grad = gradient_F(xk, A, b, lambda)

        while !x_found && i < 200
            Mi = 2.0^i * Mk
            x_next = solve_subproblem_p1(xk, grad, lambda, Mi)

            f_next = full_objective(x_next, A, b, lambda)
            diff_x = x_next .- xk
            norm_diff = norm(diff_x)
            regularization = Mi / factorial(2) * norm_diff^2

            if f_next <= f_prev - regularization
                x_found = true
                break
            end

            i += 1
        end

        if !x_found
            println("WARNING: Baseline inner loop failed at iter $k")
            x_next = solve_subproblem_p1(xk, grad, lambda, max(Mk, 1.0))
        end

        xk = copy(x_next)
        Mk = max(Mk / 2.0, M_min)

        f_xk = full_objective(xk, A, b, lambda)
        f_prev = f_xk

        iter_time = time() - iter_start
        total_time += iter_time

        stat_measure = compute_stationarity_measure(xk, A, b, lambda)

        push!(history["obj_values"], f_xk)
        push!(history["stationarity"], stat_measure)
        push!(history["iter_times"], total_time)
        push!(history["M_values"], Mk)
        push!(history["inner_iters"], i)

        history["iterations"] = k
        history["final_x"] = copy(xk)

        if k % 50 == 0 || k <= 5
            @printf("Baseline, iter=%3d: f=%.6e, S_f=%.6e, M=%.4f, inner=%d\n", 
                    k, f_xk, stat_measure, Mk, i)
        end

        if stat_measure < epsilon
            @printf("Baseline: CONVERGED at iter=%d, S_f=%.6e\n", k, stat_measure)
            break
        end
    end

    return history
end

# ============================================================================
# PART 6: Experiments
# ============================================================================

function run_experiments()
    println("="^70)
    println("NHOTA: Logistic Regression with NONCONVEX Regularization")
    println("Dataset: $(basename(DATASET_URL)) (LIBSVM)") 
    println("Regularizer: h(x) = lambda * sum_i x_i^2/(1+x_i^2)")
    println("="^70)

    A, y = download_dataset()
    n, d = size(A)

    lambda = 0.1
    epsilon = 1e-5
    max_iter = 100

    rng = MersenneTwister(42)
    x0 = rand(rng, d) * 0.01

    println("\n--- Problem parameters ---")
    println("Nonconvex regularizer: lambda=$lambda")
    println("epsilon=$epsilon, max_iter=$max_iter")
    println("ALL methods use SAME dataset and SAME initialization")

    results = Dict()

    println("\n" * "="^70)
    println("Running: Baseline PG (p=1, monotone)")
    println("="^70)
    results["Baseline_PG"] = proximal_gradient_baseline(copy(x0), A, y, lambda;
                                                        M0=1.0, M_min=1e-4, max_iter=max_iter, epsilon=epsilon)

    println("\n" * "="^70)
    println("Running: NHOTA p=1, u=0.5")
    println("="^70)
    results["NHOTA_p1_u5"] = NHOTA(copy(x0), A, y, lambda, 1, 0.5;
                                   M0=1.0, M_min=1e-4, max_iter=max_iter, epsilon=epsilon)

    for u in [0.5, 1.0]
        println("\n" * "="^70)
        println("Running: NHOTA p=2, u=$u")
        println("="^70)
        results["NHOTA_p2_u$(Int(u*10))"] = NHOTA(copy(x0), A, y, lambda, 2, u;
                                                   M0=1.0, M_min=1e-4, max_iter=max_iter, epsilon=epsilon)
    end

    for u in [0.5, 1.0]
        println("\n" * "="^70)
        println("Running: NHOTA p=3, u=$u")
        println("="^70)
        results["NHOTA_p3_u$(Int(u*10))"] = NHOTA(copy(x0), A, y, lambda, 3, u;
                                                   M0=1.0, M_min=1e-4, max_iter=max_iter, epsilon=epsilon)
    end

    println("\n" * "="^70)
    println("SUMMARY OF RESULTS")
    println("="^70)
    for (name, hist) in results
        iters = hist["iterations"]
        final_obj = hist["obj_values"][end]
        final_stat = hist["stationarity"][end]
        total_time = hist["iter_times"][end]
        @printf("%-25s: iter=%3d, f=%.6e, S_f=%.6e, time=%.3fs\n",
                name, iters, final_obj, final_stat, total_time)
    end

    return results
end

# ============================================================================
# PART 7: Clean Plotting with Transparency
# ============================================================================

function plot_comparison(results::Dict; save_prefix::String="figures/nhota_dataset")

    mkpath(dirname(save_prefix))

    default(
        size=(900, 500),
        dpi=300,
        framestyle=:box,
        legendfontsize=9,
        titlefontsize=11,
        guidefontsize=10,
        tickfontsize=9
    )

    styles = Dict(
        "NHOTA_p3_u10"       => (color=RGB(0.0, 0.0, 0.0),       alpha=1.0, marker=:none,     lw=1.0),
        "NHOTA_p1_u5"        => (color=RGB(0.2, 0.6, 0.8),       alpha=0.3, marker=:star5,    lw=3.0),
        "NHOTA_p2_u5"        => (color=RGB(0.9, 0.4, 0.1),       alpha=0.3, marker=:star5,    lw=3.0),
        "NHOTA_p2_u10"       => (color=RGB(0.8, 0.2, 0.2),       alpha=1.0, marker=:none,     lw=1.0),
        "NHOTA_p3_u5"        => (color=RGB(0.2, 0.7, 0.3),       alpha=0.3, marker=:star5,    lw=3.0),
        "Baseline_PG"       => (color=RGB(0.5, 0.2, 0.7),        alpha=1.0, marker=:none,     lw=1.0)
    )

    safe(x) = max.(x, 1e-12)

    function get_obj_range(results)
        all_vals = Float64[]
        for name in ["Baseline_PG", "NHOTA_p1_u5", "NHOTA_p2_u10", "NHOTA_p2_u5", "NHOTA_p3_u10", "NHOTA_p3_u5"]
            hist = results[name]
            append!(all_vals, hist["obj_values"])
        end
        mn, mx = minimum(all_vals), maximum(all_vals)
        pad = (mx - mn) * 0.1
        return (mn - pad, mx + pad)
    end

    yrange = get_obj_range(results)

    p1 = plot(title="Objective vs Iterations", xlabel="Iteration", ylabel="f(x)", 
              legend=:topright, grid=true, ylims=yrange)
    for name in ["Baseline_PG", "NHOTA_p1_u5", "NHOTA_p2_u10", "NHOTA_p2_u5", "NHOTA_p3_u10", "NHOTA_p3_u5"]
        hist = results[name]
        st = get(styles, name, (color=:gray, alpha=1.0, marker=:none, lw=2.0))
        iters = 0:hist["iterations"]
        obj = hist["obj_values"]
        plot!(p1, iters, obj, label=name,
              color=st.color, alpha=st.alpha,
              marker=st.marker, markersize=4, markercolor=st.color,
              linewidth=st.lw, markerstrokewidth=0)
    end
    savefig(p1, "$(save_prefix)_obj_vs_iter.png")

    p2 = plot(title="Stationarity vs Iterations", xlabel="Iteration", ylabel="||grad f(x)||", 
              yscale=:log10, legend=:topright, grid=true)
    for name in ["Baseline_PG", "NHOTA_p1_u5", "NHOTA_p2_u10", "NHOTA_p2_u5", "NHOTA_p3_u10", "NHOTA_p3_u5"]
        hist = results[name]
        st = get(styles, name, (color=:gray, alpha=1.0, marker=:none, lw=2.0))
        iters = 0:hist["iterations"]
        stat = safe(hist["stationarity"])
        plot!(p2, iters, stat, label=name,
              color=st.color, alpha=st.alpha,
              marker=st.marker, markersize=4, markercolor=st.color,
              linewidth=st.lw, markerstrokewidth=0)
    end
    savefig(p2, "$(save_prefix)_stat_vs_iter.png")

    p3 = plot(title="Objective vs CPU Time", xlabel="Time (s)", ylabel="f(x)", 
              legend=:topright, grid=true, ylims=yrange)
    for name in ["Baseline_PG", "NHOTA_p1_u5", "NHOTA_p2_u10", "NHOTA_p2_u5", "NHOTA_p3_u10", "NHOTA_p3_u5"]
        hist = results[name]
        st = get(styles, name, (color=:gray, alpha=1.0, marker=:none, lw=2.0))
        times = hist["iter_times"]
        obj = hist["obj_values"]
        plot!(p3, times, obj, label=name,
              color=st.color, alpha=st.alpha,
              marker=st.marker, markersize=4, markercolor=st.color,
              linewidth=st.lw, markerstrokewidth=0)
    end
    savefig(p3, "$(save_prefix)_obj_vs_time.png")

    p4 = plot(title="Stationarity vs CPU Time", xlabel="Time (s)", ylabel="||grad f(x)||", 
              yscale=:log10, legend=:topright, grid=true)
    for name in ["Baseline_PG", "NHOTA_p1_u5", "NHOTA_p2_u10", "NHOTA_p2_u5", "NHOTA_p3_u10", "NHOTA_p3_u5"]
        hist = results[name]
        st = get(styles, name, (color=:gray, alpha=1.0, marker=:none, lw=2.0))
        times = hist["iter_times"]
        stat = safe(hist["stationarity"])
        plot!(p4, times, stat, label=name,
              color=st.color, alpha=st.alpha,
              marker=st.marker, markersize=4, markercolor=st.color,
              linewidth=st.lw, markerstrokewidth=0)
    end
    savefig(p4, "$(save_prefix)_stat_vs_time.png")

    println("\nPlots saved to $(save_prefix)*")
end

# ============================================================================
# MAIN Execution
# ============================================================================

results = run_experiments()
plot_comparison(results)