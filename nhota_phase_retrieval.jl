# ============================================================================
# NHOTA - Nonmonotone Higher-Order Taylor Approximation
# Phase Retrieval with Quartic Objective + Elastic Net
# ============================================================================
# Objective: f(x) = sum_i (|a_i^T x|^2 - b_i)^2 + lambda_1 ||x||_1 + (lambda_2/2)||x||^2
# This is a DEGREE-4 polynomial in x, so p=3 is the first meaningful approximation.
# ============================================================================
# Methods:
#   - Baseline PG (p=1, monotone, same framework as NHOTA)
#   - NHOTA p=1, u=0.5
#   - NHOTA p=2, u=0.5 and u=1.0
#   - NHOTA p=3, u=0.5 and u=1.0
# ============================================================================
# ALL methods use the same acceptance condition:
#   f(x_{k+1}) <= reference - (M/(p+1)!) * ||x_{k+1} - x_k||^{p+1}
# where reference = f(x_k) for baseline, reference = R_k for NHOTA
# ============================================================================

using LinearAlgebra
using Statistics
using Random
using Printf
using Plots
pgfplotsx()

# ============================================================================
# PART 1: Phase Retrieval Data Generation
# ============================================================================
# We observe b_i = |a_i^T x_true|^2 + noise (no phase information!)
# Goal: recover x_true from (a_i, b_i) pairs

function generate_phase_retrieval_data(n::Int, d::Int; noise_std::Float64=0.01, seed::Int=42)
    rng = MersenneTwister(seed)

    # True signal: sparse and real-valued
    x_true = zeros(d)
    active_idx = randperm(rng, d)[1:min(10, d)]
    x_true[active_idx] = randn(rng, length(active_idx))
    x_true ./= norm(x_true)  # Normalize

    # Gaussian measurement vectors
    A = randn(rng, n, d)
    # Normalize rows
    for i in 1:n
        A[i, :] ./= norm(A[i, :])
    end

    # Observations: intensity measurements (no phase!)
    b = [(dot(A[i, :], x_true))^2 for i in 1:n]

    # Add small noise
    b .+= noise_std * randn(rng, n)
    b = max.(b, 1e-8)  # Ensure positive

    return A, b, x_true
end

# ============================================================================
# PART 2: Quartic Phase Retrieval Objective
# ============================================================================
# f(x) = (1/n) * sum_i (|a_i^T x|^2 - b_i)^2
# This is a degree-4 polynomial in x!

function phase_retrieval_loss(x::Vector{Float64}, A::Matrix{Float64}, b::Vector{Float64})
    n = size(A, 1)
    loss = 0.0
    for i in 1:n
        ai_x = dot(A[i, :], x)
        residual = ai_x^2 - b[i]
        loss += residual^2
    end
    return loss / n
end

# Gradient: nabla f(x) = (4/n) * sum_i (|a_i^T x|^2 - b_i) * (a_i^T x) * a_i
function gradient_F(x::Vector{Float64}, A::Matrix{Float64}, b::Vector{Float64})
    n, d = size(A)
    grad = zeros(d)
    for i in 1:n
        ai_x = dot(A[i, :], x)
        residual = ai_x^2 - b[i]
        coeff = 4.0 * residual * ai_x
        grad .+= coeff .* A[i, :]
    end
    return grad / n
end

# Hessian: nabla^2 f(x) = (4/n) * sum_i [(3*(a_i^T x)^2 - b_i) * a_i a_i^T]
function hessian_F(x::Vector{Float64}, A::Matrix{Float64}, b::Vector{Float64})
    n, d = size(A)
    H = zeros(d, d)
    for i in 1:n
        ai_x = dot(A[i, :], x)
        coeff = 4.0 * (3.0 * ai_x^2 - b[i])
        H .+= coeff .* (A[i, :] * A[i, :]')
    end
    return H / n
end

# For p=3, we need the tensor-vector product applied twice: T[s,s,:] where T is the 3rd derivative
# The 3rd derivative tensor at x applied to directions s1, s2, s3 is:
# T[s1,s2,s3] = (24/n) * sum_i (a_i^T x) * (a_i^T s1) * (a_i^T s2) * (a_i^T s3)
# For the subproblem, we need the vector T[s,s,:] which we can write as a matrix-vector product
function third_derivative_tensor_action(x::Vector{Float64}, A::Matrix{Float64}, b::Vector{Float64}, s::Vector{Float64})
    n, d = size(A)
    result = zeros(d)
    for i in 1:n
        ai_x = dot(A[i, :], x)
        ai_s = dot(A[i, :], s)
        coeff = 24.0 * ai_x * ai_s^2
        result .+= coeff .* A[i, :]
    end
    return result / n
end

# ============================================================================
# PART 3: Elastic Net Regularization (L1 + L2)
# ============================================================================

function h_elastic_net(x::Vector{Float64}, lambda_l1::Float64, lambda_l2::Float64)
    return lambda_l1 * sum(abs.(x)) + 0.5 * lambda_l2 * sum(x.^2)
end

function prox_elastic_net(x::Vector{Float64}, gamma_l1::Float64, gamma_l2::Float64)
    return (1.0 / (1.0 + gamma_l2)) .* prox_l1(x, gamma_l1 / (1.0 + gamma_l2))
end

function prox_l1(x::Vector{Float64}, gamma::Float64)
    return sign.(x) .* max.(abs.(x) .- gamma, 0.0)
end

function full_objective(x::Vector{Float64}, A::Matrix{Float64}, b::Vector{Float64}, 
                        lambda_l1::Float64, lambda_l2::Float64)
    return phase_retrieval_loss(x, A, b) + h_elastic_net(x, lambda_l1, lambda_l2)
end

function compute_stationarity_measure(x::Vector{Float64}, A::Matrix{Float64}, b::Vector{Float64}, 
                                      lambda_l1::Float64, lambda_l2::Float64)
    grad = gradient_F(x, A, b) + lambda_l2 * x
    x_plus = prox_elastic_net(x .- grad, lambda_l1, lambda_l2)
    return norm(x .- x_plus)
end

# ============================================================================
# PART 4: Subproblem Solvers
# ============================================================================

function solve_subproblem_p1(xk::Vector{Float64}, grad_F::Vector{Float64}, 
                              lambda_l1::Float64, lambda_l2::Float64, M::Float64)
    grad_total = grad_F + lambda_l2 * xk
    return prox_elastic_net(xk .- grad_total ./ M, lambda_l1 / M, lambda_l2 / M)
end

function solve_subproblem_p2(xk::Vector{Float64}, grad_F::Vector{Float64}, 
                              H_F::Matrix{Float64}, lambda_l1::Float64, lambda_l2::Float64, M::Float64)
    d = length(xk)
    H_total = H_F + lambda_l2 * I
    grad_total = grad_F + lambda_l2 * xk

    reg = M / 2.0
    H_reg = H_total + reg * I
    H_reg = (H_reg + H_reg') / 2

    s = zeros(d)
    try
        F_chol = cholesky(Hermitian(H_reg))
        s = F_chol \ (-grad_total)
    catch
        H_reg += 0.1 * I
        s = H_reg \ (-grad_total)
    end

    return prox_l1(xk + s, lambda_l1 / M)
end

# For p=3, we use a more sophisticated approach:
# The cubic model includes: grad^T s + 0.5 s^T H s + (1/6) T[s,s,s]
# We approximate this with an iterative proximal gradient step on the cubic model
function solve_subproblem_p3(xk::Vector{Float64}, grad_F::Vector{Float64}, 
                              H_F::Matrix{Float64}, A::Matrix{Float64}, b::Vector{Float64},
                              lambda_l1::Float64, lambda_l2::Float64, M::Float64;
                              max_inner::Int=20, tol_inner::Float64=1e-6)
    d = length(xk)

    # Start from p=2 solution as warm start
    H_total = H_F + lambda_l2 * I
    grad_total = grad_F + lambda_l2 * xk
    reg = M / 6.0
    H_reg = H_total + reg * I
    H_reg = (H_reg + H_reg') / 2

    s = zeros(d)
    try
        F_chol = cholesky(Hermitian(H_reg))
        s = F_chol \ (-grad_total)
    catch
        H_reg += 0.1 * I
        s = H_reg \ (-grad_total)
    end

    # Refine with few proximal gradient steps on the cubic model
    gamma_l1 = lambda_l1 / M
    gamma_l2 = lambda_l2 / M

    for inner in 1:max_inner
        grad_cubic = grad_total + H_total * s + third_derivative_tensor_action(xk, A, b, s)
        s_new = prox_elastic_net(s .- grad_cubic ./ M, gamma_l1, gamma_l2) .- xk
        if norm(s_new - s) < tol_inner * max(norm(s), 1e-10)
            s = s_new
            break
        end
        s = s_new
    end

    return prox_l1(xk + s, lambda_l1 / M)
end

# ============================================================================
# PART 5: NHOTA Algorithm
# ============================================================================

function NHOTA(x0::Vector{Float64}, A::Matrix{Float64}, b::Vector{Float64},
               lambda_l1::Float64, lambda_l2::Float64, p::Int, u_val::Float64;
               M0::Float64=1.0, M_min::Float64=1e-4, max_iter::Int=100, epsilon::Float64=1e-4)

    d = length(x0)
    xk = copy(x0)
    Mk = M0

    # Reference value R(x_k)
    Rk = full_objective(xk, A, b, lambda_l1, lambda_l2)

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

    total_time = 0.0

    for k in 1:max_iter
        iter_start = time()

        i = 0
        x_found = false
        x_next = copy(xk)

        grad = gradient_F(xk, A, b)

        H = nothing
        if p >= 2
            H = hessian_F(xk, A, b)
        end

        while !x_found && i < 50
            Mi = 2.0^i * Mk

            if p == 1
                x_next = solve_subproblem_p1(xk, grad, lambda_l1, lambda_l2, Mi)
            elseif p == 2
                x_next = solve_subproblem_p2(xk, grad, H, lambda_l1, lambda_l2, Mi)
            elseif p == 3
                x_next = solve_subproblem_p3(xk, grad, H, A, b, lambda_l1, lambda_l2, Mi)
            else
                error("p must be 1, 2, or 3")
            end

            # CORRECT ACCEPTANCE CONDITION (Step 5)
            f_next = full_objective(x_next, A, b, lambda_l1, lambda_l2)
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
            x_next = solve_subproblem_p1(xk, grad, lambda_l1, lambda_l2, max(Mk, 1.0))
        end

        xk = copy(x_next)
        Mk = max(Mk / 2.0, M_min)

        f_xk = full_objective(xk, A, b, lambda_l1, lambda_l2)
        Rk = (1.0 - u_val) * Rk + u_val * f_xk

        iter_time = time() - iter_start
        total_time += iter_time

        stat_measure = compute_stationarity_measure(xk, A, b, lambda_l1, lambda_l2)

        push!(history["obj_values"], f_xk)
        push!(history["ref_values"], Rk)
        push!(history["stationarity"], stat_measure)
        push!(history["iter_times"], total_time)
        push!(history["M_values"], Mk)
        push!(history["inner_iters"], i)

        history["iterations"] = k
        history["final_x"] = copy(xk)

        if k % 50 == 0 || k <= 5
            @printf("p=%d, u=%.2f, iter=%3d: f=%.6e, S_f=%.6e, M=%.4f, inner=%d
",
                    p, u_val, k, f_xk, stat_measure, Mk, i)
        end

        if stat_measure < epsilon
            @printf("p=%d, u=%.2f: CONVERGED at iter=%d, S_f=%.6e
", p, u_val, k, stat_measure)
            break
        end
    end

    return history
end

# ============================================================================
# PART 6: Baseline - SAME framework as NHOTA but monotone (u=1.0)
# ============================================================================

function proximal_gradient_baseline(x0::Vector{Float64}, A::Matrix{Float64}, b::Vector{Float64},
                                    lambda_l1::Float64, lambda_l2::Float64; 
                                    M0::Float64=1.0, M_min::Float64=1e-4,
                                    max_iter::Int=500, epsilon::Float64=1e-6)

    d = length(x0)
    xk = copy(x0)
    Mk = M0

    # Monotone reference: just f(x_k)
    f_prev = full_objective(xk, A, b, lambda_l1, lambda_l2)

    history = Dict(
        "obj_values" => Float64[],
        "stationarity" => Float64[],
        "iter_times" => Float64[],
        "M_values" => Float64[],
        "inner_iters" => Int[],
        "iterations" => 0,
        "final_x" => copy(x0)
    )

    total_time = 0.0

    for k in 1:max_iter
        iter_start = time()

        i = 0
        x_found = false
        x_next = copy(xk)

        grad = gradient_F(xk, A, b)

        while !x_found && i < 50
            Mi = 2.0^i * Mk

            # p=1 subproblem (same as NHOTA p=1)
            x_next = solve_subproblem_p1(xk, grad, lambda_l1, lambda_l2, Mi)

            # SAME acceptance condition as NHOTA, but with monotone reference
            f_next = full_objective(x_next, A, b, lambda_l1, lambda_l2)
            diff_x = x_next .- xk
            norm_diff = norm(diff_x)
            regularization = Mi / factorial(2) * norm_diff^2  # p=1: (p+1)! = 2! = 2

            if f_next <= f_prev - regularization
                x_found = true
                break
            end

            i += 1
        end

        if !x_found
            println("WARNING: Baseline inner loop failed at iter $k")
            x_next = solve_subproblem_p1(xk, grad, lambda_l1, lambda_l2, max(Mk, 1.0))
        end

        xk = copy(x_next)
        Mk = max(Mk / 2.0, M_min)

        f_xk = full_objective(xk, A, b, lambda_l1, lambda_l2)
        f_prev = f_xk  # Monotone: reference is just current f value

        iter_time = time() - iter_start
        total_time += iter_time

        stat_measure = compute_stationarity_measure(xk, A, b, lambda_l1, lambda_l2)

        push!(history["obj_values"], f_xk)
        push!(history["stationarity"], stat_measure)
        push!(history["iter_times"], total_time)
        push!(history["M_values"], Mk)
        push!(history["inner_iters"], i)

        history["iterations"] = k
        history["final_x"] = copy(xk)

        if k % 50 == 0 || k <= 5
            @printf("Baseline, iter=%3d: f=%.6e, S_f=%.6e, M=%.4f, inner=%d
", 
                    k, f_xk, stat_measure, Mk, i)
        end

        if stat_measure < epsilon
            @printf("Baseline: CONVERGED at iter=%d, S_f=%.6e
", k, stat_measure)
            break
        end
    end

    return history
end

# ============================================================================
# PART 7: Experiments
# ============================================================================

function run_experiments()
    println("="^70)
    println("NHOTA: Phase Retrieval (Quartic Objective + Elastic Net)")
    println("="^70)

    # Phase retrieval parameters - MORE MEASUREMENTS for better conditioning
    n = 800
    d = 30

    println("
--- Generating phase retrieval data ---")
    A, b, x_true = generate_phase_retrieval_data(n, d; noise_std=0.01, seed=42)
    println("n=$n measurements, d=$d dimension")
    println("True signal sparsity: $(sum(abs.(x_true) .> 1e-6))")
    println("Oversampling ratio: n/d = $(n/d)")

    # Smaller dataset for p=3 (expensive)
    n_small = 400
    A_small, b_small, x_true_small = generate_phase_retrieval_data(n_small, d; noise_std=0.01, seed=123)
    println("Small dataset: n=$n_small, d=$d")

    lambda_l1 = 0.01
    lambda_l2 = 0.0001
    epsilon = 1e-4
    max_iter = 200

    rng = MersenneTwister(42)
    x0 = randn(rng, d) * 0.5
    x0 ./= norm(x0)
    x0_small = copy(x0)

    println("
--- Problem parameters ---")
    println("L1=$lambda_l1, L2=$lambda_l2, epsilon=$epsilon")
    println("Max iterations: $max_iter")
    println("Objective is QUARTIC (degree 4) in x")

    results = Dict()

    println("
" * "="^70)
    println("Running: Baseline PG (p=1, monotone, same framework as NHOTA)")
    println("="^70)
    results["Baseline_PG"] = proximal_gradient_baseline(x0, A, b, lambda_l1, lambda_l2;
                                                        M0=1.0, M_min=1e-4, max_iter=max_iter, epsilon=epsilon)

    println("
" * "="^70)
    println("Running: NHOTA p=1, u=0.5")
    println("="^70)
    results["NHOTA_p1_u5"] = NHOTA(x0, A, b, lambda_l1, lambda_l2, 1, 0.5;
                                      M0=1.0, M_min=1e-4, max_iter=max_iter, epsilon=epsilon)

    for u in [0.5, 1.0]
        println("
" * "="^70)
        println("Running: NHOTA p=2, u=$u")
        println("="^70)
        results["NHOTA_p2_u$(Int(u*10))"] = NHOTA(x0, A, b, lambda_l1, lambda_l2, 2, u;
                                                    M0=1.0, M_min=1e-4, max_iter=max_iter, epsilon=epsilon)
    end

    for u in [0.5, 1.0]
        println("
" * "="^70)
        println("Running: NHOTA p=3, u=$u (small dataset)")
        println("="^70)
        results["NHOTA_p3_u$(Int(u*10))"] = NHOTA(x0_small, A_small, b_small, lambda_l1, lambda_l2, 3, u;
                                                    M0=1.0, M_min=1e-4, max_iter=max_iter, epsilon=epsilon)
    end

    println("
" * "="^70)
    println("Running: NHOTA p=2, u=0.5 (small dataset, for fair p=3 comparison)")
    println("="^70)
    results["NHOTA_p2_small_u5"] = NHOTA(x0_small, A_small, b_small, lambda_l1, lambda_l2, 2, 0.5;
                                          M0=1.0, M_min=1e-4, max_iter=max_iter, epsilon=epsilon)

    println("
" * "="^70)
    println("SUMMARY OF RESULTS")
    println("="^70)
    for (name, hist) in results
        iters = hist["iterations"]
        final_obj = hist["obj_values"][end]
        final_stat = hist["stationarity"][end]
        total_time = hist["iter_times"][end]
        @printf("%-25s: iter=%3d, f=%.6e, S_f=%.6e, time=%.3fs
",
                name, iters, final_obj, final_stat, total_time)
    end

    println("
--- Recovery Quality (relative error to true signal) ---")
    for (name, hist) in results
        x_recovered = hist["final_x"]
        err1 = norm(x_recovered - x_true) / norm(x_true)
        err2 = norm(x_recovered + x_true) / norm(x_true)
        rel_err = min(err1, err2)
        println("$name: relative error = $(@sprintf("%.4f", rel_err))")
    end

    return results
end

# ============================================================================
# PART 8: Plotting
# ============================================================================

function plot_comparison(results::Dict; save_prefix::String="figures/nhota_phase_retrieval")

    mkpath(dirname(save_prefix))

    default(
        size=(950, 520),
        dpi=300,
        framestyle=:box,
        fontfamily="Computer Modern"
    )

    colors = Dict(
        "Baseline_PG"       => :black,
        "NHOTA_p1_u5"        => :cyan,
        "NHOTA_p2_u5"        => :orange,
        "NHOTA_p2_u10"       => :darkred,
        "NHOTA_p3_u5"        => :green,
        "NHOTA_p3_u10"       => :blue,
        "NHOTA_p2_small_u5"  => :purple
    )

    linestyles = Dict(
        "Baseline_PG"       => :solid,
        "NHOTA_p1_u5"       => :dot,
        "NHOTA_p2_u5"       => :dot,
        "NHOTA_p2_u10"      => :dashdot,
        "NHOTA_p3_u5"       => :solid,
        "NHOTA_p3_u10"      => :dashdot,
        "NHOTA_p2_small_u5" => :dashdot
    )

    lw = 2.5
    safe(x) = max.(x, 1e-12)

    p1 = plot(title="Objective vs Iterations", xlabel="Iteration", ylabel="f(x)", 
              yscale=:log10, legend=:outerright, grid=true)
    for (name, hist) in results
        plot!(p1, 1:hist["iterations"], safe(hist["obj_values"]), label=name,
              color=get(colors, name, :gray), linestyle=get(linestyles, name, :solid), linewidth=lw)
    end
    savefig(p1, "$(save_prefix)_obj_vs_iter.pdf")

    p2 = plot(title="Stationarity vs Iterations", xlabel="Iteration", ylabel="S_f(x)", 
              yscale=:log10, legend=:outerright, grid=true)
    for (name, hist) in results
        plot!(p2, 1:hist["iterations"], safe(hist["stationarity"]), label=name,
              color=get(colors, name, :gray), linestyle=get(linestyles, name, :solid), linewidth=lw)
    end
    savefig(p2, "$(save_prefix)_stat_vs_iter.pdf")

    p3 = plot(title="Objective vs CPU Time", xlabel="Time (s)", ylabel="f(x)", 
              yscale=:log10, legend=:outerright, grid=true)
    for (name, hist) in results
        plot!(p3, hist["iter_times"], safe(hist["obj_values"]), label=name,
              color=get(colors, name, :gray), linestyle=get(linestyles, name, :solid), linewidth=lw)
    end
    savefig(p3, "$(save_prefix)_obj_vs_time.pdf")

    p4 = plot(title="Stationarity vs CPU Time", xlabel="Time (s)", ylabel="S_f(x)", 
              yscale=:log10, legend=:outerright, grid=true)
    for (name, hist) in results
        plot!(p4, hist["iter_times"], safe(hist["stationarity"]), label=name,
              color=get(colors, name, :gray), linestyle=get(linestyles, name, :solid), linewidth=lw)
    end
    savefig(p4, "$(save_prefix)_stat_vs_time.pdf")

    p5 = plot(title="Nonmonotonicity (NHOTA p=2)", xlabel="Iteration", ylabel="f(x)", 
              legend=:outerright, grid=true)
    for u_label in ["NHOTA_p2_u5", "NHOTA_p2_u10"]
        if haskey(results, u_label)
            hist = results[u_label]
            plot!(p5, 1:hist["iterations"], hist["obj_values"], label="$u_label: f(x_k)", linewidth=lw)
            plot!(p5, 1:hist["iterations"], hist["ref_values"], label="$u_label: R_k", 
                  linestyle=:dash, linewidth=1.8)
        end
    end
    savefig(p5, "$(save_prefix)_nonmonotonicity.pdf")

    p6 = plot(title="Order Comparison (Stationarity vs Iterations)", xlabel="Iteration", 
              ylabel="Stationarity", yscale=:log10, legend=:outerright, grid=true)
    for (name, hist) in results
        plot!(p6, 1:hist["iterations"], safe(hist["stationarity"]), label=name,
              color=get(colors, name, :gray), linestyle=get(linestyles, name, :solid), linewidth=lw)
    end
    savefig(p6, "$(save_prefix)_order_comparison.pdf")

    println("
Plots saved to $(save_prefix)*")
end

# ============================================================================
# MAIN
# ============================================================================

results = run_experiments()
plot_comparison(results)