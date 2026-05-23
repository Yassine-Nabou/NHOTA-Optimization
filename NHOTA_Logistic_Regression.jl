# ============================================================================
# NHOTA - Nonmonotone Higher-Order Taylor Approximation
# Logistic Regression on MNIST with Elastic Net Regularization
# ============================================================================
# Methods:
#   - Baseline PG (p=1, monotone, same condition as NHOTA)
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
# PART 1: MNIST Data Loading
# ============================================================================

function download_mnist()
    base_url = "https://ossci-datasets.s3.amazonaws.com/mnist/"
    files = Dict(
        "train_images" => "train-images-idx3-ubyte.gz",
        "train_labels" => "train-labels-idx1-ubyte.gz",
        "test_images"  => "t10k-images-idx3-ubyte.gz",
        "test_labels"  => "t10k-labels-idx1-ubyte.gz"
    )
    
    data = Dict()
    for (key, filename) in files
        println("Downloading $filename...")
        url = base_url * filename
        gzfile = tempname() * ".gz"
        run(`curl -s -o $gzfile $url`)
        run(`gunzip -f $gzfile`)
        unzipped_file = replace(gzfile, ".gz" => "")
        data[key] = read(unzipped_file)
        rm(unzipped_file)
    end
    
    function parse_images(raw)
        n = div(length(raw) - 16, 784)
        pixels = [Float64(raw[16 + i]) / 255.0 for i in 1:length(raw)-16]
        reshape(pixels, 784, n)'
    end
    
    function parse_labels(raw)
        [Int(raw[8 + i]) for i in 1:length(raw)-8]
    end
    
    X_train = parse_images(data["train_images"])
    y_train = parse_labels(data["train_labels"])
    X_test  = parse_images(data["test_images"])
    y_test  = parse_labels(data["test_labels"])
    
    return X_train, y_train, X_test, y_test
end

function prepare_binary_mnist(X_train, y_train, X_test, y_test, class1, class2; subsample=0)
    idx_train = findall(y -> y == class1 || y == class2, y_train)
    X_tr = X_train[idx_train, :]
    y_tr = [y == class1 ? 1.0 : -1.0 for y in y_train[idx_train]]
    
    idx_test = findall(y -> y == class1 || y == class2, y_test)
    X_te = X_test[idx_test, :]
    y_te = [y == class1 ? 1.0 : -1.0 for y in y_test[idx_test]]
    
    if subsample > 0 && subsample < size(X_tr, 1)
        rng = MersenneTwister(42)
        perm = randperm(rng, size(X_tr, 1))
        X_tr = X_tr[perm[1:subsample], :]
        y_tr = y_tr[perm[1:subsample]]
    end
    
    return X_tr, y_tr, X_te, y_te
end

# ============================================================================
# PART 2: Standard Logistic Loss
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

function gradient_F(x::Vector{Float64}, A::Matrix{Float64}, b::Vector{Float64})
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

function hessian_F(x::Vector{Float64}, A::Matrix{Float64}, b::Vector{Float64})
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

function full_objective(x::Vector{Float64}, A::Matrix{Float64}, Y::Vector{Float64}, 
                        lambda_l1::Float64, lambda_l2::Float64)
    return logistic_loss(x, A, Y) + h_elastic_net(x, lambda_l1, lambda_l2)
end

function compute_stationarity_measure(x::Vector{Float64}, A::Matrix{Float64}, Y::Vector{Float64}, 
                                      lambda_l1::Float64, lambda_l2::Float64)
    grad = gradient_F(x, A, Y) + lambda_l2 * x
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

function solve_subproblem_p3(xk::Vector{Float64}, grad_F::Vector{Float64}, 
                              H_F::Matrix{Float64}, lambda_l1::Float64, lambda_l2::Float64, M::Float64)
    d = length(xk)
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
    
    return prox_l1(xk + s, lambda_l1 / M)
end

# ============================================================================
# PART 5: NHOTA Algorithm
# ============================================================================

function NHOTA(x0::Vector{Float64}, A::Matrix{Float64}, Y::Vector{Float64},
               lambda_l1::Float64, lambda_l2::Float64, p::Int, u_val::Float64;
               M0::Float64=1.0, M_min::Float64=1e-4, max_iter::Int=100, epsilon::Float64=1e-4)
    
    d = length(x0)
    xk = copy(x0)
    Mk = M0
    
    # Reference value R(x_k)
    Rk = full_objective(xk, A, Y, lambda_l1, lambda_l2)
    
    history = Dict(
        "obj_values" => Float64[],
        "ref_values" => Float64[],
        "stationarity" => Float64[],
        "iter_times" => Float64[],
        "M_values" => Float64[],
        "inner_iters" => Int[],
        "iterations" => 0
    )
    
    total_time = 0.0
    
    for k in 1:max_iter
        iter_start = time()
        
        i = 0
        x_found = false
        x_next = copy(xk)
        
        grad = gradient_F(xk, A, Y)
        
        H = nothing
        if p >= 2
            H = hessian_F(xk, A, Y)
        end
        
        while !x_found && i < 50
            Mi = 2.0^i * Mk
            
            if p == 1
                x_next = solve_subproblem_p1(xk, grad, lambda_l1, lambda_l2, Mi)
            elseif p == 2
                x_next = solve_subproblem_p2(xk, grad, H, lambda_l1, lambda_l2, Mi)
            elseif p == 3
                x_next = solve_subproblem_p3(xk, grad, H, lambda_l1, lambda_l2, Mi)
            else
                error("p must be 1, 2, or 3")
            end
            
            # CORRECT ACCEPTANCE CONDITION (Step 5)
            f_next = full_objective(x_next, A, Y, lambda_l1, lambda_l2)
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
        
        f_xk = full_objective(xk, A, Y, lambda_l1, lambda_l2)
        Rk = (1.0 - u_val) * Rk + u_val * f_xk
        
        iter_time = time() - iter_start
        total_time += iter_time
        
        stat_measure = compute_stationarity_measure(xk, A, Y, lambda_l1, lambda_l2)
        
        push!(history["obj_values"], f_xk)
        push!(history["ref_values"], Rk)
        push!(history["stationarity"], stat_measure)
        push!(history["iter_times"], total_time)
        push!(history["M_values"], Mk)
        push!(history["inner_iters"], i)
        
        history["iterations"] = k
        
        if k % 50 == 0 || k <= 5
            @printf("p=%d, u=%.2f, iter=%3d: f=%.6e, S_f=%.6e, M=%.4f, inner=%d\n",
                    p, u_val, k, f_xk, stat_measure, Mk, i)
        end
        
        if stat_measure < epsilon
            @printf("p=%d, u=%.2f: CONVERGED at iter=%d, S_f=%.6e\n", p, u_val, k, stat_measure)
            break
        end
    end
    
    history["final_x"] = xk
    return history
end

# ============================================================================
# PART 6: Baseline - SAME framework as NHOTA but monotone (u=1.0)
# ============================================================================
# The baseline uses the SAME acceptance condition as NHOTA:
#   f(x_{k+1}) <= f(x_k) - (M/(p+1)!) * ||x_{k+1} - x_k||^{p+1}
# But with monotone reference: reference = f(x_k) (not R_k)
# This is equivalent to NHOTA p=1, u=1.0 with reference = f(x_k)

function proximal_gradient_baseline(x0::Vector{Float64}, A::Matrix{Float64}, Y::Vector{Float64},
                                    lambda_l1::Float64, lambda_l2::Float64; 
                                    M0::Float64=1.0, M_min::Float64=1e-4,
                                    max_iter::Int=500, epsilon::Float64=1e-6)
    
    d = length(x0)
    xk = copy(x0)
    Mk = M0
    
    # Monotone reference: just f(x_k)
    f_prev = full_objective(xk, A, Y, lambda_l1, lambda_l2)
    
    history = Dict(
        "obj_values" => Float64[],
        "stationarity" => Float64[],
        "iter_times" => Float64[],
        "M_values" => Float64[],
        "inner_iters" => Int[],
        "iterations" => 0
    )
    
    total_time = 0.0
    
    for k in 1:max_iter
        iter_start = time()
        
        i = 0
        x_found = false
        x_next = copy(xk)
        
        grad = gradient_F(xk, A, Y)
        
        while !x_found && i < 50
            Mi = 2.0^i * Mk
            
            # p=1 subproblem (same as NHOTA p=1)
            x_next = solve_subproblem_p1(xk, grad, lambda_l1, lambda_l2, Mi)
            
            # SAME acceptance condition as NHOTA, but with monotone reference
            f_next = full_objective(x_next, A, Y, lambda_l1, lambda_l2)
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
        
        f_xk = full_objective(xk, A, Y, lambda_l1, lambda_l2)
        f_prev = f_xk  # Monotone: reference is just current f value
        
        iter_time = time() - iter_start
        total_time += iter_time
        
        stat_measure = compute_stationarity_measure(xk, A, Y, lambda_l1, lambda_l2)
        
        push!(history["obj_values"], f_xk)
        push!(history["stationarity"], stat_measure)
        push!(history["iter_times"], total_time)
        push!(history["M_values"], Mk)
        push!(history["inner_iters"], i)
        
        history["iterations"] = k
        
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
# PART 7: Experiments
# ============================================================================

function run_experiments()
    println("="^70)
    println("NHOTA: Logistic Regression on MNIST (Elastic Net)")
    println("="^70)
    
    println("\n--- Loading MNIST data ---")
    X_train, y_train, X_test, y_test = download_mnist()
    println("Full MNIST: train=$(size(X_train)), test=$(size(X_test))")
    
    println("\n--- Preparing binary classification (0 vs 1) ---")
    X_tr, y_tr, X_te, y_te = prepare_binary_mnist(X_train, y_train, X_test, y_test, 0, 1, subsample = 1000)
    println("Binary train: $(size(X_tr)), test: $(size(X_te))")
    println("Class distribution: +1=$(sum(y_tr.==1)), -1=$(sum(y_tr.==-1))")
    
    X_tr_small, y_tr_small, _, _ = prepare_binary_mnist(
        X_train, y_train, X_test, y_test, 0, 1; subsample = 500)
    println("Subsampled (for p=3): $(size(X_tr_small))")
    
    d = size(X_tr, 2)
    lambda_l1 = 0.0001
    lambda_l2 = 0.01
    epsilon = 1e-5
    max_iter = 200
    
    rng = MersenneTwister(42)
    x0 = randn(rng, d) * 2.0
    x0_small = randn(rng, size(X_tr_small, 2)) * 2.0
    
    println("\n--- Problem parameters ---")
    println("Dimension: d=$d")
    println("Training samples (p=1,2): $(size(X_tr, 1))")
    println("Training samples (p=3): $(size(X_tr_small, 1))")
    println("L1=$lambda_l1, L2=$lambda_l2, epsilon=$epsilon")
    
    results = Dict()
    
    println("\n" * "="^70)
    println("Running: Baseline PG (p=1, monotone, same framework as NHOTA)")
    println("="^70)
    results["Baseline_PG"] = proximal_gradient_baseline(x0, X_tr, y_tr, lambda_l1, lambda_l2;
                                                        M0=1.0, M_min=1e-4, max_iter=max_iter, epsilon=epsilon)
    
    println("\n" * "="^70)
    println("Running: NHOTA p=1, u=0.5")
    println("="^70)
    results["NHOTA_p1_u5"] = NHOTA(x0, X_tr, y_tr, lambda_l1, lambda_l2, 1, 0.5;
                                      M0=1.0, M_min=1e-4, max_iter=max_iter, epsilon=epsilon)
    
    for u in [0.5, 1.0]
        println("\n" * "="^70)
        println("Running: NHOTA p=2, u=$u")
        println("="^70)
        results["NHOTA_p2_u$(Int(u*10))"] = NHOTA(x0, X_tr, y_tr, lambda_l1, lambda_l2, 2, u;
                                                    M0=1.0, M_min=1e-4, max_iter=max_iter, epsilon=epsilon)
    end
    
    for u in [0.5, 1.0]
        println("\n" * "="^70)
        println("Running: NHOTA p=3, u=$u (subsampled data)")
        println("="^70)
        results["NHOTA_p3_u$(Int(u*10))"] = NHOTA(x0_small, X_tr_small, y_tr_small, lambda_l1, lambda_l2, 3, u;
                                                    M0=1.0, M_min=1e-4, max_iter=max_iter, epsilon=epsilon)
    end
    
    println("\n" * "="^70)
    println("Running: NHOTA p=2, u=0.5 (subsampled data, for fair p=3 comparison)")
    println("="^70)
    results["NHOTA_p2_small_u5"] = NHOTA(x0_small, X_tr_small, y_tr_small, lambda_l1, lambda_l2, 2, 0.5;
                                          M0=1.0, M_min=1e-4, max_iter=max_iter, epsilon=epsilon)
    
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
# PART 8: Plotting
# ============================================================================

function plot_comparison(results::Dict; save_prefix::String="figures/nhota_mnist")

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
end