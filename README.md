# NHOTA-Optimization

An optimized Julia framework for Nonmonotone Higher-Order Taylor Approximation methods ($p=1,2,3$) applied to non-convex regularized regression, with matrix-free Inexact Newton-CG tensor subproblem solvers benchmarked on real-world LIBSVM datasets.

---
## Key Features

* **Multi-Order Framework ($p=1,2,3$):** Compare standard Proximal Gradient ($p=1$), Regularized Cubic Proximal Newton ($p=2$), and Quartically-Regularized Tensor-Newton ($p=3$) variants under a single, unified codebase.
* **Matrix-Free Operators:** Eliminates $O(d^3)$ dense matrix storage bottlenecks via functional vector-action closures ($v \mapsto H_k v$ and $v \mapsto J(s)v$), enabling smooth scaling up to high dimensions ($d=5000+$).
* **Inexact Newton-CG Solvers:** Handles the challenging non-convex $p=3$ quartic subproblem using nested, non-linear inner iterations coupled with localized Conjugate Gradient (CG) loops and backtracking updates.
* **Adaptive Nonmonotone Tracking ($u=0.5$):** Integrates a dynamic relaxation buffer that allows temporary objective increases to bypass severe subproblem ill-conditioning in non-convex landscapes.
* **Automated LIBSVM Pipeline:** Built-in streaming pipelines to download, decompress, parse, and normalize data straight from the LIBSVM database.

---

## 📊 Problem Formulation

The suite minimizes a composite objective function featuring a convex logistic loss paired with a sharp, non-convex rational penalty term:

$$F(x) = \frac{1}{n} \sum_{i=1}^n \log\left(1 + \exp(-b_i a_i^T x)\right) + \lambda \sum_{j=1}^d \frac{x_j^2}{1+x_j^2}$$

---

## 📈 Empirical Convergence Summary

Extensive benchmarks across varying real-world datasets demonstrate an exact outer iteration hierarchy ($p=3 < p=2 < p=1$). Implementing nonmonotone tracking ($u=0.5$) prevents line-search stagnation and drops computational runtimes drastically compared to strict monotone variants ($u=1.0$):

| Dataset | Method | Outer Iterations | Objective Value ($f$) | Stationarity ($S_f$) | CPU Time (s) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **gisette** ($d=5000$) | `NHOTA_p3_u0.5` | **9** | 0.690083 | $4.44 \times 10^{-6}$ | 41.324 |
| | `NHOTA_p2_u0.5` | 10 | 0.690083 | $8.89 \times 10^{-6}$ | **0.437** |
| | `Baseline_PG` | 100 | 0.690083 | $3.78 \times 10^{-4}$ | 28.611 |
| **duke** ($d=7129$) | `NHOTA_p3_u0.5` | **9** | 0.233441 | $2.51 \times 10^{-6}$ | 3.462 |
| | `NHOTA_p2_u0.5` | 11 | 0.233441 | $6.61 \times 10^{-6}$ | **0.016** |
| | `Baseline_PG` | 88 | 0.233441 | $8.87 \times 10^{-6}$ | 0.130 |

---

## 🛠️ Usage

To run the pipeline locally, make sure you have the required Julia dependencies (`CodecBzip2`, `Plots`, `LinearAlgebra`), clone this repository, and load the experiment environment:

```julia
include("NHOTA_Robust_regression.jl")
