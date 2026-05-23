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

Extensive benchmarks across real-world datasets (`gisette`, `duke`, `a1a`) reveal two major insights:
1. **The Iteration Hierarchy Holds:** Higher-order tracking consistently yields fewer outer iterations to reach a target stationarity tolerance $\epsilon = 10^{-5}$ ($p=3 \le p=2 \le p=1$).
2. **Nonmonotonicity is Crucial:** Implementing nonmonotone tracking ($u=0.5$) prevents line-search stagnation and drops computational runtimes drastically over strict monotone variants ($u=1.0$), positioning the $p=2$ nonmonotone configuration as the ideal computational sweet spot for high-dimensional setups.

---

## 🛠️ Usage

To run the pipeline locally, make sure you have the required Julia dependencies (`CodecBzip2`, `Plots`, `LinearAlgebra`), clone this repository, and load the experiment environment:

```julia
include("NHOTA_Robust_regression.jl")
