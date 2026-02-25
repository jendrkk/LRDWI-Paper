# Research Report: Numerical Methods for Estimating Missing Demographic Cross-Tabulations with Hierarchical Consistency Constraints

**Date:** 24 February 2026  
**Context:** GeoTERYT database, Polish administrative units (voivodeships → powiats → gminas), 1986–2025. Census benchmarks at 1988, 2002, 2011, 2021. Annual BDL marginals (e.g., age×sex) at gmina level 1995–2024. Goal: estimate full cross-tabulations (age×sex, sex×education, household size) for all intervening years, maintaining hierarchical consistency.

---

## Table of Contents

1. [Iterative Proportional Fitting (IPF)](#1-iterative-proportional-fitting-ipf)
2. [Entropy Maximization / Information-Theoretic Approaches](#2-entropy-maximization--information-theoretic-approaches)
3. [Bayesian Approaches for Small Area Estimation](#3-bayesian-approaches-for-small-area-estimation)
4. [Multi-Source Data Fusion Methods](#4-multi-source-data-fusion-methods)
5. [Optimization-Based Approaches](#5-optimization-based-approaches)
6. [Temporal Interpolation for Demographic Tables](#6-temporal-interpolation-for-demographic-tables)
7. [Python Implementations](#7-python-implementations)
8. [Comparison and Recommended Approach](#8-comparison-and-recommended-approach)

---

## 1. Iterative Proportional Fitting (IPF)

### 1.1 Overview and History

IPF (also known as **biproportional fitting**, **RAS method**, **raking**, **matrix scaling**, or **Kruithof's method**) is the most widely used algorithm for adjusting contingency tables to match known marginals. It was independently discovered by:

- **Kruithof (1937)** – telephone traffic ("double factor method")
- **Deming & Stephan (1940)** – adjusting census cross-tabulations
- **Sheleikhovskii** – traffic (reported by Bregman)

Key convergence proofs: **Sinkhorn (1964)**, **Bacharach (1965)**, **Bishop (1967)**, **Fienberg (1970)**, **Csiszár (1975)**, **Pukelsheim & Simeone (2009)**.

### 1.2 Mathematical Formulation

#### 2D Case (Classical IPF)

Given an $I \times J$ seed matrix $x_{ij}$, find $\hat{m}_{ij} = a_i b_j x_{ij}$ such that:

$$\sum_j \hat{m}_{ij} = u_i \quad \forall i, \qquad \sum_i \hat{m}_{ij} = v_j \quad \forall j$$

**Algorithm:** Initialize $\hat{m}_{ij}^{(0)} := x_{ij}$. For $\eta \geq 1$:

$$\hat{m}_{ij}^{(2\eta-1)} = \hat{m}_{ij}^{(2\eta-2)} \cdot \frac{u_i}{\sum_{k=1}^J \hat{m}_{ik}^{(2\eta-2)}}$$

$$\hat{m}_{ij}^{(2\eta)} = \hat{m}_{ij}^{(2\eta-1)} \cdot \frac{v_j}{\sum_{k=1}^I \hat{m}_{kj}^{(2\eta-1)}}$$

Repeat until convergence.

#### RAS Form

$$M^{(2\eta-1)} = R_\eta M^{(2\eta-2)}, \quad M^{(2\eta)} = M^{(2\eta-1)} S_\eta$$

where $R_\eta = \text{diag}\left(\frac{u_i}{\sum_j m_{ij}^{(2\eta-2)}}\right)$, $S_\eta = \text{diag}\left(\frac{v_j}{\sum_i m_{ij}^{(2\eta-1)}}\right)$.

#### Factor Estimation (more efficient)

Initialize $\hat{b}_j^{(0)} := 1$. For $\eta \geq 1$:

$$\hat{a}_i^{(\eta)} = \frac{u_i}{\sum_j x_{ij} \hat{b}_j^{(\eta-1)}}, \qquad \hat{b}_j^{(\eta)} = \frac{v_j}{\sum_i x_{ij} \hat{a}_i^{(\eta)}}$$

Result: $\hat{m}_{ij} = \hat{a}_i^{(\eta)} \hat{b}_j^{(\eta)} x_{ij}$

**Computational advantage:** Factor estimation requires only $2IJ + I + J$ operations per iteration vs. $I^2J + IJ^2 + 4IJ$ for classical IPF — at least one order of magnitude faster.

### 1.3 Equivalence to Cross-Entropy Minimization

IPF solves the following optimization problem:

$$\min_{x_{ij}} \sum_i \sum_j x_{ij} \log\left(\frac{x_{ij}}{z_{ij}}\right) \quad \text{s.t.} \quad \sum_j x_{ij} = y_{i\cdot}, \; \sum_i x_{ij} = y_{\cdot j}$$

The Lagrangian yields $x_{ij} = P_i z_{ij} Q_j$, i.e., $X = PZQ$ where $P, Q$ are diagonal matrices. This is **exactly equivalent to minimizing the Kullback-Leibler divergence** from the seed matrix $Z$ to the fitted matrix $X$.

### 1.4 Multi-Dimensional Extensions (3D+)

For $N$-dimensional tables, IPF generalizes by cycling through all dimensions:

- Given an $N$-dimensional array and target marginals for various dimension subsets
- Cycle: adjust along dimension 1, then dimension 2, ..., then dimension $N$, repeat
- The `ipfn` Python package and `mipfp` R package support arbitrary $N$-dimensional IPF with marginals of any dimension subset

**Key reference:** Bishop, Fienberg & Holland (1975). *Discrete Multivariate Analysis: Theory and Practice*. MIT Press.

### 1.5 Hierarchical/Nested Constraints

For our problem (gminas → powiats → voivodeships):

**Approach 1: Bottom-up IPF with post-aggregation check**
1. Run IPF at gmina level with gmina-level marginals
2. Aggregate to powiat level; compare with powiat-level data
3. Add powiat-level marginals as additional constraints and re-run

**Approach 2: Multi-level IPF**
- Include aggregation constraints as additional marginals. E.g., for each powiat $p$, the sum of gmina tables within $p$ must equal the powiat total.
- These become additional constraint dimensions in multi-dimensional IPF.

**Approach 3: Two-pass method**
1. First pass: IPF at powiat level → powiat-level tables
2. Second pass: For each powiat, IPF at gmina level constrained to sum to powiat totals

### 1.6 Convergence Guarantees

- **Sufficient condition:** If all entries of the seed matrix $x_{ij} > 0$, existence and uniqueness of the solution is guaranteed (Csiszár, 1975).
- **Convergence rate:** Linear in worst case (Fienberg, 1970), but exponential convergence has been observed (Pukelsheim & Simeone, 2009).
- If a direct estimator (closed form) exists, IPF converges after exactly 2 iterations.
- For tables with **structural zeros** (cells that must remain zero), convergence may be arbitrarily slow. However, zeros in the seed are preserved (a zero in $Z$ maps to a zero in $X$).

### 1.7 Limitations and Failure Modes

1. **Structural zeros:** If the seed matrix has zeros in cells that should be non-zero in the solution, IPF cannot fill them. Workaround: use small positive values (e.g., 0.001) instead of zeros.
2. **Inconsistent marginals:** If row and column totals don't sum to the same grand total, no solution exists.
3. **Decomposable matrices:** If the seed matrix permutes to a block-diagonal form ("separable"), uniqueness may fail.
4. **No temporal structure:** IPF is a static method — it adjusts one table at a time without exploiting temporal continuity.
5. **Non-integer results:** IPF produces real-valued tables, not integer counts. Controlled rounding may be needed.

### 1.8 Key Academic References

| Author(s) | Year | Title | Journal/Publisher |
|---|---|---|---|
| Deming, W.E. & Stephan, F.F. | 1940 | On a Least Squares Adjustment of a Sampled Frequency Table When the Expected Marginal Totals are Known | *Annals of Mathematical Statistics* 11(4): 427–444 |
| Bacharach, M. | 1965 | Estimating Nonnegative Matrices from Marginal Data | *International Economic Review* 6(3): 294–310 |
| Fienberg, S.E. | 1970 | An Iterative Procedure for Estimation in Contingency Tables | *Annals of Mathematical Statistics* 41(3): 907–917 |
| Bishop, Y.M.M., Fienberg, S.E. & Holland, P.W. | 1975 | Discrete Multivariate Analysis: Theory and Practice | MIT Press |
| Csiszár, I. | 1975 | I-Divergence of Probability Distributions and Minimization Problems | *Annals of Probability* 3(1): 146–158 |
| Pukelsheim, F. & Simeone, B. | 2009 | On the Iterative Proportional Fitting Procedure: Structure of Accumulation Points and L1-Error Analysis | Univ. Augsburg |
| Idel, M. | 2016 | A review of matrix scaling and Sinkhorn's normal form for matrices and positive maps | arXiv:1609.06349 |

---

## 2. Entropy Maximization / Information-Theoretic Approaches

### 2.1 Cross-Entropy Minimization (Minimum Discrimination Information)

The principle of **Minimum Discrimination Information (MDI)**, proposed by Kullback, states: given new information (constraints), choose the distribution $f$ that minimizes the KL divergence from the prior distribution $f_0$:

$$\min_f D_{KL}(f \| f_0) = \min_f \sum_{i,j} f_{ij} \log \frac{f_{ij}}{f_{0,ij}}$$

subject to the new marginal constraints.

**Key insight:** This is the **dual problem** of IPF. IPF's fixed point is exactly the solution to cross-entropy minimization. The duality was established by Csiszár (1975) and further explored in the information geometry framework (Amari, 2016).

### 2.2 Generalized Cross-Entropy (GCE)

GCE extends the framework to handle:
- **Noisy constraints** (constraints known with uncertainty)
- **Prior distributions** over both the table cells and the error terms
- **Weighted constraints** (some marginals more reliable than others)

**Formulation:**

$$\min_{p, e} \sum_{ij} p_{ij} \log \frac{p_{ij}}{q_{ij}} + \gamma \sum_k e_k \log \frac{e_k}{w_k}$$

subject to $Ap + e = b$ (marginal constraints with errors).

This is particularly useful for our case where BDL marginals may have measurement error or be estimated themselves.

**Key references:**
- Golan, A., Judge, G.G. & Miller, D. (1996). *Maximum Entropy Econometrics: Robust Estimation with Limited Data*. Wiley.
- Golan, A. (2018). *Foundations of Info-Metrics*. Oxford University Press.

### 2.3 Maximum Entropy for Contingency Tables

When no seed matrix is available (e.g., for years far from any census), maximum entropy provides the "least informative" table consistent with the known marginals:

$$\max_p -\sum_{ij} p_{ij} \log p_{ij} \quad \text{s.t.} \quad \text{marginal constraints}$$

The solution has the log-linear form:

$$p_{ij} = \exp\left(\sum_{k=0}^n \lambda_k f_k(i,j)\right)$$

where $\lambda_k$ are Lagrange multipliers determined by the constraints.

**Connection to IPF:** Maximum entropy with given marginals yields the **independence model** (product of marginal proportions). IPF with a non-uniform seed departs from this by preserving the **association structure** (cross-product ratios) of the seed while matching the new marginals.

### 2.4 I-Divergence Projection

Csiszár's I-projection framework provides a unified geometric view:
- The set of all tables matching the marginal constraints forms a **convex set** $\mathcal{C}$
- The set of all tables with the same interaction structure as the seed forms a **linear family** $\mathcal{L}$ (in log space)
- IPF finds the unique point in $\mathcal{C} \cap \mathcal{L}$ (or the I-projection of the seed onto $\mathcal{C}$)
- This satisfies a **Pythagorean identity** for KL divergence (Amari, 2016)

---

## 3. Bayesian Approaches for Small Area Estimation

### 3.1 Hierarchical Bayesian Models

For demographic cross-tabulations at small areas (gminas), Bayesian hierarchical models can:
- **Borrow strength** across areas and time
- **Quantify uncertainty** in the estimates
- **Incorporate spatial structure** (neighboring gminas should have similar distributions)

**General structure:**
$$y_{ij,t} | \theta_{ij,t} \sim f(y | \theta_{ij,t})$$
$$\theta_{ij,t} = g(X_{t}\beta + u_i + v_i + \phi_t)$$

where $u_i$ are structured spatial effects (e.g., CAR prior), $v_i$ are unstructured random effects, and $\phi_t$ are temporal effects.

### 3.2 Fay-Herriot Model

The Fay-Herriot (1979) area-level model is the standard for small area estimation:

$$y_i = x_i'\beta + u_i + e_i$$

where:
- $y_i$ = direct estimate for area $i$
- $x_i'\beta$ = regression on auxiliary variables (covariates)
- $u_i \sim N(0, \sigma_u^2)$ = area-specific random effect
- $e_i \sim N(0, \psi_i)$ = sampling error (variance $\psi_i$ known)

**Extensions relevant to our problem:**
- **Multivariate Fay-Herriot:** Benavent & Morales (2016). Joint estimation of multiple cross-tabulation cells, exploiting correlations between variables.
- **Spatial Fay-Herriot:** Porter, Holan, Wikle & Cressie (2013). Spatial smoothing via CAR priors or spatial kernels.
- **Temporal Fay-Herriot:** AR(1) or random walk priors on the time dimension.

### 3.3 Spatial Smoothing Priors

- **Conditional Autoregressive (CAR) priors:** $u_i | u_{-i} \sim N\left(\frac{\sum_{j \sim i} u_j}{n_i}, \frac{\sigma^2}{n_i}\right)$, where $j \sim i$ denotes neighbors.
- **Intrinsic CAR (ICAR):** Besag, York & Mollié (1991). The BYM model combines structured + unstructured spatial effects.
- **Spatial adjacency for gminas:** Poland's gmina adjacency graph can be directly used.

### 3.4 Limitations for Our Use Case

- **Computational cost:** MCMC for ~4,600 gminas × 38 years × multi-dimensional tables is extremely expensive.
- **Not designed for table constraints:** Bayesian SAE estimates individual parameters, not full contingency tables with exact marginal constraints.
- **Better suited as a complement:** Use Bayesian models to estimate association parameters or smooth the seed matrices, then apply IPF for exact marginal fitting.

### 3.5 Key References

| Author(s) | Year | Title | Journal/Publisher |
|---|---|---|---|
| Fay, R.E. & Herriot, R.A. | 1979 | Estimates of Income for Small Places: An Application of James-Stein Procedures to Census Data | *JASA* 74(366): 269–277 |
| Rao, J.N.K. & Molina, I. | 2015 | Small Area Estimation | Wiley (2nd edition) |
| Benavent, R. & Morales, D. | 2016 | Multivariate Fay-Herriot Models for Small Area Estimation | *CSDA* 94: 372–390 |
| Porter, A.T. et al. | 2013 | Spatial Fay-Herriot Models for Small Area Estimation with Functional Covariates | arXiv:1303.6668 |
| Besag, J., York, J. & Mollié, A. | 1991 | Bayesian Image Restoration, with Two Applications in Spatial Statistics | *Annals of the Institute of Statistical Mathematics* 43: 1–20 |
| Ghosh, M. & Rao, J.N.K. | 1994 | Small Area Estimation: An Appraisal | *Statistical Science* 9(1): 55–76 |

---

## 4. Multi-Source Data Fusion Methods

### 4.1 Structure-Preserving Interpolation Between Censuses

**The core challenge:** We have full cross-tabulations at census years (1988, 2002, 2011, 2021) and marginals (e.g., age×sex) for intervening years. We must interpolate the full tables.

**Method: IPF with census seed + annual marginals**
1. For year $t$ between censuses at $t_0$ and $t_1$:
2. Create a seed matrix by interpolating between the census tables (e.g., weighted average of the two nearest census tables)
3. Apply IPF using the known annual marginals for year $t$
4. The result preserves the **association structure** of the interpolated seed while matching the known marginals

**Variant: Log-linear interpolation of seeds**
- Instead of linear interpolation of cell counts, interpolate in log-space:
$$\log z_{ij}^{(t)} = \frac{t_1 - t}{t_1 - t_0} \log z_{ij}^{(t_0)} + \frac{t - t_0}{t_1 - t_0} \log z_{ij}^{(t_1)}$$
- This preserves cross-product ratios better than linear interpolation

### 4.2 Combining Survey Marginals with Census Benchmarks

**Hierarchical data fusion approach:**
1. **Level 1 (Census years):** Full tables known → use directly
2. **Level 2 (Years with BDL marginals):** Use nearest-census table as seed + IPF with BDL marginals
3. **Level 3 (Years with no data):** Interpolate seeds from Level 1 or Level 2 results, apply IPF if any partial marginals exist; otherwise, use pure interpolation

**Multi-source weighting:**
- When marginals from different sources conflict, use GCE (Section 2.2) to weight them by reliability
- BDL data (administrative, near-complete coverage) typically more reliable than survey data

### 4.3 Key References

| Author(s) | Year | Title | Journal/Publisher |
|---|---|---|---|
| Lomax, N. & Norman, P. | 2016 | Estimating Population Attribute Values in a Table: "Get Me Started in" Iterative Proportional Fitting | *The Professional Geographer* 68(3): 451–461 |
| Simpson, L. & Tranmer, M. | 2005 | Combining Sample and Census Data in Small Area Estimates: Iterative Proportional Fitting with Standard Software | *The Professional Geographer* 57(2): 222–234 |
| Birkin, M. & Clarke, M. | 1988 | SYNTHESIS — A Synthetic Spatial Information System for Urban and Regional Analysis | *Environment and Planning A* 20: 1645–1671 |
| Hermes, K. & Poulsen, M. | 2012 | A Review of Current Methods to Generate Synthetic Spatial Microdata Using Reweighting and Future Directions | *Computers, Environment and Urban Systems* 36(4): 281–290 |

---

## 5. Optimization-Based Approaches

### 5.1 Quadratic Programming (QP) Formulation

Instead of minimizing KL divergence (which yields IPF), minimize the **squared Euclidean distance** from the seed:

$$\min_x \sum_{ij} (x_{ij} - z_{ij})^2$$

subject to:
- $\sum_j x_{ij} = u_i$ (row marginals)
- $\sum_i x_{ij} = v_j$ (column marginals)
- $x_{ij} \geq 0$ (non-negativity)
- Hierarchical constraints: $\sum_{g \in p} x_{ij}^{(g)} = x_{ij}^{(p)}$ for all powiats $p$

This is a standard **convex QP** solvable by Gurobi, OSQP, or CVXPY.

**Advantages over IPF:**
- Can handle additional linear/quadratic constraints directly
- Can incorporate hierarchical constraints in a single optimization
- Guaranteed global optimum (convex problem)

**Disadvantages:**
- Does not preserve cross-product ratios (association structure)
- Squared distance is less natural than KL divergence for count data
- Larger problem size requires more memory

### 5.2 Linear Programming (LP) for Table Balancing

Minimize L1 deviation:

$$\min_x \sum_{ij} |x_{ij} - z_{ij}|$$

subject to marginal and non-negativity constraints. Can be linearized via auxiliary variables and solved as LP.

### 5.3 Entropy-Regularized Optimization

Combine the best of both worlds:

$$\min_x \sum_{ij} x_{ij} \log \frac{x_{ij}}{z_{ij}} + \lambda \cdot \text{(additional penalty terms)}$$

subject to marginal constraints + hierarchical constraints.

This is a convex problem (KL divergence is convex) and can be solved with standard nonlinear optimization solvers.

### 5.4 Gurobi Applicability

**Gurobi** (with premium license) is excellent for this problem class:

- **QP solver:** Handles quadratic objectives with linear constraints efficiently
- **LP solver:** For L1 formulations
- **Nonlinear (via piecewise linear approximation):** Can approximate entropy objectives
- **MILP:** If integer solutions required, Gurobi can add integrality constraints
- **Scale:** ~4,600 gminas × cells per table (e.g., 20 age groups × 2 sexes = 40 cells) ≈ 184,000 variables per year. Well within Gurobi's capabilities.
- **Hierarchical constraints:** Can be added directly as linear equality constraints

**Gurobi Python API:**
```python
import gurobipy as gp
from gurobipy import GRB

m = gp.Model("table_balancing")
x = m.addVars(n_gminas, n_age, n_sex, lb=0, name="x")

# Marginal constraints
for g in gminas:
    for a in ages:
        m.addConstr(gp.quicksum(x[g,a,s] for s in sexes) == marginal_age[g,a])

# Hierarchical constraints
for p in powiats:
    for a in ages:
        for s in sexes:
            m.addConstr(gp.quicksum(x[g,a,s] for g in gminas_in[p]) == table_powiat[p,a,s])

# Objective: minimize squared distance from seed
m.setObjective(gp.quicksum((x[g,a,s] - seed[g,a,s])**2
                            for g in gminas for a in ages for s in sexes),
               GRB.MINIMIZE)
m.optimize()
```

### 5.5 CVXPY Alternative

CVXPY v1.8 provides a cleaner modeling interface:

```python
import cvxpy as cp
import numpy as np

x = cp.Variable((n_rows, n_cols), nonneg=True)

# KL divergence objective (using cp.kl_div)
objective = cp.Minimize(cp.sum(cp.kl_div(x, seed)))

# Marginal constraints
constraints = [
    cp.sum(x, axis=1) == row_marginals,
    cp.sum(x, axis=0) == col_marginals,
]

# Hierarchical constraints
for p in powiats:
    constraints.append(cp.sum(x[gminas_in[p], :], axis=0) == powiat_totals[p])

prob = cp.Problem(objective, constraints)
prob.solve(solver=cp.SCS)  # or cp.GUROBI if available
```

**Key advantage of CVXPY:** Supports `cp.kl_div()` for the entropy objective, which is natively convex and can be solved via conic solvers (SCS, Clarabel) or interfaced to Gurobi.

### 5.6 scipy.optimize

For smaller subproblems:
- `scipy.optimize.minimize` with method `SLSQP` or `trust-constr` can handle nonlinear constrained optimization
- `scipy.optimize.linprog` for LP formulations (uses HiGHS solver)
- `scipy.optimize.milp` for mixed-integer formulations
- Less efficient than Gurobi for large-scale problems

### 5.7 Key References

| Author(s) | Year | Title | Journal/Publisher |
|---|---|---|---|
| Boyd, S. & Vandenberghe, L. | 2004 | Convex Optimization | Cambridge University Press |
| Diamond, S. & Boyd, S. | 2016 | CVXPY: A Python-Embedded Modeling Language for Convex Optimization | *JMLR* 17(83): 1–5 |
| Gurobi Optimization, LLC | 2024 | Gurobi Optimizer Reference Manual | gurobi.com |
| Stone, R., Champernowne, D.G. & Meade, J.E. | 1942 | The Precision of National Income Estimates | *Review of Economic Studies* 9(2): 111–125 |

---

## 6. Temporal Interpolation for Demographic Tables

### 6.1 Cohort-Component Projection as Interpolation

The **cohort-component method** is the standard demographic technique for population projection. It can be adapted for **interpolation** between censuses:

**Forward projection from $t_0$:**
$$P_{a+1,s}^{(t+1)} = P_{a,s}^{(t)} \cdot S_{a,s}^{(t)} + M_{a+1,s}^{(t)}$$

where $S_{a,s}$ is the survival rate and $M_{a,s}$ is net migration.

**Interpolation application:**
1. Use the 1988 census as the starting population
2. Apply survival rates (from life tables) and estimate migration to project forward
3. Adjust (rake) to match the 2002 census
4. The intercensal estimates maintain demographic coherence (aging structure)

**Advantages:** Reflects actual demographic processes (aging, mortality, migration, fertility).  
**Limitations:** Requires auxiliary data (life tables, fertility rates, migration estimates) at gmina level, which may not be available.

### 6.2 Log-Linear Models for Contingency Tables

The **log-linear model** represents the cell frequencies of a multi-dimensional contingency table as:

$$\log(F_{ijk}) = \lambda + \lambda_i^A + \lambda_j^B + \lambda_k^C + \lambda_{ij}^{AB} + \lambda_{ik}^{AC} + \lambda_{jk}^{BC} + \lambda_{ijk}^{ABC}$$

**Application to temporal interpolation:**
1. Fit log-linear models to the census cross-tabulations
2. Interpolate the $\lambda$ parameters over time (e.g., linear or spline interpolation of the interaction terms)
3. Reconstruct the table from the interpolated parameters
4. Apply IPF to match the known annual marginals

This approach is **structure-preserving** because it interpolates the *association structure* (interaction terms) rather than raw cell counts.

**Key reference:** Agresti, A. (2013). *Categorical Data Analysis* (3rd ed.). Wiley.

### 6.3 Spline-Based Interpolation with Marginal Constraints

**Method:**
1. For each cell $(i,j)$, fit a smooth function $f_{ij}(t)$ through the census values
2. Options: cubic splines, monotone splines, PCHIP (Piecewise Cubic Hermite Interpolating Polynomial)
3. At each year $t$, the interpolated table $\{f_{ij}(t)\}$ may not satisfy the known marginals
4. Apply IPF to the interpolated table with the known marginals → final estimate

**Variant: Constrained spline fitting**

Formulate as a single optimization:

$$\min \sum_{ij} \int \left(f_{ij}''(t)\right)^2 dt \quad \text{(smoothness)}$$

subject to:
- $f_{ij}(t_c) = z_{ij}^{(t_c)}$ for census years $t_c$ (interpolation constraints)
- $\sum_j f_{ij}(t) = u_i(t)$ for years with known row marginals (marginal constraints)
- $\sum_i f_{ij}(t) = v_j(t)$ for years with known column marginals
- $f_{ij}(t) \geq 0$ (non-negativity)

This is a **convex optimization problem** (quadratic objective with linear constraints) solvable with Gurobi or CVXPY in one shot, simultaneously for all years.

### 6.4 Composite Approach: Interpolate + IPF

The most practical approach combines temporal interpolation with cross-sectional fitting:

**Step 1: Temporal interpolation of association structure**
- Extract odds ratios / interaction parameters from census tables
- Interpolate these parameters over time using splines

**Step 2: Annual cross-sectional fitting**
- For each year with annual marginals (age×sex from BDL):
  - Construct seed from interpolated association structure × smoothly interpolated marginals
  - Apply multi-dimensional IPF with all available marginals
  - Enforce hierarchical consistency (add aggregation constraints)

**Step 3: Post-processing**
- Controlled rounding to integers if needed
- Verify hierarchical consistency: gminas → powiats → voivodeships

### 6.5 Key References

| Author(s) | Year | Title | Journal/Publisher |
|---|---|---|---|
| Agresti, A. | 2013 | Categorical Data Analysis (3rd ed.) | Wiley |
| Rees, P.H. | 1994 | Estimating and Projecting the Populations of Urban Communities | *Environment and Planning A* 26: 1671–1697 |
| Swanson, D.A. & Tayman, J. | 2012 | Subnational Population Estimates | Springer |
| Norman, P. | 1999 | Putting Iterative Proportional Fitting on the Researcher's Desk | *Working Paper, School of Geography, University of Leeds* |
| Smith, S.K., Tayman, J. & Swanson, D.A. | 2001 | State and Local Population Projections: Methodology and Analysis | Kluwer Academic/Plenum |

---

## 7. Python Implementations

### 7.1 IPF Packages

| Package | PyPI | Features | Status |
|---|---|---|---|
| **`ipfn`** | `pip install ipfn` | N-dimensional IPF, numpy and pandas support, convergence tracking | Last updated Dec 2021 (v1.4.4). Functional but unmaintained. 103 GitHub stars. |
| **`synthpop`** (various) | — | Synthetic population generation via IPF | Various implementations |
| **`mipfp`** (R) | CRAN | Multi-dimensional IPF, well-documented | Active R package (v3.2+) |

**`ipfn` usage example (3D):**
```python
from ipfn import ipfn
import numpy as np

# 3D seed array: (dma × size × age)
m = np.zeros((2, 4, 3))
# ... fill m ...

# Define marginals
xipp = np.array([52, 48])           # sum over dims 1,2
xpjp = np.array([20, 30, 35, 15])   # sum over dims 0,2
xppk = np.array([35, 40, 25])       # sum over dims 0,1
xijp = np.array([[9,17,19,7],[11,13,16,8]])  # sum over dim 2
xpjk = np.array([[7,9,4],[8,12,10],[15,12,8],[5,7,3]])  # sum over dim 0

aggregates = [xipp, xpjp, xppk, xijp, xpjk]
dimensions = [[0], [1], [2], [0,1], [1,2]]

IPF = ipfn.ipfn(m, aggregates, dimensions, convergence_rate=1e-6)
result = IPF.iteration()
```

### 7.2 Optimization Packages

| Package | Use Case | KL Divergence Support | Hierarchical Constraints |
|---|---|---|---|
| **CVXPY** (`pip install cvxpy`) | Convex optimization modeling | Yes (`cp.kl_div`) | Yes (linear constraints) |
| **Gurobi** (`pip install gurobipy`) | Large-scale LP/QP/MILP | Via piecewise linear approx. | Yes (native) |
| **SciPy** (`scipy.optimize`) | General optimization | Manual formulation | Via constraint functions |
| **PuLP** (`pip install pulp`) | LP modeling | No | Yes |
| **Google OR-Tools** | Constraint programming, LP | No | Yes |

### 7.3 Bayesian / Statistical Packages

| Package | Use Case |
|---|---|
| **PyMC** (`pip install pymc`) | Bayesian hierarchical models, spatial priors |
| **PyStan** / **CmdStanPy** | Stan interface for Bayesian computation |
| **statsmodels** | Log-linear models, GLM |
| **pysal** / **spreg** | Spatial regression models |

### 7.4 Demographic / Spatial Packages

| Package | Use Case |
|---|---|
| **geopandas** | Spatial data management, gmina geometries |
| **libpysal** | Spatial weights, neighborhood structures |
| **scipy.interpolate** | Spline interpolation (`CubicSpline`, `PchipInterpolator`) |
| **numpy** / **pandas** | Core data manipulation |

### 7.5 Custom Implementation Strategy

Given the specific requirements, a custom implementation combining existing tools is recommended:

```python
# Pseudocode for the full pipeline

import numpy as np
from scipy.interpolate import CubicSpline
from ipfn import ipfn
# or import cvxpy as cp for the optimization approach

def estimate_cross_tabulation(census_tables, annual_marginals, hierarchy):
    """
    census_tables: dict {year: {gmina_id: np.array}} for census years
    annual_marginals: dict {year: {gmina_id: {'age_sex': np.array, ...}}}
    hierarchy: dict {voivodeship: {powiat: [gmina_ids]}}
    """
    results = {}

    # Step 1: Interpolate association structure between censuses
    for gmina in all_gminas:
        log_odds = extract_log_odds_ratios(census_tables, gmina)
        spline = CubicSpline(census_years, log_odds, bc_type='natural')

        for year in all_years:
            # Step 2: Reconstruct seed from interpolated structure
            seed = reconstruct_from_log_odds(spline(year), total=get_total(gmina, year))

            # Step 3: Apply IPF with available marginals
            if year in annual_marginals:
                marginals = annual_marginals[year][gmina]
                result = apply_ipf(seed, marginals)
            else:
                result = seed

            results[(gmina, year)] = result

    # Step 4: Enforce hierarchical consistency
    for year in all_years:
        enforce_hierarchy(results, hierarchy, year)

    return results
```

---

## 8. Comparison and Recommended Approach

### 8.1 Pros/Cons Comparison

| Method | Pros | Cons | Scalability (~4,600 × 38 years) |
|---|---|---|---|
| **IPF** | Simple, fast, preserves association structure, well-understood theory, existing packages | Static (no temporal), can't handle all constraint types simultaneously, needs seed | Excellent — milliseconds per table |
| **Cross-Entropy Min.** | Equivalent to IPF, information-theoretic foundation | Same limitations as IPF | Excellent |
| **GCE** | Handles noisy constraints, uncertainty quantification | More complex to implement, slower | Good |
| **Bayesian SAE** | Uncertainty quantification, spatial smoothing, borrows strength | Very expensive computationally, doesn't produce constrained tables directly | Poor for full problem |
| **QP (Gurobi)** | Handles all constraints simultaneously, hierarchical constraints native, guaranteed optimal | Doesn't preserve association structure, less natural for count data | Very good — Gurobi handles 100K+ variables |
| **CVXPY + KL div** | Best of IPF (KL objective) + optimization (all constraints at once), clean Python API | Slower than pure IPF, requires conic solver | Good — depends on solver |
| **Log-linear interpolation** | Preserves table structure, smooth temporal evolution | Only for interpolation, still needs marginal fitting | Excellent |
| **Spline + IPF** | Smooth temporal estimates, exact marginal matching, practical to implement | Two-step (not jointly optimal), accumulated errors | Excellent |

### 8.2 Recommended Hybrid Approach

Given the specific parameters of the problem (~4,600 gminas, 38 years, hierarchical structure, census anchors, BDL marginals, Gurobi license available), I recommend a **three-layer hybrid approach**:

#### Layer 1: Temporal Interpolation of Association Structure

1. At census years (1988, 2002, 2011, 2021), compute **log-linear interaction parameters** (or equivalently, log odds ratios) for each gmina's cross-tabulations.
2. Use **cubic spline interpolation** (in log-space) of these parameters to create a smooth temporal trajectory for each gmina.
3. This produces a **seed table** for every gmina × year combination, carrying the association structure.

#### Layer 2: Cross-Sectional Fitting via IPF

4. For each year with BDL data (1995–2024), apply **multi-dimensional IPF** (`ipfn` package) to adjust each gmina's seed table to match the known annual marginals.
5. Use `ipfn` with numpy arrays for speed (handles 3D+ tables natively).
6. For years without BDL data (1986–1994), use the interpolated seed directly.

#### Layer 3: Hierarchical Consistency via Constrained Optimization (Gurobi)

7. After Layer 2, check if gmina totals sum correctly to powiat and voivodeship totals.
8. If hierarchical inconsistencies exist, formulate a **single QP** in Gurobi:
   - **Objective:** Minimize $\sum_{g,i,j} (x_{ij}^{(g)} - \hat{m}_{ij}^{(g)})^2$ where $\hat{m}$ is the Layer 2 result
   - **Constraints:** Row/column marginals, gmina-to-powiat aggregation, powiat-to-voivodeship aggregation, non-negativity
   - Solve for all gminas within a voivodeship simultaneously
9. This ensures **exact hierarchical consistency** in one optimization pass.

#### Alternative for Layer 3: CVXPY with KL Divergence

If preserving association structure is paramount, use CVXPY:
- **Objective:** $\min \sum_{g,i,j} x_{ij}^{(g)} \log \frac{x_{ij}^{(g)}}{\hat{m}_{ij}^{(g)}}$ (KL divergence from Layer 2 result)
- **Constraints:** Same as above
- Solve with SCS or Clarabel solver
- This finds the table **closest in KL divergence** to the IPF result while satisfying all hierarchical constraints

### 8.3 Computational Feasibility Estimate

| Step | Operations | Estimated Time |
|---|---|---|
| Layer 1: Spline interpolation | 4,600 gminas × 1 spline fit each | < 1 second total |
| Layer 2: IPF | 4,600 gminas × 30 years × ~10 iterations | ~5–30 seconds total (numpy) |
| Layer 3: Gurobi QP | 16 voivodeships × ~300 gminas × 40 cells ≈ 12,000 vars/voivodeship × 38 years | ~1–5 minutes per voivodeship, ~30 min total |
| **Total** | | **< 1 hour** |

### 8.4 Implementation Roadmap

1. **Phase 1:** Implement Layer 1 (spline interpolation of census tables). Test on a single voivodeship.
2. **Phase 2:** Implement Layer 2 (IPF with `ipfn`). Validate against known census years (hold-out one census).
3. **Phase 3:** Implement Layer 3 (Gurobi hierarchical consistency). Start with Gurobi QP; if association structure degradation is unacceptable, switch to CVXPY + KL.
4. **Phase 4:** Scale to full dataset. Parallelize Layer 2 across gminas (embarrassingly parallel). Batch Layer 3 by voivodeship.
5. **Phase 5:** Validation — compare estimates against held-out census data, check temporal smoothness, verify hierarchical consistency.

### 8.5 Key Packages to Install

```bash
pip install ipfn numpy scipy pandas geopandas cvxpy gurobipy
# Optional:
pip install pymc statsmodels
```

---

## Appendix: Glossary of Key Terms

| Term | Definition |
|---|---|
| **Cross-tabulation** | A contingency table showing the joint frequency distribution of two or more categorical variables |
| **Marginals** | Row/column/dimension totals of a contingency table |
| **Seed matrix** | Initial estimate of the table, to be adjusted by IPF |
| **Association structure** | The pattern of statistical dependence between variables, as captured by cross-product ratios or interaction parameters |
| **Hierarchical consistency** | The requirement that small-area estimates aggregate to match larger-area totals |
| **I-divergence** | Another name for KL divergence; the quantity minimized by IPF |
| **Raking** | Synonym for IPF in survey statistics |
| **Structural zeros** | Cells that must be zero by definition (e.g., males who gave birth) |
| **BDL** | Bank Danych Lokalnych (Bank of Local Data) — Poland's statistical database |
| **TERYT** | National Official Register of the Territorial Division of the Country (Poland) |
