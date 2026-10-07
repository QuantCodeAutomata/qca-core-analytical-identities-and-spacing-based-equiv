# Project: SD vs GMD (Standard Deviation vs Gini Mean Difference)

## Overview
Numerical experiments comparing two dispersion measures:
- **SD** = sqrt(Var[X])
- **GMD** = E[|X - X'|] = 2 * ∫ F(x) S(x) dx  (for continuous distributions on ℝ≥0)

## Repository Layout
```
exp/          – experiment scripts (each self-contained, no src/core imports)
  exp_1_core_identities.py
  exp_2_dominance_classification.py
  exp_3_mean_excess_representation.py
  exp_4_closure_and_truncation.py
  exp_5_discrete_theory.py

tests/        – pytest test files (self-contained inline helpers)
  test_exp_1.py  test_exp_2.py  test_exp_3.py  test_exp_4.py  test_exp_5.py

src/core/     – library modules (may not be used by exp scripts)

results/
  figures/    – PNG plots (written by running experiments)
  tables/     – CSV tables (written by running experiments)
  RESULTS.md  – placeholder
```

## Key Design Decisions
1. **Self-contained experiments** – Each `exp/exp_N_*.py` imports only from the
   Python standard library, numpy, scipy, pandas, and matplotlib.  No `src/core`
   imports are used so experiments can run regardless of library completeness.

2. **Self-contained tests** – Each `tests/test_exp_N.py` defines its own helper
   functions inline.  Tests do NOT import from the experiment files; they
   re-implement the minimal logic needed to verify the mathematical assertions.

3. **GMD formulas used**
   - Continuous: `GMD = 2 * ∫ F(x) S(x) dx`
   - Discrete (unit-spaced integers): `GMD = 2 * Σ_k F(k) * (1 - F(k))`  [O(K)]

4. **Survival-function identity for E[X]**: `E[X] = ∫ S(x) dx`  (nonneg support)

## Experiment Summaries

### Exp 3 – Mean-excess representation
- Distributions: Exp(1), Gamma(0.5), Gamma(2), Weibull(0.5), Weibull(2), GPD(0.2)
- Three equivalent representations of m_Y(t):
  * Direct via S_Y(t)
  * Q^F: `∫_{x≥t} F(x-t) S(x) dx / ∫_{x≥t} F(x-t) f(x) dx`
  * Q^S: `∫ S(x+t) F(x) dx / ∫ S(x+t) f(x) dx`
- Outputs: `exp3_mean_excess.csv`, `exp3_mean_excess.png`

### Exp 4 – Closure and truncation
- Part A: Lomax/Gamma-frailty mixture (DFR → SD dominant)
- Part B: Convolution – Gamma(4,1), N(0,2) (GMD dominant)
- Part C: Tail-truncation Delta(u) = SD[X_u^+] - GMD[X_u^+]
  * Example 5.4: h(x)=x·exp(-0.1x)+1; sign change near u≈9.7
  * Example 5.6: Normal scale mixture (σ1=0.5, σ2=2, q=0.75); sign change near u≈0.65
  * Makeham (A=0.00022, B=2.7e-6, c=0.124); sign change near u≈60.7
- Outputs: `exp4_partA_lomax.csv`, `exp4_partB_convolution.csv`,
           `exp4_ex54_delta.csv`, `exp4_ex56_delta.csv`, `exp4_makeham_delta.csv`,
           `exp4_closure_truncation.png`

### Exp 5 – Discrete theory
- Families: Geometric, Zipf, Poisson, NegBinomial(r=2)
- Lambda = Σ p(k)^2 = P(X=Y)
- Concentration bound = (1-Lambda)/(2*Lambda)
- Poisson: SD=GMD sign change at θ≈0.668; bound threshold at θ≈0.802
- NB(r=2): SD=GMD sign change at p≈0.654; bound threshold at p≈0.569
- Outputs: `exp5_benchmark.csv`, `exp5_poisson_sweep.csv`,
           `exp5_negbinom_sweep.csv`, `exp5_discrete_theory.png`

## Running Experiments
```bash
python exp/exp_3_mean_excess_representation.py
python exp/exp_4_closure_and_truncation.py
python exp/exp_5_discrete_theory.py
```

## Running Tests
```bash
pytest tests/ -v
```

## Notes on Numerical Accuracy
- All `scipy.integrate.quad` calls use `limit=300-500, epsabs=1e-11, epsrel=1e-9`.
- Weibull(α=0.5) has an integrable singularity at x=0; handled by integrating
  from `eps=1e-14` rather than 0.
- Makeham cutoff = 115 (S(115) ≈ 0 numerically).
- Zipf at α=3 uses K=10000 truncation (variance is formally infinite).
