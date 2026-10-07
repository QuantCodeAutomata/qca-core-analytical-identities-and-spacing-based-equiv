# Statistical / Actuarial Research Repository

A Python research library implementing statistical and actuarial identities
centred on the **Gini Mean Difference (GMD)**, **Standard Deviation (SD)**,
and their ratio **R = SD / GMD**.

---

## Repository Structure

```
project/
├── data/                          # Raw data (empty; add files here)
├── src/
│   └── core/
│       ├── distributions.py       # DistributionSpec + factory functions
│       ├── moments.py             # E[X], Var[X], SD[X], Mx[X] via quad
│       ├── gmd.py                 # GMD via 3 methods: 1D, Cov, pairwise
│       ├── spacing.py             # Y=|X-X'| moments and spacing criterion
│       ├── spacing_distribution.py# S_Y(t), m_Y(t) direct computation
│       ├── mean_excess.py         # m_Y^F and m_Y^S representations
│       ├── discrete_distributions.py  # Geometric, Zipf, Poisson, NB
│       ├── discrete_metrics.py    # GMD_discrete, Lambda, hazard functions
│       ├── mixtures.py            # Gamma frailty, normal scale mixture
│       ├── convolution.py         # FFT-based density convolution
│       ├── truncation.py          # Tail truncation X_u^+, Delta(u)
│       └── custom_examples.py     # Paper-specific examples (Ex 3.6, 4.1, …)
├── exp/
│   ├── exp_1_core_identities.py   # Experiment 1: verify analytical identities
│   └── exp_2_dominance_classification.py  # Experiment 2: parameter sweeps
├── tests/
│   ├── test_exp_1.py              # Pytest suite for Experiment 1
│   └── test_exp_2.py              # Pytest suite for Experiment 2
├── results/
│   ├── tables/                    # CSV output tables
│   └── figures/                   # PNG output figures
└── README.md
```

---

## Key Concepts

| Symbol | Definition |
|--------|-----------|
| GMD[X] | Gini Mean Difference = E\|X − X'\| |
| SD[X]  | Standard Deviation = √Var[X] |
| R[X]   | Ratio = SD[X] / GMD[X] |
| Mx[X]  | Maxiance = E[max(X,X')] − E[X] |
| Y      | Spacing Y = \|X − X'\| |

### Core Identities

- **GMD = 2 · Mx[X]** (maxiance identity)
- **GMD = 2 ∫ F(x) S(x) dx** (1-D identity)
- **GMD = 4 Cov(X, F(X))** (covariance identity)
- **E[Y²] = 2 SD²[X]** (spacing moment identity)
- **SD[X] ≥ GMD[X]  ⟺  CV[Y] ≥ 1** (spacing criterion)
- **R[X] ≥ √3/2** (universal lower bound, attained at Uniform)

### Analytical Values

| Distribution | SD | GMD | R |
|---|---|---|---|
| Exponential(λ) | 1/λ | 1/λ | 1 |
| Uniform[0,1] | 1/√12 | 1/3 | √3/2 ≈ 0.866 |
| Normal(0,1) | 1 | 2/√π ≈ 1.128 | √π/2 ≈ 0.886 |
| Logistic(0,1) | π/√3 ≈ 1.814 | 2 | π/(2√3) ≈ 0.907 |
| Laplace(0,1) | √2 ≈ 1.414 | 3/2 | 2√2/3 ≈ 0.943 |

### Dominance Pattern (Gamma / Weibull families)

- **α < 1**: SD > GMD (SD-dominant)
- **α = 1** (Exponential): SD = GMD (knife-edge)
- **α > 1**: GMD > SD (GMD-dominant)

---

## Installation

```bash
pip install numpy scipy pandas matplotlib pytest
```

---

## Running Experiments

```bash
cd /workspace/project

# Experiment 1 — core identities
python -m exp.exp_1_core_identities

# Experiment 2 — dominance classification
python -m exp.exp_2_dominance_classification
```

Results are written to `results/tables/` (CSV) and `results/figures/` (PNG).

---

## Running Tests

```bash
cd /workspace/project
pytest tests/ -v
```

---

## Module Reference

### `src/core/distributions.py`

`DistributionSpec` — dataclass holding `f`, `F`, `S`, `support`, `mean`,
`variance`, and an optional `sample` function.

Factory functions: `make_exponential`, `make_uniform`, `make_normal`,
`make_logistic`, `make_laplace`, `make_gamma`, `make_weibull`, `make_gpd`,
`make_burr`, `make_beta`, `make_gompertz`, `make_makeham`.

### `src/core/moments.py`

`compute_mean`, `compute_variance`, `compute_sd`, `compute_maxiance` —
all via `scipy.integrate.quad`.

### `src/core/gmd.py`

`compute_gmd_1d` — preferred method; `compute_gmd_cov`; `compute_gmd_pairwise`
(double integral); `compute_gmd_maxiance` (= 2·Mx).

### `src/core/spacing.py`

`compute_spacing_moments` — returns E[Y], E[Y²], Var[Y], CV[Y].
`check_spacing_criterion` — verifies sign(SD−GMD) = sign(CV[Y]−1).

### `src/core/custom_examples.py`

`make_example_3_6_4`, `make_example_4_1`, `make_example_4_8`,
`make_example_5_4`, `make_example_5_6`.
