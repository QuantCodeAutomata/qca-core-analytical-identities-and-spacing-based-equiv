"""
Experiment 5: Discrete SD vs GMD theory.

Families:
  Geometric(p)          – constant hazard, SD dominant
  Zipf(alpha)           – DFR (decreasing hazard), SD dominant
  Poisson(theta)        – sign change at theta ≈ 0.668; bound threshold ≈ 0.802
  Negative-Binomial(r=2, p) – sign change at p ≈ 0.654; bound threshold ≈ 0.569

Key quantities:
  SD    = sqrt(Var[X])
  GMD   = E[|X-Y|]  computed via the O(K) CDF formula
  Lambda = sum_k p(k)^2 = P(X=Y)
  Bound  = (1-Lambda)/(2*Lambda)  (concentration upper bound for GMD)

All helpers are self-contained; no src/core imports.
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import zeta, i0

FIGURES_DIR = "/workspace/project/results/figures"
TABLES_DIR  = "/workspace/project/results/tables"
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)

# ===========================================================================
# Core discrete helpers
# ===========================================================================

def compute_gmd_from_cdf(pmf):
    """
    GMD = 2 * sum_{k} F(k) * (1 - F(k))  [O(K), unit-spaced support].

    Works for any distribution whose support is {0,1,...} or {1,2,...} since
    each integer gap contributes 1 unit to |X-Y|.
    """
    cumf = np.cumsum(pmf)
    # The last value is 1.0, contributing 1*0=0, so we drop it.
    return 2.0 * float(np.sum(cumf[:-1] * (1.0 - cumf[:-1])))


def compute_lambda(pmf):
    """Lambda = sum_k p(k)^2 = P(X = Y) for iid X, Y."""
    return float(np.sum(np.asarray(pmf, dtype=float)**2))


def concentration_bound(lam):
    """Upper bound on GMD: (1-Lambda)/(2*Lambda)."""
    return (1.0 - lam) / (2.0 * lam)


def compute_discrete_hazard(pmf):
    """
    h(k) = P(X=k) / P(X>=k)  for k = 0, 1, ..., K.

    P(X>=k) = 1 - F(k-1) = 1 - cumsum[k-1], with P(X>=0) = 1.
    """
    pmf = np.asarray(pmf, dtype=float)
    # survival at k: P(X >= k) = 1 - F(k-1)
    sf = np.concatenate([[1.0], 1.0 - np.cumsum(pmf[:-1])])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        h = np.where(sf > 1e-15, pmf / sf, np.nan)
    return h


def compute_discrete_rev_hazard(pmf):
    """r(k) = P(X=k) / P(X<=k) = p(k) / F(k)."""
    pmf = np.asarray(pmf, dtype=float)
    cdf = np.cumsum(pmf)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = np.where(cdf > 1e-15, pmf / cdf, np.nan)
    return r


def monotonicity_label(values, tol=1e-9):
    """Return 'increasing', 'decreasing', 'constant', or 'non-monotone'."""
    finite = values[np.isfinite(values)]
    if len(finite) < 2:
        return "undefined"
    diffs = np.diff(finite)
    if np.all(diffs >= -tol):
        return "increasing" if np.any(diffs > tol) else "constant"
    if np.all(diffs <= tol):
        return "decreasing"
    return "non-monotone"


def find_sign_change(param_grid, diff_vals):
    """First parameter where diff_vals changes sign (linear interpolation)."""
    d = np.asarray(diff_vals, dtype=float)
    for i in range(len(d) - 1):
        if np.isfinite(d[i]) and np.isfinite(d[i+1]) and d[i] * d[i+1] < 0:
            u0, u1 = param_grid[i], param_grid[i+1]
            d0, d1 = d[i], d[i+1]
            return u0 - d0 * (u1 - u0) / (d1 - d0)
    return np.nan


# ===========================================================================
# Distribution builders – return (pmf, support, sd_formula)
# ===========================================================================

def geometric_pmf(p, K=500):
    """P(X=k) = p*(1-p)^k, k = 0, 1, ..., K."""
    k = np.arange(K + 1)
    pmf = p * (1.0 - p)**k
    pmf /= pmf.sum()          # normalise to account for truncation
    return pmf, k


def zipf_pmf(alpha, K=5000):
    """P(X=k) = k^{-alpha}/zeta(alpha), k = 1, 2, ..., K."""
    k = np.arange(1, K + 1)
    pmf = k.astype(float)**(-alpha)
    pmf /= pmf.sum()          # normalise (truncation correction)
    return pmf, k


def poisson_pmf(theta, K=None):
    """Poisson(theta) truncated at K (auto if None)."""
    if K is None:
        K = max(100, int(theta + 10 * np.sqrt(theta) + 50))
    k = np.arange(K + 1)
    pmf = stats.poisson.pmf(k, theta)
    pmf /= pmf.sum()
    return pmf, k


def negbinom_pmf(r, p, K=None):
    """NegBinomial(r, p): P(X=k) = C(k+r-1,k) * (1-p)^k * p^r."""
    mean = r * (1.0 - p) / p
    std  = np.sqrt(r * (1.0 - p)) / p
    if K is None:
        K = max(200, int(mean + 10 * std + 50))
    k = np.arange(K + 1)
    pmf = stats.nbinom.pmf(k, r, p)
    pmf /= pmf.sum()
    return pmf, k


# ===========================================================================
# Single-distribution analysis
# ===========================================================================

def analyse_discrete(name, pmf, support, sd_formula):
    """
    Returns a dict of summary statistics for one discrete distribution.
    """
    pmf = np.asarray(pmf, dtype=float)

    # GMD via O(K) CDF formula
    gmd = compute_gmd_from_cdf(pmf)

    lam   = compute_lambda(pmf)
    bound = concentration_bound(lam)

    h = compute_discrete_hazard(pmf)
    r = compute_discrete_rev_hazard(pmf)

    # Use only a reasonable prefix to assess monotonicity (avoid numerical noise near tail)
    K_mono = min(len(pmf), 200)
    h_label = monotonicity_label(h[:K_mono])
    r_label = monotonicity_label(r[:K_mono])

    return {
        "family":       name,
        "SD":           round(sd_formula, 6),
        "GMD":          round(gmd, 6),
        "Lambda":       round(lam, 6),
        "Bound":        round(bound, 6),
        "SD_minus_GMD": round(sd_formula - gmd, 6),
        "SD_dominant":  bool(sd_formula > gmd),
        "bound_holds":  bool(gmd <= bound + 1e-8),
        "h_monotone":   h_label,
        "r_monotone":   r_label,
    }


# ===========================================================================
# Per-family analysis
# ===========================================================================

def run_geometric():
    print("\n=== Geometric ===")
    rows = []
    for p in [0.2, 0.4, 0.6, 0.8]:
        pmf, support = geometric_pmf(p, K=800)
        mean_val = (1.0 - p) / p
        var_val  = (1.0 - p) / p**2
        sd_val   = float(np.sqrt(var_val))
        rec = analyse_discrete(f"Geometric(p={p})", pmf, support, sd_val)
        print(f"  p={p}: SD={rec['SD']:.4f}, GMD={rec['GMD']:.4f}, "
              f"Lambda={rec['Lambda']:.4f}, Bound={rec['Bound']:.4f}, "
              f"SD_dom={rec['SD_dominant']}, bound_holds={rec['bound_holds']}")
        rows.append(rec)
    return rows


def run_zipf():
    print("\n=== Zipf ===")
    rows = []
    for alpha in [3.5, 4.0, 5.0]:
        K = 10000
        pmf, support = zipf_pmf(alpha, K)
        mean_trunc = float(np.sum(support * pmf))
        e2_trunc   = float(np.sum(support**2 * pmf))
        var_trunc  = e2_trunc - mean_trunc**2
        sd_val     = float(np.sqrt(max(var_trunc, 0.0)))
        rec = analyse_discrete(f"Zipf(alpha={alpha})", pmf, support, sd_val)
        print(f"  alpha={alpha}: SD={rec['SD']:.4f}, GMD={rec['GMD']:.4f}, "
              f"SD_dom={rec['SD_dominant']}")
        rows.append(rec)

    # alpha=3 with finite truncation (variance is divergent in the limit)
    alpha = 3.0
    K = 10000
    pmf, support = zipf_pmf(alpha, K)
    mean_trunc = float(np.sum(support * pmf))
    e2_trunc   = float(np.sum(support**2 * pmf))
    var_trunc  = e2_trunc - mean_trunc**2
    sd_val     = float(np.sqrt(max(var_trunc, 0.0)))
    rec = analyse_discrete(f"Zipf(alpha=3,K={K})", pmf, support, sd_val)
    print(f"  alpha=3 (K={K}): SD={rec['SD']:.4f}, GMD={rec['GMD']:.4f}, "
          f"SD_dom={rec['SD_dominant']}, R={sd_val/rec['GMD']:.3f}")
    rows.append(rec)
    return rows


def run_poisson():
    print("\n=== Poisson ===")
    theta_grid = np.linspace(0.01, 5.0, 300)
    K_default  = 300

    sd_vals, gmd_vals, lam_vals, bound_vals = [], [], [], []

    for theta in theta_grid:
        pmf, _ = poisson_pmf(theta, K=K_default)
        gmd = compute_gmd_from_cdf(pmf)
        lam = compute_lambda(pmf)
        sd  = float(np.sqrt(theta))
        sd_vals.append(sd)
        gmd_vals.append(gmd)
        lam_vals.append(lam)
        bound_vals.append(concentration_bound(lam))

    sd_vals    = np.array(sd_vals)
    gmd_vals   = np.array(gmd_vals)
    lam_vals   = np.array(lam_vals)
    bound_vals = np.array(bound_vals)

    diff_sd_gmd   = sd_vals - gmd_vals
    diff_bound_gmd = bound_vals - gmd_vals   # positive when bound holds

    sc_actual = find_sign_change(theta_grid, diff_sd_gmd)
    sc_bound  = find_sign_change(theta_grid, diff_bound_gmd)

    print(f"  SD=GMD sign change at theta ≈ {sc_actual:.3f}  (expected ≈ 0.668)")
    print(f"  Bound=GMD threshold  at theta ≈ {sc_bound:.3f}  (expected ≈ 0.802)")

    df = pd.DataFrame({
        "theta": theta_grid, "SD": sd_vals, "GMD": gmd_vals,
        "Lambda": lam_vals, "Bound": bound_vals,
        "SD_minus_GMD": diff_sd_gmd, "Bound_minus_GMD": diff_bound_gmd,
    })
    df.to_csv(os.path.join(TABLES_DIR, "exp5_poisson_sweep.csv"), index=False)

    # Spot-check rows for benchmark table
    spot_rows = []
    for theta in [0.3, 0.668, 1.0, 2.0, 5.0]:
        pmf, support = poisson_pmf(theta)
        gmd = compute_gmd_from_cdf(pmf)
        lam = compute_lambda(pmf)
        sd  = float(np.sqrt(theta))
        h   = compute_discrete_hazard(pmf)
        r_h = compute_discrete_rev_hazard(pmf)
        spot_rows.append(analyse_discrete(f"Poisson({theta:.3f})", pmf, support, sd))

    return theta_grid, sd_vals, gmd_vals, bound_vals, sc_actual, sc_bound, spot_rows


def run_negbinom():
    print("\n=== Negative Binomial (r=2) ===")
    r = 2
    p_grid = np.linspace(0.01, 0.99, 300)

    sd_vals, gmd_vals, lam_vals, bound_vals = [], [], [], []

    for p in p_grid:
        pmf, _ = negbinom_pmf(r, p)
        gmd = compute_gmd_from_cdf(pmf)
        lam = compute_lambda(pmf)
        var = r * (1.0 - p) / p**2
        sd  = float(np.sqrt(var))
        sd_vals.append(sd)
        gmd_vals.append(gmd)
        lam_vals.append(lam)
        bound_vals.append(concentration_bound(lam))

    sd_vals    = np.array(sd_vals)
    gmd_vals   = np.array(gmd_vals)
    lam_vals   = np.array(lam_vals)
    bound_vals = np.array(bound_vals)

    diff_sd_gmd    = sd_vals - gmd_vals
    diff_bound_gmd = bound_vals - gmd_vals

    sc_actual = find_sign_change(p_grid, diff_sd_gmd)
    sc_bound  = find_sign_change(p_grid, diff_bound_gmd)

    print(f"  SD=GMD sign change at p ≈ {sc_actual:.3f}  (expected ≈ 0.654)")
    print(f"  Bound=GMD threshold  at p ≈ {sc_bound:.3f}  (expected ≈ 0.569)")

    df = pd.DataFrame({
        "p": p_grid, "SD": sd_vals, "GMD": gmd_vals,
        "Lambda": lam_vals, "Bound": bound_vals,
        "SD_minus_GMD": diff_sd_gmd, "Bound_minus_GMD": diff_bound_gmd,
    })
    df.to_csv(os.path.join(TABLES_DIR, "exp5_negbinom_sweep.csv"), index=False)

    spot_rows = []
    for p in [0.2, 0.5, 0.654, 0.8, 0.95]:
        pmf, support = negbinom_pmf(r, p)
        gmd = compute_gmd_from_cdf(pmf)
        lam = compute_lambda(pmf)
        var = r * (1.0 - p) / p**2
        sd  = float(np.sqrt(var))
        spot_rows.append(analyse_discrete(f"NB(r=2,p={p:.3f})", pmf, support, sd))

    return p_grid, sd_vals, gmd_vals, bound_vals, sc_actual, sc_bound, spot_rows


# ===========================================================================
# Main
# ===========================================================================

def main():
    # ---- Individual family benchmarks ----
    rows_geo  = run_geometric()
    rows_zipf = run_zipf()

    (theta_grid, sd_p, gmd_p, bound_p,
     sc_p_actual, sc_p_bound, rows_pois) = run_poisson()

    (p_grid, sd_nb, gmd_nb, bound_nb,
     sc_nb_actual, sc_nb_bound, rows_nb) = run_negbinom()

    # ---- Benchmark table ----
    all_rows = rows_geo + rows_zipf + rows_pois + rows_nb
    df_bench = pd.DataFrame(all_rows)
    bench_path = os.path.join(TABLES_DIR, "exp5_benchmark.csv")
    df_bench.to_csv(bench_path, index=False)
    print(f"\nBenchmark table saved to {bench_path}")
    print(df_bench.to_string())

    # ---- Figures ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1 – Poisson: SD-GMD
    ax = axes[0, 0]
    ax.plot(theta_grid, sd_p - gmd_p, "b-", linewidth=1.6, label="SD − GMD")
    ax.axhline(0, color="k", linewidth=0.8, linestyle="--")
    if np.isfinite(sc_p_actual):
        ax.axvline(sc_p_actual, color="r", linewidth=1.2, linestyle=":",
                   label=f"sign change θ≈{sc_p_actual:.3f}")
    ax.set_title("Poisson: SD − GMD vs θ")
    ax.set_xlabel("θ")
    ax.set_ylabel("SD − GMD")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 2 – Poisson: concentration bound
    ax = axes[0, 1]
    ax.plot(theta_grid, bound_p,  "g-",  linewidth=1.6, label="Bound (1−Λ)/(2Λ)")
    ax.plot(theta_grid, gmd_p,    "b--", linewidth=1.4, label="GMD")
    ax.plot(theta_grid, sd_p,     "r:",  linewidth=1.4, label="SD")
    if np.isfinite(sc_p_bound):
        ax.axvline(sc_p_bound, color="purple", linewidth=1.2, linestyle=":",
                   label=f"bound=GMD θ≈{sc_p_bound:.3f}")
    ax.set_title("Poisson: Concentration bound vs θ")
    ax.set_xlabel("θ")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 3 – NB(r=2): SD-GMD
    ax = axes[1, 0]
    ax.plot(p_grid, sd_nb - gmd_nb, "b-", linewidth=1.6, label="SD − GMD")
    ax.axhline(0, color="k", linewidth=0.8, linestyle="--")
    if np.isfinite(sc_nb_actual):
        ax.axvline(sc_nb_actual, color="r", linewidth=1.2, linestyle=":",
                   label=f"sign change p≈{sc_nb_actual:.3f}")
    ax.set_title("NB(r=2): SD − GMD vs p")
    ax.set_xlabel("p")
    ax.set_ylabel("SD − GMD")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 4 – NB(r=2): concentration bound
    ax = axes[1, 1]
    ax.plot(p_grid, bound_nb, "g-",  linewidth=1.6, label="Bound (1−Λ)/(2Λ)")
    ax.plot(p_grid, gmd_nb,  "b--", linewidth=1.4, label="GMD")
    ax.plot(p_grid, sd_nb,   "r:",  linewidth=1.4, label="SD")
    if np.isfinite(sc_nb_bound):
        ax.axvline(sc_nb_bound, color="purple", linewidth=1.2, linestyle=":",
                   label=f"bound=GMD p≈{sc_nb_bound:.3f}")
    ax.set_title("NB(r=2): Concentration bound vs p")
    ax.set_xlabel("p")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.suptitle("Experiment 5: Discrete SD vs GMD Theory", fontsize=13)
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "exp5_discrete_theory.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure saved to {fig_path}")


if __name__ == "__main__":
    main()
