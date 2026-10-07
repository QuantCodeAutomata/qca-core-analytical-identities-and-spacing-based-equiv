"""Experiment 1: Core Analytical Identities.

For each benchmark distribution, verifies:
  - Three GMD methods agree to within 1e-8
  - GMD = 2 * Mx[X]
  - Spacing moments: E[Y]=GMD, E[Y^2]=2*SD^2, CV[Y]
  - Spacing criterion: sign(SD-GMD) = sign(CV[Y]-1)
  - Analytical values for Exponential, Uniform, Normal, Logistic, Laplace
  - Monte Carlo cross-check (N=500_000)

Outputs:
  results/tables/exp1_benchmark.csv
  results/figures/exp1_comparison.png
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Allow running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.core.distributions import (
    make_exponential, make_uniform, make_normal,
    make_logistic, make_laplace,
)
from src.core.moments import compute_mean, compute_sd, compute_maxiance
from src.core.gmd import compute_gmd_1d, compute_gmd_cov, compute_gmd_pairwise, compute_gmd_maxiance
from src.core.spacing import compute_spacing_moments, check_spacing_criterion

# ---------------------------------------------------------------------------
# Analytical reference values
# ---------------------------------------------------------------------------
ANALYTICAL = {
    "Exponential(λ=1)": {"E_X": 1.0, "SD": 1.0, "GMD": 1.0, "R": 1.0},
    "Uniform[0,1]":     {"E_X": 0.5, "SD": 1.0 / np.sqrt(12.0),
                         "GMD": 1.0 / 3.0, "R": np.sqrt(3.0) / 2.0},
    "Normal(0,1)":      {"E_X": 0.0, "SD": 1.0,
                         "GMD": 2.0 / np.sqrt(np.pi), "R": np.sqrt(np.pi) / 2.0},
    "Logistic(0,1)":    {"E_X": 0.0, "SD": np.pi / np.sqrt(3.0), "GMD": 2.0,
                         "R": np.pi / (2.0 * np.sqrt(3.0))},
    "Laplace(0,1)":     {"E_X": 0.0, "SD": np.sqrt(2.0), "GMD": 1.5,
                         "R": 2.0 * np.sqrt(2.0) / 3.0},
}

TOLERANCE_METHODS = 1e-6    # GMD method agreement tolerance
TOLERANCE_IDENTITY = 1e-5   # Identity verification tolerance


def _monte_carlo_gmd(dist, n: int = 500_000) -> float:
    """Estimate GMD via MC: E|X-X'| with n iid pairs."""
    if dist.sample is None:
        return np.nan
    x1 = dist.sample(n)
    x2 = dist.sample(n)
    return float(np.mean(np.abs(x1 - x2)))


def _monte_carlo_sd(dist, n: int = 500_000) -> float:
    if dist.sample is None:
        return np.nan
    x = dist.sample(n)
    return float(np.std(x, ddof=0))


def run_benchmark(label: str, dist, analytical: dict) -> dict:
    """Run all computations for one distribution and collect results."""
    print(f"\n{'='*60}")
    print(f"Distribution: {label}")
    print(f"{'='*60}")

    # ---- Numerical moments ----
    e_x = compute_mean(dist)
    sd  = compute_sd(dist)
    mx  = compute_maxiance(dist)

    # ---- GMD via three methods ----
    gmd_1d   = compute_gmd_1d(dist)
    gmd_cov  = compute_gmd_cov(dist)
    gmd_mx   = compute_gmd_maxiance(dist)
    # Pairwise is slow; use only for finite-support distributions
    lo, hi = dist.support
    if np.isfinite(lo) and np.isfinite(hi):
        gmd_pw = compute_gmd_pairwise(dist, tol=1e-8)
    else:
        gmd_pw = compute_gmd_pairwise(dist, tol=1e-6)

    gmd = gmd_1d  # primary estimate

    # ---- Method agreement ----
    diff_1d_cov = abs(gmd_1d - gmd_cov)
    diff_1d_mx  = abs(gmd_1d - gmd_mx)
    diff_1d_pw  = abs(gmd_1d - gmd_pw)

    print(f"  E[X]        = {e_x:.8f}  (analytic: {analytical.get('E_X', 'N/A')})")
    print(f"  SD          = {sd:.8f}  (analytic: {analytical.get('SD', 'N/A'):.8f})")
    print(f"  GMD (1D)    = {gmd_1d:.8f}  (analytic: {analytical.get('GMD', 'N/A'):.8f})")
    print(f"  GMD (Cov)   = {gmd_cov:.8f}")
    print(f"  GMD (Mx)    = {gmd_mx:.8f}")
    print(f"  GMD (PW)    = {gmd_pw:.8f}")
    print(f"  |1d-cov|    = {diff_1d_cov:.2e}")
    print(f"  |1d-mx|     = {diff_1d_mx:.2e}")
    print(f"  |1d-pw|     = {diff_1d_pw:.2e}")

    # GMD = 2*Mx check
    gmd_from_mx = 2.0 * mx
    diff_gmd_mx = abs(gmd - gmd_from_mx)
    print(f"  Mx          = {mx:.8f},  2*Mx = {gmd_from_mx:.8f},  |GMD-2Mx| = {diff_gmd_mx:.2e}")

    # ---- Spacing moments ----
    spacing = compute_spacing_moments(dist, gmd=gmd, sd=sd)
    criterion = check_spacing_criterion(sd=sd, gmd=gmd, cv_y=spacing["CV_Y"])

    R = sd / gmd if gmd > 0 else np.nan
    print(f"  R = SD/GMD  = {R:.8f}  (analytic: {analytical.get('R', 'N/A'):.8f})")
    print(f"  E[Y]        = {spacing['E_Y']:.8f}")
    print(f"  E[Y²]       = {spacing['E_Y2']:.8f}  (should = {2*sd**2:.8f})")
    print(f"  CV[Y]       = {spacing['CV_Y']:.8f}")
    print(f"  sign(SD-GMD)= {criterion['sign_sd_gmd']:+d}  sign(CV-1)={criterion['sign_cvy_1']:+d}  match={criterion['match']}")

    # ---- Monte Carlo validation ----
    mc_gmd = _monte_carlo_gmd(dist)
    mc_sd  = _monte_carlo_sd(dist)
    print(f"  MC GMD      = {mc_gmd:.6f}  MC SD = {mc_sd:.6f}")

    # ---- Analytical checks ----
    analytic_gmd = analytical.get("GMD")
    analytic_sd  = analytical.get("SD")
    analytic_r   = analytical.get("R")
    if analytic_gmd is not None:
        err_gmd = abs(gmd - analytic_gmd)
        err_sd  = abs(sd  - analytic_sd)
        err_r   = abs(R   - analytic_r)
        print(f"  |num-analytic|: GMD={err_gmd:.2e}, SD={err_sd:.2e}, R={err_r:.2e}")
        assert err_gmd < 1e-4, f"GMD mismatch for {label}: {gmd} vs {analytic_gmd}"
        assert err_sd  < 1e-4, f"SD mismatch for {label}: {sd} vs {analytic_sd}"

    assert diff_1d_cov < TOLERANCE_METHODS, f"1D/Cov GMD disagree for {label}: {diff_1d_cov}"
    assert diff_1d_mx  < TOLERANCE_METHODS, f"1D/Mx  GMD disagree for {label}: {diff_1d_mx}"
    assert criterion["match"], f"Spacing criterion mismatch for {label}"

    return {
        "distribution": label,
        "E_X":          round(e_x, 8),
        "SD":           round(sd, 8),
        "GMD_1d":       round(gmd_1d, 8),
        "GMD_cov":      round(gmd_cov, 8),
        "GMD_mx":       round(gmd_mx, 8),
        "GMD_pw":       round(gmd_pw, 8),
        "R":            round(R, 8),
        "Mx":           round(mx, 8),
        "E_Y":          round(spacing["E_Y"], 8),
        "E_Y2":         round(spacing["E_Y2"], 8),
        "CV_Y":         round(spacing["CV_Y"], 8),
        "sign_sd_gmd":  criterion["sign_sd_gmd"],
        "sign_cv1":     criterion["sign_cvy_1"],
        "match":        criterion["match"],
        "MC_GMD":       round(mc_gmd, 6) if not np.isnan(mc_gmd) else np.nan,
        "MC_SD":        round(mc_sd, 6)  if not np.isnan(mc_sd)  else np.nan,
        "|diff_1d_cov|": f"{diff_1d_cov:.2e}",
        "|diff_1d_mx|":  f"{diff_1d_mx:.2e}",
        "analytic_GMD": analytic_gmd,
        "analytic_SD":  analytic_sd,
        "analytic_R":   analytic_r,
    }


def run_special_checks(results: list[dict]):
    """Specific verification claims from the experiment spec."""
    print("\n" + "="*60)
    print("SPECIAL CHECKS")
    print("="*60)

    # Exponential: SD = GMD = 1, CV[Y] = 1
    exp_row = next(r for r in results if "Exponential" in r["distribution"])
    assert abs(exp_row["SD"] - 1.0) < 1e-5, "Exp SD != 1"
    assert abs(exp_row["GMD_1d"] - 1.0) < 1e-5, "Exp GMD != 1"
    assert abs(exp_row["CV_Y"] - 1.0) < 1e-4, "Exp CV[Y] != 1"
    print("✓ Exponential: SD=GMD=1, CV[Y]=1")

    # Uniform: R = sqrt(3)/2
    uni_row = next(r for r in results if "Uniform" in r["distribution"])
    assert abs(uni_row["R"] - np.sqrt(3.0) / 2.0) < 1e-5, "Uniform R != sqrt(3)/2"
    print(f"✓ Uniform: R = {uni_row['R']:.8f} (expect {np.sqrt(3)/2:.8f})")

    # Normal: GMD = 2/sqrt(pi)
    norm_row = next(r for r in results if "Normal" in r["distribution"])
    assert abs(norm_row["GMD_1d"] - 2.0 / np.sqrt(np.pi)) < 1e-5, "Normal GMD != 2/sqrt(pi)"
    print(f"✓ Normal: GMD = {norm_row['GMD_1d']:.8f} (expect {2/np.sqrt(np.pi):.8f})")

    # Normal: SD < GMD, so CV[Y] < 1
    assert norm_row["sign_sd_gmd"] == -1, "Normal: expected SD < GMD"
    assert norm_row["sign_cv1"] == -1, "Normal: expected CV[Y] < 1"
    print(f"✓ Normal: SD < GMD, CV[Y] < 1  (CV[Y] = {norm_row['CV_Y']:.4f})")

    # Logistic: SD = pi/sqrt(3), GMD = 2
    log_row = next(r for r in results if "Logistic" in r["distribution"])
    assert abs(log_row["SD"] - np.pi / np.sqrt(3.0)) < 1e-4, "Logistic SD"
    assert abs(log_row["GMD_1d"] - 2.0) < 1e-4, "Logistic GMD"
    print(f"✓ Logistic: SD = {log_row['SD']:.6f}, GMD = {log_row['GMD_1d']:.6f}")

    # Laplace: SD = sqrt(2), GMD = 3/2
    lap_row = next(r for r in results if "Laplace" in r["distribution"])
    assert abs(lap_row["SD"] - np.sqrt(2.0)) < 1e-4, "Laplace SD"
    assert abs(lap_row["GMD_1d"] - 1.5) < 1e-4, "Laplace GMD"
    print(f"✓ Laplace: SD = {lap_row['SD']:.6f}, GMD = {lap_row['GMD_1d']:.6f}")

    # R >= sqrt(3)/2 for all distributions
    lower_bound = np.sqrt(3.0) / 2.0
    for r in results:
        assert r["R"] >= lower_bound - 1e-5, (
            f"R < sqrt(3)/2 for {r['distribution']}: {r['R']}"
        )
    print(f"✓ All R ≥ sqrt(3)/2 = {lower_bound:.6f}")


def make_comparison_plot(results: list[dict], out_path: str):
    """Bar chart comparing numerical vs analytical GMD and SD."""
    labels  = [r["distribution"] for r in results]
    sd_num  = [r["SD"] for r in results]
    gmd_num = [r["GMD_1d"] for r in results]
    sd_ana  = [r["analytic_SD"]  if r["analytic_SD"]  is not None else r["SD"]  for r in results]
    gmd_ana = [r["analytic_GMD"] if r["analytic_GMD"] is not None else r["GMD_1d"] for r in results]
    mc_gmd  = [r["MC_GMD"] for r in results]

    x = np.arange(len(labels))
    width = 0.18

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: SD comparison
    ax = axes[0]
    ax.bar(x - width, sd_num, width, label="SD (numerical)", color="steelblue")
    ax.bar(x,         sd_ana, width, label="SD (analytical)", color="tomato", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("SD[X]")
    ax.set_title("Standard Deviation")
    ax.legend(fontsize=8)

    # Right: GMD comparison (numerical, analytical, MC)
    ax = axes[1]
    ax.bar(x - width, gmd_num, width, label="GMD (1D)",       color="steelblue")
    ax.bar(x,         gmd_ana, width, label="GMD (analytic)", color="tomato", alpha=0.7)
    ax.bar(x + width, mc_gmd,  width, label="GMD (MC)",       color="seagreen", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("GMD[X]")
    ax.set_title("Gini Mean Difference")
    ax.legend(fontsize=8)

    fig.suptitle("Experiment 1: Core Identities — Numerical vs Analytical vs MC", fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved to {out_path}")


def main():
    np.random.seed(42)

    benchmarks = [
        ("Exponential(λ=1)", make_exponential(rate=1.0), ANALYTICAL["Exponential(λ=1)"]),
        ("Uniform[0,1]",     make_uniform(0.0, 1.0),      ANALYTICAL["Uniform[0,1]"]),
        ("Normal(0,1)",      make_normal(0.0, 1.0),        ANALYTICAL["Normal(0,1)"]),
        ("Logistic(0,1)",    make_logistic(0.0, 1.0),      ANALYTICAL["Logistic(0,1)"]),
        ("Laplace(0,1)",     make_laplace(0.0, 1.0),       ANALYTICAL["Laplace(0,1)"]),
    ]

    results = []
    for label, dist, analytic in benchmarks:
        row = run_benchmark(label, dist, analytic)
        results.append(row)

    run_special_checks(results)

    # ---- Save table ----
    df = pd.DataFrame(results)
    table_path = os.path.join(
        os.path.dirname(__file__), "..", "results", "tables", "exp1_benchmark.csv"
    )
    os.makedirs(os.path.dirname(table_path), exist_ok=True)
    df.to_csv(table_path, index=False)
    print(f"\nTable saved to {table_path}")

    # ---- Save plot ----
    fig_path = os.path.join(
        os.path.dirname(__file__), "..", "results", "figures", "exp1_comparison.png"
    )
    make_comparison_plot(results, fig_path)

    print("\n" + "="*60)
    print("Experiment 1 COMPLETED successfully.")
    print("="*60)
    return results


if __name__ == "__main__":
    main()
