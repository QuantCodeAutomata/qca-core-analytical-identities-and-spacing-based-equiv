"""Experiment 2: Dominance Classification.

For parametric families, sweeps parameter ranges and computes SD, GMD, R.
Verifies theoretical dominance patterns and benchmark numerical values.
Produces SD-GMD vs parameter plots and R[X] vs parameter plots.

Outputs:
  results/tables/exp2_gamma.csv
  results/tables/exp2_weibull.csv
  results/tables/exp2_gpd.csv
  results/tables/exp2_burr.csv
  results/tables/exp2_beta.csv
  results/tables/exp2_benchmarks.csv
  results/figures/exp2_gamma_weibull.png
  results/figures/exp2_r_profiles.png
"""

import sys
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.core.distributions import (
    make_gamma, make_weibull, make_gpd, make_burr, make_beta,
    make_logistic, make_laplace, make_gompertz, make_makeham,
)
from src.core.moments import compute_sd
from src.core.gmd import compute_gmd_1d
from src.core.custom_examples import (
    make_example_3_6_4, make_example_4_1, make_example_4_8,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
TABLES_DIR  = os.path.join(RESULTS_DIR, "tables")
FIGS_DIR    = os.path.join(RESULTS_DIR, "figures")

# ---------------------------------------------------------------------------
# Hazard rate monotonicity classification (numerical derivative)
# ---------------------------------------------------------------------------

def classify_hazard_monotonicity(dist, x_grid: np.ndarray) -> str:
    """Classify hazard h(x) = f(x)/S(x) as IHR / DHR / non-monotone."""
    haz = np.array([
        dist.f(x) / dist.S(x)
        if dist.S(x) > 1e-15 else np.nan
        for x in x_grid
    ])
    valid = ~np.isnan(haz)
    if valid.sum() < 2:
        return "unknown"
    h = haz[valid]
    dh = np.diff(h)
    n_inc = np.sum(dh > 0)
    n_dec = np.sum(dh < 0)
    if n_dec == 0:
        return "IHR"
    if n_inc == 0:
        return "DHR"
    return "non-monotone"


def compute_sd_gmd(dist) -> tuple[float, float]:
    sd  = compute_sd(dist)
    gmd = compute_gmd_1d(dist)
    return sd, gmd


# ---------------------------------------------------------------------------
# Family sweeps
# ---------------------------------------------------------------------------

def sweep_gamma(alphas: np.ndarray, theta: float = 1.0) -> pd.DataFrame:
    rows = []
    for alpha in alphas:
        dist = make_gamma(alpha, theta)
        sd, gmd = compute_sd_gmd(dist)
        R = sd / gmd if gmd > 0 else np.nan
        x_grid = np.linspace(1e-6, 20.0, 500)
        hr_class = classify_hazard_monotonicity(dist, x_grid)
        rows.append({
            "alpha": alpha, "theta": theta,
            "SD": sd, "GMD": gmd, "R": R,
            "g": sd - gmd,
            "dominant": "SD" if sd > gmd else "GMD",
            "hazard_class": hr_class,
        })
    return pd.DataFrame(rows)


def sweep_weibull(alphas: np.ndarray, scale: float = 1.0) -> pd.DataFrame:
    rows = []
    for alpha in alphas:
        dist = make_weibull(alpha, scale)
        sd, gmd = compute_sd_gmd(dist)
        R = sd / gmd if gmd > 0 else np.nan
        x_grid = np.linspace(1e-6, 20.0, 500)
        hr_class = classify_hazard_monotonicity(dist, x_grid)
        rows.append({
            "alpha": alpha, "scale": scale,
            "SD": sd, "GMD": gmd, "R": R,
            "g": sd - gmd,
            "dominant": "SD" if sd > gmd else "GMD",
            "hazard_class": hr_class,
        })
    return pd.DataFrame(rows)


def sweep_gpd(xis: np.ndarray, scale: float = 1.0) -> pd.DataFrame:
    rows = []
    for xi in xis:
        dist = make_gpd(xi, scale)
        sd, gmd = compute_sd_gmd(dist)
        R = sd / gmd if gmd > 0 else np.nan
        x_grid = np.linspace(1e-6, 50.0, 500)
        hr_class = classify_hazard_monotonicity(dist, x_grid)
        rows.append({
            "xi": xi, "scale": scale,
            "SD": sd, "GMD": gmd, "R": R,
            "g": sd - gmd,
            "dominant": "SD" if sd > gmd else "GMD",
            "hazard_class": hr_class,
        })
    return pd.DataFrame(rows)


def sweep_burr(c_vals: list, k_vals: list) -> pd.DataFrame:
    rows = []
    for c, k in zip(c_vals, k_vals):
        if c * k <= 2:
            continue  # need ck > 2 for finite variance
        dist = make_burr(c, k)
        sd, gmd = compute_sd_gmd(dist)
        R = sd / gmd if gmd > 0 else np.nan
        x_grid = np.linspace(1e-6, 20.0, 500)
        hr_class = classify_hazard_monotonicity(dist, x_grid)
        rows.append({
            "c": c, "k": k, "ck": c * k,
            "SD": sd, "GMD": gmd, "R": R,
            "g": sd - gmd,
            "dominant": "SD" if sd > gmd else "GMD",
            "hazard_class": hr_class,
        })
    return pd.DataFrame(rows)


def sweep_beta(alphas: np.ndarray, beta_vals: list) -> pd.DataFrame:
    rows = []
    for beta in beta_vals:
        for alpha in alphas:
            dist = make_beta(alpha, beta)
            sd, gmd = compute_sd_gmd(dist)
            R = sd / gmd if gmd > 0 else np.nan
            x_grid = np.linspace(1e-4, 1.0 - 1e-4, 300)
            hr_class = classify_hazard_monotonicity(dist, x_grid)
            rows.append({
                "alpha": alpha, "beta": beta,
                "SD": sd, "GMD": gmd, "R": R,
                "g": sd - gmd,
                "dominant": "SD" if sd > gmd else "GMD",
                "hazard_class": hr_class,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Benchmark point values
# ---------------------------------------------------------------------------

BENCHMARK_TARGETS = {
    "Logistic(0,1)":   {"SD": np.pi / np.sqrt(3.0), "GMD": 2.0,
                        "R": np.pi / (2.0 * np.sqrt(3.0))},
    "Laplace(0,1)":    {"SD": np.sqrt(2.0), "GMD": 1.5,
                        "R": 2.0 * np.sqrt(2.0) / 3.0},
    "Gompertz":        {"SD": 14.00, "GMD": 15.26, "R": 0.917},
    "Makeham":         {"SD": 12.03, "GMD": 12.27, "R": 0.981},
    "Ex3.6(4)":        {"SD": 0.757, "GMD": 0.675, "R": 1.121},
    "Ex4.1":           {"SD": 0.407, "GMD": 0.402, "R": 1.013},
    "Ex4.8":           {"SD": 0.291, "GMD": 0.335, "R": 0.8665},
}


def compute_benchmarks() -> pd.DataFrame:
    dists = {
        "Logistic(0,1)": make_logistic(0.0, 1.0),
        "Laplace(0,1)":  make_laplace(0.0, 1.0),
        "Gompertz":      make_gompertz(B=1e-4, c=0.09),
        "Makeham":       make_makeham(A=0.00022, B=2.7e-6, c=0.124),
        "Ex3.6(4)":      make_example_3_6_4(),
        "Ex4.1":         make_example_4_1(),
        "Ex4.8":         make_example_4_8(),
    }

    rows = []
    for name, dist in dists.items():
        sd, gmd = compute_sd_gmd(dist)
        R = sd / gmd if gmd > 0 else np.nan
        target = BENCHMARK_TARGETS[name]
        rows.append({
            "name":         name,
            "SD":           round(sd, 5),
            "GMD":          round(gmd, 5),
            "R":            round(R, 5),
            "target_SD":    target["SD"],
            "target_GMD":   target["GMD"],
            "target_R":     target["R"],
            "err_SD":       round(abs(sd - target["SD"]), 5),
            "err_GMD":      round(abs(gmd - target["GMD"]), 5),
            "dominant":     "SD" if sd > gmd else "GMD",
        })
        print(f"  {name:20s}  SD={sd:.4f}  GMD={gmd:.4f}  R={R:.4f}"
              f"  (target R={target['R']:.4f})")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Dominance verification
# ---------------------------------------------------------------------------

def verify_dominance(df_gamma: pd.DataFrame, df_weibull: pd.DataFrame,
                     df_gpd: pd.DataFrame, df_burr: pd.DataFrame):
    print("\n--- Dominance checks ---")

    # Gamma: alpha < 1 → SD > GMD
    below = df_gamma[df_gamma["alpha"] < 1.0 - 0.05]
    assert (below["dominant"] == "SD").all(), "Gamma alpha<1: expected SD dominant"
    print(f"✓ Gamma alpha<1:  SD dominant in all {len(below)} cases")

    # Gamma: alpha > 1 → GMD > SD
    above = df_gamma[df_gamma["alpha"] > 1.0 + 0.05]
    assert (above["dominant"] == "GMD").all(), "Gamma alpha>1: expected GMD dominant"
    print(f"✓ Gamma alpha>1:  GMD dominant in all {len(above)} cases")

    # Weibull: same pattern
    below_w = df_weibull[df_weibull["alpha"] < 1.0 - 0.05]
    assert (below_w["dominant"] == "SD").all(), "Weibull alpha<1: SD dominant"
    above_w = df_weibull[df_weibull["alpha"] > 1.0 + 0.05]
    assert (above_w["dominant"] == "GMD").all(), "Weibull alpha>1: GMD dominant"
    print(f"✓ Weibull: SD dominant for alpha<1, GMD dominant for alpha>1")

    # GPD: xi in [0.01, 0.49] → SD > GMD
    assert (df_gpd["dominant"] == "SD").all(), "GPD: expected SD dominant for all xi"
    print(f"✓ GPD xi∈[0.01,0.49]: SD dominant in all {len(df_gpd)} cases")

    # Burr: c<=1, ck>2 → SD > GMD
    burr_small_c = df_burr[df_burr["c"] <= 1.0]
    if len(burr_small_c) > 0:
        assert (burr_small_c["dominant"] == "SD").all(), "Burr c<=1: SD dominant"
        print(f"✓ Burr c≤1, ck>2: SD dominant in all {len(burr_small_c)} cases")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_gamma_weibull(df_gamma: pd.DataFrame, df_weibull: pd.DataFrame, out_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.plot(df_gamma["alpha"], df_gamma["g"], "b-", lw=2, label=r"$\Delta = SD - GMD$ (Gamma)")
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.axvline(1, color="r", lw=1.0, ls=":", label=r"$\alpha = 1$ (Exponential)")
    ax.set_xlabel(r"Shape $\alpha$")
    ax.set_ylabel(r"$SD[X] - GMD[X]$")
    ax.set_title("Gamma: SD − GMD vs α")
    ax.legend()

    ax = axes[1]
    ax.plot(df_weibull["alpha"], df_weibull["g"], "g-", lw=2, label=r"$\Delta = SD - GMD$ (Weibull)")
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.axvline(1, color="r", lw=1.0, ls=":", label=r"$\alpha = 1$ (Exponential)")
    ax.set_xlabel(r"Shape $\alpha$")
    ax.set_ylabel(r"$SD[X] - GMD[X]$")
    ax.set_title("Weibull: SD − GMD vs α")
    ax.legend()

    fig.suptitle("Experiment 2: Dominance Classification — Gamma vs Weibull", fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to {out_path}")


def plot_r_profiles(
    df_gamma: pd.DataFrame,
    df_weibull: pd.DataFrame,
    df_gpd: pd.DataFrame,
    df_beta1: pd.DataFrame,
    out_path: str,
):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    lower = np.sqrt(3.0) / 2.0

    ax = axes[0, 0]
    ax.plot(df_gamma["alpha"], df_gamma["R"], "b-", lw=2)
    ax.axhline(1, color="k", lw=0.8, ls="--")
    ax.axhline(lower, color="gray", lw=0.8, ls=":", label=r"$\sqrt{3}/2$")
    ax.axvline(1, color="r", lw=1.0, ls=":")
    ax.set_xlabel(r"$\alpha$"); ax.set_ylabel("R = SD/GMD")
    ax.set_title("Gamma: R vs α")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(df_weibull["alpha"], df_weibull["R"], "g-", lw=2)
    ax.axhline(1, color="k", lw=0.8, ls="--")
    ax.axhline(lower, color="gray", lw=0.8, ls=":", label=r"$\sqrt{3}/2$")
    ax.axvline(1, color="r", lw=1.0, ls=":")
    ax.set_xlabel(r"$\alpha$"); ax.set_ylabel("R = SD/GMD")
    ax.set_title("Weibull: R vs α")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.plot(df_gpd["xi"], df_gpd["R"], "m-", lw=2)
    ax.axhline(1, color="k", lw=0.8, ls="--")
    ax.axhline(lower, color="gray", lw=0.8, ls=":", label=r"$\sqrt{3}/2$")
    ax.set_xlabel(r"$\xi$"); ax.set_ylabel("R = SD/GMD")
    ax.set_title("GPD: R vs ξ")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    df_b1 = df_beta1[df_beta1["beta"] == 1]
    df_b2 = df_beta1[df_beta1["beta"] == 2]
    if len(df_b1):
        ax.plot(df_b1["alpha"], df_b1["R"], "b-", lw=2, label="Beta(α,1)")
    if len(df_b2):
        ax.plot(df_b2["alpha"], df_b2["R"], "r-", lw=2, label="Beta(α,2)")
    ax.axhline(1, color="k", lw=0.8, ls="--")
    ax.axhline(lower, color="gray", lw=0.8, ls=":", label=r"$\sqrt{3}/2$")
    ax.set_xlabel(r"$\alpha$"); ax.set_ylabel("R = SD/GMD")
    ax.set_title("Beta: R vs α")
    ax.legend(fontsize=8)

    fig.suptitle("Experiment 2: R = SD/GMD profiles by family", fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(FIGS_DIR, exist_ok=True)

    print("=" * 60)
    print("Experiment 2: Dominance Classification")
    print("=" * 60)

    # ---- Gamma sweep ----
    print("\n[1] Gamma family sweep ...")
    alphas_gamma = np.linspace(0.1, 5.0, 100)
    df_gamma = sweep_gamma(alphas_gamma, theta=1.0)
    df_gamma.to_csv(os.path.join(TABLES_DIR, "exp2_gamma.csv"), index=False)
    print(f"    saved exp2_gamma.csv  ({len(df_gamma)} rows)")

    # ---- Weibull sweep ----
    print("[2] Weibull family sweep ...")
    alphas_wb = np.linspace(0.1, 5.0, 100)
    df_weibull = sweep_weibull(alphas_wb, scale=1.0)
    df_weibull.to_csv(os.path.join(TABLES_DIR, "exp2_weibull.csv"), index=False)
    print(f"    saved exp2_weibull.csv  ({len(df_weibull)} rows)")

    # ---- GPD sweep ----
    print("[3] GPD family sweep ...")
    xis = np.linspace(0.01, 0.49, 50)
    df_gpd = sweep_gpd(xis, scale=1.0)
    df_gpd.to_csv(os.path.join(TABLES_DIR, "exp2_gpd.csv"), index=False)
    print(f"    saved exp2_gpd.csv  ({len(df_gpd)} rows)")

    # ---- Burr sweep ----
    print("[4] Burr family sweep ...")
    # c in [0.1, 0.5, 1.0], k chosen so ck > 2
    c_vals = [0.1, 0.1, 0.5, 0.5, 1.0, 1.0]
    k_vals = [30,  50,  6,   10,  3,   5  ]
    df_burr = sweep_burr(c_vals, k_vals)
    df_burr.to_csv(os.path.join(TABLES_DIR, "exp2_burr.csv"), index=False)
    print(f"    saved exp2_burr.csv  ({len(df_burr)} rows)")

    # ---- Beta sweep ----
    print("[5] Beta family sweep ...")
    alphas_beta = np.linspace(0.5, 5.0, 50)
    df_beta = sweep_beta(alphas_beta, beta_vals=[1, 2])
    df_beta.to_csv(os.path.join(TABLES_DIR, "exp2_beta.csv"), index=False)
    print(f"    saved exp2_beta.csv  ({len(df_beta)} rows)")

    # ---- Benchmark point values ----
    print("\n[6] Benchmark point values ...")
    df_bench = compute_benchmarks()
    df_bench.to_csv(os.path.join(TABLES_DIR, "exp2_benchmarks.csv"), index=False)
    print(f"    saved exp2_benchmarks.csv  ({len(df_bench)} rows)")

    # ---- Dominance verification ----
    verify_dominance(df_gamma, df_weibull, df_gpd, df_burr)

    # ---- Plots ----
    print("\n[7] Generating figures ...")
    plot_gamma_weibull(
        df_gamma, df_weibull,
        os.path.join(FIGS_DIR, "exp2_gamma_weibull.png"),
    )
    plot_r_profiles(
        df_gamma, df_weibull, df_gpd, df_beta,
        os.path.join(FIGS_DIR, "exp2_r_profiles.png"),
    )

    print("\n" + "=" * 60)
    print("Experiment 2 COMPLETED successfully.")
    print("=" * 60)
    return {
        "gamma": df_gamma,
        "weibull": df_weibull,
        "gpd": df_gpd,
        "burr": df_burr,
        "beta": df_beta,
        "benchmarks": df_bench,
    }


if __name__ == "__main__":
    main()
