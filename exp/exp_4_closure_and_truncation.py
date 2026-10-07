"""
Experiment 4: Closure properties under mixing, convolution, and truncation.

Part A  – Gamma frailty mixture (Lomax distribution): SD-dominant family.
Part B  – Convolution of Gamma and Normal: GMD-dominant family.
Part C  – Tail truncation examples:
           5.4  Custom hazard h(x) = x·exp(-0.1x) + 1
           5.6  Normal scale mixture
           Makeham mortality model

No src/core imports; all helpers are self-contained.
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import integrate, stats

FIGURES_DIR = "/workspace/project/results/figures"
TABLES_DIR  = "/workspace/project/results/tables"
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)

# ===========================================================================
# Generic SD and GMD helpers
# ===========================================================================

def compute_gmd_continuous(F, S, lower, upper, limit=500):
    """GMD = 2 ∫ F(x) S(x) dx  for a distribution supported on [lower, upper]."""
    val, _ = integrate.quad(lambda x: 2.0 * F(x) * S(x), lower, upper,
                            limit=limit, epsabs=1e-11, epsrel=1e-9)
    return val


def compute_sd_continuous(f, S, lower, upper, mean=None, limit=500):
    """SD from the variance identity  Var = 2 ∫ x S(x) dx - (∫ S(x) dx)^2."""
    ex,  _ = integrate.quad(lambda x: S(x),       lower, upper, limit=limit)
    ex2, _ = integrate.quad(lambda x: 2.0*x*S(x), lower, upper, limit=limit)
    if mean is not None:
        ex = mean
    return float(np.sqrt(max(ex2 - ex**2, 0.0)))


def compute_truncated_stats(f, F, S, u, cutoff, limit=300):
    """
    Returns (SD, GMD) for the left-truncated variable X_u^+ = (X | X > u).

    SD computed from first two moments of f_u(x) = f(x)/S(u).
    GMD = (2/S(u)^2) * ∫_u^C (S(u) - S(x)) · S(x) dx.
    """
    Su = float(S(u))
    if Su < 1e-14:
        return np.nan, np.nan

    # First two moments
    ex,  _ = integrate.quad(lambda x: x * f(x), u, cutoff, limit=limit)
    ex2, _ = integrate.quad(lambda x: x**2 * f(x), u, cutoff, limit=limit)
    ex  /= Su
    ex2 /= Su
    var  = ex2 - ex**2
    sd   = float(np.sqrt(max(var, 0.0)))

    # GMD via survival-function identity
    int_S,  _ = integrate.quad(lambda x: S(x),       u, cutoff, limit=limit)
    int_S2, _ = integrate.quad(lambda x: S(x)**2,    u, cutoff, limit=limit)
    gmd = 2.0 / Su**2 * (Su * int_S - int_S2)

    return sd, gmd


# ===========================================================================
# Part A – Lomax / Gamma frailty mixture
# ===========================================================================

def lomax_pdf(alpha, beta, x):
    return alpha * beta**alpha * (x + beta)**(-(alpha + 1))

def lomax_cdf(alpha, beta, x):
    return 1.0 - (beta / (x + beta))**alpha

def lomax_sf(alpha, beta, x):
    return (beta / (x + beta))**alpha

def lomax_sd_formula(alpha, beta):
    """Closed-form SD: requires alpha > 2."""
    var = alpha * beta**2 / ((alpha - 1)**2 * (alpha - 2))
    return float(np.sqrt(var))

def run_part_A():
    print("\n=== Part A: Lomax (Gamma frailty mixture) ===")
    rows = []
    for alpha in [3.0, 4.0, 5.0]:
        beta   = 1.0
        cutoff = 300.0 / alpha          # generous tail cutoff
        f  = lambda x, a=alpha, b=beta: lomax_pdf(a, b, x)
        Fv = lambda x, a=alpha, b=beta: lomax_cdf(a, b, x)
        Sv = lambda x, a=alpha, b=beta: lomax_sf(a, b, x)

        sd  = lomax_sd_formula(alpha, beta)
        gmd = compute_gmd_continuous(Fv, Sv, 0, cutoff)
        rows.append({
            "alpha": alpha, "beta": beta,
            "SD": round(sd, 6), "GMD": round(gmd, 6),
            "SD/GMD": round(sd / gmd, 4),
            "SD_dominant": sd > gmd,
        })
        print(f"  Lomax(alpha={alpha:.0f}): SD={sd:.4f}, GMD={gmd:.4f}, "
              f"SD>GMD={sd > gmd}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(TABLES_DIR, "exp4_partA_lomax.csv"), index=False)
    return df


# ===========================================================================
# Part B – Convolution
# ===========================================================================

def run_part_B():
    print("\n=== Part B: Convolution ===")
    rows = []

    # Gamma(2)+Gamma(2) = Gamma(4,1)
    rv_g = stats.gamma(a=4, scale=1.0)
    cutoff_g = 30.0
    sd_g  = float(np.sqrt(rv_g.var()))
    gmd_g = compute_gmd_continuous(rv_g.cdf, rv_g.sf, 0.0, cutoff_g)
    rows.append({
        "distribution": "Gamma(4,1) = Gamma(2)+Gamma(2)",
        "SD": round(sd_g, 6), "GMD": round(gmd_g, 6),
        "GMD/SD": round(gmd_g / sd_g, 4), "GMD_dominant": gmd_g > sd_g,
    })
    print(f"  Gamma(4,1): SD={sd_g:.4f}, GMD={gmd_g:.4f}, GMD>SD={gmd_g > sd_g}")

    # N(0,1)+N(0,1) = N(0,2)
    sigma2 = 2.0
    sd_n  = float(np.sqrt(sigma2))
    gmd_n = 2.0 * np.sqrt(sigma2) / np.sqrt(np.pi)   # GMD[N(0,sigma^2)] = 2sigma/sqrt(pi)
    rows.append({
        "distribution": "N(0,2) = N(0,1)+N(0,1)",
        "SD": round(sd_n, 6), "GMD": round(gmd_n, 6),
        "GMD/SD": round(gmd_n / sd_n, 4), "GMD_dominant": gmd_n > sd_n,
    })
    print(f"  N(0,2):    SD={sd_n:.4f}, GMD={gmd_n:.4f}, GMD>SD={gmd_n > sd_n}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(TABLES_DIR, "exp4_partB_convolution.csv"), index=False)
    return df


# ===========================================================================
# Part C – Tail truncation
# ===========================================================================

# ---- Example 5.4: h(x) = x·exp(-0.1x) + 1 --------------------------------

def ex54_hazard(x):
    return x * np.exp(-0.1 * x) + 1.0


def ex54_cum_hazard(x):
    """H(x) = ∫_0^x h(t) dt = 100 - 10x·exp(-0.1x) - 100·exp(-0.1x) + x."""
    return 100.0 - 10.0*x*np.exp(-0.1*x) - 100.0*np.exp(-0.1*x) + x


def run_ex54(u_grid, cutoff=60.0):
    """Compute Delta(u) = SD[X_u^+] - GMD[X_u^+] for Example 5.4."""

    def S(x):
        return np.exp(-ex54_cum_hazard(np.asarray(x, dtype=float)))

    def f(x):
        x = np.asarray(x, dtype=float)
        return ex54_hazard(x) * S(x)

    def F(x):
        return 1.0 - S(x)

    deltas, sds, gmds = [], [], []
    for u in u_grid:
        sd, gmd = compute_truncated_stats(f, F, S, u, cutoff)
        deltas.append(sd - gmd if np.isfinite(sd) and np.isfinite(gmd) else np.nan)
        sds.append(sd)
        gmds.append(gmd)

    return np.array(deltas), np.array(sds), np.array(gmds)


# ---- Example 5.6: Normal scale mixture ------------------------------------

def ex56_make_dist(sigma1=0.5, sigma2=2.0, q=0.75):
    rv1 = stats.norm(0, sigma1)
    rv2 = stats.norm(0, sigma2)

    def f(x):
        return q * rv1.pdf(x) + (1.0 - q) * rv2.pdf(x)

    def F(x):
        return q * rv1.cdf(x) + (1.0 - q) * rv2.cdf(x)

    def S(x):
        return 1.0 - F(x)

    return f, F, S


def run_ex56(u_grid, cutoff=15.0):
    """
    Compute full-distribution GMD, then Delta(u) for the truncated variable.
    """
    sigma1, sigma2, q = 0.5, 2.0, 0.75
    f, F, S = ex56_make_dist(sigma1, sigma2, q)

    var_mix = q * sigma1**2 + (1.0 - q) * sigma2**2   # = 19/16
    sd_full  = float(np.sqrt(var_mix))

    # Full GMD  (symmetric distribution – integrate on full real line)
    gmd_full, _ = integrate.quad(lambda x: 2.0*F(x)*S(x), -cutoff, cutoff,
                                 limit=500, epsabs=1e-11)
    r_ratio = sd_full / gmd_full

    print(f"\n=== Example 5.6: Normal scale mixture ===")
    print(f"  Var = {var_mix:.6f} (expected {19/16:.6f})")
    print(f"  SD  = {sd_full:.6f}  (expected ≈ 1.0897)")
    print(f"  GMD = {gmd_full:.6f} (expected ≈ 1.0752)")
    print(f"  R   = {r_ratio:.4f}  (expected ≈ 1.013)")

    deltas, sds, gmds = [], [], []
    for u in u_grid:
        sd, gmd = compute_truncated_stats(f, F, S, u, cutoff)
        deltas.append(sd - gmd if np.isfinite(sd) and np.isfinite(gmd) else np.nan)
        sds.append(sd)
        gmds.append(gmd)

    return np.array(deltas), np.array(sds), np.array(gmds), sd_full, gmd_full


# ---- Makeham mortality model -----------------------------------------------

def makeham_make_dist(A=0.00022, B=2.7e-6, c=0.124):
    def h(x):
        return A + B * np.exp(c * x)

    def H(x):
        return A * x + B / c * (np.exp(c * x) - 1.0)

    def S(x):
        return np.exp(-H(np.asarray(x, dtype=float)))

    def f(x):
        x = np.asarray(x, dtype=float)
        return h(x) * S(x)

    def F(x):
        return 1.0 - S(x)

    return f, F, S


def run_makeham(u_grid, cutoff=115.0):
    f, F, S = makeham_make_dist()

    # Full distribution SD and GMD
    sd_full = compute_sd_continuous(f, S, 0.0, cutoff)
    gmd_full = compute_gmd_continuous(F, S, 0.0, cutoff)
    print(f"\n=== Makeham distribution ===")
    print(f"  SD  = {sd_full:.4f},  GMD = {gmd_full:.4f},  SD>GMD = {sd_full > gmd_full}")

    deltas, sds, gmds = [], [], []
    for u in u_grid:
        sd, gmd = compute_truncated_stats(f, F, S, u, cutoff)
        deltas.append(sd - gmd if np.isfinite(sd) and np.isfinite(gmd) else np.nan)
        sds.append(sd)
        gmds.append(gmd)

    return np.array(deltas), np.array(sds), np.array(gmds), sd_full, gmd_full


# ===========================================================================
# Main
# ===========================================================================

def _find_sign_change(u_grid, deltas):
    """Return the first u where Delta changes sign (np.nan if not found)."""
    d = np.asarray(deltas, dtype=float)
    for i in range(len(d) - 1):
        if np.isfinite(d[i]) and np.isfinite(d[i+1]) and d[i] * d[i+1] < 0:
            # Linear interpolation
            u0, u1 = u_grid[i], u_grid[i+1]
            d0, d1 = d[i], d[i+1]
            return u0 - d0 * (u1 - u0) / (d1 - d0)
    return np.nan


def main():
    # ---- Part A ----
    df_A = run_part_A()

    # ---- Part B ----
    df_B = run_part_B()

    # ---- Part C ----
    print("\n=== Part C: Tail truncation ===")

    # Example 5.4
    u_grid_54 = np.linspace(0.0, 20.0, 80)
    delta_54, sd_54, gmd_54 = run_ex54(u_grid_54, cutoff=60.0)
    sc_54 = _find_sign_change(u_grid_54, delta_54)
    print(f"\n  Example 5.4: sign-change Delta at u ≈ {sc_54:.2f} (expected ≈ 9.7)")

    df_54 = pd.DataFrame({
        "u": u_grid_54, "SD": sd_54, "GMD": gmd_54, "Delta": delta_54
    })
    df_54.to_csv(os.path.join(TABLES_DIR, "exp4_ex54_delta.csv"), index=False)

    # Example 5.6
    u_grid_56 = np.linspace(0.0, 3.0, 60)
    delta_56, sd_56, gmd_56, sd_full_56, gmd_full_56 = run_ex56(u_grid_56, cutoff=15.0)
    sc_56 = _find_sign_change(u_grid_56, delta_56)
    print(f"\n  Example 5.6: sign-change Delta at u ≈ {sc_56:.2f} (expected ≈ 0.65)")

    df_56 = pd.DataFrame({
        "u": u_grid_56, "SD": sd_56, "GMD": gmd_56, "Delta": delta_56
    })
    df_56.to_csv(os.path.join(TABLES_DIR, "exp4_ex56_delta.csv"), index=False)

    # Makeham
    u_grid_mk = np.linspace(0.0, 100.0, 80)
    delta_mk, sd_mk, gmd_mk, sd_full_mk, gmd_full_mk = run_makeham(u_grid_mk, cutoff=115.0)
    sc_mk = _find_sign_change(u_grid_mk, delta_mk)
    print(f"\n  Makeham: sign-change Delta at u ≈ {sc_mk:.2f} (expected ≈ 60.7)")

    df_mk = pd.DataFrame({
        "u": u_grid_mk, "SD": sd_mk, "GMD": gmd_mk, "Delta": delta_mk
    })
    df_mk.to_csv(os.path.join(TABLES_DIR, "exp4_makeham_delta.csv"), index=False)

    # ---- Figures ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel A – Lomax SD vs GMD bar chart
    ax = axes[0, 0]
    alphas = df_A["alpha"].values
    x_pos  = np.arange(len(alphas))
    width  = 0.35
    ax.bar(x_pos - width/2, df_A["SD"],  width, label="SD",  color="steelblue", alpha=0.8)
    ax.bar(x_pos + width/2, df_A["GMD"], width, label="GMD", color="darkorange", alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"α={a:.0f}" for a in alphas])
    ax.set_title("Part A: Lomax — SD vs GMD")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Panel B – Delta(u) Example 5.4
    ax = axes[0, 1]
    ax.plot(u_grid_54, delta_54, "b-", linewidth=1.8)
    ax.axhline(0, color="k", linewidth=0.8, linestyle="--")
    if np.isfinite(sc_54):
        ax.axvline(sc_54, color="r", linewidth=1.2, linestyle=":",
                   label=f"sign change u≈{sc_54:.1f}")
        ax.legend(fontsize=9)
    ax.set_title("Example 5.4: Δ(u) = SD[X_u⁺] − GMD[X_u⁺]")
    ax.set_xlabel("u")
    ax.set_ylabel("Δ(u)")
    ax.grid(True, alpha=0.3)

    # Panel C – Delta(u) Example 5.6
    ax = axes[1, 0]
    ax.plot(u_grid_56, delta_56, "b-", linewidth=1.8)
    ax.axhline(0, color="k", linewidth=0.8, linestyle="--")
    if np.isfinite(sc_56):
        ax.axvline(sc_56, color="r", linewidth=1.2, linestyle=":",
                   label=f"sign change u≈{sc_56:.2f}")
        ax.legend(fontsize=9)
    ax.set_title("Example 5.6: Normal scale mixture Δ(u)")
    ax.set_xlabel("u")
    ax.set_ylabel("Δ(u)")
    ax.grid(True, alpha=0.3)

    # Panel D – Delta(u) Makeham
    ax = axes[1, 1]
    ax.plot(u_grid_mk, delta_mk, "b-", linewidth=1.8)
    ax.axhline(0, color="k", linewidth=0.8, linestyle="--")
    if np.isfinite(sc_mk):
        ax.axvline(sc_mk, color="r", linewidth=1.2, linestyle=":",
                   label=f"sign change u≈{sc_mk:.1f}")
        ax.legend(fontsize=9)
    ax.set_title("Makeham: Δ(u) = SD[X_u⁺] − GMD[X_u⁺]")
    ax.set_xlabel("u (age)")
    ax.set_ylabel("Δ(u)")
    ax.grid(True, alpha=0.3)

    plt.suptitle("Experiment 4: Closure Properties and Truncation", fontsize=13)
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "exp4_closure_truncation.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nFigure saved to {fig_path}")


if __name__ == "__main__":
    main()
