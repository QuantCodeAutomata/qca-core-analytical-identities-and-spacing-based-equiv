"""
Experiment 3: Mean-excess representation of Y = |X - X'|.

For a non-negative random variable X with iid copy X', defines Y = |X - X'| and
studies its survival function S_Y, mean-excess function m_Y, and two weighted-
measure representations m_Y^F (via Q^F) and m_Y^S (via Q^S).

All distributions and helper functions are defined inline; no src/core imports.
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
TABLES_DIR = "/workspace/project/results/tables"
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Distribution factories
# Each dict contains: name, f, F, S, cutoff
# ---------------------------------------------------------------------------

def make_exponential(rate=1.0, cutoff=50.0):
    rv = stats.expon(scale=1.0 / rate)
    return {
        "name": f"Exp(rate={rate})",
        "f": rv.pdf,
        "F": rv.cdf,
        "S": rv.sf,
        "cutoff": cutoff,
    }


def make_gamma(alpha, rate=1.0, cutoff=50.0):
    rv = stats.gamma(a=alpha, scale=1.0 / rate)
    return {
        "name": f"Gamma(a={alpha})",
        "f": rv.pdf,
        "F": rv.cdf,
        "S": rv.sf,
        "cutoff": cutoff,
    }


def make_weibull(alpha, cutoff=None):
    """Weibull with F(x) = 1 - exp(-x^alpha), x >= 0."""
    if cutoff is None:
        # Weibull(0.5): heavy-ish tail; Weibull(2): very light tail
        cutoff = 100.0 if alpha < 1.0 else 8.0
    rv = stats.weibull_min(c=alpha, scale=1.0)
    return {
        "name": f"Weibull(a={alpha})",
        "f": rv.pdf,
        "F": rv.cdf,
        "S": rv.sf,
        "cutoff": cutoff,
    }


def make_gpd(xi=0.2, sigma=1.0, cutoff=100.0):
    """Generalised Pareto: F(x) = 1 - (1 + xi*x/sigma)^{-1/xi}, x >= 0."""
    rv = stats.genpareto(c=xi, scale=sigma)
    return {
        "name": f"GPD(xi={xi})",
        "f": rv.pdf,
        "F": rv.cdf,
        "S": rv.sf,
        "cutoff": cutoff,
    }


# ---------------------------------------------------------------------------
# Core computations
# ---------------------------------------------------------------------------

def compute_S_Y(dist, t, eps=1e-14):
    """
    S_Y(t) = P(|X - X'| > t) = 2 * int_0^C f(x) S(x+t) dx.

    For Weibull(0.5) the integrand has an integrable singularity at x=0;
    scipy.quad handles it with the points hint.
    """
    f, S, cutoff = dist["f"], dist["S"], dist["cutoff"]
    upper = max(cutoff - t, eps)

    def integrand(x):
        return f(x) * S(x + t)

    result, _ = integrate.quad(
        integrand, eps, upper,
        limit=400, epsabs=1e-11, epsrel=1e-9,
    )
    return 2.0 * result


def build_S_Y_grid(dist, n_points=300):
    """Pre-compute S_Y on a fine grid from 0 to cutoff/2 for fast interpolation."""
    cutoff = dist["cutoff"]
    t_grid = np.linspace(0.0, cutoff / 2.0, n_points)
    S_Y_vals = np.array([compute_S_Y(dist, t) for t in t_grid])
    return t_grid, S_Y_vals


def compute_gmd_from_grid(t_grid, S_Y_vals):
    """GMD = E[|X - X'|] = int_0^inf S_Y(t) dt, via trapz on pre-computed grid."""
    return float(np.trapz(S_Y_vals, t_grid))


def compute_m_Y_from_grid(t, t_grid, S_Y_vals):
    """
    m_Y(t) = int_t^inf S_Y(w) dw / S_Y(t)  (mean-excess function of Y).
    Uses linear interpolation on the pre-computed S_Y grid.
    """
    S_Y_t = float(np.interp(t, t_grid, S_Y_vals))
    if S_Y_t < 1e-15:
        return np.nan
    mask = t_grid >= t
    tail = float(np.trapz(S_Y_vals[mask], t_grid[mask]))
    return tail / S_Y_t


def compute_m_Y_QF(dist, t, eps=1e-14):
    """
    m_Y^F(t) via the Q^F weighted measure:

        m_Y^F(t) = int_{x>=t} F(x-t) S(x) dx
                 / int_{x>=t} F(x-t) f(x) dx
    """
    f, F, S, cutoff = dist["f"], dist["F"], dist["S"], dist["cutoff"]

    def num_integrand(x):
        return F(x - t) * S(x)

    def den_integrand(x):
        return F(x - t) * f(x)

    num, _ = integrate.quad(num_integrand, t + eps, cutoff, limit=400, epsabs=1e-11)
    den, _ = integrate.quad(den_integrand, t + eps, cutoff, limit=400, epsabs=1e-11)
    return num / den if den > 1e-15 else np.nan


def compute_m_Y_QS(dist, t, eps=1e-14):
    """
    m_Y^S(t) via the Q^S weighted measure:

        m_Y^S(t) = int_0^C S(x+t) F(x) dx
                 / int_0^C S(x+t) f(x) dx
    """
    f, F, S, cutoff = dist["f"], dist["F"], dist["S"], dist["cutoff"]

    def num_integrand(x):
        return S(x + t) * F(x)

    def den_integrand(x):
        return S(x + t) * f(x)

    num, _ = integrate.quad(num_integrand, eps, cutoff, limit=400, epsabs=1e-11)
    den, _ = integrate.quad(den_integrand, eps, cutoff, limit=400, epsabs=1e-11)
    return num / den if den > 1e-15 else np.nan


# ---------------------------------------------------------------------------
# Analysis pipeline for one distribution
# ---------------------------------------------------------------------------

def analyse_distribution(dist, t_table, t_plot_grid):
    """
    Returns
    -------
    gmd       : float   – Gini mean difference E[|X-X'|]
    m_Y_plot  : ndarray – m_Y(t) evaluated on t_plot_grid
    rows      : list[dict] – table rows for the comparison table
    """
    # Precompute S_Y grid (300 points) for fast integrals
    t_grid, S_Y_vals = build_S_Y_grid(dist, n_points=300)

    gmd = compute_gmd_from_grid(t_grid, S_Y_vals)

    # m_Y on plotting grid
    m_Y_plot = np.array([compute_m_Y_from_grid(t, t_grid, S_Y_vals) for t in t_plot_grid])

    # Table rows at specified t values
    rows = []
    for t in t_table:
        m_direct = compute_m_Y_from_grid(t, t_grid, S_Y_vals)
        m_QF = compute_m_Y_QF(dist, t)
        m_QS = compute_m_Y_QS(dist, t)

        finite_vals = [v for v in [m_QF, m_QS] if np.isfinite(v)]
        max_err = max(abs(m_direct - v) for v in finite_vals) if finite_vals else np.nan

        rows.append({
            "family":      dist["name"],
            "t":           t,
            "m_Y_direct":  round(m_direct, 6),
            "m_Y_QF":      round(m_QF,     6) if np.isfinite(m_QF) else np.nan,
            "m_Y_QS":      round(m_QS,     6) if np.isfinite(m_QS) else np.nan,
            "max_error":   round(max_err,  8) if np.isfinite(max_err) else np.nan,
            "GMD":         round(gmd,      6),
        })

    return gmd, m_Y_plot, rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    distributions = [
        make_exponential(rate=1.0, cutoff=50.0),
        make_gamma(alpha=0.5,  rate=1.0, cutoff=50.0),
        make_gamma(alpha=2.0,  rate=1.0, cutoff=50.0),
        make_weibull(alpha=0.5, cutoff=100.0),
        make_weibull(alpha=2.0, cutoff=8.0),
        make_gpd(xi=0.2, sigma=1.0, cutoff=100.0),
    ]

    t_table = [0.5, 1.0, 2.0, 3.0]
    t_plot  = np.linspace(0.05, 5.0, 60)

    all_rows = []

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.ravel()

    for idx, dist in enumerate(distributions):
        print(f"  Analysing {dist['name']} ...", flush=True)
        gmd, m_Y_plot, rows = analyse_distribution(dist, t_table, t_plot)
        all_rows.extend(rows)

        ax = axes[idx]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ax.plot(t_plot, m_Y_plot, "b-", linewidth=1.8, label=r"$m_Y(t)$")
        ax.axhline(gmd, color="r", linestyle="--", linewidth=1.2, label=f"GMD = {gmd:.3f}")
        ax.set_title(dist["name"], fontsize=11)
        ax.set_xlabel("t")
        ax.set_ylabel(r"$m_Y(t)$")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 5)

        print(f"    GMD = {gmd:.5f}", flush=True)

    plt.suptitle("Experiment 3: Mean-excess function $m_Y(t)$ of $Y = |X - X'|$", fontsize=13)
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "exp3_mean_excess.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure saved to {fig_path}")

    df = pd.DataFrame(all_rows)
    csv_path = os.path.join(TABLES_DIR, "exp3_mean_excess.csv")
    df.to_csv(csv_path, index=False)
    print(f"Table saved to {csv_path}")
    print(df.to_string())


if __name__ == "__main__":
    main()
