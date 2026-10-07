import numpy as np
from scipy.integrate import quad

from src.core.distributions import DistributionSpec


def compute_truncated_moments(
    dist: DistributionSpec,
    u: float,
    tol: float = 1e-12,
) -> dict:
    """Compute moments of X_u^+ = (X | X > u).

    Conditional density: f_+(x) = f(x) / S(u) for x > u.
    """
    lo, hi = dist.support
    su = dist.S(u)
    if su <= 0.0:
        return {"mean": np.nan, "e2": np.nan, "sd": np.nan, "gmd": np.nan, "su": 0.0}

    def f_plus(x):
        return dist.f(x) / su

    def F_plus(x):
        # P(X_u^+ <= x) = (F(x) - F(u)) / S(u) for x > u
        return (dist.F(x) - dist.F(u)) / su

    def S_plus(x):
        return dist.S(x) / su

    # Mean
    mean_val, _ = quad(lambda x: x * f_plus(x), u, hi,
                       limit=200, epsabs=tol, epsrel=tol)
    # E[X^2]
    e2, _ = quad(lambda x: x ** 2 * f_plus(x), u, hi,
                 limit=200, epsabs=tol, epsrel=tol)
    var_val = e2 - mean_val ** 2
    sd_val = np.sqrt(max(var_val, 0.0))

    # GMD via 1D identity: 2 * integral F_+(x) * S_+(x) dx over (u, hi)
    def gmd_integrand(x):
        return F_plus(x) * S_plus(x)

    gmd_val, _ = quad(gmd_integrand, u, hi,
                      limit=200, epsabs=tol, epsrel=tol)
    gmd_val *= 2.0

    return {"mean": mean_val, "e2": e2, "sd": sd_val, "gmd": gmd_val, "su": su}


def compute_delta_u(
    dist: DistributionSpec,
    u_grid: np.ndarray,
    tol: float = 1e-12,
) -> np.ndarray:
    """Delta(u) = SD[X_u^+] - GMD[X_u^+] for each u in u_grid."""
    deltas = np.empty(len(u_grid))
    for i, u in enumerate(u_grid):
        mom = compute_truncated_moments(dist, u, tol=tol)
        deltas[i] = mom["sd"] - mom["gmd"]
    return deltas
