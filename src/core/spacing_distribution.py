import numpy as np
from scipy.integrate import quad

from src.core.distributions import DistributionSpec
from src.core.moments import _quad_full, _split_point


def compute_S_Y(dist: DistributionSpec, t: float, tol: float = 1e-12) -> float:
    """S_Y(t) = P(|X - X'| > t).

    Identity: S_Y(t) = 2 * integral_support f(x) * S(x + t) dx.
    Valid for nonneg continuous X (and by extension for distributions
    where S(x+t) is well-defined over the full support).
    """
    lo, hi = dist.support
    lo_eff = max(lo, -1e6)
    split = _split_point(dist)
    result = _quad_full(
        lambda x: dist.f(x) * dist.S(x + t),
        lo_eff, hi, split, tol,
    )
    return 2.0 * result


def compute_m_Y_direct(
    dist: DistributionSpec,
    t: float,
    t_grid: np.ndarray,
    tol: float = 1e-12,
) -> float:
    """m_Y(t) = E[Y - t | Y > t] = integral_t^inf S_Y(w) dw / S_Y(t).

    Numerically integrate S_Y over (t, max(t_grid)) using the precomputed
    t_grid for efficiency; then divide by S_Y(t).
    """
    # Restrict t_grid to values > t
    mask = t_grid > t
    if not np.any(mask):
        return 0.0

    t_upper = t_grid[mask]
    # Compute S_Y on the upper tail grid
    sy_vals = np.array([compute_S_Y(dist, w, tol=tol) for w in t_upper])

    # Integrate S_Y from t to t_grid[-1] via trapezoidal rule
    t_ext = np.concatenate([[t], t_upper])
    sy_ext = np.concatenate([[compute_S_Y(dist, t, tol=tol)], sy_vals])
    numerator = np.trapz(sy_ext, t_ext)

    sy_t = sy_ext[0]
    if sy_t <= 0.0:
        return 0.0
    return numerator / sy_t
