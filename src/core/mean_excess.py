import numpy as np
from scipy.integrate import quad

from src.core.distributions import DistributionSpec
from src.core.moments import _quad_full, _split_point


def compute_m_Y_QF(dist: DistributionSpec, t: float, tol: float = 1e-12) -> float:
    """m_Y^F(t) via the F-weighted representation.

    C_X(x, t) = F(x - t) / F(x)
    Numerator:   2 * integral C_X(x,t) * S(x) * F(x) dx
    Denominator: 2 * integral C_X(x,t) * F(x) * f(x) dx

    Result: Numerator / Denominator
    """
    lo, hi = dist.support
    # C_X(x, t) is nonzero only where F(x) > 0 and x - t is inside the support
    lo_eff = max(lo, lo + t)
    split = _split_point(dist)
    split_eff = split if (split is not None and lo_eff < split < hi) else None

    def _C(x):
        fx = dist.F(x)
        if fx <= 0.0:
            return 0.0
        return dist.F(x - t) / fx

    num = _quad_full(lambda x: _C(x) * dist.S(x) * dist.F(x), lo_eff, hi, split_eff, tol)
    den = _quad_full(lambda x: _C(x) * dist.F(x) * dist.f(x), lo_eff, hi, split_eff, tol)

    if abs(den) < tol:
        return np.nan
    return num / den


def compute_m_Y_QS(dist: DistributionSpec, t: float, tol: float = 1e-12) -> float:
    """m_Y^S(t) via the S-weighted representation.

    D_X(x, t) = S(x + t) / S(x)
    Numerator:   2 * integral D_X(x,t) * F(x) * S(x) dx
    Denominator: 2 * integral D_X(x,t) * S(x) * f(x) dx

    Result: Numerator / Denominator
    """
    lo, hi = dist.support
    split = _split_point(dist)

    def _D(x):
        sx = dist.S(x)
        if sx <= 0.0:
            return 0.0
        return dist.S(x + t) / sx

    num = _quad_full(lambda x: _D(x) * dist.F(x) * dist.S(x), lo, hi, split, tol)
    den = _quad_full(lambda x: _D(x) * dist.S(x) * dist.f(x), lo, hi, split, tol)

    if abs(den) < tol:
        return np.nan
    return num / den
