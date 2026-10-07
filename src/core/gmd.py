import numpy as np
from scipy.integrate import quad, dblquad

from src.core.distributions import DistributionSpec
from src.core.moments import compute_mean, compute_maxiance, _quad_full, _split_point


def compute_gmd_pairwise(dist: DistributionSpec, tol: float = 1e-8) -> float:
    """GMD[X] = 2 * integral_{x < x'} (x'-x)*f(x)*f(x') dx dx'."""
    lo, hi = dist.support

    # Clip infinite bounds for dblquad to a large finite range
    lo_eff = max(lo, -1000.0)
    hi_eff = min(hi, 1000.0)

    def inner(x_prime, x):
        return (x_prime - x) * dist.f(x) * dist.f(x_prime)

    result, _ = dblquad(
        inner,
        lo_eff, hi_eff,
        lambda x: x, lambda x: hi_eff,
        epsabs=tol, epsrel=tol,
    )
    return 2.0 * result


def compute_gmd_1d(dist: DistributionSpec, tol: float = 1e-12) -> float:
    """GMD[X] = 2 * integral F(x)*S(x) dx over support."""
    lo, hi = dist.support
    split = _split_point(dist)
    result = _quad_full(lambda x: dist.F(x) * dist.S(x), lo, hi, split, tol)
    return 2.0 * result


def compute_gmd_cov(dist: DistributionSpec, tol: float = 1e-12) -> float:
    """GMD[X] = 4*(E[X*F(X)] - E[X]*E[F(X)]) = 4*(E[X*F(X)] - E[X]/2)
    since for continuous X, E[F(X)] = 1/2.
    """
    lo, hi = dist.support
    split = _split_point(dist)
    mu = compute_mean(dist, tol=tol)
    e_xFx = _quad_full(lambda x: x * dist.F(x) * dist.f(x), lo, hi, split, tol)
    return 4.0 * (e_xFx - mu * 0.5)


def compute_gmd_maxiance(dist: DistributionSpec, tol: float = 1e-12) -> float:
    """GMD[X] = 2 * Mx[X]."""
    return 2.0 * compute_maxiance(dist, tol=tol)
