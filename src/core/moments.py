import numpy as np
from scipy.integrate import quad

from src.core.distributions import DistributionSpec


def _quad_full(f, lo: float, hi: float, split: float | None, tol: float) -> float:
    """Integrate f from lo to hi, splitting at `split` if provided.

    scipy.quad does not accept `points` when bounds are ±infinity, so we
    split the integral manually at the given breakpoint.
    """
    kw = dict(limit=200, epsabs=tol, epsrel=tol)
    if split is None or not (lo < split < hi):
        val, _ = quad(f, lo, hi, **kw)
        return val
    # Both pieces may still have infinite bounds, but each is one-sided
    val_lo, _ = quad(f, lo, split, **kw)
    val_hi, _ = quad(f, split, hi, **kw)
    return val_lo + val_hi


def _split_point(dist: DistributionSpec) -> float | None:
    """Return a natural split point for bilaterlal infinite-support distributions."""
    lo, hi = dist.support
    if lo == -np.inf and hi == np.inf:
        return 0.0
    return None


def compute_mean(dist: DistributionSpec, tol: float = 1e-12) -> float:
    lo, hi = dist.support
    split = _split_point(dist)
    return _quad_full(lambda x: x * dist.f(x), lo, hi, split, tol)


def compute_variance(dist: DistributionSpec, tol: float = 1e-12) -> float:
    lo, hi = dist.support
    split = _split_point(dist)
    mu = compute_mean(dist, tol=tol)
    e2 = _quad_full(lambda x: x ** 2 * dist.f(x), lo, hi, split, tol)
    return e2 - mu ** 2


def compute_sd(dist: DistributionSpec, tol: float = 1e-12) -> float:
    return np.sqrt(compute_variance(dist, tol=tol))


def compute_maxiance(dist: DistributionSpec, tol: float = 1e-12) -> float:
    """Mx[X] = E[max(X,X')] - E[X] = integral x * 2*F(x)*f(x) dx - E[X]."""
    lo, hi = dist.support
    split = _split_point(dist)
    mu = compute_mean(dist, tol=tol)
    e_max = _quad_full(lambda x: x * 2.0 * dist.F(x) * dist.f(x), lo, hi, split, tol)
    return e_max - mu
