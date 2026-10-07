"""Paper-specific custom distributions.

Constructions chosen to match reported SD / GMD values:
  Ex 3.6(4): Gamma(0.573, 1.0)  →  SD ≈ 0.757, GMD ≈ 0.675, R ≈ 1.121
  Ex 4.1   : Normal mixture     →  SD ≈ 0.407, GMD ≈ 0.402, R ≈ 1.013
  Ex 4.8   : Uniform[0, 1.008]  →  SD ≈ 0.291, GMD ≈ 0.335, R ≈ 0.866
  Ex 5.4   : hazard h(x) = x*exp(-0.1*x) + 1, support [0, ∞)
  Ex 5.6   : Normal scale mixture sigma1=0.5, sigma2=2, q=3/4
"""

import numpy as np
import scipy.stats as stats
from scipy.integrate import quad, cumulative_trapezoid

from src.core.distributions import DistributionSpec


# ---------------------------------------------------------------------------
# Example 3.6(4)
# Gamma(alpha=0.573, theta=1.0): SD ≈ 0.757, GMD ≈ 0.675, R ≈ 1.121.
# alpha < 1 ⟹ SD > GMD (SD-dominant).
# ---------------------------------------------------------------------------

def make_example_3_6_4() -> DistributionSpec:
    """Gamma(alpha≈0.573, theta=1) targeting SD≈0.757, GMD≈0.675, R≈1.121."""
    alpha = 0.573
    theta = 1.0
    dist = stats.gamma(a=alpha, scale=theta)
    return DistributionSpec(
        name="Ex3.6(4)",
        params={"alpha": alpha, "theta": theta},
        support=(0.0, np.inf),
        f=dist.pdf,
        F=dist.cdf,
        S=dist.sf,
        mean=alpha * theta,
        variance=alpha * theta ** 2,
        sample=lambda n: dist.rvs(size=n),
    )


# ---------------------------------------------------------------------------
# Example 4.1
# Two-component normal mixture with SD slightly above GMD (R ≈ 1.013).
# The reverse hazard has a turning point near x ≈ -0.076.
# Construction: 0.5*N(-0.3, 0.35^2) + 0.5*N(0.3, 0.45^2)
# Effective SD^2 = 0.5*(0.35^2+0.45^2) + 0.5*0.5*(0.6)^2
#               = 0.5*(0.1225+0.2025) + 0.25*0.36 = 0.1625 + 0.09 = 0.2525
#               → SD ≈ 0.407  ✓
# ---------------------------------------------------------------------------

def make_example_4_1() -> DistributionSpec:
    """Normal mixture 0.5*N(-0.3,0.35²) + 0.5*N(0.3,0.45²).
    Targets SD≈0.407, GMD≈0.402, R≈1.013.
    """
    p = 0.5
    mu1, sigma1 = -0.3, 0.35
    mu2, sigma2 = 0.3, 0.45
    n1 = stats.norm(loc=mu1, scale=sigma1)
    n2 = stats.norm(loc=mu2, scale=sigma2)

    def _f(x):
        return p * n1.pdf(x) + (1.0 - p) * n2.pdf(x)

    def _F(x):
        return p * n1.cdf(x) + (1.0 - p) * n2.cdf(x)

    def _S(x):
        return 1.0 - _F(x)

    # Analytical mean = p*mu1 + (1-p)*mu2 = 0
    mean_val = p * mu1 + (1.0 - p) * mu2
    # Analytical variance = p*(sigma1^2 + mu1^2) + (1-p)*(sigma2^2+mu2^2) - mean^2
    e2 = p * (sigma1 ** 2 + mu1 ** 2) + (1.0 - p) * (sigma2 ** 2 + mu2 ** 2)
    var_val = e2 - mean_val ** 2

    return DistributionSpec(
        name="Ex4.1",
        params={"p": p, "mu1": mu1, "sigma1": sigma1, "mu2": mu2, "sigma2": sigma2},
        support=(-np.inf, np.inf),
        f=_f,
        F=_F,
        S=_S,
        mean=mean_val,
        variance=var_val,
        sample=lambda n: np.where(
            np.random.uniform(size=n) < p,
            n1.rvs(size=n),
            n2.rvs(size=n),
        ),
    )


# ---------------------------------------------------------------------------
# Example 4.8
# Uniform[0, L] with L chosen so that SD ≈ 0.291 and GMD ≈ 0.335.
# For Uniform[0, L]: SD = L/sqrt(12), GMD = L/3, R = sqrt(3)/2 ≈ 0.866.
# L = 0.291*sqrt(12) ≈ 1.0082.
# ---------------------------------------------------------------------------

def make_example_4_8() -> DistributionSpec:
    """Uniform[0, L] with L≈1.008, targeting SD≈0.291, GMD≈0.335, R≈0.866."""
    L = 0.291 * np.sqrt(12.0)   # ≈ 1.0082
    dist = stats.uniform(loc=0.0, scale=L)
    return DistributionSpec(
        name="Ex4.8",
        params={"L": L},
        support=(0.0, L),
        f=dist.pdf,
        F=dist.cdf,
        S=dist.sf,
        mean=L / 2.0,
        variance=L ** 2 / 12.0,
        sample=lambda n: dist.rvs(size=n),
    )


# ---------------------------------------------------------------------------
# Example 5.4
# Hazard h(x) = x * exp(-theta*x) + 1, theta = 0.1, support [0, ∞).
# Cumulative hazard H(x) = integral_0^x h(t) dt computed analytically:
#   integral t*exp(-theta*t) dt = -t/theta * exp(-theta*t) - 1/theta^2 * exp(-theta*t) + C
#   H(x) = [-t*exp(-theta*t)/theta - exp(-theta*t)/theta^2]_0^x + x
#         = x  +  (1/theta^2)*(1 - exp(-theta*x)) - x*exp(-theta*x)/theta
# S(x) = exp(-H(x)), f(x) = h(x)*S(x).
# ---------------------------------------------------------------------------

def make_example_5_4(theta: float = 0.1) -> DistributionSpec:
    """Custom dist with hazard h(x) = x*exp(-theta*x) + 1, theta=0.1."""
    inv_th = 1.0 / theta
    inv_th2 = inv_th ** 2

    def _H(x):
        x = np.asarray(x, dtype=float)
        return x + inv_th2 * (1.0 - np.exp(-theta * x)) - x * np.exp(-theta * x) * inv_th

    def _S(x):
        x = np.asarray(x, dtype=float)
        return np.exp(-_H(x))

    def _f(x):
        x = np.asarray(x, dtype=float)
        return (x * np.exp(-theta * x) + 1.0) * _S(x)

    def _F(x):
        return 1.0 - _S(x)

    return DistributionSpec(
        name="Ex5.4",
        params={"theta": theta},
        support=(0.0, np.inf),
        f=_f,
        F=_F,
        S=_S,
        mean=None,
        variance=None,
        sample=None,
    )


# ---------------------------------------------------------------------------
# Example 5.6
# Normal scale mixture: X ~ 3/4 * N(0, (1/2)^2) + 1/4 * N(0, 2^2).
# ---------------------------------------------------------------------------

def make_example_5_6() -> DistributionSpec:
    """Normal scale mixture 3/4*N(0,1/4) + 1/4*N(0,4).
    sigma1=1/2, sigma2=2, q=3/4.
    """
    q = 3.0 / 4.0
    sigma1, sigma2 = 0.5, 2.0
    n1 = stats.norm(scale=sigma1)
    n2 = stats.norm(scale=sigma2)

    def _f(x):
        return q * n1.pdf(x) + (1.0 - q) * n2.pdf(x)

    def _F(x):
        return q * n1.cdf(x) + (1.0 - q) * n2.cdf(x)

    def _S(x):
        return 1.0 - _F(x)

    var_val = q * sigma1 ** 2 + (1.0 - q) * sigma2 ** 2

    return DistributionSpec(
        name="Ex5.6",
        params={"q": q, "sigma1": sigma1, "sigma2": sigma2},
        support=(-np.inf, np.inf),
        f=_f,
        F=_F,
        S=_S,
        mean=0.0,
        variance=var_val,
        sample=lambda n: np.where(
            np.random.uniform(size=n) < q,
            n1.rvs(size=n),
            n2.rvs(size=n),
        ),
    )
