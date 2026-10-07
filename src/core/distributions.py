from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple

import numpy as np
import scipy.stats as stats
from scipy.special import gamma as gamma_fn


@dataclass
class DistributionSpec:
    name: str
    params: dict
    support: Tuple[float, float]
    f: Callable[[float], float]        # PDF
    F: Callable[[float], float]        # CDF
    S: Callable[[float], float]        # Survival = 1 - F
    mean: Optional[float] = None
    variance: Optional[float] = None
    sample: Optional[Callable[[int], np.ndarray]] = None


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def make_exponential(rate: float = 1.0) -> DistributionSpec:
    dist = stats.expon(scale=1.0 / rate)
    return DistributionSpec(
        name="Exponential",
        params={"rate": rate},
        support=(0.0, np.inf),
        f=dist.pdf,
        F=dist.cdf,
        S=dist.sf,
        mean=1.0 / rate,
        variance=1.0 / rate ** 2,
        sample=lambda n: dist.rvs(size=n),
    )


def make_uniform(a: float = 0.0, b: float = 1.0) -> DistributionSpec:
    dist = stats.uniform(loc=a, scale=b - a)
    return DistributionSpec(
        name="Uniform",
        params={"a": a, "b": b},
        support=(a, b),
        f=dist.pdf,
        F=dist.cdf,
        S=dist.sf,
        mean=(a + b) / 2.0,
        variance=(b - a) ** 2 / 12.0,
        sample=lambda n: dist.rvs(size=n),
    )


def make_normal(mu: float = 0.0, sigma: float = 1.0) -> DistributionSpec:
    dist = stats.norm(loc=mu, scale=sigma)
    return DistributionSpec(
        name="Normal",
        params={"mu": mu, "sigma": sigma},
        support=(-np.inf, np.inf),
        f=dist.pdf,
        F=dist.cdf,
        S=dist.sf,
        mean=mu,
        variance=sigma ** 2,
        sample=lambda n: dist.rvs(size=n),
    )


def make_logistic(mu: float = 0.0, s: float = 1.0) -> DistributionSpec:
    dist = stats.logistic(loc=mu, scale=s)
    return DistributionSpec(
        name="Logistic",
        params={"mu": mu, "s": s},
        support=(-np.inf, np.inf),
        f=dist.pdf,
        F=dist.cdf,
        S=dist.sf,
        mean=mu,
        variance=np.pi ** 2 * s ** 2 / 3.0,
        sample=lambda n: dist.rvs(size=n),
    )


def make_laplace(mu: float = 0.0, b: float = 1.0) -> DistributionSpec:
    dist = stats.laplace(loc=mu, scale=b)
    return DistributionSpec(
        name="Laplace",
        params={"mu": mu, "b": b},
        support=(-np.inf, np.inf),
        f=dist.pdf,
        F=dist.cdf,
        S=dist.sf,
        mean=mu,
        variance=2.0 * b ** 2,
        sample=lambda n: dist.rvs(size=n),
    )


def make_gamma(alpha: float, theta: float = 1.0) -> DistributionSpec:
    # shape=alpha, scale=theta (so mean = alpha*theta)
    dist = stats.gamma(a=alpha, scale=theta)
    return DistributionSpec(
        name="Gamma",
        params={"alpha": alpha, "theta": theta},
        support=(0.0, np.inf),
        f=dist.pdf,
        F=dist.cdf,
        S=dist.sf,
        mean=alpha * theta,
        variance=alpha * theta ** 2,
        sample=lambda n: dist.rvs(size=n),
    )


def make_weibull(alpha: float, scale: float = 1.0) -> DistributionSpec:
    # Weibull: F(x) = 1 - exp(-(x/scale)^alpha)
    # scipy.stats.weibull_min(c=alpha, scale=scale)
    dist = stats.weibull_min(c=alpha, scale=scale)
    mean_val = scale * gamma_fn(1.0 + 1.0 / alpha)
    var_val = scale ** 2 * (gamma_fn(1.0 + 2.0 / alpha) - gamma_fn(1.0 + 1.0 / alpha) ** 2)
    return DistributionSpec(
        name="Weibull",
        params={"alpha": alpha, "scale": scale},
        support=(0.0, np.inf),
        f=dist.pdf,
        F=dist.cdf,
        S=dist.sf,
        mean=mean_val,
        variance=var_val,
        sample=lambda n: dist.rvs(size=n),
    )


def make_gpd(xi: float, scale: float = 1.0) -> DistributionSpec:
    # Generalized Pareto: F(x) = 1 - (1 + xi*x/scale)^{-1/xi}, xi > 0
    # scipy shape parameter c = xi, scale = scale
    dist = stats.genpareto(c=xi, scale=scale)
    mean_val = scale / (1.0 - xi) if xi < 1.0 else None
    var_val = scale ** 2 / ((1.0 - xi) ** 2 * (1.0 - 2.0 * xi)) if xi < 0.5 else None
    return DistributionSpec(
        name="GPD",
        params={"xi": xi, "scale": scale},
        support=(0.0, np.inf),
        f=dist.pdf,
        F=dist.cdf,
        S=dist.sf,
        mean=mean_val,
        variance=var_val,
        sample=lambda n: dist.rvs(size=n),
    )


def make_burr(c: float, k: float) -> DistributionSpec:
    # Burr XII: F(x) = 1 - (1 + x^c)^{-k}, x >= 0
    # f(x) = c*k*x^{c-1}*(1+x^c)^{-k-1}
    def _f(x):
        x = np.asarray(x, dtype=float)
        return np.where(x > 0, c * k * x ** (c - 1) * (1 + x ** c) ** (-k - 1), 0.0)

    def _F(x):
        x = np.asarray(x, dtype=float)
        return np.where(x > 0, 1.0 - (1.0 + x ** c) ** (-k), 0.0)

    def _S(x):
        x = np.asarray(x, dtype=float)
        return np.where(x > 0, (1.0 + x ** c) ** (-k), 1.0)

    # Mean: exists if ck > 1; Variance: exists if ck > 2
    mean_val = None
    var_val = None
    if c * k > 1:
        mean_val = k * gamma_fn(1.0 + 1.0 / c) * gamma_fn(k - 1.0 / c) / gamma_fn(k + 1.0)
    if c * k > 2:
        e2 = k * gamma_fn(1.0 + 2.0 / c) * gamma_fn(k - 2.0 / c) / gamma_fn(k + 1.0)
        var_val = e2 - mean_val ** 2 if mean_val is not None else None

    dist_sc = stats.burr12(c=c, d=k)

    return DistributionSpec(
        name="Burr",
        params={"c": c, "k": k},
        support=(0.0, np.inf),
        f=_f,
        F=_F,
        S=_S,
        mean=mean_val,
        variance=var_val,
        sample=lambda n: dist_sc.rvs(size=n),
    )


def make_beta(alpha: float, beta: float) -> DistributionSpec:
    dist = stats.beta(a=alpha, b=beta)
    return DistributionSpec(
        name="Beta",
        params={"alpha": alpha, "beta": beta},
        support=(0.0, 1.0),
        f=dist.pdf,
        F=dist.cdf,
        S=dist.sf,
        mean=alpha / (alpha + beta),
        variance=alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1)),
        sample=lambda n: dist.rvs(size=n),
    )


def make_gompertz(B: float = 1e-4, c: float = 0.09) -> DistributionSpec:
    # Gompertz: h(x) = B*exp(c*x), S(x) = exp(B/c*(1 - exp(c*x)))
    # log f(x) = log(B) + c*x + B/c*(1 - exp(c*x))
    _log_B = np.log(B)
    _Bc = B / c

    def _log_S_val(x):
        """Log of survival; clip cx so exp never overflows."""
        return _Bc * (1.0 - np.exp(np.minimum(c * x, 709.0)))

    def _S(x):
        x = np.asarray(x, dtype=float)
        return np.exp(np.clip(_log_S_val(x), -750.0, 0.0))

    def _f(x):
        x = np.asarray(x, dtype=float)
        log_f = _log_B + c * x + _log_S_val(x)
        return np.exp(np.clip(log_f, -750.0, 0.0)) * (log_f > -750.0)

    def _F(x):
        return 1.0 - _S(x)

    return DistributionSpec(
        name="Gompertz",
        params={"B": B, "c": c},
        support=(0.0, np.inf),
        f=_f,
        F=_F,
        S=_S,
        mean=None,
        variance=None,
        sample=None,
    )


def make_makeham(A: float = 0.00022, B: float = 2.7e-6, c: float = 0.124) -> DistributionSpec:
    # Makeham: h(x) = A + B*exp(c*x), S(x) = exp(-A*x - B/c*(exp(c*x)-1))
    # log S(x) = -A*x - B/c*(exp(c*x)-1)
    _Bc = B / c

    def _exp_cx(x):
        return np.exp(np.minimum(c * x, 709.0))

    def _log_S_val(x):
        return -A * x - _Bc * (_exp_cx(x) - 1.0)

    def _S(x):
        x = np.asarray(x, dtype=float)
        return np.exp(np.clip(_log_S_val(x), -750.0, 0.0))

    def _f(x):
        x = np.asarray(x, dtype=float)
        log_s = _log_S_val(x)
        ecx = _exp_cx(x)
        hazard = A + B * ecx
        mask = log_s > -750.0
        result = np.where(mask, hazard * np.exp(np.clip(log_s, -750.0, 0.0)), 0.0)
        return float(result) if result.ndim == 0 else result

    def _F(x):
        return 1.0 - _S(x)

    return DistributionSpec(
        name="Makeham",
        params={"A": A, "B": B, "c": c},
        support=(0.0, np.inf),
        f=_f,
        F=_F,
        S=_S,
        mean=None,
        variance=None,
        sample=None,
    )
