import numpy as np
import scipy.stats as stats


def _build_result(support: np.ndarray, pmf: np.ndarray) -> dict:
    """Normalise PMF and build CDF, tail, etc."""
    pmf = pmf / pmf.sum()
    cdf = np.cumsum(pmf)
    tail = 1.0 - np.concatenate([[0.0], cdf[:-1]])  # P(X >= k)
    return {
        "support": support,
        "pmf": pmf,
        "cdf": cdf,
        "tail": tail,  # P(X >= x) for each x in support
    }


def make_geometric_pmf(p: float, K: int = 500) -> dict:
    """Geometric(p): P(X=k) = (1-p)^{k-1} * p, k = 1,2,...,K."""
    support = np.arange(1, K + 1, dtype=float)
    pmf = stats.geom.pmf(support.astype(int), p)
    return _build_result(support, pmf)


def make_zipf_pmf(alpha: float, K: int = 10_000) -> dict:
    """Zipf(alpha): P(X=k) proportional to k^{-alpha}, k = 1,...,K."""
    support = np.arange(1, K + 1, dtype=float)
    pmf = support ** (-alpha)
    return _build_result(support, pmf)


def make_poisson_pmf(theta: float, K: int = 500) -> dict:
    """Poisson(theta): P(X=k) = e^{-theta}*theta^k/k!, k = 0,1,...,K."""
    support = np.arange(0, K + 1, dtype=float)
    pmf = stats.poisson.pmf(support.astype(int), theta)
    return _build_result(support, pmf)


def make_negbinom_pmf(r: int, p: float, K: int = 500) -> dict:
    """Negative Binomial(r, p): number of failures before r-th success.
    P(X=k) = C(r+k-1, k) * (1-p)^k * p^r, k = 0,1,...,K.
    """
    support = np.arange(0, K + 1, dtype=float)
    pmf = stats.nbinom.pmf(support.astype(int), r, p)
    return _build_result(support, pmf)
