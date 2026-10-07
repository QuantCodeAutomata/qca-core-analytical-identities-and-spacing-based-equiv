import numpy as np


def compute_discrete_gmd(pmf: np.ndarray, support: np.ndarray) -> float:
    """GMD = 2 * sum_{i < j} (x_j - x_i) * p_i * p_j.

    Uses vectorised outer products for efficiency.
    """
    n = len(support)
    x = np.asarray(support, dtype=float)
    p = np.asarray(pmf, dtype=float)

    # diff[i,j] = x[j] - x[i], only for j > i
    diff = x[np.newaxis, :] - x[:, np.newaxis]  # shape (n, n)
    pp = np.outer(p, p)                           # shape (n, n)

    # Upper triangle (i < j)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    return 2.0 * np.sum(diff[mask] * pp[mask])


def compute_lambda(pmf: np.ndarray) -> float:
    """Lambda = sum_k p(k)^2."""
    p = np.asarray(pmf, dtype=float)
    return float(np.sum(p ** 2))


def compute_discrete_hazard(pmf: np.ndarray, support: np.ndarray) -> np.ndarray:
    """h(x) = p(x) / P(X >= x) = p(x) / tail(x).

    tail(x_i) = P(X >= x_i) = sum_{j >= i} p(x_j).
    """
    p = np.asarray(pmf, dtype=float)
    tail = np.cumsum(p[::-1])[::-1]  # P(X >= x_i)
    with np.errstate(divide="ignore", invalid="ignore"):
        h = np.where(tail > 0, p / tail, 0.0)
    return h


def compute_discrete_reverse_hazard(pmf: np.ndarray, support: np.ndarray) -> np.ndarray:
    """r(x) = p(x) / F(x).

    F(x_i) = P(X <= x_i) = sum_{j <= i} p(x_j).
    """
    p = np.asarray(pmf, dtype=float)
    cdf = np.cumsum(p)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(cdf > 0, p / cdf, 0.0)
    return r


def compute_concentration_bound(gmd: float, lam: float) -> float:
    """Returns (1 - Lambda) / (2 * Lambda) for comparison with GMD."""
    return (1.0 - lam) / (2.0 * lam)
