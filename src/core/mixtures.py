import numpy as np
import scipy.stats as stats


def gamma_frailty_mixture(alpha: float, beta: float, x_grid: np.ndarray) -> dict:
    """X | Lambda ~ Exp(Lambda), Lambda ~ Gamma(alpha, beta).

    Marginal: Pareto-II (Lomax).
    f(x) = alpha * beta / (1 + beta*x)^{alpha+1}
    F(x) = 1 - (1 + beta*x)^{-alpha}
    S(x) = (1 + beta*x)^{-alpha}
    """
    x = np.asarray(x_grid, dtype=float)
    denom = (1.0 + beta * x) ** (alpha + 1.0)
    f = alpha * beta / denom
    S = (1.0 + beta * x) ** (-alpha)
    F = 1.0 - S
    return {"x": x, "f": f, "F": F, "S": S}


def normal_scale_mixture(
    sigma1: float,
    sigma2: float,
    q: float,
    x_grid: np.ndarray,
) -> dict:
    """X ~ q * N(0, sigma1^2) + (1-q) * N(0, sigma2^2).

    f(x) = q*phi(x/sigma1)/sigma1 + (1-q)*phi(x/sigma2)/sigma2
    F(x) = q*Phi(x/sigma1) + (1-q)*Phi(x/sigma2)
    S(x) = 1 - F(x)
    """
    x = np.asarray(x_grid, dtype=float)
    n1 = stats.norm(scale=sigma1)
    n2 = stats.norm(scale=sigma2)
    f = q * n1.pdf(x) + (1.0 - q) * n2.pdf(x)
    F = q * n1.cdf(x) + (1.0 - q) * n2.cdf(x)
    S = 1.0 - F
    return {"x": x, "f": f, "F": F, "S": S}
