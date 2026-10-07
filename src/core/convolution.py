import numpy as np
from scipy.integrate import quad

from src.core.distributions import DistributionSpec
from src.core.moments import compute_mean, compute_variance


def fft_convolve_densities(
    f1: np.ndarray,
    f2: np.ndarray,
    dx: float,
) -> np.ndarray:
    """Convolve two PDFs on a uniform grid with spacing dx using FFT.

    Parameters
    ----------
    f1, f2 : array of density values on the same uniform grid
    dx     : grid spacing

    Returns
    -------
    Convolved density (same length as f1+f2-1, i.e. full convolution).
    """
    n = len(f1) + len(f2) - 1
    F1 = np.fft.rfft(f1, n=n)
    F2 = np.fft.rfft(f2, n=n)
    conv = np.fft.irfft(F1 * F2, n=n)
    conv = np.maximum(conv, 0.0)
    # Normalise so the density integrates to 1
    total = conv.sum() * dx
    if total > 0:
        conv /= total
    return conv


def compute_convolution_sd_gmd(
    dist1: DistributionSpec,
    dist2: DistributionSpec,
    x_min: float,
    x_max: float,
    n_grid: int = 10_000,
) -> dict:
    """Compute SD and GMD of X1 + X2 via FFT density convolution.

    The sum X1+X2 has support roughly on [2*x_min, 2*x_max].
    We evaluate both densities on a common uniform grid [x_min, x_max],
    then convolve.
    """
    x_grid = np.linspace(x_min, x_max, n_grid)
    dx = x_grid[1] - x_grid[0]

    f1_vals = np.array([dist1.f(x) for x in x_grid])
    f2_vals = np.array([dist2.f(x) for x in x_grid])

    conv = fft_convolve_densities(f1_vals, f2_vals, dx)

    # Build grid for the sum: [2*x_min, 2*x_max]
    n_conv = len(conv)
    x_sum = np.linspace(2.0 * x_min, 2.0 * x_max, n_conv)
    dx_sum = x_sum[1] - x_sum[0]

    # Mean and E[X^2] of sum via numerical integration on grid
    mean_sum = np.sum(x_sum * conv) * dx_sum
    e2_sum = np.sum(x_sum ** 2 * conv) * dx_sum
    var_sum = e2_sum - mean_sum ** 2
    sd_sum = np.sqrt(max(var_sum, 0.0))

    # CDF and GMD of sum
    cdf_sum = np.cumsum(conv) * dx_sum
    cdf_sum = np.clip(cdf_sum, 0.0, 1.0)
    sf_sum = 1.0 - cdf_sum
    gmd_sum = 2.0 * np.sum(cdf_sum * sf_sum * conv) * dx_sum

    return {
        "mean": mean_sum,
        "sd": sd_sum,
        "gmd": gmd_sum,
        "x_grid": x_sum,
        "density": conv,
    }
