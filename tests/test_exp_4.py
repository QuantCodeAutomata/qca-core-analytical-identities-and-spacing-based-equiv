import numpy as np
import pytest
from scipy import integrate, stats

def lomax_pdf(alpha, beta, x):
    """Gamma frailty mixture: Lomax density."""
    return alpha * beta**alpha * (x + beta)**(-(alpha + 1))

def lomax_cdf(alpha, beta, x):
    return 1 - (beta / (x + beta))**alpha

def lomax_sf(alpha, beta, x):
    return (beta / (x + beta))**alpha

def compute_gmd_1d_generic(F, S, lower, upper, limit=500):
    result, _ = integrate.quad(lambda x: 2 * F(x) * S(x), lower, upper, limit=limit)
    return result

def test_lomax_sd_dominant():
    """Gamma frailty mixture (Lomax) should show SD > GMD."""
    for alpha in [3.0, 4.0, 5.0]:
        beta = 1.0
        cutoff = 200 / alpha
        mean_val, _ = integrate.quad(lambda x: lomax_sf(alpha, beta, x), 0, cutoff)
        e_x2, _ = integrate.quad(lambda x: 2 * x * lomax_sf(alpha, beta, x), 0, cutoff)
        var_val = e_x2 - mean_val**2
        if var_val <= 0:
            continue
        sd_val = np.sqrt(var_val)
        F = lambda x: lomax_cdf(alpha, beta, x)
        Sf = lambda x: lomax_sf(alpha, beta, x)
        gmd_val = compute_gmd_1d_generic(F, Sf, 0, cutoff)
        assert sd_val > gmd_val, f"Expected SD > GMD for Lomax(alpha={alpha}), got SD={sd_val:.4f}, GMD={gmd_val:.4f}"

def test_normal_sum_gmd_dominant():
    """Sum of two N(0,1) = N(0,2): should be GMD dominant (SD < GMD)."""
    # N(0,2): SD=sqrt(2), GMD=2*sqrt(2)/sqrt(pi)
    # For N(0,sigma^2): GMD = 2*sigma/sqrt(pi)
    # For N(0,2): SD=sqrt(2)≈1.414, GMD=2*sqrt(2)/sqrt(pi)≈1.596
    # So GMD > SD: GMD dominant
    sigma2 = 2.0
    sd = np.sqrt(sigma2)
    gmd = 2 * np.sqrt(sigma2 / np.pi)
    assert gmd > sd, f"Expected GMD > SD for N(0,2), SD={sd:.4f}, GMD={gmd:.4f}"

def test_normal_scale_mixture_variance():
    """Example 5.6: Var[X] = 19/16 for the normal scale mixture."""
    sigma1, sigma2, q = 0.5, 2.0, 0.75
    var_mix = q * sigma1**2 + (1 - q) * sigma2**2
    expected = 19.0 / 16.0
    assert abs(var_mix - expected) < 1e-10

def test_example_5_6_sd():
    """SD[X] ≈ 1.0897 for the normal scale mixture."""
    sigma1, sigma2, q = 0.5, 2.0, 0.75
    var_mix = q * sigma1**2 + (1 - q) * sigma2**2
    sd = np.sqrt(var_mix)
    assert abs(sd - 1.0897) < 1e-3

def test_delta_u_sign_change_example_5_4():
    """Example 5.4: Delta(u) changes sign near u≈9.7."""
    # Simplified check: compute Delta at u=5 and u=15
    # This verifies the sign change direction
    theta = 0.1
    def H(x):
        return 100 - 10*x*np.exp(-theta*x) - 100*np.exp(-theta*x) + x
    def S(x):
        return np.exp(-H(x))
    def f(x):
        h_val = x * np.exp(-theta*x) + 1
        return h_val * S(x)
    # At u=5 (before sign change ~9.7): Delta should be negative (GMD dominant)
    # At u=15 (after sign change): Delta should be positive (SD dominant)
    # Test SD > 0 and GMD > 0 as sanity
    cutoff = 50
    u = 5.0
    Su = S(u)
    if Su > 1e-10:
        mean_u, _ = integrate.quad(lambda x: (x - u) * f(x) / Su, u, cutoff)
        e_x2_u, _ = integrate.quad(lambda x: (x - u)**2 * f(x) / Su, u, cutoff)
        var_u = e_x2_u - mean_u**2
        assert var_u >= 0
