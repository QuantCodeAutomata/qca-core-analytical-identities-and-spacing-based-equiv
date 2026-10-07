import numpy as np
import pytest
from scipy.special import zeta
from scipy.stats import poisson, nbinom

def compute_pmf_gmd(pmf, support):
    """GMD = 2 * sum_{i<j} (support[j]-support[i]) * pmf[i] * pmf[j]"""
    gmd = 0.0
    n = len(pmf)
    for i in range(n):
        for j in range(i+1, n):
            gmd += (support[j] - support[i]) * pmf[i] * pmf[j]
    return 2 * gmd

def compute_lambda(pmf):
    return np.sum(pmf**2)

def test_poisson_low_theta_sd_dominant():
    """For small theta, Poisson should be SD dominant (SD > GMD)."""
    theta = 0.3  # well below the sign change at 0.668
    K = 100
    support = np.arange(K+1)
    pmf = poisson.pmf(support, theta)
    pmf /= pmf.sum()
    sd = np.sqrt(theta)
    gmd = compute_pmf_gmd(pmf, support)
    assert sd > gmd, f"Expected SD > GMD for Poisson(theta={theta}): SD={sd:.4f}, GMD={gmd:.4f}"

def test_poisson_high_theta_gmd_dominant():
    """For large theta, Poisson should be GMD dominant (GMD > SD)."""
    theta = 2.0  # well above the sign change at 0.668
    K = 200
    support = np.arange(K+1)
    pmf = poisson.pmf(support, theta)
    pmf /= pmf.sum()
    sd = np.sqrt(theta)
    gmd = compute_pmf_gmd(pmf[:50], support[:50])  # truncate for speed
    assert gmd > sd, f"Expected GMD > SD for Poisson(theta={theta}): SD={sd:.4f}, GMD={gmd:.4f}"

def test_concentration_bound_poisson():
    """For Poisson(theta=1), concentration bound should hold (GMD <= (1-Lambda)/(2*Lambda))."""
    theta = 1.0  # above 0.802 threshold
    K = 100
    support = np.arange(K+1)
    pmf = poisson.pmf(support, theta)
    pmf /= pmf.sum()
    lam = compute_lambda(pmf)
    bound = (1 - lam) / (2 * lam)
    gmd = compute_pmf_gmd(pmf[:30], support[:30])
    assert gmd <= bound + 0.01, f"Concentration bound should hold for Poisson({theta}): GMD={gmd:.4f}, bound={bound:.4f}"

def test_zipf_sd_dominant():
    """Zipf distribution should be SD dominant (heavy tail)."""
    alpha = 3.0
    K = 1000
    support = np.arange(1, K+1)
    z = zeta(alpha)
    pmf = support.astype(float)**(-alpha) / z
    # PMF may not sum to exactly 1 with finite truncation
    pmf_trunc = pmf / pmf.sum()
    sd_sq = np.sum(support**2 * pmf_trunc) - np.sum(support * pmf_trunc)**2
    if sd_sq <= 0:
        pytest.skip("Variance not finite at alpha=3 with this truncation")
    sd = np.sqrt(sd_sq)
    gmd = compute_pmf_gmd(pmf_trunc[:100], support[:100])
    # Expect R = SD/GMD >> 1
    assert sd > gmd, f"Zipf(3) should be SD dominant: SD={sd:.4f}, GMD={gmd:.4f}"

def test_negbinom_low_p_sd_dominant():
    """NB(r=2) with small p should be SD dominant."""
    r, p = 2, 0.9  # small failure prob -> large mean, heavy tail
    K = 500
    support = np.arange(K+1)
    pmf = nbinom.pmf(support, r, p)
    pmf /= pmf.sum()
    var = r*(1-p)/p**2
    sd = np.sqrt(var)
    gmd = compute_pmf_gmd(pmf[:50], support[:50])
    assert sd > gmd, f"NB(r=2,p={p}) should be SD dominant: SD={sd:.4f}, GMD={gmd:.4f}"

def test_lambda_is_probability():
    """Lambda = sum p(x)^2 should be in (0,1]."""
    for theta in [0.5, 1.0, 2.0]:
        K = 100
        pmf = poisson.pmf(np.arange(K+1), theta)
        lam = compute_lambda(pmf / pmf.sum())
        assert 0 < lam <= 1, f"Lambda={lam} not in (0,1]"

def test_discrete_gmd_nonneg():
    """GMD should be nonneg for all distributions."""
    for theta in [0.1, 1.0, 3.0]:
        K = 50
        support = np.arange(K+1)
        pmf = poisson.pmf(support, theta)
        pmf /= pmf.sum()
        gmd = compute_pmf_gmd(pmf[:20], support[:20])
        assert gmd >= 0
