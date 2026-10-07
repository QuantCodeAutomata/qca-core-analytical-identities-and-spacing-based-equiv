"""Tests for Experiment 2: Dominance Classification."""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.core.distributions import (
    make_gamma, make_weibull, make_gpd,
    make_logistic, make_laplace, make_gompertz, make_makeham,
)
from src.core.moments import compute_sd
from src.core.gmd import compute_gmd_1d


def _sd_gmd(dist):
    return compute_sd(dist), compute_gmd_1d(dist)


# Test 1: Gamma alpha<1 → SD dominant -----------------------------------

def test_gamma_dominance_below_1():
    """Gamma(alpha=0.5, theta=1): SD > GMD."""
    dist = make_gamma(alpha=0.5, theta=1.0)
    sd, gmd = _sd_gmd(dist)
    assert sd > gmd, f"Gamma(0.5): SD={sd:.6f}, GMD={gmd:.6f}, expected SD>GMD"


# Test 2: Gamma alpha>1 → GMD dominant ----------------------------------

def test_gamma_dominance_above_1():
    """Gamma(alpha=2, theta=1): GMD > SD."""
    dist = make_gamma(alpha=2.0, theta=1.0)
    sd, gmd = _sd_gmd(dist)
    assert gmd > sd, f"Gamma(2): SD={sd:.6f}, GMD={gmd:.6f}, expected GMD>SD"


# Test 3: Gamma knife-edge alpha=1 → SD ≈ GMD ---------------------------

def test_gamma_knife_edge():
    """Gamma(alpha=1, theta=1) = Exponential: SD ≈ GMD within 1e-4."""
    dist = make_gamma(alpha=1.0, theta=1.0)
    sd, gmd = _sd_gmd(dist)
    assert abs(sd - gmd) < 1e-4, f"Gamma(1): SD={sd:.6f}, GMD={gmd:.6f}, |diff|={abs(sd-gmd):.2e}"


# Test 4: Weibull dominance pattern -------------------------------------

def test_weibull_dominance():
    """Weibull(alpha=0.5): SD>GMD; Weibull(alpha=2): GMD>SD."""
    dist_low = make_weibull(alpha=0.5, scale=1.0)
    sd_l, gmd_l = _sd_gmd(dist_low)
    assert sd_l > gmd_l, f"Weibull(0.5): SD={sd_l:.6f}, GMD={gmd_l:.6f}"

    dist_high = make_weibull(alpha=2.0, scale=1.0)
    sd_h, gmd_h = _sd_gmd(dist_high)
    assert gmd_h > sd_h, f"Weibull(2): SD={sd_h:.6f}, GMD={gmd_h:.6f}"


# Test 5: Weibull knife-edge alpha=1 ------------------------------------

def test_weibull_knife_edge():
    """Weibull(alpha=1) = Exponential: SD ≈ GMD within 1e-4."""
    dist = make_weibull(alpha=1.0, scale=1.0)
    sd, gmd = _sd_gmd(dist)
    assert abs(sd - gmd) < 1e-4, f"Weibull(1): |SD-GMD|={abs(sd-gmd):.2e}"


# Test 6: GPD SD dominant for xi in (0, 0.5) ----------------------------

def test_gpd_sd_dominant():
    """GPD(xi=0.2): SD > GMD."""
    dist = make_gpd(xi=0.2, scale=1.0)
    sd, gmd = _sd_gmd(dist)
    assert sd > gmd, f"GPD(xi=0.2): SD={sd:.6f}, GMD={gmd:.6f}"


@pytest.mark.parametrize("xi", [0.05, 0.1, 0.2, 0.3, 0.45])
def test_gpd_sd_dominant_parametric(xi):
    """GPD SD > GMD for all xi in (0, 0.5)."""
    dist = make_gpd(xi=xi, scale=1.0)
    sd, gmd = _sd_gmd(dist)
    assert sd > gmd, f"GPD(xi={xi}): SD={sd:.6f}, GMD={gmd:.6f}"


# Test 7: Gompertz GMD dominant -----------------------------------------

def test_gompertz_gmd_dominant():
    """Gompertz(B=1e-4, c=0.09): GMD > SD  (R < 1)."""
    dist = make_gompertz(B=1e-4, c=0.09)
    sd, gmd = _sd_gmd(dist)
    R = sd / gmd
    assert R < 1.0, f"Gompertz: R={R:.4f}, expected R<1"


# Test 8: Makeham GMD dominant -----------------------------------------

def test_makeham_gmd_dominant():
    """Makeham(A=0.00022, B=2.7e-6, c=0.124): GMD > SD  (R < 1)."""
    dist = make_makeham(A=0.00022, B=2.7e-6, c=0.124)
    sd, gmd = _sd_gmd(dist)
    R = sd / gmd
    assert R < 1.0, f"Makeham: R={R:.4f}, expected R<1"


# Test 9: Logistic analytical values ------------------------------------

def test_logistic_values():
    """Logistic(0,1): SD=π/√3, GMD=2, within 1e-4."""
    dist = make_logistic(0.0, 1.0)
    sd, gmd = _sd_gmd(dist)
    assert abs(sd  - np.pi / np.sqrt(3.0)) < 1e-4, f"Logistic SD={sd}"
    assert abs(gmd - 2.0) < 1e-4, f"Logistic GMD={gmd}"


# Test 10: Laplace analytical values ------------------------------------

def test_laplace_values():
    """Laplace(0,1): SD=√2, GMD=3/2, within 1e-4."""
    dist = make_laplace(0.0, 1.0)
    sd, gmd = _sd_gmd(dist)
    assert abs(sd  - np.sqrt(2.0)) < 1e-4, f"Laplace SD={sd}"
    assert abs(gmd - 1.5) < 1e-4, f"Laplace GMD={gmd}"


# Test 11: Gamma sweep monotonicity of R ---------------------------------

def test_gamma_R_monotone():
    """R[Gamma(alpha)] is decreasing in alpha (SD-dominant → GMD-dominant)."""
    alphas = np.linspace(0.3, 4.0, 10)
    Rs = []
    for a in alphas:
        sd, gmd = _sd_gmd(make_gamma(a, 1.0))
        Rs.append(sd / gmd)
    # R should be monotone decreasing
    for i in range(len(Rs) - 1):
        assert Rs[i] > Rs[i + 1] - 1e-4, (
            f"R not monotone at alpha={alphas[i]:.2f}->{alphas[i+1]:.2f}: "
            f"R={Rs[i]:.4f}->{Rs[i+1]:.4f}"
        )


# Test 12: Scaling invariance of R ----------------------------------------

@pytest.mark.parametrize("scale", [0.5, 1.0, 2.0, 5.0])
def test_exponential_scale_invariance(scale):
    """R[Exponential(rate)] should equal 1 regardless of rate."""
    dist = make_gamma(alpha=1.0, theta=scale)
    sd, gmd = _sd_gmd(dist)
    R = sd / gmd
    assert abs(R - 1.0) < 1e-4, f"Exponential(scale={scale}): R={R:.6f}"


# Test 13: R >= sqrt(3)/2 for all parametric families ----------------------

@pytest.mark.parametrize("make_dist,label", [
    (lambda: make_gamma(0.3, 1.0),    "Gamma(0.3)"),
    (lambda: make_gamma(0.5, 1.0),    "Gamma(0.5)"),
    (lambda: make_weibull(0.5, 1.0),  "Weibull(0.5)"),
    (lambda: make_gpd(0.1, 1.0),      "GPD(0.1)"),
    (lambda: make_logistic(0.0, 1.0), "Logistic"),
    (lambda: make_laplace(0.0, 1.0),  "Laplace"),
])
def test_r_lower_bound_parametric(make_dist, label):
    dist = make_dist()
    sd, gmd = _sd_gmd(dist)
    R = sd / gmd
    lower = np.sqrt(3.0) / 2.0
    assert R >= lower - 1e-5, f"{label}: R={R:.6f} < sqrt(3)/2={lower:.6f}"
