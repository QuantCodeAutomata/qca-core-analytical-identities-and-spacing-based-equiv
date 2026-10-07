"""Tests for Experiment 1: Core Analytical Identities."""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.core.distributions import (
    make_exponential, make_uniform, make_normal,
    make_logistic, make_laplace,
)
from src.core.moments import compute_mean, compute_sd, compute_variance, compute_maxiance
from src.core.gmd import (
    compute_gmd_1d, compute_gmd_cov, compute_gmd_pairwise, compute_gmd_maxiance,
)
from src.core.spacing import compute_spacing_moments, check_spacing_criterion

# Fixtures ----------------------------------------------------------------

@pytest.fixture
def exp_dist():
    return make_exponential(rate=1.0)

@pytest.fixture
def uniform_dist():
    return make_uniform(0.0, 1.0)

@pytest.fixture
def normal_dist():
    return make_normal(0.0, 1.0)

@pytest.fixture
def logistic_dist():
    return make_logistic(0.0, 1.0)

@pytest.fixture
def laplace_dist():
    return make_laplace(0.0, 1.0)


# Test 1: SD^2 = (1/2)*E[(X-X')^2] for Exponential -----------------------

def test_identity_sd_squared(exp_dist):
    """Verify SD^2 = (1/2)*E[(X-X')^2] = (1/2)*2*Var[X] = Var[X]."""
    sd = compute_sd(exp_dist)
    var = compute_variance(exp_dist)
    # E[(X-X')^2] = 2*Var[X] by iid independence
    e_y2 = 2.0 * var
    gmd = compute_gmd_1d(exp_dist)
    spacing = compute_spacing_moments(exp_dist, gmd=gmd, sd=sd)
    assert abs(spacing["E_Y2"] - e_y2) < 1e-10, (
        f"E[Y^2]={spacing['E_Y2']} != 2*Var={e_y2}"
    )
    # Also verify SD^2 == Var from moments
    assert abs(sd ** 2 - var) < 1e-10


# Test 2: Three GMD methods agree within 1e-6 ----------------------------

@pytest.mark.parametrize("make_dist,label", [
    (lambda: make_exponential(rate=1.0), "Exponential"),
    (lambda: make_normal(0.0, 1.0),     "Normal"),
    (lambda: make_uniform(0.0, 1.0),    "Uniform"),
])
def test_gmd_methods_agree(make_dist, label):
    dist = make_dist()
    gmd_1d  = compute_gmd_1d(dist)
    gmd_cov = compute_gmd_cov(dist)
    gmd_mx  = compute_gmd_maxiance(dist)
    assert abs(gmd_1d - gmd_cov) < 1e-6, (
        f"{label}: |1D-Cov|={abs(gmd_1d-gmd_cov):.2e}"
    )
    assert abs(gmd_1d - gmd_mx) < 1e-6, (
        f"{label}: |1D-Mx|={abs(gmd_1d-gmd_mx):.2e}"
    )


# Test 3: GMD = 2*Mx for Exponential ------------------------------------

def test_gmd_maxiance(exp_dist):
    gmd = compute_gmd_1d(exp_dist)
    mx  = compute_maxiance(exp_dist)
    assert abs(gmd - 2.0 * mx) < 1e-6, f"GMD={gmd}, 2*Mx={2*mx}"


# Test 4: Spacing criterion sign matches for all benchmarks -------------

@pytest.mark.parametrize("make_dist,label", [
    (lambda: make_exponential(1.0), "Exponential"),
    (lambda: make_uniform(0.0, 1.0), "Uniform"),
    (lambda: make_normal(0.0, 1.0),  "Normal"),
    (lambda: make_logistic(0.0, 1.0), "Logistic"),
    (lambda: make_laplace(0.0, 1.0),  "Laplace"),
])
def test_spacing_criterion_sign(make_dist, label):
    dist = make_dist()
    sd   = compute_sd(dist)
    gmd  = compute_gmd_1d(dist)
    spacing = compute_spacing_moments(dist, gmd=gmd, sd=sd)
    result  = check_spacing_criterion(sd=sd, gmd=gmd, cv_y=spacing["CV_Y"])
    assert result["match"], (
        f"{label}: sign(SD-GMD)={result['sign_sd_gmd']}, "
        f"sign(CV-1)={result['sign_cvy_1']}, match={result['match']}"
    )


# Test 5: Exponential knife-edge: SD=GMD=1, CV[Y]=1 --------------------

def test_exponential_knife_edge(exp_dist):
    sd  = compute_sd(exp_dist)
    gmd = compute_gmd_1d(exp_dist)
    assert abs(sd - 1.0) < 1e-6,  f"Exp SD={sd}"
    assert abs(gmd - 1.0) < 1e-6, f"Exp GMD={gmd}"

    spacing = compute_spacing_moments(exp_dist, gmd=gmd, sd=sd)
    assert abs(spacing["CV_Y"] - 1.0) < 1e-4, f"Exp CV[Y]={spacing['CV_Y']}"


# Test 6: Uniform lower bound R = sqrt(3)/2 --------------------------------

def test_uniform_lower_bound(uniform_dist):
    sd  = compute_sd(uniform_dist)
    gmd = compute_gmd_1d(uniform_dist)
    R = sd / gmd
    target = np.sqrt(3.0) / 2.0
    assert abs(R - target) < 1e-6, f"Uniform R={R}, expected {target}"


# Test 7: Normal GMD = 2/sqrt(pi) ----------------------------------------

def test_normal_gmd(normal_dist):
    gmd = compute_gmd_1d(normal_dist)
    target = 2.0 / np.sqrt(np.pi)
    assert abs(gmd - target) < 1e-6, f"Normal GMD={gmd}, expected {target}"


# Test 8: R[X] >= sqrt(3)/2 for all tested distributions ----------------

@pytest.mark.parametrize("make_dist,label", [
    (lambda: make_exponential(1.0),  "Exponential"),
    (lambda: make_uniform(0.0, 1.0), "Uniform"),
    (lambda: make_normal(0.0, 1.0),  "Normal"),
    (lambda: make_logistic(0.0, 1.0),"Logistic"),
    (lambda: make_laplace(0.0, 1.0), "Laplace"),
    (lambda: make_exponential(2.0),  "Exponential(λ=2)"),
    (lambda: make_uniform(0.0, 5.0), "Uniform[0,5]"),
])
def test_r_bounds(make_dist, label):
    dist = make_dist()
    sd  = compute_sd(dist)
    gmd = compute_gmd_1d(dist)
    R   = sd / gmd
    lower = np.sqrt(3.0) / 2.0
    assert R >= lower - 1e-5, f"{label}: R={R} < sqrt(3)/2={lower}"


# Test 9: Edge-case — zero-variance distribution (degenerate) ------------

def test_empty_edge_cases():
    """Test that the uniform spacing criterion still works for small support."""
    dist = make_uniform(0.0, 0.01)
    sd  = compute_sd(dist)
    gmd = compute_gmd_1d(dist)
    assert sd > 0
    assert gmd > 0
    R = sd / gmd
    assert abs(R - np.sqrt(3.0) / 2.0) < 1e-5


# Additional verification: analytical SD and GMD values ------------------

def test_logistic_analytical(logistic_dist):
    sd  = compute_sd(logistic_dist)
    gmd = compute_gmd_1d(logistic_dist)
    assert abs(sd  - np.pi / np.sqrt(3.0)) < 1e-4, f"Logistic SD={sd}"
    assert abs(gmd - 2.0) < 1e-4, f"Logistic GMD={gmd}"


def test_laplace_analytical(laplace_dist):
    sd  = compute_sd(laplace_dist)
    gmd = compute_gmd_1d(laplace_dist)
    assert abs(sd  - np.sqrt(2.0)) < 1e-4, f"Laplace SD={sd}"
    assert abs(gmd - 1.5) < 1e-4, f"Laplace GMD={gmd}"


def test_spacing_moments_structure(exp_dist):
    """Verify spacing moments dict has expected keys and E[Y]=GMD."""
    gmd = compute_gmd_1d(exp_dist)
    sd  = compute_sd(exp_dist)
    sp  = compute_spacing_moments(exp_dist, gmd=gmd, sd=sd)
    assert set(sp.keys()) == {"E_Y", "E_Y2", "Var_Y", "CV_Y"}
    assert abs(sp["E_Y"] - gmd) < 1e-12  # by definition
    assert sp["E_Y2"] > 0
    assert sp["Var_Y"] >= 0
    assert sp["CV_Y"] >= 0
