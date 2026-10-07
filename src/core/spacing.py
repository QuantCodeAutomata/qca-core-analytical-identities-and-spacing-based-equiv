import numpy as np


def compute_spacing_moments(dist, gmd: float, sd: float) -> dict:
    """Compute moments of Y = |X - X'|.

    E[Y]    = GMD[X]
    E[Y^2]  = E[(X-X')^2] = 2 * Var[X] = 2 * SD[X]^2
    Var[Y]  = E[Y^2] - E[Y]^2
    CV[Y]   = sqrt(Var[Y]) / E[Y]
    """
    e_y = gmd
    e_y2 = 2.0 * sd ** 2
    var_y = e_y2 - e_y ** 2
    cv_y = np.sqrt(max(var_y, 0.0)) / e_y if e_y > 0 else np.nan
    return {
        "E_Y": e_y,
        "E_Y2": e_y2,
        "Var_Y": var_y,
        "CV_Y": cv_y,
    }


def check_spacing_criterion(sd: float, gmd: float, cv_y: float) -> dict:
    """Verify: SD[X] >= GMD[X]  iff  CV[Y] >= 1.

    sign_sd_gmd: +1 if SD > GMD, -1 if SD < GMD, 0 if equal
    sign_cvy_1:  +1 if CV[Y] > 1, -1 if CV[Y] < 1, 0 if equal
    match: True if both signs agree
    """
    eps = 1e-10
    diff_sd_gmd = sd - gmd
    diff_cv = cv_y - 1.0

    sign_sd_gmd = int(np.sign(diff_sd_gmd)) if abs(diff_sd_gmd) > eps else 0
    sign_cvy_1 = int(np.sign(diff_cv)) if abs(diff_cv) > eps else 0
    match = sign_sd_gmd == sign_cvy_1

    return {
        "sign_sd_gmd": sign_sd_gmd,
        "sign_cvy_1": sign_cvy_1,
        "match": match,
    }
