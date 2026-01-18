from __future__ import annotations

import numpy as np


def chi2_bao(z: np.ndarray, dv_over_rd_obs: np.ndarray, sigma: np.ndarray, dv_over_rd_model_func) -> float:
    """
    MVP BAO: compara D_V(z)/r_d.
    Si aun no modelas r_d, tratalo como constante absorbida en el dataset demo.
    """
    z = np.asarray(z, float)
    y = np.asarray(dv_over_rd_obs, float)
    s = np.asarray(sigma, float)
    ym = dv_over_rd_model_func(z)
    r = (ym - y) / s
    return float(np.sum(r * r))
