from __future__ import annotations

import numpy as np


def chi2_hz(z_data: np.ndarray, H_data: np.ndarray, sigma: np.ndarray, H_model_func) -> float:
    z_data = np.asarray(z_data, float)
    H_data = np.asarray(H_data, float)
    sigma = np.asarray(sigma, float)
    Hm = H_model_func(z_data)
    r = (Hm - H_data) / sigma
    return float(np.sum(r * r))
