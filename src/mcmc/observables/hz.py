from __future__ import annotations

import numpy as np


def _chi2_from_cov(r: np.ndarray, cov: np.ndarray | None, cov_inv: np.ndarray | None) -> float:
    if cov_inv is not None:
        return float(r.T @ cov_inv @ r)
    if cov is not None:
        cinv = np.linalg.inv(cov)
        return float(r.T @ cinv @ r)
    raise ValueError("cov/cov_inv expected")


def chi2_hz(z_data: np.ndarray, H_data: np.ndarray, sigma: np.ndarray, H_model_func, *, cov=None, cov_inv=None) -> float:
    z_data = np.asarray(z_data, float)
    H_data = np.asarray(H_data, float)
    sigma = np.asarray(sigma, float)

    Hm = H_model_func(z_data)
    r = (Hm - H_data)

    if cov is not None or cov_inv is not None:
        return _chi2_from_cov(r, cov, cov_inv)

    r = r / sigma
    return float(np.sum(r * r))
