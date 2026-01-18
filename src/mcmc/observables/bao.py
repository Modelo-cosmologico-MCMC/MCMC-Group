from __future__ import annotations

import numpy as np


def _chi2_from_cov(r: np.ndarray, cov: np.ndarray | None, cov_inv: np.ndarray | None) -> float:
    if cov_inv is not None:
        return float(r.T @ cov_inv @ r)
    if cov is not None:
        cinv = np.linalg.inv(cov)
        return float(r.T @ cinv @ r)
    raise ValueError("cov/cov_inv expected")


def chi2_bao(z: np.ndarray, dv_over_rd_obs: np.ndarray, sigma: np.ndarray, dv_over_rd_model_func, *, cov=None, cov_inv=None) -> float:
    z = np.asarray(z, float)
    y = np.asarray(dv_over_rd_obs, float)
    s = np.asarray(sigma, float)

    ym = dv_over_rd_model_func(z)
    r = (ym - y)

    if cov is not None or cov_inv is not None:
        return _chi2_from_cov(r, cov, cov_inv)

    r = r / s
    return float(np.sum(r * r))
