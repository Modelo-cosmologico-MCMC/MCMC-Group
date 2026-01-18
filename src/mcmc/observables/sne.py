from __future__ import annotations

import numpy as np


def _chi2_from_cov(r: np.ndarray, cov: np.ndarray | None, cov_inv: np.ndarray | None) -> float:
    if cov_inv is not None:
        return float(r.T @ cov_inv @ r)
    if cov is not None:
        cinv = np.linalg.inv(cov)
        return float(r.T @ cinv @ r)
    raise ValueError("cov/cov_inv expected")


def chi2_sne(z: np.ndarray, mu_obs: np.ndarray, sigma_mu: np.ndarray, mu_model_func, *, cov=None, cov_inv=None) -> float:
    z = np.asarray(z, float)
    mu_obs = np.asarray(mu_obs, float)
    sigma_mu = np.asarray(sigma_mu, float)

    mu_m = mu_model_func(z)
    r = (mu_m - mu_obs)

    if cov is not None or cov_inv is not None:
        return _chi2_from_cov(r, cov, cov_inv)

    r = r / sigma_mu
    return float(np.sum(r * r))
