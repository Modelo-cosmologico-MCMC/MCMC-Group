from __future__ import annotations

import numpy as np


def chi2_sne(z: np.ndarray, mu_obs: np.ndarray, sigma_mu: np.ndarray, mu_model_func) -> float:
    z = np.asarray(z, float)
    mu_obs = np.asarray(mu_obs, float)
    sigma_mu = np.asarray(sigma_mu, float)
    mu_m = mu_model_func(z)
    r = (mu_m - mu_obs) / sigma_mu
    return float(np.sum(r * r))
