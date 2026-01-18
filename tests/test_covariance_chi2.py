import numpy as np
from mcmc.observables.sne import chi2_sne


def test_chi2_with_cov_inv_matches_diagonal():
    z = np.array([0.1, 0.2, 0.5])
    mu_obs = np.array([38.0, 40.0, 42.0])
    sigma = np.array([0.2, 0.3, 0.25])

    # Diagonal covariance
    cov = np.diag(sigma**2)
    cov_inv = np.diag(1.0 / (sigma**2))

    def mu_model(_z):
        return np.array([38.1, 39.8, 42.2])

    chi_diag = chi2_sne(z, mu_obs, sigma, mu_model)
    chi_cov = chi2_sne(z, mu_obs, sigma, mu_model, cov=cov)
    chi_cinv = chi2_sne(z, mu_obs, sigma, mu_model, cov_inv=cov_inv)

    assert abs(chi_diag - chi_cov) < 1e-10
    assert abs(chi_diag - chi_cinv) < 1e-10
