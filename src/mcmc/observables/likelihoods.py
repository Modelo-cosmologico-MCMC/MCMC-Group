"""Likelihood functions for post-Big-Bang observables.

CRITICAL: These likelihoods are ONLY valid for post-BB observables (z ≥ 0).
The observable universe begins at S = S_BB = 1.001 (the Big Bang).

Observables:
- H(z): Hubble parameter vs redshift
- μ(z): Distance modulus vs redshift (SNe Ia)
- DV/rd(z): BAO angular scale vs redshift
"""
from __future__ import annotations

import numpy as np

from .hz import chi2_hz
from .sne import chi2_sne
from .bao import chi2_bao


def _validate_postbb_redshifts(z: np.ndarray, name: str) -> None:
    """Validate that all redshifts are in post-BB regime (z ≥ 0).

    Args:
        z: Array of redshifts
        name: Dataset name for error message

    Raises:
        ValueError: If any z < 0
    """
    z_arr = np.asarray(z, float)
    if np.any(z_arr < 0):
        raise ValueError(
            f"{name}: Redshifts must be z >= 0 (post-BB regime). "
            f"Found min(z) = {z_arr.min():.4f}"
        )


def loglike_total(datasets: dict, model: dict, validate: bool = True) -> float:
    """Compute total log-likelihood for post-BB observables.

    This function computes the combined log-likelihood from H(z), SNe Ia,
    and BAO datasets. All observables are defined in the post-BB regime
    (z >= 0, corresponding to S > S_BB = 1.001).

    Args:
        datasets: Dictionary with keys 'hz', 'sne', 'bao' containing
                  observation data (z, values, sigma, optional cov/cov_inv)
        model: Dictionary with keys 'H(z)', 'mu(z)', 'DVrd(z)' containing
               callable model functions
        validate: If True, validate that all z >= 0 (post-BB)

    Returns:
        Total log-likelihood = -0.5 * sum(chi2)

    Raises:
        ValueError: If validate=True and any dataset has z < 0
    """
    chi2 = 0.0

    if "hz" in datasets:
        d = datasets["hz"]
        if validate:
            _validate_postbb_redshifts(d["z"], "H(z)")
        chi2 += chi2_hz(
            d["z"], d["H"], d.get("sigma"), model["H(z)"],
            cov=d.get("cov"), cov_inv=d.get("cov_inv"),
        )

    if "sne" in datasets:
        d = datasets["sne"]
        if validate:
            _validate_postbb_redshifts(d["z"], "SNe Ia")
        chi2 += chi2_sne(
            d["z"], d["mu"], d.get("sigma"), model["mu(z)"],
            cov=d.get("cov"), cov_inv=d.get("cov_inv"),
        )

    if "bao" in datasets:
        d = datasets["bao"]
        if validate:
            _validate_postbb_redshifts(d["z"], "BAO")
        chi2 += chi2_bao(
            d["z"], d["dv_rd"], d.get("sigma"), model["DVrd(z)"],
            cov=d.get("cov"), cov_inv=d.get("cov_inv"),
        )

    return -0.5 * chi2


def loglike_hz(z: np.ndarray, H_obs: np.ndarray, sigma: np.ndarray,
               H_model_func, validate: bool = True, **cov_kwargs) -> float:
    """Log-likelihood for H(z) measurements.

    ONLY valid for post-BB regime (z >= 0).

    Args:
        z: Redshift array
        H_obs: Observed H(z) values
        sigma: Uncertainties (or provide cov/cov_inv)
        H_model_func: Model function H(z)
        validate: If True, validate z >= 0
        **cov_kwargs: Optional cov or cov_inv matrix

    Returns:
        Log-likelihood = -0.5 * chi2
    """
    if validate:
        _validate_postbb_redshifts(z, "H(z)")
    chi2 = chi2_hz(z, H_obs, sigma, H_model_func, **cov_kwargs)
    return -0.5 * chi2


def loglike_sne(z: np.ndarray, mu_obs: np.ndarray, sigma: np.ndarray,
                mu_model_func, validate: bool = True, **cov_kwargs) -> float:
    """Log-likelihood for SNe Ia distance modulus measurements.

    ONLY valid for post-BB regime (z >= 0).

    Args:
        z: Redshift array
        mu_obs: Observed distance modulus values
        sigma: Uncertainties (or provide cov/cov_inv)
        mu_model_func: Model function mu(z)
        validate: If True, validate z >= 0
        **cov_kwargs: Optional cov or cov_inv matrix

    Returns:
        Log-likelihood = -0.5 * chi2
    """
    if validate:
        _validate_postbb_redshifts(z, "SNe Ia")
    chi2 = chi2_sne(z, mu_obs, sigma, mu_model_func, **cov_kwargs)
    return -0.5 * chi2


def loglike_bao(z: np.ndarray, dvrd_obs: np.ndarray, sigma: np.ndarray,
                dvrd_model_func, validate: bool = True, **cov_kwargs) -> float:
    """Log-likelihood for BAO measurements.

    ONLY valid for post-BB regime (z >= 0).

    Args:
        z: Redshift array
        dvrd_obs: Observed DV/rd values
        sigma: Uncertainties (or provide cov/cov_inv)
        dvrd_model_func: Model function DVrd(z)
        validate: If True, validate z >= 0
        **cov_kwargs: Optional cov or cov_inv matrix

    Returns:
        Log-likelihood = -0.5 * chi2
    """
    if validate:
        _validate_postbb_redshifts(z, "BAO")
    chi2 = chi2_bao(z, dvrd_obs, sigma, dvrd_model_func, **cov_kwargs)
    return -0.5 * chi2
