"""Halo mass function for MCMC cosmology.

Computes the halo mass function dn/dM, which gives the number density
of halos as a function of mass.

MCMC modification: The mass function depends on S through:
1. Modified growth factor D(S)
2. MCV contribution to effective matter density
3. Stratified collapse threshold
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.integrate import quad

from mcmc.core.ontology import S_0, OMEGA_M, H_0


@dataclass(frozen=True)
class MassFunctionParams:
    """Parameters for halo mass function calculation.

    Attributes:
        M_min: Minimum halo mass [M_sun]
        M_max: Maximum halo mass [M_sun]
        n_bins: Number of mass bins
        sigma8: RMS fluctuation at 8 Mpc/h
        n_s: Spectral index
        S_current: Current entropic coordinate
        use_mcmc_correction: Whether to apply MCMC S-dependent corrections
    """
    M_min: float = 1e8
    M_max: float = 1e15
    n_bins: int = 50
    sigma8: float = 0.811
    n_s: float = 0.965
    S_current: float = S_0
    use_mcmc_correction: bool = True


def sigma_M(M: float | np.ndarray, params: MassFunctionParams) -> float | np.ndarray:
    """RMS fluctuation sigma(M) for mass M.

    sigma(M) propto M^(-alpha) with alpha ~ (n_s + 3) / 6

    Args:
        M: Halo mass [M_sun]
        params: Mass function parameters

    Returns:
        sigma(M)
    """
    M_arr = np.asarray(M)

    # Normalization: sigma(M_8) = sigma_8 where M_8 ~ 5e13 M_sun
    M_8 = 5e13  # M_sun (mass within 8 Mpc/h sphere)
    alpha = (params.n_s + 3) / 6

    sigma = params.sigma8 * (M_arr / M_8) ** (-alpha)
    return float(sigma) if np.isscalar(M) else sigma


def dsigma_dM(M: float | np.ndarray, params: MassFunctionParams) -> float | np.ndarray:
    """Derivative d(sigma)/dM.

    Args:
        M: Halo mass [M_sun]
        params: Mass function parameters

    Returns:
        d(sigma)/dM
    """
    M_arr = np.asarray(M)
    alpha = (params.n_s + 3) / 6

    dsig = -alpha * sigma_M(M_arr, params) / M_arr
    return float(dsig) if np.isscalar(M) else dsig


def growth_factor_mcmc(S: float, S_0: float = S_0) -> float:
    """MCMC growth factor D(S).

    The growth factor is modified by the entropic evolution:
        D(S) / D(S_0) = f(S / S_0)

    For S < S_0 (earlier times), D < D(S_0).

    Args:
        S: Entropic coordinate
        S_0: Present entropic coordinate

    Returns:
        D(S) / D(S_0)
    """
    # Simple model: D grows approximately as S^0.5 in matter-dominated era
    S_trans = 65.0  # Transition to Lambda domination
    if S < S_trans:
        return (S / S_0) ** 0.5
    else:
        # Slower growth during Lambda domination
        return (S_trans / S_0) ** 0.5 * (1 + 0.3 * np.log(S / S_trans))


def delta_c_mcmc(S: float, params: MassFunctionParams) -> float:
    """MCMC-corrected collapse threshold.

    The critical overdensity for collapse depends on S:
        delta_c(S) = delta_c_0 * f(S)

    In MCMC, tensorial effects modify the collapse threshold.

    Args:
        S: Entropic coordinate
        params: Mass function parameters

    Returns:
        Collapse threshold delta_c
    """
    delta_c_0 = 1.686  # Standard spherical collapse value

    if not params.use_mcmc_correction:
        return delta_c_0

    # MCMC correction: tensorial effects lower the threshold slightly
    # for intermediate S (during structure formation)
    S_peak = 47.5  # Peak of structure formation
    correction = 1 - 0.05 * np.exp(-0.5 * ((S - S_peak) / 20) ** 2)

    return delta_c_0 * correction


def press_schechter(
    M: float | np.ndarray,
    params: MassFunctionParams,
    z: float = 0.0,
) -> float | np.ndarray:
    """Press-Schechter mass function.

    dn/dM = sqrt(2/pi) * (rho_m/M) * |d(ln sigma)/d(ln M)|
            * (delta_c/sigma) * exp(-delta_c^2 / (2*sigma^2))

    Args:
        M: Halo mass [M_sun]
        params: Mass function parameters
        z: Redshift

    Returns:
        dn/dM [Mpc^-3 M_sun^-1]
    """
    M_arr = np.asarray(M)

    # Mean matter density [M_sun / Mpc^3]
    h = H_0 / 100
    rho_crit = 2.775e11 * h ** 2  # M_sun / Mpc^3
    rho_m = OMEGA_M * rho_crit * (1 + z) ** 3

    # Variance and derivative
    sig = sigma_M(M_arr, params)
    dsig = dsigma_dM(M_arr, params)

    # Growth factor correction
    if params.use_mcmc_correction:
        D_ratio = growth_factor_mcmc(params.S_current)
        sig = sig * D_ratio

    # Collapse threshold
    delta_c = delta_c_mcmc(params.S_current, params)

    # Press-Schechter formula
    nu = delta_c / sig
    prefactor = np.sqrt(2 / np.pi) * (rho_m / M_arr)
    dln_sig_dln_M = M_arr * dsig / sig

    dn_dM = prefactor * np.abs(dln_sig_dln_M) * nu * np.exp(-nu ** 2 / 2)

    return float(dn_dM) if np.isscalar(M) else dn_dM


def sheth_tormen(
    M: float | np.ndarray,
    params: MassFunctionParams,
    z: float = 0.0,
) -> float | np.ndarray:
    """Sheth-Tormen mass function (improved Press-Schechter).

    Includes ellipsoidal collapse corrections:
        f(nu) = A * sqrt(2a/pi) * [1 + (nu^2/a)^p] * nu * exp(-a*nu^2/2)

    Args:
        M: Halo mass [M_sun]
        params: Mass function parameters
        z: Redshift

    Returns:
        dn/dM [Mpc^-3 M_sun^-1]
    """
    M_arr = np.asarray(M)

    # Sheth-Tormen parameters
    A = 0.3222
    a = 0.707
    p = 0.3

    # Mean matter density
    h = H_0 / 100
    rho_crit = 2.775e11 * h ** 2
    rho_m = OMEGA_M * rho_crit * (1 + z) ** 3

    # Variance
    sig = sigma_M(M_arr, params)
    dsig = dsigma_dM(M_arr, params)

    # Growth factor correction
    if params.use_mcmc_correction:
        D_ratio = growth_factor_mcmc(params.S_current)
        sig = sig * D_ratio

    # Collapse threshold
    delta_c = delta_c_mcmc(params.S_current, params)

    nu = delta_c / sig

    # Sheth-Tormen multiplicity function
    f_nu = A * np.sqrt(2 * a / np.pi) * (1 + (nu ** 2 / a) ** p) * nu * np.exp(-a * nu ** 2 / 2)

    # Mass function
    prefactor = rho_m / M_arr
    dln_sig_dln_M = M_arr * dsig / sig
    dn_dM = prefactor * np.abs(dln_sig_dln_M) * f_nu

    return float(dn_dM) if np.isscalar(M) else dn_dM


def compute_mass_function_mcmc(
    M: float | np.ndarray,
    params: MassFunctionParams,
    z: float = 0.0,
    method: str = "sheth_tormen",
) -> float | np.ndarray:
    """Compute MCMC-corrected halo mass function.

    Args:
        M: Halo mass [M_sun]
        params: Mass function parameters
        z: Redshift
        method: "press_schechter" or "sheth_tormen"

    Returns:
        dn/dM [Mpc^-3 M_sun^-1]
    """
    if method == "press_schechter":
        return press_schechter(M, params, z)
    else:
        return sheth_tormen(M, params, z)


class HaloMassFunction:
    """Calculator for halo mass function with MCMC corrections."""

    def __init__(self, params: MassFunctionParams | None = None):
        """Initialize mass function calculator.

        Args:
            params: Calculation parameters
        """
        self.params = params or MassFunctionParams()

    def dn_dM(
        self,
        M: float | np.ndarray,
        z: float = 0.0,
        method: str = "sheth_tormen",
    ) -> float | np.ndarray:
        """Compute mass function dn/dM.

        Args:
            M: Halo mass [M_sun]
            z: Redshift
            method: "press_schechter" or "sheth_tormen"

        Returns:
            dn/dM [Mpc^-3 M_sun^-1]
        """
        return compute_mass_function_mcmc(M, self.params, z, method)

    def dn_dlog10M(
        self,
        M: float | np.ndarray,
        z: float = 0.0,
        method: str = "sheth_tormen",
    ) -> float | np.ndarray:
        """Mass function per log10(M) bin.

        dn/dlog10(M) = M * ln(10) * dn/dM

        Args:
            M: Halo mass [M_sun]
            z: Redshift
            method: Method to use

        Returns:
            dn/dlog10(M) [Mpc^-3]
        """
        M_arr = np.asarray(M)
        dn_dM = self.dn_dM(M_arr, z, method)
        return M_arr * np.log(10) * dn_dM

    def n_above(
        self,
        M_min: float,
        z: float = 0.0,
        method: str = "sheth_tormen",
    ) -> float:
        """Number density of halos above mass threshold.

        n(>M) = integral_M^infty dn/dM' dM'

        Args:
            M_min: Minimum mass [M_sun]
            z: Redshift
            method: Method to use

        Returns:
            n(>M) [Mpc^-3]
        """
        def integrand(logM):
            M = 10 ** logM
            return self.dn_dlog10M(M, z, method)

        log_M_min = np.log10(M_min)
        log_M_max = np.log10(self.params.M_max)

        result, _ = quad(integrand, log_M_min, log_M_max)
        return result

    def compare_lcdm(
        self,
        M: np.ndarray,
        z: float = 0.0,
    ) -> dict:
        """Compare MCMC and LCDM mass functions.

        Args:
            M: Mass array [M_sun]
            z: Redshift

        Returns:
            Dictionary with comparison results
        """
        # MCMC prediction
        dn_mcmc = self.dn_dlog10M(M, z)

        # Standard LCDM (no MCMC correction)
        lcdm_params = MassFunctionParams(
            M_min=self.params.M_min,
            M_max=self.params.M_max,
            sigma8=self.params.sigma8,
            n_s=self.params.n_s,
            use_mcmc_correction=False,
        )
        lcdm_calc = HaloMassFunction(lcdm_params)
        dn_lcdm = lcdm_calc.dn_dlog10M(M, z)

        ratio = dn_mcmc / np.maximum(dn_lcdm, 1e-30)

        return {
            "M": M,
            "dn_mcmc": dn_mcmc,
            "dn_lcdm": dn_lcdm,
            "ratio": ratio,
            "log_ratio": np.log10(ratio),
        }
