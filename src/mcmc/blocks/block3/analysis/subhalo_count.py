"""Subhalo abundance analysis for MCMC simulations.

The "missing satellites problem" in CDM is the over-prediction of
small subhalos compared to observed dwarf satellites.

MCMC prediction: The MCV (dark matter equivalent) dynamics naturally
suppresses small-scale structure, potentially resolving this tension.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.integrate import quad

from mcmc.core.ontology import S_0
from mcmc.blocks.block3.analysis.mass_function import (
    MassFunctionParams,
    sigma_M,
)


@dataclass(frozen=True)
class SubhaloParams:
    """Parameters for subhalo abundance calculation.

    Attributes:
        M_host: Host halo mass [M_sun]
        M_sub_min: Minimum subhalo mass [M_sun]
        M_sub_max: Maximum subhalo mass [M_sun]
        n_bins: Number of mass bins
        z_infall: Average infall redshift
        S_host: Entropic coordinate of host
        tidal_strength: Tidal stripping efficiency (0-1)
        use_mcmc_suppression: Apply MCMC small-scale suppression
    """
    M_host: float = 1e12       # Milky Way-like
    M_sub_min: float = 1e6     # M_sun
    M_sub_max: float = 1e11    # M_sun
    n_bins: int = 30
    z_infall: float = 2.0
    S_host: float = S_0
    tidal_strength: float = 0.5
    use_mcmc_suppression: bool = True


def unevolved_subhalo_mf(
    M_sub: float | np.ndarray,
    M_host: float,
    params: SubhaloParams,
) -> float | np.ndarray:
    """Unevolved subhalo mass function (at infall).

    Based on extended Press-Schechter theory:
        dN/dM = (M_host/M_sub) * f(M_sub/M_host) * |d(ln sigma)/d(ln M)|

    Args:
        M_sub: Subhalo mass [M_sun]
        M_host: Host halo mass [M_sun]
        params: Subhalo parameters

    Returns:
        dN/dM_sub [M_sun^-1]
    """
    M_arr = np.asarray(M_sub)

    # Mass ratio
    x = M_arr / M_host

    # Variance at subhalo and host mass
    mf_params = MassFunctionParams(S_current=params.S_host)
    sig_sub = sigma_M(M_arr, mf_params)
    sig_host = sigma_M(M_host, mf_params)

    # Conditional variance
    delta_sig2 = sig_sub ** 2 - sig_host ** 2
    delta_sig2 = np.maximum(delta_sig2, 1e-10)
    delta_sig = np.sqrt(delta_sig2)

    # Conditional mass function (Lacey & Cole 1993)
    delta_c = 1.686
    delta_c / delta_sig

    # Simplified form
    prefactor = 0.1 * (M_host / M_arr)
    alpha = 0.9  # Slope of subhalo MF
    dn_dM = prefactor * x ** (-alpha) * np.exp(-x / 0.1)

    return float(dn_dM) if np.isscalar(M_sub) else dn_dM


def mcmc_suppression_factor(
    M: float | np.ndarray,
    S: float,
    M_suppress: float = 1e8,
) -> float | np.ndarray:
    """MCMC small-scale suppression factor.

    In MCMC, the MCV dynamics suppresses small-scale structure below
    a characteristic mass M_suppress(S).

    The suppression mass depends on S:
        M_suppress(S) ~ M_0 * (S / S_0)^beta

    Args:
        M: Halo mass [M_sun]
        S: Entropic coordinate
        M_suppress: Suppression mass scale at S_0 [M_sun]

    Returns:
        Suppression factor (0 to 1)
    """
    M_arr = np.asarray(M)

    # S-dependent suppression mass
    beta = 1.5
    M_supp_S = M_suppress * (S / S_0) ** beta

    # Smooth suppression (error function-like)
    x = np.log10(M_arr / M_supp_S)
    width = 0.5  # Width of transition in dex
    suppression = 0.5 * (1 + np.tanh(x / width))

    return float(suppression) if np.isscalar(M) else suppression


def tidal_stripping_factor(
    M_sub: float | np.ndarray,
    M_host: float,
    tidal_strength: float = 0.5,
) -> float | np.ndarray:
    """Mass loss factor due to tidal stripping.

    Subhalos lose mass after falling into the host:
        M_final / M_infall = f(M_sub/M_host)

    Args:
        M_sub: Initial subhalo mass [M_sun]
        M_host: Host halo mass [M_sun]
        tidal_strength: Stripping efficiency (0-1)

    Returns:
        Mass retention fraction
    """
    M_arr = np.asarray(M_sub)
    x = M_arr / M_host

    # Smaller subhalos are more stripped
    f_retain = 1 - tidal_strength * (1 - x ** 0.3)
    f_retain = np.clip(f_retain, 0.1, 1.0)

    return float(f_retain) if np.isscalar(M_sub) else f_retain


def subhalo_mass_function(
    M_sub: float | np.ndarray,
    params: SubhaloParams,
) -> float | np.ndarray:
    """Evolved subhalo mass function.

    Includes:
    1. Unevolved (infall) mass function
    2. Tidal stripping
    3. MCMC small-scale suppression

    Args:
        M_sub: Subhalo mass [M_sun]
        params: Subhalo parameters

    Returns:
        dN/dM_sub [M_sun^-1]
    """
    M_arr = np.asarray(M_sub)

    # Unevolved mass function
    dn_dM = unevolved_subhalo_mf(M_arr, params.M_host, params)

    # Tidal stripping correction
    f_tidal = tidal_stripping_factor(M_arr, params.M_host, params.tidal_strength)
    dn_dM = dn_dM * f_tidal

    # MCMC suppression
    if params.use_mcmc_suppression:
        f_mcmc = mcmc_suppression_factor(M_arr, params.S_host)
        dn_dM = dn_dM * f_mcmc

    return float(dn_dM) if np.isscalar(M_sub) else dn_dM


def count_subhalos(
    params: SubhaloParams,
    M_min: float | None = None,
    M_max: float | None = None,
) -> float:
    """Count total number of subhalos in mass range.

    N(<M) = integral_M_min^M_max dN/dM dM

    Args:
        params: Subhalo parameters
        M_min: Minimum mass (default: params.M_sub_min)
        M_max: Maximum mass (default: params.M_sub_max)

    Returns:
        Total number of subhalos
    """
    if M_min is None:
        M_min = params.M_sub_min
    if M_max is None:
        M_max = params.M_sub_max

    def integrand(logM):
        M = 10 ** logM
        return M * np.log(10) * subhalo_mass_function(M, params)

    result, _ = quad(integrand, np.log10(M_min), np.log10(M_max))
    return result


class SubhaloCounter:
    """Subhalo abundance calculator with MCMC corrections."""

    def __init__(self, params: SubhaloParams | None = None):
        """Initialize subhalo counter.

        Args:
            params: Calculation parameters
        """
        self.params = params or SubhaloParams()

    def dn_dM(
        self,
        M_sub: float | np.ndarray,
    ) -> float | np.ndarray:
        """Subhalo mass function dN/dM.

        Args:
            M_sub: Subhalo mass [M_sun]

        Returns:
            dN/dM [M_sun^-1]
        """
        return subhalo_mass_function(M_sub, self.params)

    def dn_dlog10M(
        self,
        M_sub: float | np.ndarray,
    ) -> float | np.ndarray:
        """Subhalo mass function per log10(M) bin.

        Args:
            M_sub: Subhalo mass [M_sun]

        Returns:
            dN/dlog10(M)
        """
        M_arr = np.asarray(M_sub)
        return M_arr * np.log(10) * self.dn_dM(M_arr)

    def total_count(
        self,
        M_min: float | None = None,
        M_max: float | None = None,
    ) -> float:
        """Total number of subhalos.

        Args:
            M_min: Minimum mass
            M_max: Maximum mass

        Returns:
            Total subhalo count
        """
        return count_subhalos(self.params, M_min, M_max)

    def cumulative(
        self,
        M: np.ndarray,
    ) -> np.ndarray:
        """Cumulative number of subhalos N(>M).

        Args:
            M: Mass array [M_sun]

        Returns:
            N(>M) array
        """
        N_cum = np.zeros_like(M)
        for i, M_val in enumerate(M):
            N_cum[i] = self.total_count(M_val, self.params.M_sub_max)
        return N_cum

    def compare_cdm(
        self,
        M: np.ndarray,
    ) -> dict:
        """Compare MCMC and CDM subhalo abundances.

        Args:
            M: Mass array [M_sun]

        Returns:
            Dictionary with comparison results
        """
        # MCMC prediction
        dn_mcmc = self.dn_dlog10M(M)
        N_mcmc = self.cumulative(M)

        # Standard CDM (no MCMC suppression)
        cdm_params = SubhaloParams(
            M_host=self.params.M_host,
            M_sub_min=self.params.M_sub_min,
            M_sub_max=self.params.M_sub_max,
            S_host=self.params.S_host,
            use_mcmc_suppression=False,
        )
        cdm_counter = SubhaloCounter(cdm_params)
        dn_cdm = cdm_counter.dn_dlog10M(M)
        N_cdm = cdm_counter.cumulative(M)

        return {
            "M": M,
            "dn_mcmc": dn_mcmc,
            "dn_cdm": dn_cdm,
            "N_mcmc": N_mcmc,
            "N_cdm": N_cdm,
            "suppression": dn_mcmc / np.maximum(dn_cdm, 1e-30),
        }

    def missing_satellites_prediction(self) -> dict:
        """Predict resolution of missing satellites problem.

        Returns:
            Dictionary with predictions
        """
        # Observed MW satellites: ~60 above ~10^5 M_sun
        M_obs_min = 1e5
        N_observed = 60

        # CDM prediction (no suppression)
        cdm_params = SubhaloParams(
            M_host=self.params.M_host,
            use_mcmc_suppression=False,
        )
        N_cdm = count_subhalos(cdm_params, M_obs_min, self.params.M_sub_max)

        # MCMC prediction
        N_mcmc = self.total_count(M_obs_min, self.params.M_sub_max)

        return {
            "N_observed": N_observed,
            "N_cdm_predicted": N_cdm,
            "N_mcmc_predicted": N_mcmc,
            "cdm_overproduction": N_cdm / N_observed,
            "mcmc_overproduction": N_mcmc / N_observed,
            "mcmc_suppression": N_cdm / max(N_mcmc, 1),
            "tension_resolved": N_mcmc < 2 * N_observed,
        }
