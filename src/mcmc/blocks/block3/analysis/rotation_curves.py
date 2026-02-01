"""Rotation curve analysis for MCMC N-body simulations.

Computes and analyzes galaxy rotation curves, comparing MCMC predictions
with standard CDM and observed data.

Key MCMC prediction: The MCV (dark matter equivalent) produces cored
profiles in dwarf galaxies, leading to slowly rising rotation curves.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.optimize import curve_fit
from typing import Literal

from mcmc.core.ontology import S_0
from mcmc.blocks.block3.profiles import NFWProfile, BurkertProfile, ZhaoMCMCProfile
from mcmc.blocks.block3.profiles.nfw import NFWParams
from mcmc.blocks.block3.profiles.burkert import BurkertParams
from mcmc.blocks.block3.profiles.zhao_mcmc import ZhaoMCMCParams


@dataclass(frozen=True)
class RotationCurveParams:
    """Parameters for rotation curve analysis.

    Attributes:
        r_min: Minimum radius [kpc]
        r_max: Maximum radius [kpc]
        n_bins: Number of radial bins
        include_baryons: Whether to include baryonic contribution
        M_star: Stellar mass [M_sun] (if include_baryons)
        r_disk: Disk scale length [kpc]
        S_galaxy: Entropic coordinate of galaxy
    """
    r_min: float = 0.1      # kpc
    r_max: float = 30.0     # kpc
    n_bins: int = 50
    include_baryons: bool = True
    M_star: float = 1e10    # M_sun
    r_disk: float = 3.0     # kpc
    S_galaxy: float = S_0


@dataclass
class RotationCurveData:
    """Container for rotation curve data.

    Attributes:
        r: Radii [kpc]
        v_obs: Observed velocity [km/s]
        v_err: Velocity error [km/s]
        v_model: Model velocity [km/s] (optional)
        v_disk: Disk contribution [km/s] (optional)
        v_halo: Halo contribution [km/s] (optional)
    """
    r: np.ndarray
    v_obs: np.ndarray
    v_err: np.ndarray
    v_model: np.ndarray | None = None
    v_disk: np.ndarray | None = None
    v_halo: np.ndarray | None = None


def disk_velocity(r: np.ndarray, M_star: float, r_disk: float) -> np.ndarray:
    """Exponential disk rotation curve contribution.

    v_disk^2 = G * M_star * y^2 * [I_0*K_0 - I_1*K_1]
    where y = r / (2*r_disk)

    Simplified approximation for flat disk.

    Args:
        r: Radii [kpc]
        M_star: Stellar mass [M_sun]
        r_disk: Disk scale length [kpc]

    Returns:
        Disk velocity contribution [km/s]
    """
    G_units = 4.302e-6  # kpc * (km/s)^2 / M_sun
    y = r / (2 * r_disk)
    y = np.maximum(y, 1e-3)

    # Approximation: v_disk^2 ~ G*M / r * f(y)
    # where f(y) captures the disk geometry
    # Using Freeman (1970) approximation
    from scipy.special import i0, i1, k0, k1
    bessel_term = i0(y) * k0(y) - i1(y) * k1(y)
    v2 = G_units * M_star * y ** 2 * bessel_term / r_disk

    return np.sqrt(np.maximum(v2, 0))


def compute_rotation_curve(
    r: np.ndarray,
    profile: NFWProfile | BurkertProfile | ZhaoMCMCProfile,
    params: RotationCurveParams | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute total rotation curve from halo profile.

    v_total^2 = v_disk^2 + v_halo^2

    Args:
        r: Radii [kpc]
        profile: Halo density profile
        params: Analysis parameters

    Returns:
        Tuple of (v_total, v_disk, v_halo) in km/s
    """
    if params is None:
        params = RotationCurveParams()

    # Convert r from kpc to Mpc for profile calculations
    r_mpc = r * 1e-3

    # Halo contribution from profile
    v_halo = profile.velocity(r_mpc)

    # Disk contribution
    if params.include_baryons:
        v_disk = disk_velocity(r, params.M_star, params.r_disk)
    else:
        v_disk = np.zeros_like(r)

    # Total velocity (quadrature sum)
    v_total = np.sqrt(v_halo ** 2 + v_disk ** 2)

    return v_total, v_disk, v_halo


def fit_rotation_curve(
    data: RotationCurveData,
    profile_type: Literal["nfw", "burkert", "zhao_mcmc"] = "zhao_mcmc",
    params: RotationCurveParams | None = None,
) -> tuple[dict, np.ndarray]:
    """Fit rotation curve data with specified profile.

    Args:
        data: Observed rotation curve data
        profile_type: Type of profile to fit
        params: Analysis parameters

    Returns:
        Tuple of (best_fit_params, model_curve)
    """
    if params is None:
        params = RotationCurveParams()

    r_kpc = data.r
    r_kpc * 1e-3
    v_obs = data.v_obs
    v_err = np.maximum(data.v_err, 1.0)  # Minimum error of 1 km/s

    # Define fitting function based on profile type
    if profile_type == "nfw":
        def model_func(r, log_rho_s, r_s):
            rho_s = 10 ** log_rho_s
            profile = NFWProfile(NFWParams(rho_s=rho_s, r_s=r_s))
            v_tot, _, _ = compute_rotation_curve(r, profile, params)
            return v_tot

        p0 = [7.0, 0.02]  # log(rho_s), r_s
        bounds = ([4.0, 0.001], [12.0, 0.5])

    elif profile_type == "burkert":
        def model_func(r, log_rho_0, r_0):
            rho_0 = 10 ** log_rho_0
            r_0_mpc = r_0 * 1e-3  # Input in kpc, convert to Mpc
            profile = BurkertProfile(BurkertParams(rho_0=rho_0, r_0=r_0_mpc))
            v_tot, _, _ = compute_rotation_curve(r, profile, params)
            return v_tot

        p0 = [8.0, 5.0]  # log(rho_0), r_0 [kpc]
        bounds = ([5.0, 0.1], [12.0, 50.0])

    else:  # zhao_mcmc
        def model_func(r, log_rho_s, r_s, delta_gamma):
            rho_s = 10 ** log_rho_s
            profile = ZhaoMCMCProfile(ZhaoMCMCParams(
                rho_s=rho_s,
                r_s=r_s,
                delta_gamma=delta_gamma,
                S_halo=params.S_galaxy,
            ))
            v_tot, _, _ = compute_rotation_curve(r, profile, params)
            return v_tot

        p0 = [7.0, 0.02, 0.5]  # log(rho_s), r_s, delta_gamma
        bounds = ([4.0, 0.001, 0.0], [12.0, 0.5, 1.0])

    # Perform fit
    try:
        popt, pcov = curve_fit(
            model_func,
            r_kpc,
            v_obs,
            p0=p0,
            sigma=v_err,
            bounds=bounds,
            maxfev=5000,
        )
        perr = np.sqrt(np.diag(pcov))
    except RuntimeError:
        # Fit failed, return initial guess
        popt = np.array(p0)
        perr = np.zeros_like(popt)

    # Compute best-fit model
    v_model = model_func(r_kpc, *popt)

    # Package results
    if profile_type == "nfw":
        result = {
            "log_rho_s": popt[0],
            "rho_s": 10 ** popt[0],
            "r_s": popt[1],
            "errors": {"log_rho_s": perr[0], "r_s": perr[1]},
        }
    elif profile_type == "burkert":
        result = {
            "log_rho_0": popt[0],
            "rho_0": 10 ** popt[0],
            "r_0": popt[1],
            "errors": {"log_rho_0": perr[0], "r_0": perr[1]},
        }
    else:
        result = {
            "log_rho_s": popt[0],
            "rho_s": 10 ** popt[0],
            "r_s": popt[1],
            "delta_gamma": popt[2],
            "errors": {"log_rho_s": perr[0], "r_s": perr[1], "delta_gamma": perr[2]},
        }

    # Compute chi-squared
    chi2 = np.sum(((v_obs - v_model) / v_err) ** 2)
    dof = len(v_obs) - len(popt)
    result["chi2"] = chi2
    result["chi2_reduced"] = chi2 / dof if dof > 0 else chi2
    result["profile_type"] = profile_type

    return result, v_model


class RotationCurveAnalyzer:
    """Analyzer for galaxy rotation curves.

    Compares MCMC predictions with observations and standard CDM.
    """

    def __init__(self, params: RotationCurveParams | None = None):
        """Initialize analyzer.

        Args:
            params: Analysis parameters
        """
        self.params = params or RotationCurveParams()

    def analyze(
        self,
        data: RotationCurveData,
    ) -> dict:
        """Analyze rotation curve with multiple profiles.

        Fits NFW, Burkert, and Zhao-MCMC profiles and compares.

        Args:
            data: Observed rotation curve

        Returns:
            Dictionary with fit results for each profile
        """
        results = {}

        for profile_type in ["nfw", "burkert", "zhao_mcmc"]:
            fit_result, v_model = fit_rotation_curve(
                data,
                profile_type=profile_type,
                params=self.params,
            )
            results[profile_type] = {
                "params": fit_result,
                "v_model": v_model,
            }

        # Determine best profile by chi2
        chi2_values = {pt: results[pt]["params"]["chi2_reduced"] for pt in results}
        best_profile = min(chi2_values, key=chi2_values.get)
        results["best_profile"] = best_profile
        results["chi2_comparison"] = chi2_values

        return results

    def compute_inner_slope(
        self,
        data: RotationCurveData,
        r_inner: float = 1.0,  # kpc
    ) -> float:
        """Compute inner slope of rotation curve.

        d(log v) / d(log r) at r = r_inner

        Args:
            data: Rotation curve data
            r_inner: Radius for slope measurement [kpc]

        Returns:
            Inner slope (0 for rising, 0.5 for flat, ~0.5 for declining)
        """
        r = data.r
        v = data.v_obs

        # Find points near r_inner
        idx = np.argmin(np.abs(r - r_inner))
        if idx == 0:
            idx = 1

        # Local log-log slope
        log_r = np.log10(r[idx - 1:idx + 2])
        log_v = np.log10(np.maximum(v[idx - 1:idx + 2], 1e-3))

        slope = np.polyfit(log_r, log_v, 1)[0]
        return slope

    def cusp_core_test(
        self,
        data: RotationCurveData,
    ) -> dict:
        """Test whether rotation curve prefers cusp or core.

        Compares NFW (cusp) vs Burkert (core) fits.

        Args:
            data: Rotation curve data

        Returns:
            Dictionary with test results
        """
        nfw_result, v_nfw = fit_rotation_curve(data, "nfw", self.params)
        burkert_result, v_burkert = fit_rotation_curve(data, "burkert", self.params)

        chi2_nfw = nfw_result["chi2_reduced"]
        chi2_burkert = burkert_result["chi2_reduced"]

        # Bayesian evidence ratio approximation (BIC difference)
        n_data = len(data.v_obs)
        n_params_nfw = 2
        n_params_burkert = 2
        bic_nfw = chi2_nfw * n_data + n_params_nfw * np.log(n_data)
        bic_burkert = chi2_burkert * n_data + n_params_burkert * np.log(n_data)

        delta_bic = bic_nfw - bic_burkert  # Positive = Burkert preferred

        return {
            "chi2_nfw": chi2_nfw,
            "chi2_burkert": chi2_burkert,
            "delta_chi2": chi2_nfw - chi2_burkert,
            "bic_nfw": bic_nfw,
            "bic_burkert": bic_burkert,
            "delta_bic": delta_bic,
            "prefers_core": delta_bic > 2,  # Positive = core preferred
            "inner_slope": self.compute_inner_slope(data),
        }
