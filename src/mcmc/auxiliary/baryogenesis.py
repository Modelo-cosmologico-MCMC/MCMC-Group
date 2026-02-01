"""Baryogenesis in MCMC cosmology.

MCMC Ontology: Baryogenesis occurs during the pre-geometric to
geometric transition (S ~ S_GEOM), where:
1. Out-of-equilibrium conditions from rapid S evolution
2. B-L violation from tensorial interactions
3. CP violation from MCMC field phases

The baryon asymmetry eta_B depends on the transition dynamics.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from mcmc.core.ontology import S_GEOM, OMEGA_B


@dataclass(frozen=True)
class BaryogenesisParams:
    """Parameters for MCMC baryogenesis.

    Attributes:
        S_baryogenesis: S value where baryogenesis occurs
        delta_S_width: Width of transition region in S
        epsilon_CP: CP violation parameter
        Gamma_BL: B-L violation rate
        eta_B_observed: Observed baryon asymmetry
        use_mcmc_mechanism: Use MCMC-specific mechanism
    """
    S_baryogenesis: float = S_GEOM  # Big Bang transition
    delta_S_width: float = 0.1     # Width of transition
    epsilon_CP: float = 1e-8       # CP violation
    Gamma_BL: float = 1e-3         # B-L violation rate
    eta_B_observed: float = 6.1e-10  # Observed value
    use_mcmc_mechanism: bool = True


def sakharov_conditions(S: float, params: BaryogenesisParams) -> dict:
    """Check Sakharov conditions at given S.

    The three Sakharov conditions for baryogenesis:
    1. Baryon number violation
    2. C and CP violation
    3. Out of equilibrium

    Args:
        S: Entropic coordinate
        params: Baryogenesis parameters

    Returns:
        Dictionary with condition status
    """
    S_bg = params.S_baryogenesis
    dS = params.delta_S_width

    # In MCMC, conditions are satisfied near the Big Bang transition
    # where dS/dt is large and tensorial fields are strong

    # 1. B-L violation: occurs at S < S_GEOM
    B_violation = S < S_GEOM + dS

    # 2. CP violation: from MCMC field phases
    # Strongest near the transition
    CP_violation = np.exp(-((S - S_bg) / dS) ** 2) > 0.1

    # 3. Out of equilibrium: rapid S evolution
    # Near transition, dS/dt is maximal
    out_of_eq = np.abs(S - S_bg) < 2 * dS

    return {
        "B_violation": B_violation,
        "CP_violation": CP_violation,
        "out_of_equilibrium": out_of_eq,
        "all_satisfied": B_violation and CP_violation and out_of_eq,
        "S": S,
        "distance_to_transition": abs(S - S_bg),
    }


def cp_violation_mcmc(
    S: float,
    params: BaryogenesisParams,
) -> float:
    """CP violation parameter in MCMC.

    In MCMC, CP violation arises from the phase of the
    tensorial field Phi_ten during the transition.

    epsilon_CP(S) = epsilon_0 * exp(-(S - S_bg)^2 / delta_S^2)

    Args:
        S: Entropic coordinate
        params: Baryogenesis parameters

    Returns:
        Effective CP violation parameter
    """
    S_bg = params.S_baryogenesis
    dS = params.delta_S_width

    # CP violation peaks at the transition
    epsilon = params.epsilon_CP * np.exp(-((S - S_bg) / dS) ** 2)

    return float(epsilon)


def bl_violation_rate(
    S: float,
    params: BaryogenesisParams,
) -> float:
    """B-L violation rate in MCMC.

    In the pre-geometric regime, B-L is not conserved due to
    tensorial interactions. The rate depends on S.

    Gamma_BL(S) = Gamma_0 * H(S_GEOM - S)

    where H is the Heaviside function (smoothed).

    Args:
        S: Entropic coordinate
        params: Baryogenesis parameters

    Returns:
        B-L violation rate
    """
    S_bg = params.S_baryogenesis
    dS = params.delta_S_width

    # Smoothed Heaviside: large for S < S_GEOM
    x = (S_bg - S) / dS
    H_smooth = 0.5 * (1 + np.tanh(x))

    return params.Gamma_BL * H_smooth


def departure_from_equilibrium(
    S: float,
    params: BaryogenesisParams,
) -> float:
    """Departure from equilibrium factor.

    In MCMC, the rapid evolution of S near the transition
    drives the system out of equilibrium.

    delta(S) = |dS/dt| / H

    where H is the Hubble rate.

    Args:
        S: Entropic coordinate
        params: Baryogenesis parameters

    Returns:
        Departure factor (0 = equilibrium, 1 = strongly out of eq)
    """
    S_bg = params.S_baryogenesis
    dS = params.delta_S_width

    # Model: departure peaks at transition
    delta = np.exp(-((S - S_bg) / dS) ** 2)

    return float(delta)


def eta_B_of_S(
    S: float,
    params: BaryogenesisParams,
) -> float:
    """Baryon asymmetry produced at given S.

    eta_B = n_B / s ≈ epsilon_CP * delta * Gamma_BL / (expansion rate)

    In MCMC, the asymmetry is generated during the transition.

    Args:
        S: Entropic coordinate
        params: Baryogenesis parameters

    Returns:
        Baryon-to-entropy ratio
    """
    # Get all factors
    epsilon = cp_violation_mcmc(S, params)
    delta = departure_from_equilibrium(S, params)
    Gamma = bl_violation_rate(S, params)

    # Simplified model: eta_B ∝ epsilon * delta * Gamma
    # Normalize to give observed value at transition
    eta_max = params.eta_B_observed / (
        params.epsilon_CP * 1.0 * params.Gamma_BL
    )
    eta_B = eta_max * epsilon * delta * Gamma

    return float(eta_B)


def integrate_eta_B(
    params: BaryogenesisParams,
    S_min: float = 0.5,
    S_max: float = 2.0,
    n_points: int = 100,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Integrate baryon asymmetry over S evolution.

    Total eta_B = integral of d(eta_B)/dS over transition.

    Args:
        params: Baryogenesis parameters
        S_min: Integration start
        S_max: Integration end
        n_points: Number of integration points

    Returns:
        Tuple of (total_eta_B, S_array, eta_B_array)
    """
    S_arr = np.linspace(S_min, S_max, n_points)
    eta_arr = np.array([eta_B_of_S(S, params) for S in S_arr])

    # Trapezoidal integration (use trapezoid for numpy >= 2.0)
    try:
        total_eta = np.trapezoid(eta_arr, S_arr)
    except AttributeError:
        total_eta = np.trapz(eta_arr, S_arr)

    return total_eta, S_arr, eta_arr


class BaryogenesisModel:
    """MCMC baryogenesis model."""

    def __init__(self, params: BaryogenesisParams | None = None):
        """Initialize model.

        Args:
            params: Model parameters
        """
        self.params = params or BaryogenesisParams()

    def eta_B(self, S: float) -> float:
        """Baryon asymmetry at S."""
        return eta_B_of_S(S, self.params)

    def sakharov(self, S: float) -> dict:
        """Check Sakharov conditions at S."""
        return sakharov_conditions(S, self.params)

    def compute_final_asymmetry(self) -> dict:
        """Compute final baryon asymmetry from MCMC evolution.

        Returns:
            Dictionary with asymmetry results
        """
        total_eta, S_arr, eta_arr = integrate_eta_B(self.params)

        # Find peak contribution
        i_peak = np.argmax(eta_arr)
        S_peak = S_arr[i_peak]

        return {
            "eta_B_total": total_eta,
            "eta_B_observed": self.params.eta_B_observed,
            "ratio_to_observed": total_eta / self.params.eta_B_observed,
            "S_peak_contribution": S_peak,
            "peak_eta_B": eta_arr[i_peak],
            "S_array": S_arr,
            "eta_B_array": eta_arr,
        }

    def calibrate_to_observed(self) -> BaryogenesisParams:
        """Calibrate parameters to match observed asymmetry.

        Adjusts epsilon_CP to reproduce eta_B_observed.

        Returns:
            Calibrated parameters
        """
        # First compute with default parameters
        total_eta, _, _ = integrate_eta_B(self.params)

        if total_eta < 1e-20:
            # Avoid division by zero
            scale_factor = 1.0
        else:
            scale_factor = self.params.eta_B_observed / total_eta

        # Scale epsilon_CP
        new_epsilon = self.params.epsilon_CP * scale_factor

        return BaryogenesisParams(
            S_baryogenesis=self.params.S_baryogenesis,
            delta_S_width=self.params.delta_S_width,
            epsilon_CP=new_epsilon,
            Gamma_BL=self.params.Gamma_BL,
            eta_B_observed=self.params.eta_B_observed,
            use_mcmc_mechanism=self.params.use_mcmc_mechanism,
        )

    def omega_b_prediction(self) -> float:
        """Predict Omega_b from MCMC baryogenesis.

        Omega_b ∝ eta_B * (entropy per baryon)

        Returns:
            Predicted Omega_b
        """
        result = self.compute_final_asymmetry()
        eta_B = result["eta_B_total"]

        # Simplified relation: Omega_b ~ eta_B * (some factor)
        # Calibrated to give Omega_b ~ 0.05 for eta_B ~ 6e-10
        conversion_factor = OMEGA_B / self.params.eta_B_observed

        return float(eta_B * conversion_factor)

    def compare_to_observation(self) -> dict:
        """Compare MCMC predictions to observations.

        Returns:
            Dictionary with comparison
        """
        result = self.compute_final_asymmetry()
        omega_b_pred = self.omega_b_prediction()

        return {
            "eta_B_predicted": result["eta_B_total"],
            "eta_B_observed": self.params.eta_B_observed,
            "eta_B_ratio": result["ratio_to_observed"],
            "Omega_b_predicted": omega_b_pred,
            "Omega_b_observed": OMEGA_B,
            "Omega_b_ratio": omega_b_pred / OMEGA_B,
            "agreement": 0.5 < result["ratio_to_observed"] < 2.0,
        }
