"""Mass gap extraction from lattice correlators.

MCMC Ontology: The mass gap depends on S through beta(S).
In the pre-geometric regime (S << S_GEOM), strong coupling
produces a large mass gap (confinement).
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.optimize import curve_fit

from mcmc.blocks.block4.config import MassGapParams
from mcmc.blocks.block4.correlators import CorrelatorData, effective_mass


@dataclass
class MassGapResult:
    """Result of mass gap extraction.

    Attributes:
        mass: Extracted mass gap
        mass_err: Statistical error
        chi2: Chi-squared of fit
        chi2_dof: Chi-squared per degree of freedom
        fit_range: (t_min, t_max) used in fit
        fit_params: All fit parameters
        plateau_quality: Quality measure of effective mass plateau
    """
    mass: float
    mass_err: float
    chi2: float
    chi2_dof: float
    fit_range: tuple[int, int]
    fit_params: dict
    plateau_quality: float


def single_exponential(t: np.ndarray, A: float, m: float) -> np.ndarray:
    """Single exponential decay.

    C(t) = A * exp(-m * t)

    Args:
        t: Time array
        A: Amplitude
        m: Mass (decay rate)

    Returns:
        Correlator values
    """
    return A * np.exp(-m * t)


def double_exponential(
    t: np.ndarray, A1: float, m1: float, A2: float, m2: float
) -> np.ndarray:
    """Double exponential decay.

    C(t) = A1 * exp(-m1 * t) + A2 * exp(-m2 * t)

    Args:
        t: Time array
        A1, m1: Ground state amplitude and mass
        A2, m2: Excited state amplitude and mass

    Returns:
        Correlator values
    """
    return A1 * np.exp(-m1 * t) + A2 * np.exp(-m2 * t)


def cosh_form(t: np.ndarray, A: float, m: float, Nt: int) -> np.ndarray:
    """Cosh form for periodic BC.

    C(t) = A * (exp(-m*t) + exp(-m*(Nt-t)))
         = A * cosh(m * (t - Nt/2)) * exp(-m * Nt/2) * 2

    Args:
        t: Time array
        A: Amplitude
        m: Mass
        Nt: Temporal extent

    Returns:
        Correlator values
    """
    return A * (np.exp(-m * t) + np.exp(-m * (Nt - t)))


def fit_exponential_decay(
    correlator: CorrelatorData,
    params: MassGapParams | None = None,
    Nt: int | None = None,
) -> MassGapResult:
    """Fit correlator to extract mass gap.

    Args:
        correlator: Correlator data
        params: Fit parameters
        Nt: Temporal extent (for cosh fit)

    Returns:
        MassGapResult with extracted mass
    """
    if params is None:
        params = MassGapParams()

    t = correlator.t
    C = correlator.C
    C_err = correlator.C_err if correlator.C_err is not None else np.ones_like(C) * 0.01 * np.abs(C).max()

    # Determine fit range
    t_min = params.t_min
    t_max = params.t_max if params.t_max is not None else len(t) - 2

    # Select fit range
    mask = (t >= t_min) & (t <= t_max) & (np.abs(C) > 1e-30)
    t_fit = t[mask]
    C_fit = np.abs(C[mask])  # Use absolute value
    C_err_fit = C_err[mask]

    if len(t_fit) < 3:
        # Not enough points
        return MassGapResult(
            mass=0.0,
            mass_err=0.0,
            chi2=np.inf,
            chi2_dof=np.inf,
            fit_range=(t_min, t_max),
            fit_params={},
            plateau_quality=0.0,
        )

    # Perform fit based on method
    try:
        if params.fit_method == "single_exp":
            # Initial guess from effective mass
            m_init = np.mean(np.diff(np.log(np.maximum(C_fit, 1e-30))))
            m_init = max(0.1, min(5.0, abs(m_init)))
            A_init = C_fit[0] * np.exp(m_init * t_fit[0])

            popt, pcov = curve_fit(
                single_exponential,
                t_fit,
                C_fit,
                p0=[A_init, m_init],
                sigma=C_err_fit,
                bounds=([0, 0], [np.inf, 10]),
                maxfev=5000,
            )
            A, m = popt
            perr = np.sqrt(np.diag(pcov))
            m_err = perr[1]
            fit_params = {"A": A, "m": m}
            C_model = single_exponential(t_fit, A, m)

        elif params.fit_method == "double_exp":
            m1_init = 0.5
            m2_init = 1.5
            A1_init = C_fit[0] * 0.8
            A2_init = C_fit[0] * 0.2

            popt, pcov = curve_fit(
                double_exponential,
                t_fit,
                C_fit,
                p0=[A1_init, m1_init, A2_init, m2_init],
                sigma=C_err_fit,
                bounds=([0, 0, 0, 0], [np.inf, 10, np.inf, 10]),
                maxfev=5000,
            )
            A1, m1, A2, m2 = popt
            perr = np.sqrt(np.diag(pcov))
            m = min(m1, m2)  # Ground state is smaller mass
            m_err = perr[1] if m1 < m2 else perr[3]
            fit_params = {"A1": A1, "m1": m1, "A2": A2, "m2": m2}
            C_model = double_exponential(t_fit, A1, m1, A2, m2)

        else:  # cosh
            if Nt is None:
                Nt = int(2 * t.max())
            m_init = 0.5
            A_init = C_fit[0] / 2

            def cosh_fit(t, A, m):
                return cosh_form(t, A, m, Nt)

            popt, pcov = curve_fit(
                cosh_fit,
                t_fit,
                C_fit,
                p0=[A_init, m_init],
                sigma=C_err_fit,
                bounds=([0, 0], [np.inf, 10]),
                maxfev=5000,
            )
            A, m = popt
            perr = np.sqrt(np.diag(pcov))
            m_err = perr[1]
            fit_params = {"A": A, "m": m, "Nt": Nt}
            C_model = cosh_fit(t_fit, A, m)

        # Compute chi-squared
        residuals = (C_fit - C_model) / C_err_fit
        chi2 = np.sum(residuals ** 2)
        dof = len(t_fit) - len(popt)
        chi2_dof = chi2 / dof if dof > 0 else chi2

        # Plateau quality: stability of effective mass
        _, m_eff, _ = effective_mass(correlator)
        m_eff_range = m_eff[t_min:t_max]
        plateau_quality = 1.0 / (1.0 + np.std(m_eff_range) / np.mean(np.abs(m_eff_range) + 1e-10))

    except (RuntimeError, ValueError):
        # Fit failed
        m = 0.0
        m_err = 0.0
        chi2 = np.inf
        chi2_dof = np.inf
        fit_params = {}
        plateau_quality = 0.0

    return MassGapResult(
        mass=m,
        mass_err=m_err,
        chi2=chi2,
        chi2_dof=chi2_dof,
        fit_range=(t_min, t_max),
        fit_params=fit_params,
        plateau_quality=plateau_quality,
    )


def extract_mass_gap(
    correlator: CorrelatorData,
    params: MassGapParams | None = None,
) -> MassGapResult:
    """Extract mass gap with automatic fit range optimization.

    Tries multiple fit ranges and selects the best by chi2/dof.

    Args:
        correlator: Correlator data
        params: Fit parameters

    Returns:
        Best MassGapResult
    """
    if params is None:
        params = MassGapParams()

    t = correlator.t
    t_max_data = len(t) - 2

    # Try different fit ranges
    best_result = None
    best_quality = -np.inf

    for t_min in range(1, min(5, t_max_data)):
        for t_max in range(t_min + 3, t_max_data + 1):
            test_params = MassGapParams(
                t_min=t_min,
                t_max=t_max,
                fit_method=params.fit_method,
            )
            result = fit_exponential_decay(correlator, test_params)

            # Quality metric: good chi2/dof + plateau quality
            if result.chi2_dof < np.inf:
                quality = result.plateau_quality - 0.1 * max(0, result.chi2_dof - 1)
                if quality > best_quality:
                    best_quality = quality
                    best_result = result

    if best_result is None:
        # Return default failed result
        return fit_exponential_decay(correlator, params)

    return best_result


class MassGapExtractor:
    """Class for mass gap extraction with various methods."""

    def __init__(self, params: MassGapParams | None = None):
        """Initialize extractor.

        Args:
            params: Fit parameters
        """
        self.params = params or MassGapParams()

    def extract(self, correlator: CorrelatorData) -> MassGapResult:
        """Extract mass gap from correlator.

        Args:
            correlator: Correlator data

        Returns:
            MassGapResult
        """
        return extract_mass_gap(correlator, self.params)

    def effective_mass_analysis(
        self,
        correlator: CorrelatorData,
    ) -> dict:
        """Analyze effective mass for plateau.

        Args:
            correlator: Correlator data

        Returns:
            Dictionary with effective mass analysis
        """
        t, m_eff, m_err = effective_mass(correlator)

        # Find plateau region
        # Look for region where m_eff is approximately constant
        window = 3
        stability = np.zeros(len(m_eff) - window)
        for i in range(len(stability)):
            stability[i] = np.std(m_eff[i:i + window])

        # Best plateau is where stability is minimum
        best_start = np.argmin(stability) if len(stability) > 0 else 0
        plateau_mass = np.mean(m_eff[best_start:best_start + window]) if best_start + window <= len(m_eff) else np.nan

        return {
            "t": t,
            "m_eff": m_eff,
            "m_err": m_err,
            "plateau_start": best_start,
            "plateau_mass": plateau_mass,
            "plateau_stability": stability,
        }

    def bootstrap_error(
        self,
        configs: list,
        correlator_func,
        n_bootstrap: int | None = None,
    ) -> tuple[float, float]:
        """Estimate mass gap error via bootstrap.

        Args:
            configs: List of gauge configurations
            correlator_func: Function to compute correlator from configs
            n_bootstrap: Number of bootstrap samples

        Returns:
            Tuple of (mass, mass_error)
        """
        if n_bootstrap is None:
            n_bootstrap = self.params.n_bootstrap

        n_configs = len(configs)
        masses = []

        for _ in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.randint(0, n_configs, n_configs)
            sample_configs = [configs[i] for i in indices]

            # Compute correlator and extract mass
            corr = correlator_func(sample_configs)
            result = self.extract(corr)
            if result.mass > 0:
                masses.append(result.mass)

        if len(masses) < 2:
            return 0.0, 0.0

        return np.mean(masses), np.std(masses)
