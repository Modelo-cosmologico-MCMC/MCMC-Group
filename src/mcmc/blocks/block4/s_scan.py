"""S-dependent scanning of mass gap and phase transitions.

MCMC Ontology: Scans the mass gap as a function of S to identify:
1. Pre-geometric confinement phase (S < S_GEOM)
2. Deconfinement transition at S ~ S_GEOM (Big Bang)
3. Geometric phase (S > S_GEOM)
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from mcmc.core.ontology import THRESHOLDS
from mcmc.blocks.block4.config import (
    LatticeParams,
    WilsonParams,
    MonteCarloParams,
    MassGapParams,
    SScanParams,
)
from mcmc.blocks.block4.wilson_action import WilsonAction, LatticeConfiguration
from mcmc.blocks.block4.monte_carlo import HeatBathSampler
from mcmc.blocks.block4.correlators import gluon_correlator, polyakov_loop
from mcmc.blocks.block4.mass_gap import MassGapExtractor


@dataclass
class SScanResult:
    """Result of S-scan analysis.

    Attributes:
        S_values: Array of S values scanned
        beta_values: Corresponding beta(S) values
        mass_gaps: Extracted mass gaps
        mass_errors: Mass gap errors
        plaquettes: Average plaquettes
        polyakov: Average Polyakov loop modulus
        chi2_values: Fit chi-squared values
        transitions: Detected transition S values
    """
    S_values: np.ndarray
    beta_values: np.ndarray
    mass_gaps: np.ndarray
    mass_errors: np.ndarray
    plaquettes: np.ndarray
    polyakov: np.ndarray
    chi2_values: np.ndarray
    transitions: list[float]


def scan_mass_gap_vs_S(
    lattice: LatticeParams,
    mc_params: MonteCarloParams | None = None,
    mg_params: MassGapParams | None = None,
    scan_params: SScanParams | None = None,
) -> SScanResult:
    """Scan mass gap as a function of S.

    For each S value:
    1. Set beta(S)
    2. Thermalize configuration
    3. Collect measurements
    4. Extract mass gap

    Args:
        lattice: Lattice parameters
        mc_params: Monte Carlo parameters
        mg_params: Mass gap parameters
        scan_params: S-scan parameters

    Returns:
        SScanResult with scan data
    """
    if mc_params is None:
        mc_params = MonteCarloParams(n_thermalize=500, n_sweeps=200, n_skip=5)
    if mg_params is None:
        mg_params = MassGapParams()
    if scan_params is None:
        scan_params = SScanParams()

    # Generate S values
    if scan_params.log_scale:
        S_values = np.logspace(
            np.log10(scan_params.S_min),
            np.log10(scan_params.S_max),
            scan_params.n_S_points,
        )
    else:
        S_values = np.linspace(
            scan_params.S_min,
            scan_params.S_max,
            scan_params.n_S_points,
        )

    # Initialize arrays
    n_S = len(S_values)
    beta_values = np.zeros(n_S)
    mass_gaps = np.zeros(n_S)
    mass_errors = np.zeros(n_S)
    plaquettes = np.zeros(n_S)
    polyakov_values = np.zeros(n_S)
    chi2_values = np.zeros(n_S)

    # Mass gap extractor
    extractor = MassGapExtractor(mg_params)

    # Scan over S values
    for i, S in enumerate(S_values):
        # Create Wilson action with this S
        wilson = WilsonParams(S_current=S, use_mcmc_beta=True)
        action = WilsonAction(lattice, wilson)
        beta_values[i] = action.beta()

        # Initialize and thermalize
        config = LatticeConfiguration(lattice, cold_start=True)
        sampler = HeatBathSampler(action, mc_params)

        # Thermalize
        for _ in range(mc_params.n_thermalize):
            sampler.sweep(config)

        # Collect configurations
        configs = []
        plaq_sum = 0.0
        poly_sum = 0.0

        for _ in range(mc_params.n_sweeps):
            for _ in range(mc_params.n_skip):
                sampler.sweep(config)

            # Measurements
            plaq_sum += action.average_plaquette(config)
            L = polyakov_loop(config)
            poly_sum += np.mean(np.abs(L))

            # Store configuration (copy)
            configs.append(LatticeConfiguration(lattice, cold_start=True))
            configs[-1].U = config.U.copy()

        plaquettes[i] = plaq_sum / mc_params.n_sweeps
        polyakov_values[i] = poly_sum / mc_params.n_sweeps

        # Extract mass gap
        if len(configs) >= 10:
            corr = gluon_correlator(configs)
            result = extractor.extract(corr)
            mass_gaps[i] = result.mass
            mass_errors[i] = result.mass_err
            chi2_values[i] = result.chi2_dof
        else:
            mass_gaps[i] = 0.0
            mass_errors[i] = 0.0
            chi2_values[i] = np.inf

    # Detect transitions
    transitions = []
    if scan_params.detect_transitions:
        transitions = phase_transition_finder(
            S_values,
            mass_gaps,
            polyakov_values,
            scan_params.transition_threshold,
        )

    return SScanResult(
        S_values=S_values,
        beta_values=beta_values,
        mass_gaps=mass_gaps,
        mass_errors=mass_errors,
        plaquettes=plaquettes,
        polyakov=polyakov_values,
        chi2_values=chi2_values,
        transitions=transitions,
    )


def phase_transition_finder(
    S_values: np.ndarray,
    mass_gaps: np.ndarray,
    polyakov: np.ndarray,
    threshold: float = 10.0,
) -> list[float]:
    """Find phase transitions from scan data.

    Looks for:
    1. Rapid changes in mass gap
    2. Polyakov loop order parameter changes
    3. Known ontological thresholds (S_GEOM, S_PRE_*)

    Args:
        S_values: Array of S values
        mass_gaps: Corresponding mass gaps
        polyakov: Polyakov loop values
        threshold: Chi-squared threshold for transition

    Returns:
        List of transition S values
    """
    transitions = []

    # Look for sharp changes in mass gap
    if len(mass_gaps) > 2:
        dm_dS = np.abs(np.diff(mass_gaps) / np.diff(S_values))
        dm_mean = np.mean(dm_dS)
        dm_std = np.std(dm_dS)

        for i, dm in enumerate(dm_dS):
            if dm > dm_mean + 2 * dm_std:
                S_trans = (S_values[i] + S_values[i + 1]) / 2
                transitions.append(S_trans)

    # Look for Polyakov loop transitions (confinement/deconfinement)
    if len(polyakov) > 2:
        dL_dS = np.abs(np.diff(polyakov) / np.diff(S_values))
        dL_mean = np.mean(dL_dS)
        dL_std = np.std(dL_dS)

        for i, dL in enumerate(dL_dS):
            if dL > dL_mean + 2 * dL_std:
                S_trans = (S_values[i] + S_values[i + 1]) / 2
                if S_trans not in transitions:
                    transitions.append(S_trans)

    # Check for proximity to known ontological thresholds
    known_thresholds = [
        THRESHOLDS.S_PRE_0,  # 0.001
        THRESHOLDS.S_PRE_1,  # 0.01
        THRESHOLDS.S_PRE_2,  # 0.1
        THRESHOLDS.S_PRE_3,  # 0.5
        THRESHOLDS.S_GEOM,   # 1.001 (Big Bang)
    ]

    for S_th in known_thresholds:
        if S_values.min() < S_th < S_values.max():
            # Find nearest transition
            closest = min(transitions, key=lambda x: abs(x - S_th), default=None)
            if closest is None or abs(closest - S_th) > 0.1 * S_th:
                transitions.append(S_th)

    return sorted(set(transitions))


class SScanAnalyzer:
    """Analyzer for S-dependent lattice simulations."""

    def __init__(
        self,
        lattice: LatticeParams | None = None,
        mc_params: MonteCarloParams | None = None,
        mg_params: MassGapParams | None = None,
        scan_params: SScanParams | None = None,
    ):
        """Initialize analyzer.

        Args:
            lattice: Lattice parameters
            mc_params: Monte Carlo parameters
            mg_params: Mass gap parameters
            scan_params: S-scan parameters
        """
        self.lattice = lattice or LatticeParams(Nx=8, Ny=8, Nz=8, Nt=16)
        self.mc_params = mc_params or MonteCarloParams()
        self.mg_params = mg_params or MassGapParams()
        self.scan_params = scan_params or SScanParams()

    def run_scan(self) -> SScanResult:
        """Run full S-scan.

        Returns:
            SScanResult with all data
        """
        return scan_mass_gap_vs_S(
            self.lattice,
            self.mc_params,
            self.mg_params,
            self.scan_params,
        )

    def quick_scan(self, n_points: int = 10) -> SScanResult:
        """Run quick scan with fewer points.

        Args:
            n_points: Number of S values

        Returns:
            SScanResult
        """
        quick_scan = SScanParams(
            S_min=self.scan_params.S_min,
            S_max=self.scan_params.S_max,
            n_S_points=n_points,
            log_scale=True,
        )
        quick_mc = MonteCarloParams(
            n_thermalize=100,
            n_sweeps=50,
            n_skip=2,
        )
        return scan_mass_gap_vs_S(
            self.lattice,
            quick_mc,
            self.mg_params,
            quick_scan,
        )

    def analyze_transition(
        self,
        S_trans: float,
        n_points: int = 10,
        delta_S: float = 0.1,
    ) -> dict:
        """Detailed analysis around a transition.

        Args:
            S_trans: Approximate transition S value
            n_points: Number of points around transition
            delta_S: Half-width of scan region

        Returns:
            Dictionary with transition analysis
        """
        # Scan around transition
        S_min = S_trans * (1 - delta_S)
        S_max = S_trans * (1 + delta_S)

        local_scan = SScanParams(
            S_min=S_min,
            S_max=S_max,
            n_S_points=n_points,
            log_scale=False,
        )
        result = scan_mass_gap_vs_S(
            self.lattice,
            self.mc_params,
            self.mg_params,
            local_scan,
        )

        # Find transition location more precisely
        # Use derivative of mass gap
        dm = np.gradient(result.mass_gaps, result.S_values)
        i_trans = np.argmax(np.abs(dm))
        S_precise = result.S_values[i_trans]

        # Estimate critical exponents (simplified)
        # Mass gap ~ |S - S_c|^nu
        below = result.S_values < S_precise
        above = result.S_values >= S_precise

        if np.sum(below) > 2 and np.sum(above) > 2:
            # Fit power law below transition
            log_dS_below = np.log(np.abs(result.S_values[below] - S_precise) + 1e-10)
            log_m_below = np.log(np.abs(result.mass_gaps[below]) + 1e-10)
            if len(log_dS_below) > 1:
                nu_below = np.polyfit(log_dS_below, log_m_below, 1)[0]
            else:
                nu_below = np.nan
        else:
            nu_below = np.nan

        return {
            "S_transition": S_precise,
            "S_values": result.S_values,
            "mass_gaps": result.mass_gaps,
            "polyakov": result.polyakov,
            "nu_exponent": nu_below,
            "order": "first" if np.max(np.abs(dm)) > 10 * np.mean(np.abs(dm)) else "crossover",
        }

    def compare_ontology(self, result: SScanResult) -> dict:
        """Compare scan results with MCMC ontological predictions.

        Args:
            result: S-scan result

        Returns:
            Dictionary with comparison
        """
        # Check if transitions align with ontological thresholds
        ontology_thresholds = [
            ("S_PRE_0 (Planck)", THRESHOLDS.S_PRE_0),
            ("S_PRE_1 (GUT)", THRESHOLDS.S_PRE_1),
            ("S_PRE_2", THRESHOLDS.S_PRE_2),
            ("S_PRE_3 (EW)", THRESHOLDS.S_PRE_3),
            ("S_GEOM (Big Bang)", THRESHOLDS.S_GEOM),
        ]

        matches = []
        for name, S_th in ontology_thresholds:
            # Check if scan covers this threshold
            if result.S_values.min() <= S_th <= result.S_values.max():
                # Find nearest detected transition
                for S_trans in result.transitions:
                    if abs(S_trans - S_th) < 0.2 * S_th:
                        matches.append({
                            "threshold": name,
                            "S_expected": S_th,
                            "S_detected": S_trans,
                            "deviation": abs(S_trans - S_th) / S_th,
                        })
                        break

        # Phase classification
        phases = []
        for i, S in enumerate(result.S_values):
            if S < THRESHOLDS.S_GEOM:
                phase = "pre-geometric"
            else:
                phase = "geometric"
            phases.append(phase)

        return {
            "n_transitions_detected": len(result.transitions),
            "n_ontology_matches": len(matches),
            "matches": matches,
            "phases": phases,
            "confinement_S": result.S_values[np.argmax(result.mass_gaps)] if len(result.mass_gaps) > 0 else None,
        }
