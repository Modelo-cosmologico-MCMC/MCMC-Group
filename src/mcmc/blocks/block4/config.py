"""Configuration dataclasses for Block 4: Lattice-Gauge Simulations.

MCMC Ontology: These configurations support lattice QCD/QFT simulations
with S-dependent coupling beta(S).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from mcmc.core.ontology import S_0


@dataclass(frozen=True)
class LatticeParams:
    """Parameters for the lattice geometry.

    Attributes:
        Nx, Ny, Nz: Spatial lattice dimensions
        Nt: Temporal lattice dimension
        a: Lattice spacing [fm or dimensionless]
        gauge_group: SU(N) gauge group
        periodic_bc: Use periodic boundary conditions
    """
    Nx: int = 16
    Ny: int = 16
    Nz: int = 16
    Nt: int = 32
    a: float = 0.1  # fm
    gauge_group: Literal["SU2", "SU3"] = "SU3"
    periodic_bc: bool = True

    @property
    def volume(self) -> int:
        """Total lattice volume."""
        return self.Nx * self.Ny * self.Nz * self.Nt

    @property
    def spatial_volume(self) -> int:
        """Spatial lattice volume."""
        return self.Nx * self.Ny * self.Nz


@dataclass(frozen=True)
class WilsonParams:
    """Parameters for Wilson gauge action.

    The Wilson action with MCMC coupling:
        S_W = beta(S) * sum_P [1 - Re Tr(U_P) / N_c]

    where beta(S) depends on the entropic coordinate.

    Attributes:
        beta_0: Base coupling at S_0
        S_current: Current entropic coordinate
        use_mcmc_beta: Whether to use S-dependent beta
        beta_pre_geom: Coupling in pre-geometric regime
        improved: Use Symanzik improvement
        c_1: Improvement coefficient (rectangle)
    """
    beta_0: float = 6.0  # Typical QCD value
    S_current: float = S_0
    use_mcmc_beta: bool = True
    beta_pre_geom: float = 0.1  # Strong coupling in pre-geom
    improved: bool = False
    c_1: float = -0.083  # 1-loop Symanzik coefficient


@dataclass(frozen=True)
class MonteCarloParams:
    """Parameters for Monte Carlo sampling.

    Attributes:
        n_thermalize: Number of thermalization sweeps
        n_sweeps: Number of measurement sweeps
        n_skip: Sweeps between measurements
        algorithm: Sampling algorithm
        epsilon: HMC step size
        n_steps: HMC trajectory length
        seed: Random seed
    """
    n_thermalize: int = 1000
    n_sweeps: int = 5000
    n_skip: int = 10
    algorithm: Literal["metropolis", "heatbath", "hmc"] = "heatbath"
    epsilon: float = 0.01  # HMC step size
    n_steps: int = 20      # HMC steps per trajectory
    seed: int = 42


@dataclass(frozen=True)
class MassGapParams:
    """Parameters for mass gap extraction.

    Attributes:
        t_min: Minimum time slice for fit
        t_max: Maximum time slice for fit (None = Nt/2)
        n_bootstrap: Number of bootstrap samples
        fit_method: Fitting method
        correlator_type: Type of correlator to use
    """
    t_min: int = 2
    t_max: int | None = None
    n_bootstrap: int = 100
    fit_method: Literal["single_exp", "double_exp", "cosh"] = "single_exp"
    correlator_type: Literal["gluon", "polyakov", "meson"] = "gluon"


@dataclass(frozen=True)
class SScanParams:
    """Parameters for S-dependent scans.

    Attributes:
        S_min: Minimum entropic coordinate
        S_max: Maximum entropic coordinate
        n_S_points: Number of S values to scan
        log_scale: Use logarithmic spacing in S
        detect_transitions: Whether to detect phase transitions
        transition_threshold: Chi-squared threshold for transition
    """
    S_min: float = 0.001  # Deep pre-geometric
    S_max: float = S_0    # Present
    n_S_points: int = 20
    log_scale: bool = True
    detect_transitions: bool = True
    transition_threshold: float = 10.0
