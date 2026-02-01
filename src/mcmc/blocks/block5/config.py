"""Configuration dataclasses for Block 5: Qubit Tensorial MCMC.

MCMC Ontology: Configuration for quantum simulation of MCMC dynamics.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from mcmc.core.ontology import S_0, S_GEOM


@dataclass(frozen=True)
class HilbertSpaceParams:
    """Parameters for qudit Hilbert space.

    Attributes:
        d: Local dimension (number of levels per site)
        n_sites: Number of sites in the system
        S_min: Minimum entropic coordinate (maps to ground state)
        S_max: Maximum entropic coordinate (maps to highest level)
        truncation: Maximum occupation number (for bosonic modes)
    """
    d: int = 10                 # d=10 for S in [0, 100] with dS=10
    n_sites: int = 1           # Single site by default
    S_min: float = 0.0         # Maps to |0>
    S_max: float = 100.0       # Maps to |d-1>
    truncation: int = 20       # For Fock space truncation


@dataclass(frozen=True)
class OperatorParams:
    """Parameters for tensorial operators.

    Attributes:
        S_current: Current entropic coordinate
        coupling_strength: Operator coupling constant
        decoherence_rate: Rate of S-induced decoherence
        use_lindblad: Use Lindblad master equation
    """
    S_current: float = S_0
    coupling_strength: float = 1.0
    decoherence_rate: float = 0.01
    use_lindblad: bool = True


@dataclass(frozen=True)
class HamiltonianParams:
    """Parameters for MCMC Hamiltonian.

    The MCMC Hamiltonian has the form:
        H = H_kinetic + H_potential + H_interaction
        H = sum_i omega_i * n_i + V(S) + g * sum_<ij> (a_i^dag a_j + h.c.)

    Attributes:
        omega: Base frequency (energy scale)
        g: Hopping/interaction strength
        V_amplitude: Potential amplitude
        S_transition: S value for potential transition
        include_decoherence: Add decoherence terms
        decoherence_gamma: Decoherence rate
    """
    omega: float = 1.0         # Base frequency
    g: float = 0.1             # Interaction strength
    V_amplitude: float = 0.5   # Potential strength
    S_transition: float = S_GEOM  # Transition S value
    include_decoherence: bool = True
    decoherence_gamma: float = 0.01


@dataclass(frozen=True)
class GateParams:
    """Parameters for quantum gates.

    Attributes:
        gate_time: Duration of gate operation
        error_rate: Gate error probability
        n_trotter: Trotter steps for time evolution
    """
    gate_time: float = 1.0
    error_rate: float = 0.001
    n_trotter: int = 10


@dataclass(frozen=True)
class SimulationParams:
    """Parameters for qudit simulation.

    Attributes:
        hilbert: Hilbert space parameters
        hamiltonian: Hamiltonian parameters
        t_max: Maximum simulation time
        dt: Time step
        n_trajectories: Number of quantum trajectories (for jump method)
        measure_interval: Measurement interval
        observable: Observable to measure
        use_sparse: Use sparse matrices
    """
    hilbert: HilbertSpaceParams = field(default_factory=HilbertSpaceParams)
    hamiltonian: HamiltonianParams = field(default_factory=HamiltonianParams)
    t_max: float = 10.0
    dt: float = 0.1
    n_trajectories: int = 100
    measure_interval: float = 0.5
    observable: Literal["S", "n", "coherence"] = "S"
    use_sparse: bool = True
