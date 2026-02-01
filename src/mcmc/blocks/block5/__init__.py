"""Block 5: Qubit Tensorial MCMC.

MCMC Ontology: This block implements quantum information aspects of MCMC:
1. Tensorial states as qudit superpositions
2. S-dependent decoherence and collapse
3. Quantum simulation of MCMC dynamics

The entropic coordinate S can be interpreted as measuring the
degree of quantum coherence vs classical determinacy.

Modules:
    - config: Configuration dataclasses
    - hilbert_space: Qudit Hilbert space construction
    - operators: Tensorial operators (creation, annihilation, S-measurement)
    - hamiltonian: MCMC Hamiltonian H(S)
    - gates: Quantum gates for MCMC simulation
    - simulation: Qudit simulator for MCMC dynamics
"""
from __future__ import annotations

from mcmc.blocks.block5.config import (
    HilbertSpaceParams,
    OperatorParams,
    HamiltonianParams,
    SimulationParams,
)
from mcmc.blocks.block5.hilbert_space import (
    QuditState,
    QuditBasis,
    tensor_product,
    create_vacuum,
    create_superposition,
)
from mcmc.blocks.block5.operators import (
    TensorialOperator,
    creation_operator,
    annihilation_operator,
    number_operator,
    S_operator,
    collapse_operator,
)
from mcmc.blocks.block5.hamiltonian import (
    MCMCHamiltonian,
    kinetic_term,
    potential_term,
    interaction_term,
    time_evolution,
)

__all__ = [
    # Config
    "HilbertSpaceParams",
    "OperatorParams",
    "HamiltonianParams",
    "SimulationParams",
    # Hilbert space
    "QuditState",
    "QuditBasis",
    "tensor_product",
    "create_vacuum",
    "create_superposition",
    # Operators
    "TensorialOperator",
    "creation_operator",
    "annihilation_operator",
    "number_operator",
    "S_operator",
    "collapse_operator",
    # Hamiltonian
    "MCMCHamiltonian",
    "kinetic_term",
    "potential_term",
    "interaction_term",
    "time_evolution",
]
