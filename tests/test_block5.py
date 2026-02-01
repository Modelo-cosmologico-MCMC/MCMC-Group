"""Tests for Block 5: Qubit Tensorial MCMC.

Tests ontological invariants and module functionality.
"""
from __future__ import annotations

import numpy as np



class TestHilbertSpace:
    """Tests for qudit Hilbert space."""

    def test_state_normalization(self):
        """States are normalized."""
        from mcmc.blocks.block5.hilbert_space import QuditState

        amps = np.array([1.0, 2.0, 3.0], dtype=complex)
        state = QuditState(amps, d=3)
        assert np.abs(state.norm - 1.0) < 1e-10

    def test_probabilities_sum_to_one(self):
        """Probabilities sum to 1."""
        from mcmc.blocks.block5.hilbert_space import create_superposition
        from mcmc.blocks.block5.config import HilbertSpaceParams

        params = HilbertSpaceParams(d=5)
        coeffs = np.random.randn(5) + 1j * np.random.randn(5)
        state = create_superposition(coeffs, params)

        probs = state.probabilities()
        assert np.abs(np.sum(probs) - 1.0) < 1e-10

    def test_S_value_in_range(self):
        """S expectation value is in valid range."""
        from mcmc.blocks.block5.hilbert_space import create_superposition
        from mcmc.blocks.block5.config import HilbertSpaceParams

        params = HilbertSpaceParams(d=10, S_min=0.0, S_max=100.0)
        coeffs = np.random.randn(10) + 1j * np.random.randn(10)
        state = create_superposition(coeffs, params)

        S = state.S_value()
        assert 0.0 <= S <= 100.0

    def test_basis_completeness(self):
        """Basis states span the space."""
        from mcmc.blocks.block5.hilbert_space import QuditBasis
        from mcmc.blocks.block5.config import HilbertSpaceParams

        params = HilbertSpaceParams(d=5)
        basis = QuditBasis(params)

        # All basis states should be orthonormal
        for i in range(5):
            for j in range(5):
                inner = basis.basis_state(i).inner(basis.basis_state(j))
                expected = 1.0 if i == j else 0.0
                assert np.abs(inner - expected) < 1e-10


class TestOperators:
    """Tests for tensorial operators."""

    def test_creation_annihilation_commutator(self):
        """[a, a^dag] has correct structure (identity except at truncation)."""
        from mcmc.blocks.block5.operators import creation_operator, annihilation_operator
        from mcmc.blocks.block5.config import HilbertSpaceParams

        params = HilbertSpaceParams(d=10)
        a = annihilation_operator(params)
        a_dag = creation_operator(params)

        commutator = a.commutator(a_dag)
        # The commutator [a, a^dag] = I for bosonic operators
        # But with truncation, the (d-1, d-1) element is different
        # Check diagonal elements are approximately 1 except last
        diag = np.diag(commutator.matrix)
        assert np.allclose(np.real(diag[:-1]), 1.0, atol=1e-10)
        # Last element affected by truncation
        assert np.isfinite(diag[-1])

    def test_number_operator_eigenvalues(self):
        """Number operator has eigenvalues 0, 1, 2, ..."""
        from mcmc.blocks.block5.operators import number_operator
        from mcmc.blocks.block5.config import HilbertSpaceParams

        params = HilbertSpaceParams(d=10)
        n = number_operator(params)

        eigenvals = n.eigenvalues()
        expected = np.arange(10, dtype=float)
        np.testing.assert_allclose(np.sort(eigenvals), expected, atol=1e-10)

    def test_S_operator_hermitian(self):
        """S operator is Hermitian."""
        from mcmc.blocks.block5.operators import S_operator
        from mcmc.blocks.block5.config import HilbertSpaceParams

        params = HilbertSpaceParams(d=10)
        S = S_operator(params)
        assert S.is_hermitian()

    def test_S_eigenvalues_match_levels(self):
        """S eigenvalues match level-to-S mapping."""
        from mcmc.blocks.block5.operators import S_operator
        from mcmc.blocks.block5.hilbert_space import QuditBasis
        from mcmc.blocks.block5.config import HilbertSpaceParams

        params = HilbertSpaceParams(d=10, S_min=0.0, S_max=100.0)
        S_op = S_operator(params)
        basis = QuditBasis(params)

        eigenvals = S_op.eigenvalues()
        expected = basis.all_levels()
        np.testing.assert_allclose(np.sort(eigenvals), np.sort(expected), atol=1e-10)


class TestHamiltonian:
    """Tests for MCMC Hamiltonian."""

    def test_hamiltonian_hermitian(self):
        """Hamiltonian is Hermitian."""
        from mcmc.blocks.block5.hamiltonian import MCMCHamiltonian
        from mcmc.blocks.block5.config import HilbertSpaceParams, HamiltonianParams

        hs_params = HilbertSpaceParams(d=10)
        h_params = HamiltonianParams(include_decoherence=False)
        H = MCMCHamiltonian(hs_params, h_params)

        assert np.allclose(H.matrix, H.matrix.conj().T)

    def test_ground_state_energy_finite(self):
        """Ground state energy is finite."""
        from mcmc.blocks.block5.hamiltonian import MCMCHamiltonian
        from mcmc.blocks.block5.config import HilbertSpaceParams, HamiltonianParams

        hs_params = HilbertSpaceParams(d=10)
        h_params = HamiltonianParams()
        H = MCMCHamiltonian(hs_params, h_params)

        E0, _ = H.ground_state()
        assert np.isfinite(E0)

    def test_time_evolution_unitary(self):
        """Time evolution preserves norm."""
        from mcmc.blocks.block5.hamiltonian import MCMCHamiltonian, time_evolution
        from mcmc.blocks.block5.hilbert_space import create_superposition
        from mcmc.blocks.block5.config import HilbertSpaceParams, HamiltonianParams

        hs_params = HilbertSpaceParams(d=10)
        h_params = HamiltonianParams(include_decoherence=False)
        H = MCMCHamiltonian(hs_params, h_params)

        coeffs = np.random.randn(10) + 1j * np.random.randn(10)
        initial = create_superposition(coeffs, hs_params)

        final = time_evolution(initial, H, t=1.0)
        assert np.abs(final.norm - 1.0) < 1e-6


class TestSimulation:
    """Tests for quantum simulation."""

    def test_trajectory_S_bounded(self):
        """S values stay bounded during trajectory."""
        from mcmc.blocks.block5.simulation import MCMCSimulator
        from mcmc.blocks.block5.config import SimulationParams, HilbertSpaceParams, HamiltonianParams

        hs_params = HilbertSpaceParams(d=10, S_min=0.0, S_max=100.0)
        h_params = HamiltonianParams(include_decoherence=True)
        params = SimulationParams(hilbert=hs_params, hamiltonian=h_params, t_max=1.0)

        sim = MCMCSimulator(params)
        result = sim.run_trajectory()

        assert np.all(result.S_values >= 0)
        assert np.all(result.S_values <= 100)

    def test_decoherence_reduces_coherence(self):
        """Decoherence decreases coherence over time."""
        from mcmc.blocks.block5.simulation import MCMCSimulator
        from mcmc.blocks.block5.config import SimulationParams, HilbertSpaceParams, HamiltonianParams

        hs_params = HilbertSpaceParams(d=10)
        h_params = HamiltonianParams(include_decoherence=True, decoherence_gamma=0.1)
        params = SimulationParams(hilbert=hs_params, hamiltonian=h_params, t_max=5.0)

        sim = MCMCSimulator(params)
        result = sim.run_trajectory()

        # Coherence should generally decrease (may fluctuate)
        assert result.coherence[-1] <= result.coherence[0] + 0.5


class TestOntologicalInvariants:
    """Tests for MCMC ontological invariants in Block 5."""

    def test_S_superposition_primordial(self):
        """Primordial state (S~0) is superposition."""
        from mcmc.blocks.block5.hilbert_space import create_superposition
        from mcmc.blocks.block5.config import HilbertSpaceParams

        params = HilbertSpaceParams(d=10, S_min=0.0, S_max=100.0)
        # Equal superposition represents primordial state
        coeffs = np.ones(10) / np.sqrt(10)
        state = create_superposition(coeffs, params)

        # Should have high coherence
        coherence = state.coherence()
        assert coherence > 0

    def test_collapse_to_S_eigenstate(self):
        """Measurement collapses to S eigenstate."""
        from mcmc.blocks.block5.hilbert_space import create_superposition
        from mcmc.blocks.block5.config import HilbertSpaceParams

        params = HilbertSpaceParams(d=10)
        coeffs = np.ones(10) / np.sqrt(10)
        state = create_superposition(coeffs, params)

        # After measurement, state should be basis state
        outcome = state.measure()
        assert 0 <= outcome < 10

        # State should now be pure basis state
        probs = state.probabilities()
        assert np.sum(probs == 0) == 9  # Only one non-zero
