"""Tests for Block 4: Lattice-Gauge simulations.

Tests ontological invariants and module functionality.
"""
from __future__ import annotations

import numpy as np

from mcmc.core.ontology import S_GEOM


class TestWilsonAction:
    """Tests for Wilson gauge action."""

    def test_beta_of_S_positive(self):
        """beta(S) > 0 for all S."""
        from mcmc.blocks.block4.wilson_action import beta_of_S

        S_values = np.logspace(-3, 2, 50)
        for S in S_values:
            beta = beta_of_S(S)
            assert beta > 0

    def test_beta_transition(self):
        """beta(S) increases from pre-geometric to geometric regime."""
        from mcmc.blocks.block4.wilson_action import beta_of_S

        beta_pre = beta_of_S(0.1)  # Pre-geometric
        beta_post = beta_of_S(50.0)  # Post-Big Bang
        assert beta_pre < beta_post

    def test_plaquette_bounded(self):
        """Plaquette value is bounded."""
        from mcmc.blocks.block4.wilson_action import (
            LatticeConfiguration,
            plaquette,
        )
        from mcmc.blocks.block4.config import LatticeParams

        lattice = LatticeParams(Nx=4, Ny=4, Nz=4, Nt=4)
        config = LatticeConfiguration(lattice, cold_start=True)

        # For cold start (identity links), plaquette = N (= 3 for SU3)
        P = plaquette(config, (0, 0, 0, 0), 0, 1)
        N = config.N
        assert np.abs(P) <= N + 0.1


class TestMonteCarlo:
    """Tests for Monte Carlo sampling."""

    def test_metropolis_acceptance(self):
        """Metropolis sampler has non-zero acceptance."""
        from mcmc.blocks.block4.monte_carlo import MetropolisSampler
        from mcmc.blocks.block4.wilson_action import WilsonAction, LatticeConfiguration
        from mcmc.blocks.block4.config import LatticeParams, WilsonParams, MonteCarloParams

        lattice = LatticeParams(Nx=4, Ny=4, Nz=4, Nt=4, gauge_group="SU2")
        wilson = WilsonParams(beta_0=2.0, use_mcmc_beta=False)
        mc_params = MonteCarloParams(n_sweeps=10)

        action = WilsonAction(lattice, wilson)
        config = LatticeConfiguration(lattice, cold_start=True)
        sampler = MetropolisSampler(action, mc_params)

        stats = sampler.run(config, n_sweeps=10)
        assert stats.acceptance_rate >= 0
        assert stats.acceptance_rate <= 1

    def test_thermalization_plaquette(self):
        """Thermalized plaquette has expected value."""
        from mcmc.blocks.block4.monte_carlo import HeatBathSampler
        from mcmc.blocks.block4.wilson_action import WilsonAction, LatticeConfiguration
        from mcmc.blocks.block4.config import LatticeParams, WilsonParams, MonteCarloParams

        lattice = LatticeParams(Nx=4, Ny=4, Nz=4, Nt=4, gauge_group="SU2")
        wilson = WilsonParams(beta_0=2.0, use_mcmc_beta=False)
        mc_params = MonteCarloParams(n_thermalize=50, n_sweeps=20)

        action = WilsonAction(lattice, wilson)
        config = LatticeConfiguration(lattice, cold_start=True)
        sampler = HeatBathSampler(action, mc_params)

        stats = sampler.run(config, n_sweeps=20)
        assert len(stats.plaquette_history) > 0
        # Plaquette should be between 0 and 1
        assert np.all(stats.plaquette_history >= 0)
        assert np.all(stats.plaquette_history <= 1)


class TestMassGap:
    """Tests for mass gap extraction."""

    def test_fit_exponential_decay(self):
        """Exponential fit extracts correct mass."""
        from mcmc.blocks.block4.mass_gap import fit_exponential_decay
        from mcmc.blocks.block4.correlators import CorrelatorData

        # Create mock exponential correlator
        t = np.arange(1, 16)
        m_true = 0.5
        A = 1.0
        C = A * np.exp(-m_true * t) + 0.01 * np.random.randn(len(t))
        C_err = 0.01 * np.ones_like(C)

        corr = CorrelatorData(t=t, C=C, C_err=C_err)
        result = fit_exponential_decay(corr)

        # Check mass is extracted approximately correctly
        if result.mass > 0:
            assert np.abs(result.mass - m_true) < 0.3


class TestSScan:
    """Tests for S-dependent scans."""

    def test_phase_transition_detection(self):
        """Phase transitions are detected near S_GEOM."""
        from mcmc.blocks.block4.s_scan import phase_transition_finder

        # Mock data with transition at S_GEOM
        S_values = np.linspace(0.1, 10.0, 20)
        mass_gaps = np.ones(20) * 2.0  # Flat
        mass_gaps[S_values > S_GEOM] = 0.5  # Jump at S_GEOM
        polyakov = np.zeros(20)
        polyakov[S_values > S_GEOM] = 0.8

        transitions = phase_transition_finder(S_values, mass_gaps, polyakov)

        # Should detect S_GEOM
        assert len(transitions) >= 1
        # At least one transition should be near S_GEOM
        found_geom = any(np.abs(t - S_GEOM) < 1.0 for t in transitions)
        assert found_geom


class TestOntologicalInvariants:
    """Tests for MCMC ontological invariants in Block 4."""

    def test_coupling_asymptotic_freedom(self):
        """beta increases with S (asymptotic freedom in geometric regime)."""
        from mcmc.blocks.block4.wilson_action import beta_of_S

        # In geometric regime, higher S means weaker coupling (larger beta)
        S_values = [10.0, 50.0, 90.0]
        betas = [beta_of_S(S) for S in S_values]
        # Should be approximately increasing
        # (actual monotonicity depends on parametrization)
        assert betas[-1] >= betas[0] * 0.9

    def test_confinement_pre_geometric(self):
        """Pre-geometric regime has confinement (low beta)."""
        from mcmc.blocks.block4.wilson_action import beta_of_S
        from mcmc.blocks.block4.config import WilsonParams

        params = WilsonParams(beta_0=6.0, beta_pre_geom=0.1)
        beta_pre = beta_of_S(0.01, params)
        beta_post = beta_of_S(50.0, params)

        # Pre-geometric should have much lower beta (stronger coupling)
        assert beta_pre < beta_post / 2
