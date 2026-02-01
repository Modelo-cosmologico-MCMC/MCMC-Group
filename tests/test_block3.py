"""Tests for Block 3: N-body Cronos simulations.

Tests ontological invariants and module functionality.
"""
from __future__ import annotations

import numpy as np

from mcmc.core.ontology import S_0, S_GEOM


class TestCronosTimestep:
    """Tests for Cronos timestep module."""

    def test_lapse_function_positive(self):
        """N(S) > 0 for all S in valid range."""
        from mcmc.blocks.block3.timestep_cronos import lapse_function_default

        S_values = np.linspace(S_GEOM, S_0, 100)
        N_values = lapse_function_default(S_values)
        assert np.all(N_values > 0)

    def test_lapse_function_finite(self):
        """N(S) is finite for all valid S."""
        from mcmc.blocks.block3.timestep_cronos import lapse_function_default

        S_values = np.linspace(0.1, 99.0, 100)
        N_values = lapse_function_default(S_values)
        assert np.all(np.isfinite(N_values))

    def test_cronos_timestep_positive(self):
        """dt_Cronos > 0."""
        from mcmc.blocks.block3.timestep_cronos import CronosTimestep

        ts = CronosTimestep()
        dt = ts.dt_cronos()
        assert dt > 0

    def test_timestep_at_present(self):
        """At S_0, dt_Cronos ~ dt_Newton."""
        from mcmc.blocks.block3.timestep_cronos import CronosTimestep, TimestepCronosParams

        params = TimestepCronosParams(S_current=S_0, dt_base=0.01)
        ts = CronosTimestep(params)
        dt = ts.dt_cronos()
        # At present, N(S_0) ~ 1
        assert 0.005 < dt < 0.02


class TestPoissonModified:
    """Tests for MCMC-modified Poisson solver."""

    def test_poisson_solver_periodic(self):
        """Poisson solver works with periodic BC."""
        from mcmc.blocks.block3.poisson_modified import PoissonModified, PoissonModifiedParams

        params = PoissonModifiedParams(grid_size=16, box_size=10.0)
        solver = PoissonModified(params)

        # Point mass density
        rho = np.zeros((16, 16, 16))
        rho[8, 8, 8] = 1.0

        potential = solver.solve(rho)
        assert potential.shape == (16, 16, 16)
        assert np.all(np.isfinite(potential))

    def test_mcv_density_positive(self):
        """MCV density contribution is positive."""
        from mcmc.blocks.block3.poisson_modified import rho_lat_simple

        for S in [10.0, 50.0, 90.0]:
            rho = rho_lat_simple(S)
            assert rho >= 0


class TestProfiles:
    """Tests for halo density profiles."""

    def test_nfw_density_positive(self):
        """NFW density is positive."""
        from mcmc.blocks.block3.profiles import NFWProfile

        profile = NFWProfile()
        r = np.logspace(-3, 0, 50)
        rho = profile.density(r)
        assert np.all(rho > 0)

    def test_nfw_mass_increasing(self):
        """NFW enclosed mass is increasing."""
        from mcmc.blocks.block3.profiles import NFWProfile

        profile = NFWProfile()
        r = np.logspace(-3, 0, 50)
        mass = profile.mass(r)
        assert np.all(np.diff(mass) > 0)

    def test_burkert_cored(self):
        """Burkert profile has finite central density."""
        from mcmc.blocks.block3.profiles import BurkertProfile

        profile = BurkertProfile()
        rho_center = profile.density(1e-6)
        assert np.isfinite(rho_center)
        assert rho_center > 0

    def test_zhao_mcmc_interpolates(self):
        """Zhao-MCMC interpolates between cuspy and cored."""
        from mcmc.blocks.block3.profiles import ZhaoMCMCProfile
        from mcmc.blocks.block3.profiles.zhao_mcmc import ZhaoMCMCParams

        # High S (present): cuspy
        params_cusp = ZhaoMCMCParams(delta_gamma=0.0, S_halo=S_0)
        profile_cusp = ZhaoMCMCProfile(params_cusp)
        assert profile_cusp.is_cuspy()

        # With delta_gamma and low S_local: cored
        params_core = ZhaoMCMCParams(delta_gamma=1.0, S_halo=S_0)
        profile_core = ZhaoMCMCProfile(params_core)
        # At low S_local
        assert profile_core.is_cored(S_local=S_GEOM)


class TestAnalysis:
    """Tests for analysis modules."""

    def test_mass_function_positive(self):
        """Halo mass function is positive."""
        from mcmc.blocks.block3.analysis import HaloMassFunction

        mf = HaloMassFunction()
        M = np.logspace(10, 14, 20)
        dn_dM = mf.dn_dM(M)
        assert np.all(dn_dM >= 0)

    def test_subhalo_count_finite(self):
        """Subhalo count is finite."""
        from mcmc.blocks.block3.analysis import SubhaloCounter

        counter = SubhaloCounter()
        N = counter.total_count()
        assert np.isfinite(N)
        assert N >= 0


class TestOntologicalInvariants:
    """Tests for MCMC ontological invariants in Block 3."""

    def test_S_range_respected(self):
        """S values stay in [0, 100] range."""
        from mcmc.blocks.block3.timestep_cronos import CronosTimestep, TimestepCronosParams

        for S in [0.1, 1.0, 50.0, 95.0]:
            params = TimestepCronosParams(S_current=S)
            ts = CronosTimestep(params)
            assert 0 <= ts.state.S_current <= 100

    def test_present_stratified(self):
        """S_local < S_global in dense regions (stratified present)."""
        from mcmc.blocks.block3.profiles import NFWProfile

        profile = NFWProfile()
        # Inner region should have lower effective S
        S_global = S_0
        rho_inner = profile.density_mcmc(0.001, S_local=50.0)
        rho_outer = profile.density_mcmc(0.001, S_local=S_global)
        # Lower S_local gives higher effective density
        assert rho_inner > rho_outer
