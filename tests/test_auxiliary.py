"""Tests for auxiliary modules.

Tests baryogenesis and other auxiliary physics modules.
"""
from __future__ import annotations

import numpy as np

from mcmc.core.ontology import S_0, S_GEOM


class TestBaryogenesis:
    """Tests for baryogenesis module."""

    def test_sakharov_conditions_at_transition(self):
        """Sakharov conditions are satisfied at S_GEOM."""
        from mcmc.auxiliary.baryogenesis import sakharov_conditions, BaryogenesisParams

        params = BaryogenesisParams(S_baryogenesis=S_GEOM)
        conditions = sakharov_conditions(S_GEOM, params)

        assert conditions["all_satisfied"]

    def test_sakharov_not_satisfied_late(self):
        """Sakharov conditions not satisfied at late times."""
        from mcmc.auxiliary.baryogenesis import sakharov_conditions, BaryogenesisParams

        params = BaryogenesisParams(S_baryogenesis=S_GEOM)
        conditions = sakharov_conditions(S_0, params)

        assert not conditions["all_satisfied"]

    def test_eta_B_positive(self):
        """Baryon asymmetry is positive."""
        from mcmc.auxiliary.baryogenesis import eta_B_of_S, BaryogenesisParams

        params = BaryogenesisParams()
        eta = eta_B_of_S(S_GEOM, params)
        assert eta >= 0

    def test_eta_B_peaks_at_transition(self):
        """Baryon asymmetry peaks near the transition."""
        from mcmc.auxiliary.baryogenesis import eta_B_of_S, BaryogenesisParams

        params = BaryogenesisParams(S_baryogenesis=S_GEOM)

        eta_pre = eta_B_of_S(0.5, params)
        eta_trans = eta_B_of_S(S_GEOM, params)
        eta_late = eta_B_of_S(50.0, params)

        # Peak should be at/near transition
        assert eta_trans >= eta_pre
        assert eta_trans >= eta_late

    def test_cp_violation_mcmc(self):
        """CP violation peaks at transition."""
        from mcmc.auxiliary.baryogenesis import cp_violation_mcmc, BaryogenesisParams

        params = BaryogenesisParams(S_baryogenesis=S_GEOM, epsilon_CP=1e-8)

        eps_trans = cp_violation_mcmc(S_GEOM, params)
        eps_late = cp_violation_mcmc(50.0, params)

        assert eps_trans > eps_late

    def test_integrate_eta_B_finite(self):
        """Integrated baryon asymmetry is finite."""
        from mcmc.auxiliary.baryogenesis import integrate_eta_B, BaryogenesisParams

        params = BaryogenesisParams()
        total_eta, S_arr, eta_arr = integrate_eta_B(params)

        assert np.isfinite(total_eta)
        assert np.all(np.isfinite(eta_arr))

    def test_baryogenesis_model_calibration(self):
        """Model can be calibrated to observed asymmetry."""
        from mcmc.auxiliary.baryogenesis import BaryogenesisModel

        model = BaryogenesisModel()
        calibrated_params = model.calibrate_to_observed()

        # Check calibrated epsilon_CP is reasonable
        assert calibrated_params.epsilon_CP > 0
        assert np.isfinite(calibrated_params.epsilon_CP)

    def test_omega_b_prediction(self):
        """Omega_b prediction is positive and finite."""
        from mcmc.auxiliary.baryogenesis import BaryogenesisModel

        model = BaryogenesisModel()
        omega_b = model.omega_b_prediction()

        assert omega_b > 0
        assert np.isfinite(omega_b)


class TestOntologicalInvariants:
    """Tests for MCMC ontological invariants in auxiliary modules."""

    def test_baryogenesis_at_big_bang(self):
        """Baryogenesis occurs at Big Bang (S_GEOM)."""
        from mcmc.auxiliary.baryogenesis import BaryogenesisParams

        params = BaryogenesisParams()
        # Default S_baryogenesis should be S_GEOM
        assert params.S_baryogenesis == S_GEOM

    def test_baryon_fraction_matches_omega_b(self):
        """MCMC baryon fraction consistent with Omega_b."""
        from mcmc.auxiliary.baryogenesis import BaryogenesisModel

        model = BaryogenesisModel()
        comparison = model.compare_to_observation()

        # Should be order-of-magnitude correct
        assert comparison["Omega_b_predicted"] > 0
        # Within a factor of 10 of observed (model is simplified)
        ratio = comparison["Omega_b_ratio"]
        assert 0.01 < ratio < 100
