"""Tests for PR-FIX-03: Separate pre-BB and post-BB solvers.

Verifies:
1. solve_prebb operates only in S ≤ S_BB regime
2. solve_postbb operates only in z ≥ 0 (post-BB) regime
3. Boundary conditions at S_BB are correctly extracted
4. Post-BB observables are well-defined
"""
from __future__ import annotations

import numpy as np

from mcmc.core.ontology import THRESHOLDS
from mcmc.core.solve_prebb import solve_prebb, get_prebb_boundary_conditions
from mcmc.core.solve_postbb import (
    solve_postbb,
    PostBBParams,
    evaluate_postbb_at_z,
    get_postbb_time_mapping,
)


class TestSolvePreBB:
    """Test pre-BB solver."""

    def test_default_params(self) -> None:
        """solve_prebb works with default parameters."""
        result = solve_prebb()
        assert result.S is not None
        assert len(result.S) > 0

    def test_S_range_is_prebb(self) -> None:
        """S grid stays in pre-BB regime (S ≤ S_BB)."""
        result = solve_prebb()
        assert np.max(result.S) <= THRESHOLDS.S_BB + 1e-6

    def test_t_anchor_at_BB(self) -> None:
        """t(S_BB) ≈ 0 at Big Bang."""
        result = solve_prebb()
        assert abs(result.t_BB) < 1e-6

    def test_a_rel_normalized_at_BB(self) -> None:
        """a_rel(S_BB) = 1 by normalization (ontological, NOT FRW)."""
        result = solve_prebb()
        assert abs(result.a_rel_BB - 1.0) < 1e-6

    def test_t_negative_before_BB(self) -> None:
        """t < 0 for S < S_BB (pre-Big-Bang regime)."""
        result = solve_prebb()
        # t should be negative for most of the pre-BB regime
        # (except at S_BB where t=0)
        pre_bb_mask = result.S < THRESHOLDS.S_BB - 0.01
        if np.any(pre_bb_mask):
            assert np.all(result.t[pre_bb_mask] < 0)

    def test_boundary_conditions(self) -> None:
        """Boundary conditions are correctly extracted."""
        result = solve_prebb()
        bc = get_prebb_boundary_conditions(result)

        assert "a_rel_BB" in bc
        assert "t_BB" in bc
        assert "Mp_pre" in bc
        assert "Ep_pre" in bc
        assert abs(bc["t_BB"]) < 1e-6


class TestSolvePostBB:
    """Test post-BB solver."""

    def test_default_params(self) -> None:
        """solve_postbb works with default parameters."""
        result = solve_postbb()
        assert result.H_of_z is not None
        assert result.mu_of_z is not None
        assert result.DVrd_of_z is not None

    def test_H0_at_z_zero(self) -> None:
        """H(z=0) = H0."""
        params = PostBBParams(H0=70.0)
        result = solve_postbb(params)
        H_at_0 = result.H_of_z(np.array([0.0]))[0]
        assert abs(H_at_0 - 70.0) < 1e-6

    def test_H_increases_with_z(self) -> None:
        """H(z) generally increases with z (matter/DE dominated)."""
        result = solve_postbb()
        z = np.linspace(0, 2, 50)
        H = result.H_of_z(z)
        # H should generally increase (not strictly due to transitions)
        assert H[-1] > H[0]

    def test_mu_at_z_zero(self) -> None:
        """μ(z=0) should be well-defined (may be -inf or small)."""
        result = solve_postbb()
        z = np.array([0.01])  # Avoid z=0 singularity
        mu = result.mu_of_z(z)
        assert np.isfinite(mu[0])

    def test_mu_increases_with_z(self) -> None:
        """μ(z) increases with z (further objects are fainter)."""
        result = solve_postbb()
        z = np.linspace(0.01, 2, 50)
        mu = result.mu_of_z(z)
        dmu = np.diff(mu)
        assert np.all(dmu > 0), "mu should increase with z"

    def test_DVrd_positive(self) -> None:
        """DV/rd(z) > 0 for z > 0."""
        result = solve_postbb()
        z = np.linspace(0.1, 2, 20)
        dvrd = result.DVrd_of_z(z)
        assert np.all(dvrd > 0)

    def test_age_of_universe_positive(self) -> None:
        """Age of universe t0 > 0."""
        result = solve_postbb()
        assert result.t0 > 0


class TestEvaluatePostBB:
    """Test evaluation helpers."""

    def test_evaluate_at_z(self) -> None:
        """evaluate_postbb_at_z returns all observables."""
        result = solve_postbb()
        z = np.array([0.1, 0.5, 1.0])
        obs = evaluate_postbb_at_z(result, z)

        assert "z" in obs
        assert "H" in obs
        assert "mu" in obs
        assert "DVrd" in obs
        assert len(obs["H"]) == 3
        assert len(obs["mu"]) == 3
        assert len(obs["DVrd"]) == 3

    def test_time_mapping(self) -> None:
        """get_postbb_time_mapping returns valid times."""
        result = solve_postbb()
        z = np.array([0.0, 0.5, 1.0])
        times = get_postbb_time_mapping(result, z)

        assert "t_lookback" in times
        assert "t_cosmic" in times

        # t_lookback(z=0) = 0
        assert abs(times["t_lookback"][0]) < 1e-10

        # t_cosmic(z=0) = t0
        assert abs(times["t_cosmic"][0] - result.t0) < 1e-10

        # t_lookback increases with z
        assert times["t_lookback"][1] > times["t_lookback"][0]
        assert times["t_lookback"][2] > times["t_lookback"][1]

        # t_cosmic decreases with z
        assert times["t_cosmic"][1] < times["t_cosmic"][0]
        assert times["t_cosmic"][2] < times["t_cosmic"][1]


class TestSolverSeparation:
    """Test that solvers are properly separated."""

    def test_prebb_does_not_use_Hz_observables(self) -> None:
        """Pre-BB solver should not produce H(z) observables."""
        result = solve_prebb()
        # H_ref is in S-space, not z-space
        assert hasattr(result, "H_ref")
        # Should NOT have H_of_z callable
        assert not callable(getattr(result, "H_of_z", None))

    def test_postbb_does_not_use_S_grid(self) -> None:
        """Post-BB solver should not produce S grid."""
        result = solve_postbb()
        # Should NOT have S array
        assert not hasattr(result, "S") or result.S is None if hasattr(result, "S") else True

    def test_both_solvers_independent(self) -> None:
        """Both solvers can run independently."""
        # Pre-BB doesn't need post-BB
        prebb = solve_prebb()
        assert prebb.t_BB == 0.0

        # Post-BB doesn't need pre-BB
        postbb = solve_postbb()
        assert postbb.H0 > 0
