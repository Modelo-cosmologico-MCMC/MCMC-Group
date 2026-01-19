"""Tests for PR-FIX-02: Chronos law and S↔t↔z mapping.

Verifies:
1. t(S_BB) = 0 (Big Bang anchor)
2. Monotonicity of t(S)
3. Invertibility: S ≈ S_of_t(t_of_S(S))
4. Monotonicity of t_lookback(z)
"""
from __future__ import annotations

import numpy as np

from mcmc.core.ontology import THRESHOLDS
from mcmc.core.chronos import ChronosParams, t_of_S, S_of_t, dt_dS, assert_monotonic
from mcmc.core.mapping import t_lookback_of_z, t_cosmic_of_z, z_of_t_lookback


class TestChronosAnchor:
    """Test that Chronos anchors time at Big Bang."""

    def test_t_of_S_at_BB_is_zero(self) -> None:
        """t(S_BB) must equal 0."""
        p = ChronosParams()
        t_bb = t_of_S(np.array([THRESHOLDS.S_BB]), p)[0]
        assert abs(t_bb) < 1e-10, f"t(S_BB) = {t_bb}, expected 0"

    def test_t_negative_pre_bb(self) -> None:
        """t < 0 for S < S_BB (pre-Big-Bang)."""
        p = ChronosParams()
        S_pre = np.array([0.5, 0.8, 1.0])
        t = t_of_S(S_pre, p)
        assert np.all(t < 0), f"Expected t < 0 for pre-BB, got {t}"

    def test_t_positive_post_bb(self) -> None:
        """t > 0 for S > S_BB (post-Big-Bang)."""
        p = ChronosParams()
        S_post = np.array([1.002, 1.1, 1.5])
        t = t_of_S(S_post, p)
        assert np.all(t > 0), f"Expected t > 0 for post-BB, got {t}"


class TestChronosMonotonicity:
    """Test that Chronos t(S) is monotonically increasing."""

    def test_dt_dS_positive(self) -> None:
        """dt/dS = T(S) * N(S) > 0."""
        p = ChronosParams()
        S = np.linspace(0.01, 2.0, 200)
        dtds = dt_dS(S, p)
        assert np.all(dtds > 0), f"dt/dS should be positive, min={np.min(dtds)}"

    def test_t_monotonic(self) -> None:
        """t(S) is strictly increasing."""
        p = ChronosParams()
        S = np.linspace(0.01, 2.0, 500)
        t = t_of_S(S, p)
        dt = np.diff(t)
        assert np.all(dt > 0), f"t(S) not monotonic, min(dt)={np.min(dt)}"

    def test_assert_monotonic_passes(self) -> None:
        """assert_monotonic should pass with default params."""
        p = ChronosParams()
        assert_monotonic(p, S_range=(0.01, 2.0))


class TestChronosInversion:
    """Test S_of_t inverts t_of_S."""

    def test_roundtrip_accuracy(self) -> None:
        """S ≈ S_of_t(t_of_S(S))."""
        p = ChronosParams()
        S_orig = np.array([0.5, THRESHOLDS.S_BB, 1.2, 1.5])
        t = t_of_S(S_orig, p)
        S_recovered = S_of_t(t, p, bracket=(0.01, 3.0))

        diff = np.abs(S_recovered - S_orig)
        assert np.max(diff) < 1e-3, f"Roundtrip error: max diff = {np.max(diff)}"

    def test_inversion_at_bb(self) -> None:
        """S_of_t(0) = S_BB."""
        p = ChronosParams()
        S_bb = S_of_t(np.array([0.0]), p)[0]
        assert abs(S_bb - THRESHOLDS.S_BB) < 1e-3, f"S_of_t(0) = {S_bb}, expected {THRESHOLDS.S_BB}"


class TestMappingLookback:
    """Test cosmological time mappings."""

    @staticmethod
    def simple_H(z: np.ndarray) -> np.ndarray:
        """Simple H(z) = H0 * (1+z) for testing."""
        return 70.0 * (1.0 + z)

    def test_t_lookback_zero_at_z_zero(self) -> None:
        """t_lookback(z=0) = 0."""
        z = np.array([0.0])
        t = t_lookback_of_z(z, self.simple_H)
        assert abs(float(t[0])) < 1e-12

    def test_t_lookback_monotonic(self) -> None:
        """t_lookback increases with z."""
        z = np.linspace(0, 2, 50)
        t = t_lookback_of_z(z, self.simple_H)
        dt = np.diff(t)
        assert np.all(dt >= -1e-12), f"t_lookback not monotonic, min(dt)={np.min(dt)}"

    def test_z_of_t_lookback_monotonic(self) -> None:
        """z_of_t_lookback is monotonic (larger t -> larger z)."""
        t_test = np.linspace(0.0, 0.01, 20)  # Small range of lookback times
        z_test = z_of_t_lookback(t_test, self.simple_H, zmax=5.0, n_grid=500)

        # z should increase with t_lookback
        dz = np.diff(z_test)
        assert np.all(dz >= -1e-6), f"z should increase with t_lookback, min(dz)={np.min(dz)}"


class TestMappingCosmic:
    """Test cosmic time mapping."""

    @staticmethod
    def simple_H(z: np.ndarray) -> np.ndarray:
        """Simple H(z) for testing."""
        return 70.0 * (1.0 + z)

    def test_t_cosmic_decreases_with_z(self) -> None:
        """t_cosmic(z) decreases as z increases (further in past)."""
        z = np.linspace(0, 2, 50)
        t0 = 1.0  # Arbitrary age
        t_cosmic = t_cosmic_of_z(z, self.simple_H, t0)

        # At z=0, t_cosmic = t0
        assert abs(t_cosmic[0] - t0) < 1e-10

        # t_cosmic decreases with z
        dt = np.diff(t_cosmic)
        assert np.all(dt <= 1e-12), "t_cosmic should decrease with z"
