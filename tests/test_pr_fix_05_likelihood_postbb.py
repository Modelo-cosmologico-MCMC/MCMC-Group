"""Tests for PR-FIX-05: Likelihood only for post-BB observables.

Verifies:
1. Likelihoods validate z >= 0 (post-BB regime)
2. Negative redshifts raise ValueError
3. Validation can be disabled when needed
4. All likelihood functions work correctly for valid data
"""
from __future__ import annotations

import numpy as np
import pytest

from mcmc.observables.likelihoods import (
    loglike_total,
    loglike_hz,
    loglike_sne,
    loglike_bao,
    _validate_postbb_redshifts,
)


class TestPostBBValidation:
    """Test redshift validation for post-BB regime."""

    def test_valid_redshifts_pass(self) -> None:
        """z >= 0 passes validation."""
        z = np.array([0.0, 0.1, 0.5, 1.0, 2.0])
        # Should not raise
        _validate_postbb_redshifts(z, "test")

    def test_negative_redshifts_fail(self) -> None:
        """z < 0 raises ValueError."""
        z = np.array([-0.1, 0.1, 0.5])
        with pytest.raises(ValueError, match="post-BB regime"):
            _validate_postbb_redshifts(z, "test")

    def test_all_negative_fails(self) -> None:
        """All negative z raises ValueError."""
        z = np.array([-1.0, -0.5, -0.1])
        with pytest.raises(ValueError, match="post-BB regime"):
            _validate_postbb_redshifts(z, "test")

    def test_error_message_contains_min_z(self) -> None:
        """Error message includes minimum z value."""
        z = np.array([-0.5, 0.1, 0.5])
        with pytest.raises(ValueError, match="-0.5"):
            _validate_postbb_redshifts(z, "test")


class TestLoglikeHz:
    """Test H(z) likelihood function."""

    @staticmethod
    def simple_H(z: np.ndarray) -> np.ndarray:
        return 70.0 * (1.0 + z)

    def test_valid_redshifts(self) -> None:
        """loglike_hz works with valid z >= 0."""
        z = np.array([0.1, 0.5, 1.0])
        H_obs = self.simple_H(z) + np.array([1.0, -1.0, 0.5])
        sigma = np.ones_like(z) * 5.0

        ll = loglike_hz(z, H_obs, sigma, self.simple_H)
        assert np.isfinite(ll)
        assert ll <= 0  # Log-likelihood should be non-positive

    def test_negative_z_raises(self) -> None:
        """loglike_hz raises for z < 0."""
        z = np.array([-0.1, 0.5, 1.0])
        H_obs = self.simple_H(np.abs(z))
        sigma = np.ones_like(z) * 5.0

        with pytest.raises(ValueError, match="post-BB regime"):
            loglike_hz(z, H_obs, sigma, self.simple_H)

    def test_validation_can_be_disabled(self) -> None:
        """loglike_hz with validate=False skips check."""
        z = np.array([-0.1, 0.5, 1.0])
        H_obs = self.simple_H(np.abs(z))
        sigma = np.ones_like(z) * 5.0

        # Should not raise
        ll = loglike_hz(z, H_obs, sigma, self.simple_H, validate=False)
        assert np.isfinite(ll)


class TestLoglikeSne:
    """Test SNe Ia likelihood function."""

    @staticmethod
    def simple_mu(z: np.ndarray) -> np.ndarray:
        return 5.0 * np.log10((1.0 + z) * 1000.0) + 25.0

    def test_valid_redshifts(self) -> None:
        """loglike_sne works with valid z >= 0."""
        z = np.array([0.1, 0.5, 1.0])
        mu_obs = self.simple_mu(z) + np.array([0.1, -0.1, 0.05])
        sigma = np.ones_like(z) * 0.1

        ll = loglike_sne(z, mu_obs, sigma, self.simple_mu)
        assert np.isfinite(ll)

    def test_negative_z_raises(self) -> None:
        """loglike_sne raises for z < 0."""
        z = np.array([-0.1, 0.5, 1.0])
        mu_obs = self.simple_mu(np.abs(z))
        sigma = np.ones_like(z) * 0.1

        with pytest.raises(ValueError, match="SNe Ia"):
            loglike_sne(z, mu_obs, sigma, self.simple_mu)


class TestLoglikeBao:
    """Test BAO likelihood function."""

    @staticmethod
    def simple_dvrd(z: np.ndarray) -> np.ndarray:
        return 10.0 * z

    def test_valid_redshifts(self) -> None:
        """loglike_bao works with valid z >= 0."""
        z = np.array([0.3, 0.6, 1.0])
        dvrd_obs = self.simple_dvrd(z) + np.array([0.1, -0.1, 0.05])
        sigma = np.ones_like(z) * 0.5

        ll = loglike_bao(z, dvrd_obs, sigma, self.simple_dvrd)
        assert np.isfinite(ll)

    def test_negative_z_raises(self) -> None:
        """loglike_bao raises for z < 0."""
        z = np.array([-0.1, 0.5, 1.0])
        dvrd_obs = self.simple_dvrd(np.abs(z))
        sigma = np.ones_like(z) * 0.5

        with pytest.raises(ValueError, match="BAO"):
            loglike_bao(z, dvrd_obs, sigma, self.simple_dvrd)


class TestLoglikeTotal:
    """Test combined likelihood function."""

    @staticmethod
    def make_model():
        return {
            "H(z)": lambda z: 70.0 * (1.0 + z),
            "mu(z)": lambda z: 5.0 * np.log10((1.0 + z) * 1000.0) + 25.0,
            "DVrd(z)": lambda z: 10.0 * z,
        }

    def test_valid_datasets(self) -> None:
        """loglike_total works with valid datasets."""
        model = self.make_model()
        z = np.array([0.1, 0.5, 1.0])

        datasets = {
            "hz": {
                "z": z,
                "H": model["H(z)"](z) + 1.0,
                "sigma": np.ones_like(z) * 5.0,
            },
            "sne": {
                "z": z,
                "mu": model["mu(z)"](z) + 0.1,
                "sigma": np.ones_like(z) * 0.1,
            },
            "bao": {
                "z": z,
                "dv_rd": model["DVrd(z)"](z) + 0.1,
                "sigma": np.ones_like(z) * 0.5,
            },
        }

        ll = loglike_total(datasets, model)
        assert np.isfinite(ll)
        assert ll < 0

    def test_negative_z_in_any_dataset_raises(self) -> None:
        """loglike_total raises if any dataset has z < 0."""
        model = self.make_model()
        z_good = np.array([0.1, 0.5, 1.0])
        z_bad = np.array([-0.1, 0.5, 1.0])

        datasets = {
            "hz": {
                "z": z_good,
                "H": model["H(z)"](z_good),
                "sigma": np.ones_like(z_good) * 5.0,
            },
            "sne": {
                "z": z_bad,  # This dataset has negative z
                "mu": model["mu(z)"](np.abs(z_bad)),
                "sigma": np.ones_like(z_bad) * 0.1,
            },
        }

        with pytest.raises(ValueError, match="SNe Ia"):
            loglike_total(datasets, model)

    def test_validation_can_be_disabled(self) -> None:
        """loglike_total with validate=False skips check."""
        model = self.make_model()
        z_bad = np.array([-0.1, 0.5, 1.0])

        datasets = {
            "hz": {
                "z": z_bad,
                "H": model["H(z)"](np.abs(z_bad)),
                "sigma": np.ones_like(z_bad) * 5.0,
            },
        }

        # Should not raise
        ll = loglike_total(datasets, model, validate=False)
        assert np.isfinite(ll)


class TestPostBBDocumentation:
    """Test that module documents post-BB requirement."""

    def test_module_docstring_mentions_postbb(self) -> None:
        """Module docstring mentions post-BB requirement."""
        import mcmc.observables.likelihoods as mod
        assert "post-BB" in mod.__doc__.lower() or "post-big-bang" in mod.__doc__.lower()

    def test_module_mentions_s_bb(self) -> None:
        """Module docstring mentions S_BB = 1.001."""
        import mcmc.observables.likelihoods as mod
        assert "1.001" in mod.__doc__
