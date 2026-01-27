"""Tests para el canal latente ρ_lat(S)."""
from __future__ import annotations

import numpy as np
import pytest

from mcmc.channels.rho_lat import (
    LatentChannel,
    LatentChannelParams,
    hubble_correction_lat,
)


class TestLatentChannelParams:
    """Tests para parámetros del canal latente."""

    def test_default_params(self):
        """Parámetros por defecto son válidos."""
        params = LatentChannelParams()
        assert params.enabled is True
        assert params.kappa_0 > 0
        assert params.S_star == 1.001
        assert params.dS_star > 0
        assert params.rho_lat_star > 0

    def test_disabled_params(self):
        """Parámetros deshabilitados."""
        params = LatentChannelParams(enabled=False)
        assert params.enabled is False


class TestLatentChannel:
    """Tests para la clase LatentChannel."""

    @pytest.fixture
    def channel(self):
        """Canal latente con parámetros por defecto."""
        return LatentChannel()

    @pytest.fixture
    def disabled_channel(self):
        """Canal latente deshabilitado."""
        return LatentChannel(LatentChannelParams(enabled=False))

    def test_kappa_lat_at_S_star(self, channel):
        """κ_lat en S_star es cercano a 0."""
        kappa = channel.kappa_lat(channel.params.S_star)
        assert abs(kappa) < channel.params.kappa_0 * 0.1

    def test_kappa_lat_increases_with_S(self, channel):
        """κ_lat aumenta con S > S_star."""
        S1 = channel.params.S_star + 0.001
        S2 = channel.params.S_star + 0.01
        assert channel.kappa_lat(S2) > channel.kappa_lat(S1)

    def test_kappa_lat_disabled(self, disabled_channel):
        """κ_lat es 0 cuando está deshabilitado."""
        assert disabled_channel.kappa_lat(1.001) == 0.0

    def test_rho_lat_at_S_star(self, channel):
        """ρ_lat en S_star es aproximadamente rho_lat_star."""
        rho = channel.rho_lat(channel.params.S_star)
        assert np.isclose(rho, channel.params.rho_lat_star, rtol=0.1)

    def test_rho_lat_decays_with_S(self, channel):
        """ρ_lat decae para S > S_star."""
        S1 = channel.params.S_star + 0.001
        S2 = channel.params.S_star + 0.01
        assert channel.rho_lat(S2) < channel.rho_lat(S1)

    def test_rho_lat_array(self, channel):
        """rho_lat_array funciona para arrays."""
        S = np.linspace(1.0005, 1.002, 10)
        rho = channel.rho_lat_array(S)
        assert len(rho) == len(S)
        # Debe ser monótonamente decreciente para S > S_star
        assert np.all(np.diff(rho[S > channel.params.S_star]) <= 0)

    def test_rho_lat_disabled(self, disabled_channel):
        """ρ_lat es 0 cuando está deshabilitado."""
        assert disabled_channel.rho_lat(1.001) == 0.0

    def test_drho_lat_dS_negative(self, channel):
        """dρ_lat/dS es negativo (decaimiento)."""
        S = channel.params.S_star + 0.005
        drho = channel.drho_lat_dS(S)
        assert drho < 0

    def test_delta_lat_positive(self, channel):
        """δ_lat es positivo cuando activo."""
        S = channel.params.S_star + 0.005
        rho_tot = 1.0
        delta = channel.delta_lat(S, rho_tot)
        assert delta > 0

    def test_w_lat_near_minus_one(self, channel):
        """w_lat es cercano a -1 (comportamiento DE-like)."""
        S = channel.params.S_star + 0.005
        dS_dz = 0.001
        w = channel.w_lat(S, dS_dz)
        assert -1.5 < w < -0.5

    def test_f_lat_cmb_constraint(self, channel):
        """f_lat es pequeño (consistente con CMB)."""
        S = channel.params.S_star + 0.001
        f = channel.f_lat(S)
        assert f < 0.1  # f_lat < 10% de ρ_crit


class TestHubbleCorrection:
    """Tests para corrección al Hubble."""

    def test_correction_increases_H(self):
        """Corrección positiva de ρ_lat aumenta H."""
        z = np.array([0.0, 0.5, 1.0])
        H_LCDM = np.array([67.4, 85.0, 100.0])
        rho_lat = np.array([0.01, 0.01, 0.01])

        H_MCMC = hubble_correction_lat(z, H_LCDM, rho_lat)

        assert np.all(H_MCMC >= H_LCDM)

    def test_zero_correction(self):
        """Sin corrección si ρ_lat = 0."""
        z = np.array([0.0, 0.5, 1.0])
        H_LCDM = np.array([67.4, 85.0, 100.0])
        rho_lat = np.zeros(3)

        H_MCMC = hubble_correction_lat(z, H_LCDM, rho_lat)

        np.testing.assert_allclose(H_MCMC, H_LCDM)

    def test_with_delta_rho_id(self):
        """Corrección con Δρ_id adicional."""
        z = np.array([0.0, 0.5])
        H_LCDM = np.array([67.4, 85.0])
        rho_lat = np.array([0.01, 0.01])
        delta_rho_id = np.array([0.01, 0.01])

        H_MCMC = hubble_correction_lat(z, H_LCDM, rho_lat, delta_rho_id)

        # Con más corrección, H debería ser mayor
        H_only_lat = hubble_correction_lat(z, H_LCDM, rho_lat)
        assert np.all(H_MCMC >= H_only_lat)
