"""Tests para perfiles de halos N-body."""
from __future__ import annotations

import numpy as np
import pytest

from mcmc.nbody.profiles import (
    BurkertParams,
    ZhaoParams,
    NFWParams,
    SlocParams,
    rho_burkert,
    mass_burkert,
    V_burkert,
    rho_zhao,
    V_zhao,
    rho_nfw,
    mass_nfw,
    V_nfw,
    halo_core_params_from_Sloc,
    burkert_from_Sloc,
)


class TestBurkertProfile:
    """Tests para perfil Burkert."""

    @pytest.fixture
    def params(self):
        """Parámetros Burkert típicos."""
        return BurkertParams(rho0=1e7, r0=2.0)

    def test_rho_central(self, params):
        """Densidad central finita."""
        r = np.array([0.001])  # Muy cerca del centro
        rho = rho_burkert(r, params)
        # Burkert tiene núcleo finito
        assert np.isfinite(rho[0])
        assert rho[0] < params.rho0 * 10  # No diverge

    def test_rho_decreases(self, params):
        """Densidad decrece con radio."""
        r = np.array([0.1, 1.0, 5.0, 10.0])
        rho = rho_burkert(r, params)
        assert np.all(np.diff(rho) < 0)

    def test_mass_increases(self, params):
        """Masa encerrada aumenta con radio."""
        r = np.array([0.1, 1.0, 5.0, 10.0])
        M = mass_burkert(r, params)
        assert np.all(np.diff(M) > 0)

    def test_V_circ_shape(self, params):
        """V circular tiene forma esperada."""
        r = np.logspace(-1, 2, 50)
        V = V_burkert(r, params)

        # V debe ser positivo y finito
        assert np.all(V > 0)
        assert np.all(np.isfinite(V))

        # V sube, alcanza máximo, y luego cae (para r grande)
        idx_max = np.argmax(V)
        assert idx_max > 0
        assert idx_max < len(V) - 1


class TestZhaoProfile:
    """Tests para perfil Zhao generalizado."""

    @pytest.fixture
    def params_cored(self):
        """Perfil Zhao cored (γ=0)."""
        return ZhaoParams(rho_s=1e7, r_s=5.0, alpha=2.0, beta=3.0, gamma=0.0)

    @pytest.fixture
    def params_cuspy(self):
        """Perfil Zhao cuspy tipo NFW (γ=1)."""
        return ZhaoParams(rho_s=1e7, r_s=10.0, alpha=1.0, beta=3.0, gamma=1.0)

    def test_cored_finite_center(self, params_cored):
        """Perfil cored tiene densidad finita en el centro."""
        r = np.array([0.001])
        rho = rho_zhao(r, params_cored)
        assert np.isfinite(rho[0])

    def test_cuspy_diverges(self, params_cuspy):
        """Perfil cuspy diverge hacia el centro."""
        r = np.array([0.01, 0.001])
        rho = rho_zhao(r, params_cuspy)
        assert rho[1] > rho[0]  # Aumenta hacia el centro

    def test_V_zhao_positive(self, params_cored):
        """V circular es positivo."""
        r = np.array([1.0, 5.0, 10.0])
        V = V_zhao(r, params_cored)
        assert np.all(V > 0)


class TestNFWProfile:
    """Tests para perfil NFW."""

    @pytest.fixture
    def params(self):
        """Parámetros NFW típicos."""
        return NFWParams(rho_s=1e7, r_s=10.0)

    def test_rho_diverges_center(self, params):
        """NFW diverge en r→0."""
        r = np.array([0.1, 0.01, 0.001])
        rho = rho_nfw(r, params)
        assert rho[1] > rho[0]
        assert rho[2] > rho[1]

    def test_mass_nfw_analytic(self, params):
        """Masa NFW tiene forma analítica correcta."""
        r = np.array([params.r_s])  # En r_s
        M = mass_nfw(r, params)

        # M(r_s) = 4π ρ_s r_s³ [ln(2) - 1/2]
        M_expected = 4 * np.pi * params.rho_s * params.r_s**3 * (np.log(2) - 0.5)
        np.testing.assert_allclose(M[0], M_expected, rtol=1e-5)

    def test_V_nfw_shape(self, params):
        """V circular NFW tiene forma esperada."""
        r = np.logspace(-1, 2, 50)
        V = V_nfw(r, params)

        assert np.all(V > 0)
        # NFW V tiene máximo
        idx_max = np.argmax(V)
        assert 0 < idx_max < len(V) - 1


class TestSlocDependence:
    """Tests para dependencia en S_loc."""

    @pytest.fixture
    def sloc_params(self):
        """Parámetros de referencia."""
        return SlocParams(
            S_star=1.001,
            rho_star=1e7,
            r_star=2.0,
            alpha_rho=0.4,
            alpha_r=0.3,
        )

    def test_reference_values(self, sloc_params):
        """En S_loc = S_star, devuelve valores de referencia."""
        rho0, r_core = halo_core_params_from_Sloc(
            sloc_params.S_star, sloc_params
        )
        np.testing.assert_allclose(rho0, sloc_params.rho_star, rtol=1e-5)
        np.testing.assert_allclose(r_core, sloc_params.r_star, rtol=1e-5)

    def test_higher_Sloc_lower_rho(self, sloc_params):
        """Mayor S_loc → menor ρ₀."""
        S_low = 1.0005
        S_high = 1.002

        rho_low, _ = halo_core_params_from_Sloc(S_low, sloc_params)
        rho_high, _ = halo_core_params_from_Sloc(S_high, sloc_params)

        assert rho_high < rho_low

    def test_higher_Sloc_larger_core(self, sloc_params):
        """Mayor S_loc → mayor r_core."""
        S_low = 1.0005
        S_high = 1.002

        _, r_low = halo_core_params_from_Sloc(S_low, sloc_params)
        _, r_high = halo_core_params_from_Sloc(S_high, sloc_params)

        assert r_high > r_low

    def test_burkert_from_Sloc(self, sloc_params):
        """Crea parámetros Burkert desde S_loc."""
        S_loc = 1.001
        params = burkert_from_Sloc(S_loc, sloc_params)

        assert isinstance(params, BurkertParams)
        assert params.rho0 > 0
        assert params.r0 > 0
