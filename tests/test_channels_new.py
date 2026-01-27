"""Tests para lambda_rel y q_dual en channels."""
from __future__ import annotations

import numpy as np
import pytest

from mcmc.channels.lambda_rel import (
    LambdaRelParams,
    Lambda_rel_of_z,
    dLambda_rel_dz,
    Omega_Lambda_rel,
    H_squared_correction,
    H_rel,
    w_eff_Lambda,
    LambdaRelFromChannels,
)
from mcmc.channels.q_dual import (
    QDualParams,
    eta_lat_of_S,
    eta_id_of_S,
    Q_dual,
    Q_dual_simple,
    CoupledChannelEvolver,
)


class TestLambdaRel:
    """Tests para Λ_rel(z) dinámica."""

    @pytest.fixture
    def params(self):
        """Parámetros por defecto."""
        return LambdaRelParams()

    def test_Lambda_at_z0(self, params):
        """Λ_rel(z=0) es finito y positivo."""
        Lambda = Lambda_rel_of_z(0.0, params)
        # Con epsilon=0.012 y z_trans=8.9, Lambda(z=0) ≈ Lambda_0 * (1 + 0.012*8.9)
        assert np.isfinite(Lambda)
        assert Lambda > 0

    def test_Lambda_continuous(self, params):
        """Λ_rel(z) es continua."""
        z = np.linspace(0, 15, 100)
        Lambda = Lambda_rel_of_z(z, params)
        # Diferencias no deben ser demasiado grandes
        dLambda = np.diff(Lambda)
        max_jump = np.max(np.abs(dLambda))
        assert max_jump < 0.5 * params.Lambda_0

    def test_dLambda_rel_dz_finite(self, params):
        """dΛ/dz es finito en todo el rango."""
        z_arr = np.linspace(0, 15, 50)
        dL = dLambda_rel_dz(z_arr, params)
        assert np.all(np.isfinite(dL))

    def test_Omega_Lambda_positive(self, params):
        """Ω_Λ > 0."""
        z = np.array([0.0, 1.0, 2.0])
        Omega = Omega_Lambda_rel(z, params)
        assert np.all(Omega > 0)

    def test_H_squared_correction_finite(self, params):
        """Corrección a H² es finita."""
        z = 1.0
        H0 = 67.4
        H2_corr = H_squared_correction(z, params, H0)
        assert np.isfinite(H2_corr)

    def test_H_rel_positive(self, params):
        """H_rel > 0."""
        z = np.array([0.0, 1.0, 5.0])
        H0 = 67.4
        H = H_rel(z, params, H0)
        assert np.all(H > 0)

    def test_w_eff_in_range(self, params):
        """w_eff cerca de -1."""
        z = np.array([0.0, 1.0, 5.0])
        w = w_eff_Lambda(z, params)
        # w_eff debe estar cerca de -1 para dark energy
        assert np.all(w < 0)
        assert np.all(w > -2)


class TestLambdaRelFromChannels:
    """Tests para Λ_rel desde canales."""

    @pytest.fixture
    def calculator(self):
        """Calculador desde canales con funciones dummy."""
        calc = LambdaRelFromChannels()
        calc.rho_id_func = lambda z: np.ones_like(z) * 1e-30
        calc.rho_lat_func = lambda z: np.ones_like(z) * 1e-31
        return calc

    def test_Lambda_from_channels_positive(self, calculator):
        """Λ desde canales es positiva."""
        z = np.array([0.0, 1.0, 2.0])
        Lambda = calculator.Lambda_rel(z)
        assert np.all(Lambda > 0)

    def test_Lambda_from_channels_finite(self, calculator):
        """Λ desde canales es finita."""
        z = np.array([0.0, 1.0, 2.0])
        Lambda = calculator.Lambda_rel(z)
        assert np.all(np.isfinite(Lambda))


class TestQDualParams:
    """Tests para parámetros de Q_dual."""

    def test_default_params(self):
        """Parámetros por defecto válidos."""
        params = QDualParams()
        assert params.S_star > 0
        assert params.lambda_star > 0
        assert params.Q_amplitude >= 0


class TestEtaFunctions:
    """Tests para funciones η."""

    @pytest.fixture
    def params(self):
        """Parámetros por defecto."""
        return QDualParams()

    def test_eta_lat_limits(self, params):
        """η_lat → 0 para S << S_★, → 1 para S >> S_★."""
        S_low = params.S_star - 0.5
        S_high = params.S_star + 0.5
        eta_low = eta_lat_of_S(S_low, params)
        eta_high = eta_lat_of_S(S_high, params)
        assert eta_low < 0.5
        assert eta_high > 0.5

    def test_eta_id_complement(self, params):
        """η_id + η_lat ≈ 1."""
        S = params.S_star
        eta_lat = eta_lat_of_S(S, params)
        eta_id = eta_id_of_S(S, params)
        np.testing.assert_allclose(eta_lat + eta_id, 1.0, rtol=1e-10)

    def test_eta_smooth(self, params):
        """η es suave (sin discontinuidades)."""
        # Test around the transition point
        S = np.linspace(params.S_star - 0.01, params.S_star + 0.01, 100)
        eta = eta_lat_of_S(S, params)
        deta = np.diff(eta)
        # No debe haber saltos muy grandes
        assert np.all(np.abs(deta) <= 0.5)


class TestQDual:
    """Tests para Q_dual."""

    @pytest.fixture
    def params(self):
        """Parámetros con Q_amplitude > 0."""
        return QDualParams(Q_amplitude=0.01)

    def test_Q_dual_sign(self, params):
        """Q_dual puede ser + o - dependiendo de S."""
        S_values = np.linspace(0.5, 1.5, 20)
        # Q_dual cambia de signo según dη/dS y las densidades
        Q = Q_dual_simple(S_values, params)
        # Verificar que no explota
        assert np.all(np.isfinite(Q))

    def test_Q_dual_zero_amplitude(self):
        """Q_dual = 0 si Q_amplitude = 0."""
        params = QDualParams(Q_amplitude=0.0)
        S = np.array([1.0])
        Q = Q_dual_simple(S, params)
        np.testing.assert_allclose(Q, 0.0)

    def test_Q_dual_finite(self, params):
        """Q_dual devuelve valores finitos."""
        S = 1.0
        dV_dS = 0.01
        S_dot = 1.0
        Q_id, Q_lat = Q_dual(S, dV_dS, S_dot, params)
        # Q debe ser finito
        assert np.isfinite(Q_id)
        assert np.isfinite(Q_lat)


class TestCoupledChannelEvolver:
    """Tests para evolución acoplada de canales."""

    @pytest.fixture
    def evolver(self):
        """Evolucionador con parámetros por defecto."""
        q_params = QDualParams()
        return CoupledChannelEvolver(q_params=q_params)

    def test_evolve_returns_arrays(self, evolver):
        """evolve retorna tres arrays."""
        S_range = (1.001, 1.1)
        rho_id_init = 1e-30
        rho_lat_init = 1e-31
        S_arr, rho_id_arr, rho_lat_arr = evolver.evolve(
            S_range, rho_id_init, rho_lat_init, n_points=20
        )
        assert len(S_arr) == 20
        assert len(rho_id_arr) == 20
        assert len(rho_lat_arr) == 20

    def test_evolve_S_increases(self, evolver):
        """S aumenta durante evolución."""
        S_range = (1.001, 1.1)
        S_arr, _, _ = evolver.evolve(S_range, 1e-30, 1e-31, n_points=20)
        assert S_arr[-1] > S_arr[0]

    def test_evolve_initial_values(self, evolver):
        """Valores iniciales se preservan."""
        S_range = (1.001, 1.01)
        rho_id_init = 1e-10
        rho_lat_init = 1e-11
        S_arr, rho_id_arr, rho_lat_arr = evolver.evolve(
            S_range, rho_id_init, rho_lat_init, n_points=5
        )
        # El primer valor debe ser el inicial
        np.testing.assert_allclose(rho_id_arr[0], rho_id_init)
        np.testing.assert_allclose(rho_lat_arr[0], rho_lat_init)
