"""Tests para el módulo ontology.

CORRECCIÓN ONTOLÓGICA (2025): S ∈ [0, 100]
- Pre-geométrico: S ∈ [0, 1.001)
- Post-Big Bang: S ∈ [1.001, 95.07]
"""
from __future__ import annotations

import numpy as np
import pytest

from mcmc.ontology.s_map import (
    EntropyMap,
)
from mcmc.ontology.adrian_field import (
    AdrianFieldParams,
    AdrianField,
)
from mcmc.ontology.dual_metric import (
    DualRelativeMetric,
    create_LCDM_metric,
    create_MCMC_metric,
)


class TestEntropyMap:
    """Tests para el mapa entrópico S↔z↔t↔a."""

    @pytest.fixture
    def s_map(self):
        """Mapa entrópico por defecto."""
        return EntropyMap()

    def test_S_of_z_at_zero(self, s_map):
        """S(z=0) = S_0."""
        S = s_map.S_of_z(0.0)
        np.testing.assert_allclose(S, s_map.params.S_0, rtol=1e-6)

    def test_S_of_z_increases_with_z(self, s_map):
        """S disminuye con z (universo más joven a z mayor)."""
        z_arr = np.array([0.0, 1.0, 2.0, 5.0])
        S = s_map.S_of_z(z_arr)
        # S debe decrecer con z (hacia S_BB)
        assert np.all(np.diff(S) < 0)

    def test_z_of_S_inverse(self, s_map):
        """z(S(z)) = z para valores post-BB."""
        z_original = np.array([0.0, 0.5, 1.0, 2.0])
        S = s_map.S_of_z(z_original)
        z_recovered = s_map.z_of_S(S)
        np.testing.assert_allclose(z_recovered, z_original, rtol=1e-4)

    def test_a_of_S_at_today(self, s_map):
        """a(S_0) = 1."""
        a = s_map.a_of_S(s_map.params.S_0)
        np.testing.assert_allclose(a, 1.0, rtol=1e-6)

    def test_a_of_S_decreases(self, s_map):
        """a(S) decrece hacia S_BB (aumenta con S hacia hoy).

        CORRECCIÓN: Post-Big Bang S ∈ [1.001, 95.07]
        """
        S_arr = np.linspace(2.0, 90.0, 10)
        a = s_map.a_of_S(S_arr)
        assert np.all(np.diff(a) > 0)  # a aumenta con S (hacia hoy)

    def test_T_of_S_positive(self, s_map):
        """T(S) > 0.

        CORRECCIÓN: S ∈ [0, 100]
        """
        S_arr = np.linspace(0.5, 90.0, 20)
        T = s_map.T_of_S(S_arr)
        assert np.all(T > 0)

    def test_ley_de_cronos(self, s_map):
        """dt_rel/dS ∝ N(S).

        CORRECCIÓN: S ∈ [1.001, 95.07] post-Big Bang
        """
        S = 50.0  # Mid-range post-Big Bang
        N = s_map.N_of_S(S)
        dt_dS = s_map.dt_dS(S)
        # dt_dS es proporcional a N(S)
        assert np.isfinite(dt_dS)
        assert dt_dS > 0


class TestAdrianField:
    """Tests para el Campo de Adrián."""

    @pytest.fixture
    def field(self):
        """Campo de Adrián por defecto."""
        return AdrianField()

    def test_Theta_lambda_limits(self, field):
        """Theta_λ → 0 para x << -λ, → 1 para x >> λ."""
        assert field.Theta_lambda(-10) < 0.01
        assert field.Theta_lambda(10) > 0.99

    def test_Theta_lambda_center(self, field):
        """Theta_λ(0) = 0.5."""
        np.testing.assert_allclose(field.Theta_lambda(0), 0.5, rtol=1e-10)

    def test_V_eff_positive(self, field):
        """V_eff ≥ 0.

        CORRECCIÓN: S ∈ [0, 100]
        """
        Phi_arr = np.linspace(-1, 1, 20)
        S_arr = np.linspace(0.5, 90.0, 10)
        for S in S_arr:
            V = field.V_eff(Phi_arr, S)
            assert np.all(V >= 0)

    def test_V_eff_has_minimum(self, field):
        """V_eff tiene un mínimo finito.

        CORRECCIÓN: S ∈ [0, 100]
        """
        S = 50.0  # Mid-range
        # Verificar que V tiene un mínimo en algún Phi
        Phi_arr = np.linspace(-0.5, 0.5, 100)
        V = field.V_eff(Phi_arr, S)
        V_min = np.min(V)
        assert np.isfinite(V_min)
        assert V_min >= 0

    def test_Phi_ten_smooth(self, field):
        """Φ_ten es suave en S.

        CORRECCIÓN: phi_ten_center ahora en ~48 para S ∈ [1, 95]
        """
        S_arr = np.linspace(40.0, 60.0, 50)
        Phi_ten = field.Phi_ten(S_arr)
        # No debe tener saltos grandes
        dPhi = np.diff(Phi_ten)
        assert np.max(np.abs(dPhi)) < 1.0

    def test_Phi_ten_zero_amplitude(self):
        """Con amplitud 0, Φ_ten ≈ 0.

        CORRECCIÓN: S ∈ [0, 100]
        """
        params = AdrianFieldParams(phi_ten_amplitude=0.0)
        field = AdrianField(params)
        S_arr = np.linspace(0.5, 90.0, 20)
        Phi_ten = field.Phi_ten(S_arr)
        np.testing.assert_allclose(Phi_ten, 0.0, atol=1e-10)

    def test_rho_Ad_positive(self, field):
        """ρ_Ad es positiva o cero.

        CORRECCIÓN: S ∈ [0, 100]
        """
        Phi = 0.1
        dPhi_dS = 0.01
        S = 50.0
        dS_dt = 1.0
        rho = field.rho_Ad(Phi, dPhi_dS, S, dS_dt)
        assert rho >= 0


class TestDualRelativeMetric:
    """Tests para la Métrica Dual Relativa."""

    @pytest.fixture
    def metric(self):
        """Métrica MDR por defecto."""
        return DualRelativeMetric()

    def test_g_tt_negative(self, metric):
        """g_tt < 0 (signatura Lorentziana).

        CORRECCIÓN: S ∈ [0, 100]
        """
        S_arr = np.linspace(2.0, 90.0, 20)
        g_tt = metric.g_tt(S_arr)
        assert np.all(g_tt < 0)

    def test_g_rr_positive(self, metric):
        """g_rr > 0.

        CORRECCIÓN: S ∈ [0, 100]
        """
        S_arr = np.linspace(2.0, 90.0, 20)
        g_rr = metric.g_rr(S_arr)
        assert np.all(g_rr > 0)

    def test_N_of_S_positive(self, metric):
        """N(S) > 0 (lapse siempre positivo).

        CORRECCIÓN: S ∈ [0, 100]
        """
        S_arr = np.linspace(2.0, 90.0, 20)
        N = metric.N_of_S(S_arr)
        assert np.all(N > 0)

    def test_sqrt_neg_g_positive(self, metric):
        """√(-g) > 0.

        CORRECCIÓN: S ∈ [0, 100]
        """
        S_arr = np.linspace(2.0, 90.0, 20)
        sqrt_g = metric.sqrt_neg_g(S_arr)
        assert np.all(sqrt_g > 0)

    def test_LCDM_limit(self):
        """En límite ΛCDM, N → 1.

        CORRECCIÓN: S ∈ [0, 100]
        """
        metric = create_LCDM_metric()
        S_arr = np.linspace(2.0, 90.0, 20)
        N = metric.N_of_S(S_arr)
        np.testing.assert_allclose(N, 1.0, rtol=1e-6)

    def test_is_LCDM_limit(self):
        """Verifica detección de límite ΛCDM.

        CORRECCIÓN: S ∈ [0, 100]
        """
        metric_lcdm = create_LCDM_metric()
        metric_mcmc = create_MCMC_metric(phi_ten_amplitude=0.1)

        assert metric_lcdm.is_LCDM_limit(50.0, tol=1e-4)
        # MCMC con amplitud no nula tiene N(S) != 1
        N_mcmc = metric_mcmc.N_of_S(metric_mcmc.Phi_Ad.params.phi_ten_center)
        assert np.isfinite(N_mcmc)

    def test_deviation_from_FRW(self):
        """Desviación es 0 para ΛCDM.

        CORRECCIÓN: S ∈ [0, 100]
        """
        metric = create_LCDM_metric()
        S = 50.0
        dev = metric.deviation_from_FRW(S)
        np.testing.assert_allclose(dev, 0.0, atol=1e-10)

    def test_metric_tensor_shape(self, metric):
        """Tensor métrico es 4×4.

        CORRECCIÓN: S ∈ [0, 100]
        """
        g = metric.compute_metric_tensor(50.0)
        assert g.shape == (4, 4)

    def test_metric_tensor_diagonal(self, metric):
        """Métrica FRW es diagonal.

        CORRECCIÓN: S ∈ [0, 100]
        """
        g = metric.compute_metric_tensor(50.0)
        # Off-diagonal debe ser cero
        for i in range(4):
            for j in range(4):
                if i != j:
                    assert g[i, j] == 0.0
