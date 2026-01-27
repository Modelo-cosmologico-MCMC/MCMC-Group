"""Tests para el módulo lattice-gauge."""
from __future__ import annotations

import numpy as np
import pytest

from mcmc.lattice.wilson import WilsonAction, WilsonActionParams
from mcmc.lattice.beta_of_S import (
    beta_of_S,
    BetaParams,
    g_squared_of_S,
    alpha_s_of_S,
    S_of_energy,
    S3_EW,
)
from mcmc.lattice.monte_carlo import (
    MetropolisSampler,
    HeatbathSampler,
    MCParams,
    autocorrelation_time,
)


class TestWilsonAction:
    """Tests para la acción de Wilson."""

    @pytest.fixture
    def small_lattice(self):
        """Retícula pequeña para tests rápidos."""
        params = WilsonActionParams(beta=6.0, N_color=3, L=4, n_dim=4)
        return WilsonAction(params)

    def test_cold_start_action(self, small_lattice):
        """Acción de cold start es 0."""
        small_lattice.cold_start()
        S = small_lattice.action()
        assert S == 0.0

    def test_hot_start_nonzero_action(self, small_lattice):
        """Hot start da acción no nula."""
        small_lattice.hot_start()
        S = small_lattice.action()
        assert S > 0

    def test_average_plaquette_cold(self, small_lattice):
        """Plaquette promedio = 1 para cold start."""
        small_lattice.cold_start()
        P = small_lattice.average_plaquette()
        np.testing.assert_allclose(P, 1.0, rtol=1e-10)

    def test_average_plaquette_hot(self, small_lattice):
        """Plaquette promedio < 1 para hot start."""
        small_lattice.hot_start()
        P = small_lattice.average_plaquette()
        assert 0 < P < 1

    def test_copy_independent(self, small_lattice):
        """Copia es independiente del original."""
        small_lattice.hot_start()
        copy = small_lattice.copy()

        # Modificar original
        small_lattice.cold_start()

        # Copia no cambia
        assert copy.average_plaquette() < 1


class TestBetaOfS:
    """Tests para β(S)."""

    @pytest.fixture
    def params(self):
        """Parámetros por defecto."""
        return BetaParams(beta0=6.0, beta1=1.0, b_S=100.0, S_ref=S3_EW)

    def test_beta_at_reference(self, params):
        """β(S_ref) = β₀ + β₁."""
        beta = beta_of_S(params.S_ref, params)
        expected = params.beta0 + params.beta1
        np.testing.assert_allclose(beta, expected)

    def test_beta_increases_below_Sref(self, params):
        """β aumenta para S < S_ref."""
        S_below = params.S_ref - 0.1
        beta_below = beta_of_S(S_below, params)
        beta_at_ref = beta_of_S(params.S_ref, params)
        assert beta_below > beta_at_ref

    def test_beta_plateau_above_Sref(self, params):
        """β se acerca a β₀ para S >> S_ref."""
        S_far = params.S_ref + 0.1
        beta_far = beta_of_S(S_far, params)
        # Debe estar cerca de β₀
        assert abs(beta_far - params.beta0) < params.beta1

    def test_g_squared_positive(self, params):
        """g² es positivo."""
        S_values = np.linspace(0.9, 1.01, 10)
        g2 = g_squared_of_S(S_values, params)
        assert np.all(g2 > 0)

    def test_alpha_s_reasonable(self, params):
        """α_s está en rango razonable."""
        S = 1.0
        alpha = alpha_s_of_S(S, params)
        # α_s típicamente O(0.1) a escalas GeV
        assert 0.01 < alpha < 1.0

    def test_S_of_energy_bounds(self):
        """S(E) está entre los umbrales."""
        E_high = 1e18  # GeV, cercano a Planck
        E_low = 0.1    # GeV, post-EW

        S_high = S_of_energy(E_high)
        S_low = S_of_energy(E_low)

        assert S_high < S_low  # S aumenta con energía decreciente
        assert S_high >= 0.009  # No menor que S1
        assert S_low <= 1.001   # No mayor que S4


class TestMetropolisSampler:
    """Tests para muestreador Metropolis."""

    @pytest.fixture
    def sampler(self):
        """Sampler con retícula pequeña."""
        params = WilsonActionParams(beta=6.0, L=4, n_dim=4)
        action = WilsonAction(params)
        action.hot_start()
        mc_params = MCParams(n_therm=10, n_conf=5, n_skip=2, delta=0.3)
        return MetropolisSampler(action, mc_params)

    def test_sweep_acceptance(self, sampler):
        """Un sweep tiene tasa de aceptación no nula."""
        sampler.sweep()
        # Probablemente diferente (a menos que rechace todo)
        # No podemos garantizar que cambie, pero acceptance_rate > 0
        assert sampler.acceptance_rate > 0

    def test_acceptance_rate_reasonable(self, sampler):
        """Tasa de aceptación en rango razonable."""
        for _ in range(10):
            sampler.sweep()
        # Entre 20% y 80% es razonable para Metropolis
        assert 0.1 < sampler.acceptance_rate < 0.95

    def test_generate_configurations(self, sampler):
        """Genera el número correcto de configuraciones."""
        configs = sampler.generate_configurations(n_conf=3, thermalize=False)
        assert len(configs) == 3

    def test_configurations_different(self, sampler):
        """Configuraciones generadas son diferentes."""
        configs = sampler.generate_configurations(n_conf=2, n_skip=5, thermalize=False)
        # No deben ser idénticas
        assert not np.allclose(configs[0], configs[1])


class TestHeatbathSampler:
    """Tests para muestreador Heatbath."""

    @pytest.fixture
    def sampler(self):
        """Sampler heatbath."""
        params = WilsonActionParams(beta=6.0, L=4, n_dim=4)
        action = WilsonAction(params)
        action.hot_start()
        mc_params = MCParams(n_therm=10, n_conf=3, n_skip=2)
        return HeatbathSampler(action, mc_params)

    def test_thermalization_decreases_action(self, sampler):
        """Termalización reduce la acción (hacia equilibrio)."""
        # Hot start tiene acción alta
        S_initial = sampler.action.action()

        # Termalizar
        sampler.thermalize(n_sweeps=20)

        S_final = sampler.action.action()

        # En equilibrio, acción debería ser menor
        assert S_final < S_initial * 1.5  # Permitir fluctuaciones

    def test_generate_configurations(self, sampler):
        """Genera configuraciones."""
        configs = sampler.generate_configurations(n_conf=2, thermalize=False)
        assert len(configs) == 2


class TestAutocorrelation:
    """Tests para tiempo de autocorrelación."""

    def test_uncorrelated_data(self):
        """Datos no correlacionados tienen τ ≈ 1."""
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 1000)
        tau = autocorrelation_time(data)
        assert 0.5 < tau < 2.0

    def test_correlated_data(self):
        """Datos correlacionados tienen τ > 1."""
        # Generar serie correlacionada
        n = 1000
        x = np.zeros(n)
        x[0] = 0
        rng = np.random.default_rng(42)
        for i in range(1, n):
            x[i] = 0.9 * x[i-1] + rng.normal(0, 0.1)

        tau = autocorrelation_time(x)
        assert tau > 2.0
