"""
Tests de humo (smoke tests) para el likelihood.

Verifica que el sistema de likelihood funciona end-to-end
sin verificar valores exactos.
"""

import pytest
import numpy as np

from src.mcmc.observables.distances import DistanceCalculator, H_lcdm
from src.mcmc.observables.bao import get_boss_dr12_data
from src.mcmc.observables.hz import get_cosmic_chronometers_data
from src.mcmc.observables.sne import get_pantheon_binned_data
from src.mcmc.observables.likelihoods import (
    LikelihoodConfig,
    LikelihoodResult,
    CombinedLikelihood,
    log_prior_uniform,
)
from src.mcmc.observables.info_criteria import (
    compute_AIC,
    compute_BIC,
    compute_all_criteria,
    compare_models,
)


class TestLikelihoodConfig:
    """Tests para LikelihoodConfig."""

    def test_default_config(self):
        """Verifica configuración por defecto."""
        config = LikelihoodConfig()
        assert config.use_bao is True
        assert config.use_Hz is True
        assert config.use_sne is True

    def test_custom_config(self):
        """Verifica configuración personalizada."""
        config = LikelihoodConfig(use_bao=False, use_Hz=True, use_sne=False)
        assert config.use_bao is False
        assert config.use_Hz is True
        assert config.use_sne is False


class TestCombinedLikelihood:
    """Tests de humo para CombinedLikelihood."""

    def setup_method(self):
        """Setup para cada test."""
        self.H0 = 67.4
        self.Omega_m = 0.3

        def H_func(z):
            return H_lcdm(z, self.H0, self.Omega_m)

        self.H_func = H_func
        self.dist_calc = DistanceCalculator(H_func=H_func, H0=self.H0)

        # Cargar datos
        self.bao_data = get_boss_dr12_data()
        self.Hz_data = get_cosmic_chronometers_data()
        self.sne_data = get_pantheon_binned_data()

    def test_likelihood_creation(self):
        """Verifica creación del likelihood."""
        config = LikelihoodConfig()
        likelihood = CombinedLikelihood(
            config=config,
            bao_data=self.bao_data,
            Hz_data=self.Hz_data,
            sne_data=self.sne_data
        )

        assert likelihood.n_bao > 0
        assert likelihood.n_Hz > 0
        assert likelihood.n_sne > 0

    def test_chi2_bao_computes(self):
        """Verifica que χ² BAO se calcula sin error."""
        config = LikelihoodConfig(use_bao=True, use_Hz=False, use_sne=False)
        likelihood = CombinedLikelihood(
            config=config,
            bao_data=self.bao_data
        )

        chi2 = likelihood.compute_chi2_bao(self.dist_calc, self.H_func)
        assert np.isfinite(chi2)
        assert chi2 >= 0

    def test_chi2_Hz_computes(self):
        """Verifica que χ² H(z) se calcula sin error."""
        config = LikelihoodConfig(use_bao=False, use_Hz=True, use_sne=False)
        likelihood = CombinedLikelihood(
            config=config,
            Hz_data=self.Hz_data
        )

        chi2 = likelihood.compute_chi2_Hz(self.H_func)
        assert np.isfinite(chi2)
        assert chi2 >= 0

    def test_chi2_sne_computes(self):
        """Verifica que χ² SNe se calcula sin error."""
        config = LikelihoodConfig(use_bao=False, use_Hz=False, use_sne=True)
        likelihood = CombinedLikelihood(
            config=config,
            sne_data=self.sne_data
        )

        chi2 = likelihood.compute_chi2_sne(self.dist_calc)
        assert np.isfinite(chi2)
        assert chi2 >= 0

    def test_full_likelihood_computes(self):
        """Verifica que el likelihood completo se calcula."""
        config = LikelihoodConfig()
        likelihood = CombinedLikelihood(
            config=config,
            bao_data=self.bao_data,
            Hz_data=self.Hz_data,
            sne_data=self.sne_data
        )

        result = likelihood(self.dist_calc, self.H_func)

        assert isinstance(result, LikelihoodResult)
        assert np.isfinite(result.chi2_total)
        assert np.isfinite(result.log_likelihood)
        assert result.chi2_total >= 0

    def test_log_likelihood_is_negative(self):
        """Verifica que log(L) = -χ²/2 < 0 para χ² > 0."""
        config = LikelihoodConfig()
        likelihood = CombinedLikelihood(
            config=config,
            bao_data=self.bao_data,
            Hz_data=self.Hz_data,
            sne_data=self.sne_data
        )

        ll = likelihood.log_likelihood(self.dist_calc, self.H_func)

        # log(L) = -χ²/2, así que si χ² > 0, log(L) < 0
        assert ll < 0 or np.isclose(ll, 0)


class TestPriors:
    """Tests para funciones de prior."""

    def test_uniform_prior_inside_bounds(self):
        """Verifica prior uniforme dentro de límites."""
        params = np.array([70, 0.3])
        bounds = [(60, 80), (0.1, 0.5)]

        lp = log_prior_uniform(params, bounds)
        assert lp == 0.0

    def test_uniform_prior_outside_bounds(self):
        """Verifica prior uniforme fuera de límites."""
        params = np.array([50, 0.3])  # H0 fuera de límites
        bounds = [(60, 80), (0.1, 0.5)]

        lp = log_prior_uniform(params, bounds)
        assert lp == -np.inf


class TestInformationCriteria:
    """Tests para criterios de información."""

    def test_AIC_formula(self):
        """Verifica fórmula de AIC."""
        chi2 = 100
        n_params = 5
        AIC = compute_AIC(chi2, n_params)

        expected = chi2 + 2 * n_params
        assert AIC == expected

    def test_BIC_formula(self):
        """Verifica fórmula de BIC."""
        chi2 = 100
        n_params = 5
        n_data = 1000

        BIC = compute_BIC(chi2, n_params, n_data)
        expected = chi2 + n_params * np.log(n_data)

        assert np.isclose(BIC, expected)

    def test_compute_all_criteria(self):
        """Verifica cálculo de todos los criterios."""
        chi2 = 100
        n_params = 5
        n_data = 1000

        result = compute_all_criteria(chi2, n_params, n_data)

        assert result.chi2 == chi2
        assert result.n_params == n_params
        assert result.n_data == n_data
        assert np.isfinite(result.AIC)
        assert np.isfinite(result.BIC)
        assert np.isfinite(result.AICc)

    def test_compare_models(self):
        """Verifica comparación de modelos."""
        model_criteria = compute_all_criteria(chi2=100, n_params=5, n_data=100)
        ref_criteria = compute_all_criteria(chi2=105, n_params=2, n_data=100)

        comparison = compare_models(model_criteria, ref_criteria)

        assert 'interpretation' in comparison.__dict__
        assert np.isfinite(comparison.delta_AIC)
        assert np.isfinite(comparison.delta_BIC)


class TestSmokeEndToEnd:
    """Test de humo end-to-end."""

    def test_full_pipeline_runs(self):
        """Verifica que todo el pipeline corre sin errores."""
        # 1. Crear modelo
        H0 = 67.4
        Omega_m = 0.3

        def H_func(z):
            return H_lcdm(z, H0, Omega_m)

        dist_calc = DistanceCalculator(H_func=H_func, H0=H0)

        # 2. Cargar datos
        bao_data = get_boss_dr12_data()
        Hz_data = get_cosmic_chronometers_data()
        sne_data = get_pantheon_binned_data()

        # 3. Crear likelihood
        config = LikelihoodConfig()
        likelihood = CombinedLikelihood(
            config=config,
            bao_data=bao_data,
            Hz_data=Hz_data,
            sne_data=sne_data
        )

        # 4. Evaluar
        result = likelihood(dist_calc, H_func)

        # 5. Calcular criterios
        criteria = compute_all_criteria(
            chi2=result.chi2_total,
            n_params=2,  # ΛCDM: H0, Omega_m
            n_data=result.n_points_total
        )

        # 6. Verificar que todo es finito y razonable
        assert np.isfinite(result.chi2_total)
        assert np.isfinite(criteria.AIC)
        assert np.isfinite(criteria.BIC)

        # χ² reducido debería estar en un rango razonable
        chi2_red = result.chi2_total / result.n_points_total
        assert 0.1 < chi2_red < 10, f"χ² reducido = {chi2_red} fuera de rango razonable"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
