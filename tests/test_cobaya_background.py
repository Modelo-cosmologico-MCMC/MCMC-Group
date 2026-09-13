"""Tests del componente de fondo estilo Cobaya (Prioridad 1).

Verifica:
- Limite LCDM exacto contra una implementacion independiente (quad).
- Consistencia interna de las distancias (D_H, D_L, D_A, D_V, mu).
- Invariantes de estabilidad: H^2 > 0, H(0)=H0, D_C monotona.
- w_eff_id = -1 exacto en el limite LCDM.
- Consistencia con el backend 'effective' existente.
- Degeneracion de reescala conjunta rho_b0/rho0 (documentada).
"""
import numpy as np
import pytest

from mcmc.cobaya_interface.background import (
    BackgroundCalculator,
    BackgroundParams,
    C_LIGHT,
)
from mcmc.cobaya_interface.lcdm_reference import LCDMReference

Z_TEST = np.array([0.05, 0.1, 0.32, 0.51, 0.7, 1.0, 1.5, 2.33, 3.0])


@pytest.fixture(scope="module")
def lcdm_pair():
    H0, Om = 67.4, 0.315
    calc = BackgroundCalculator.lcdm(H0=H0, Omega_m=Om, n_grid=8001)
    ref = LCDMReference(H0=H0, Omega_m=Om)
    return calc, ref


class TestLCDMLimit:
    def test_H_matches_analytic(self, lcdm_pair):
        calc, ref = lcdm_pair
        np.testing.assert_allclose(calc.H(Z_TEST), ref.H(Z_TEST), rtol=1e-8)

    def test_E_normalized_at_zero(self, lcdm_pair):
        calc, _ = lcdm_pair
        assert calc.E(0.0)[0] == pytest.approx(1.0, abs=1e-12)

    def test_DM_matches_quad(self, lcdm_pair):
        calc, ref = lcdm_pair
        np.testing.assert_allclose(calc.D_M(Z_TEST), ref.D_M(Z_TEST), rtol=2e-6)

    def test_DL_matches_quad(self, lcdm_pair):
        calc, ref = lcdm_pair
        np.testing.assert_allclose(calc.D_L(Z_TEST), ref.D_L(Z_TEST), rtol=2e-6)

    def test_DV_matches_quad(self, lcdm_pair):
        calc, ref = lcdm_pair
        np.testing.assert_allclose(calc.D_V(Z_TEST), ref.D_V(Z_TEST), rtol=2e-6)

    def test_mu_matches_quad(self, lcdm_pair):
        calc, ref = lcdm_pair
        np.testing.assert_allclose(calc.mu(Z_TEST), ref.mu(Z_TEST), atol=1e-5)

    def test_w_eff_is_minus_one(self, lcdm_pair):
        calc, _ = lcdm_pair
        w = calc.w_eff_id(np.linspace(0.0, 3.0, 50))
        np.testing.assert_allclose(w, -1.0, atol=1e-12)


@pytest.fixture(scope="module")
def calc():
    return BackgroundCalculator(BackgroundParams())


class TestInternalConsistency:

    def test_DH_is_c_over_H(self, calc):
        np.testing.assert_allclose(
            calc.D_H(Z_TEST), C_LIGHT / calc.H(Z_TEST), rtol=1e-12
        )

    def test_DL_DA_relation(self, calc):
        # Reciprocidad de Etherington: D_L = (1+z)^2 D_A
        np.testing.assert_allclose(
            calc.D_L(Z_TEST), (1.0 + Z_TEST) ** 2 * calc.D_A(Z_TEST), rtol=1e-12
        )

    def test_DV_definition(self, calc):
        dv_manual = (
            calc.D_M(Z_TEST) ** 2 * C_LIGHT * Z_TEST / calc.H(Z_TEST)
        ) ** (1.0 / 3.0)
        np.testing.assert_allclose(calc.D_V(Z_TEST), dv_manual, rtol=1e-12)

    def test_bao_ratios(self, calc):
        rd = calc.params.rd
        np.testing.assert_allclose(calc.DV_rd(Z_TEST), calc.D_V(Z_TEST) / rd)
        np.testing.assert_allclose(calc.DM_rd(Z_TEST), calc.D_M(Z_TEST) / rd)
        np.testing.assert_allclose(calc.DH_rd(Z_TEST), calc.D_H(Z_TEST) / rd)

    def test_table_keys(self, calc):
        table = calc.table(Z_TEST)
        for key in ("z", "H", "E", "D_C", "D_M", "D_H", "D_A", "D_L", "D_V",
                    "mu", "DV_rd", "w_eff_id"):
            assert key in table
            assert np.isfinite(table[key]).all()


class TestStability:
    def test_H_positive_across_priors(self):
        # Barrido por el rango de priors: H^2(z) > 0 debe sostenerse
        rng = np.random.default_rng(7)
        for _ in range(50):
            p = BackgroundParams(
                H0=rng.uniform(50, 90),
                rho_b0=rng.uniform(0.01, 5.0),
                rho0=rng.uniform(0.01, 5.0),
                z_trans=rng.uniform(0.0, 5.0),
                eps=rng.uniform(0.0, 1.0),
            )
            calc = BackgroundCalculator(p, n_grid=501)
            assert np.all(calc._H_grid > 0)

    def test_H0_normalization_exact(self):
        calc = BackgroundCalculator(BackgroundParams(H0=71.2))
        assert calc.H(0.0)[0] == pytest.approx(71.2, rel=1e-10)

    def test_rejects_negative_z(self):
        calc = BackgroundCalculator(BackgroundParams())
        with pytest.raises(ValueError, match="post-BB"):
            calc.H(-0.1)

    def test_rejects_z_beyond_grid(self):
        calc = BackgroundCalculator(BackgroundParams(), z_max=2.0)
        with pytest.raises(ValueError, match="fuera de la malla"):
            calc.D_L(2.5)


class TestConsistencyWithEffectiveBackend:
    def test_H_matches_effective_model(self):
        # El calculador debe reproducir exactamente el backend 'effective'
        from mcmc.models.effective_model import EffectiveModelConfig, build_effective_model

        p = BackgroundParams()
        calc = BackgroundCalculator(p)
        model = build_effective_model(
            EffectiveModelConfig(
                H0=p.H0, rho_b0=p.rho_b0, rho0=p.rho0,
                z_trans=p.z_trans, eps=p.eps, rd=p.rd, M=p.M,
            )
        )
        np.testing.assert_allclose(calc.H(Z_TEST), model["H(z)"](Z_TEST), rtol=1e-10)


class TestKnownDegeneracy:
    def test_joint_rescaling_leaves_H_invariant(self):
        # Degeneracion documentada: (rho_b0, rho0) -> (k rho_b0, k rho0)
        # deja H(z) invariante por la normalizacion H(0) = H0.
        base = BackgroundCalculator(BackgroundParams(rho_b0=0.3, rho0=0.7))
        scaled = BackgroundCalculator(BackgroundParams(rho_b0=1.2, rho0=2.8))
        np.testing.assert_allclose(base.H(Z_TEST), scaled.H(Z_TEST), rtol=1e-12)
