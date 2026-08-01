"""Tests del ajuste sintetico (Prioridad 2).

La version CI usa MAP (Nelder-Mead) y una cadena emcee corta para mantener
el tiempo de ejecucion bajo; el script scripts/run_synthetic_recovery.py
ejecuta la version completa.
"""
import json

import numpy as np
import pytest

from mcmc.cobaya_interface.background import BackgroundParams
from mcmc.validation.synthetic import SyntheticSpec, generate_synthetic_datasets
from mcmc.validation.recovery import (
    PARAM_NAMES,
    fit_map,
    make_log_prob,
    params_from_theta,
    run_recovery,
    s_map_roundtrip_check,
    theta_from_params,
)

TRUTH = BackgroundParams(
    H0=67.4, rho_b0=0.30, rho0=0.70, z_trans=1.0, eps=0.05, rd=147.0, M=-19.3
)


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    outdir = tmp_path_factory.mktemp("synthetic")
    spec = SyntheticSpec(truth=TRUTH, seed=20260801)
    return generate_synthetic_datasets(spec, outdir=outdir)


class TestGeneration:
    def test_files_and_truth_written(self, synth, tmp_path_factory):
        for key in ("hz", "sne", "bao"):
            assert key in synth
            assert len(synth[key]["z"]) > 0
            assert np.isfinite(synth[key]["sigma"]).all()
            assert np.all(synth[key]["sigma"] > 0)

    def test_hashes_recorded(self, synth):
        for key in ("hz", "sne", "bao"):
            assert len(synth["hashes"][key]["sha256"]) == 64

    def test_reproducible_with_same_seed(self, tmp_path):
        spec = SyntheticSpec(truth=TRUTH, seed=123)
        a = generate_synthetic_datasets(spec, outdir=tmp_path / "a")
        b = generate_synthetic_datasets(spec, outdir=tmp_path / "b")
        np.testing.assert_array_equal(a["hz"]["H"], b["hz"]["H"])
        np.testing.assert_array_equal(a["sne"]["mu"], b["sne"]["mu"])
        np.testing.assert_array_equal(a["bao"]["dv_rd"], b["bao"]["dv_rd"])

    def test_truth_json_matches(self, synth):
        truth_path = synth["hashes"]["hz"]["path"].replace("hz.csv", "truth.json")
        payload = json.loads(open(truth_path, encoding="utf-8").read())
        assert payload["truth"]["H0"] == TRUTH.H0
        assert payload["seed"] == 20260801


class TestThetaMapping:
    def test_roundtrip(self):
        theta = theta_from_params(TRUTH)
        p = params_from_theta(theta)
        # La normalizacion absoluta no es observable: se compara la fraccion
        assert p.rho_b0 / (p.rho_b0 + p.rho0) == pytest.approx(
            TRUTH.rho_b0 / (TRUTH.rho_b0 + TRUTH.rho0), rel=1e-12
        )
        assert p.H0 == TRUTH.H0 and p.rd == TRUTH.rd and p.M == TRUTH.M


class TestMAPRecovery:
    def test_map_recovers_truth(self, synth):
        datasets = {k: synth[k] for k in ("hz", "sne", "bao")}
        theta_true = theta_from_params(TRUTH)
        # Arranque perturbado: el MAP debe volver cerca de la verdad
        x0 = theta_true * np.array([1.03, 0.9, 1.2, 1.5, 0.97, 1.01])
        theta_map = fit_map(datasets, x0)

        # Tolerancias acordes al nivel de ruido inyectado
        tol = {"H0": 0.02, "f_m": 0.10, "z_trans": 0.35, "eps": 1.0,
               "rd": 0.03, "M": 0.01}
        for i, name in enumerate(PARAM_NAMES):
            scale = max(abs(theta_true[i]), 0.05)
            rel_err = abs(theta_map[i] - theta_true[i]) / scale
            assert rel_err < tol[name], (
                f"{name}: MAP={theta_map[i]:.4f} verdad={theta_true[i]:.4f}"
            )

    def test_logprob_finite_at_truth(self, synth):
        datasets = {k: synth[k] for k in ("hz", "sne", "bao")}
        lp = make_log_prob(datasets)(theta_from_params(TRUTH))
        assert np.isfinite(lp)

    def test_logprob_rejects_out_of_prior(self, synth):
        datasets = {k: synth[k] for k in ("hz", "sne", "bao")}
        theta = theta_from_params(TRUTH)
        theta[0] = 200.0  # H0 fuera del prior
        assert make_log_prob(datasets)(theta) == -np.inf


class TestShortChainRecovery:
    def test_truth_within_posterior(self, synth):
        # Cadena corta (CI): verifica que la verdad cae dentro de 2 sigma
        datasets = {k: synth[k] for k in ("hz", "sne", "bao")}
        result = run_recovery(datasets, TRUTH, nwalkers=16, nsteps=400, seed=1)
        assert result.all_recovered(), (
            f"Parametros fuera de 2 sigma: "
            f"{[n for n, ok in result.within_2sigma.items() if not ok]} "
            f"pulls={result.pull}"
        )


class TestSMapRoundtrip:
    def test_s_z_mapping_consistent(self):
        check = s_map_roundtrip_check()
        assert check["monotonic_decreasing"]
        assert check["passed"], f"S<->z round-trip: {check}"
