"""Recuperacion de parametros sobre datos sinteticos.

Parametrizacion del ajuste
--------------------------
El fondo efectivo normaliza H(0) = H0, de modo que solo la razon
rho_b0 : rho0 es observable: una reescala conjunta (rho_b0, rho0) ->
(k rho_b0, k rho0) deja H(z) exactamente invariante. Ajustar ambas
amplitudes seria una degeneracion perfecta (deteccion documentada de la
Prioridad 2). Por ello el vector de ajuste usa la fraccion de materia

    f_m = rho_b0 / (rho_b0 + rho0),  con rho_b0 = f_m, rho0 = 1 - f_m.

theta = [H0, f_m, z_trans, eps, rd, M]
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from mcmc.cobaya_interface.background import BackgroundCalculator, BackgroundParams
from mcmc.observables.likelihoods import loglike_total
from mcmc.inference.emcee_fit import run_emcee
from mcmc.inference.postprocess import summarize_chain

PARAM_NAMES = ["H0", "f_m", "z_trans", "eps", "rd", "M"]

# Priors uniformes del ajuste sintetico (consistentes con config/priors.yaml)
BOUNDS = {
    "H0": (50.0, 90.0),
    "f_m": (0.01, 0.99),
    "z_trans": (0.0, 5.0),
    "eps": (0.0, 1.0),
    "rd": (80.0, 220.0),
    "M": (-20.5, -18.0),
}


def theta_from_params(p: BackgroundParams) -> np.ndarray:
    """Convierte BackgroundParams al vector de ajuste (normalizado)."""
    total = p.rho_b0 + p.rho0
    if total <= 0:
        raise ValueError("rho_b0 + rho0 debe ser > 0")
    return np.array([p.H0, p.rho_b0 / total, p.z_trans, p.eps, p.rd, p.M])


def params_from_theta(theta: np.ndarray) -> BackgroundParams:
    H0, f_m, z_trans, eps, rd, M = [float(v) for v in theta]
    return BackgroundParams(
        H0=H0, rho_b0=f_m, rho0=1.0 - f_m, z_trans=z_trans, eps=eps, rd=rd, M=M
    )


def build_model(theta: np.ndarray, *, z_max: float = 5.0) -> dict:
    """Modelo legacy {H(z), mu(z), DVrd(z)} desde el vector de ajuste."""
    calc = BackgroundCalculator(params_from_theta(theta), z_max=z_max)
    return {
        "H(z)": calc.H,
        "mu(z)": calc.mu,
        "DVrd(z)": calc.DV_rd,
    }


def log_prior(theta: np.ndarray) -> float:
    for value, name in zip(theta, PARAM_NAMES):
        lo, hi = BOUNDS[name]
        if not (lo <= value <= hi):
            return -np.inf
    return 0.0


def make_log_prob(datasets: dict):
    """Construye log-posterior sobre los datasets sinteticos."""

    def log_prob(theta: np.ndarray) -> float:
        lp = log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        try:
            model = build_model(theta)
        except ValueError:
            return -np.inf
        ll = loglike_total(datasets, model)
        if not np.isfinite(ll):
            return -np.inf
        return lp + float(ll)

    return log_prob


def fit_map(datasets: dict, x0: np.ndarray) -> np.ndarray:
    """Maximo a posteriori por Nelder-Mead (rapido, sin cadenas)."""
    log_prob = make_log_prob(datasets)

    def neg(theta):
        val = log_prob(theta)
        return -val if np.isfinite(val) else 1e12

    res = minimize(neg, x0, method="Nelder-Mead",
                   options={"maxiter": 8000, "xatol": 1e-6, "fatol": 1e-8})
    return np.asarray(res.x, float)


@dataclass(frozen=True)
class RecoveryResult:
    """Resultado del test de recuperacion sintetica."""
    truth: dict
    map_estimate: dict
    p16: dict
    p50: dict
    p84: dict
    pull: dict            # (p50 - truth) / (0.5 * (p84 - p16))
    within_2sigma: dict
    n_steps: int
    n_walkers: int
    seed: int

    def all_recovered(self) -> bool:
        return all(self.within_2sigma.values())

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.__dict__, indent=2, default=float), encoding="utf-8"
        )


def run_recovery(
    datasets: dict,
    truth: BackgroundParams,
    *,
    nwalkers: int = 32,
    nsteps: int = 1500,
    seed: int = 42,
    burn_frac: float = 0.3,
) -> RecoveryResult:
    """Pipeline completo de recuperacion: MAP -> emcee -> resumen y pulls."""
    theta_true = theta_from_params(truth)
    theta_map = fit_map(datasets, theta_true)

    log_prob = make_log_prob(datasets)
    sampler = run_emcee(log_prob, theta_map, nwalkers=nwalkers, nsteps=nsteps, seed=seed)
    chain = sampler.get_chain()

    burn = max(10, int(nsteps * burn_frac))
    summary = summarize_chain(chain, burn=burn, thin=2)

    def as_dict(values):
        return {n: float(v) for n, v in zip(PARAM_NAMES, values)}

    p16, p50, p84 = summary["p16"], summary["p50"], summary["p84"]
    pull, within = {}, {}
    for i, name in enumerate(PARAM_NAMES):
        half_width = 0.5 * (p84[i] - p16[i])
        if half_width <= 0:
            half_width = np.finfo(float).eps
        pull[name] = float((p50[i] - theta_true[i]) / half_width)
        within[name] = bool(abs(pull[name]) <= 2.0)

    return RecoveryResult(
        truth=as_dict(theta_true),
        map_estimate=as_dict(theta_map),
        p16=as_dict(p16),
        p50=as_dict(p50),
        p84=as_dict(p84),
        pull=pull,
        within_2sigma=within,
        n_steps=nsteps,
        n_walkers=nwalkers,
        seed=seed,
    )


def s_map_roundtrip_check(z_max: float = 3.0, n: int = 200, atol: float = 1e-8) -> dict:
    """Comprueba la consistencia del mapeo entropico S <-> z.

    Verifica que z -> S(z) -> z recupera z (deteccion de errores del
    mapeo, exigida por la Prioridad 2) y que S(z) es monotona decreciente.
    """
    from mcmc.ontology.s_map import EntropyMap

    s_map = EntropyMap()
    z = np.linspace(0.0, z_max, n)
    S = s_map.S_of_z(z)
    z_back = s_map.z_of_S(S)
    max_err = float(np.max(np.abs(z_back - z)))
    monotonic = bool(np.all(np.diff(S) < 0))
    return {
        "max_roundtrip_error": max_err,
        "monotonic_decreasing": monotonic,
        "passed": bool(max_err < atol and monotonic),
    }
