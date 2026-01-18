from __future__ import annotations

import argparse
import yaml
import numpy as np
import pandas as pd

from mcmc.core.s_grid import create_default_grid
from mcmc.core.background import BackgroundParams, solve_background
from mcmc.core.mapping import make_interpolators
from mcmc.core.checks import assert_background_ok

from mcmc.channels.rho_id_parametric import RhoIDParams
from mcmc.observables.distances import distance_modulus
from mcmc.observables.likelihoods import loglike_total
from mcmc.inference.emcee_fit import run_emcee
from mcmc.inference.postprocess import summarize_chain


def load_demo_csv(path: str, kind: str) -> dict:
    df = pd.read_csv(path)
    if kind == "hz":
        return {"z": df["z"].to_numpy(), "H": df["H"].to_numpy(), "sigma": df["sigma"].to_numpy()}
    if kind == "sne":
        return {"z": df["z"].to_numpy(), "mu": df["mu"].to_numpy(), "sigma": df["sigma"].to_numpy()}
    if kind == "bao":
        return {"z": df["z"].to_numpy(), "dv_rd": df["dv_rd"].to_numpy(), "sigma": df["sigma"].to_numpy()}
    raise ValueError("kind desconocido")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="src/mcmc/config/defaults.yaml")
    ap.add_argument("--nsteps", type=int, default=None)
    ap.add_argument("--nwalkers", type=int, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))

    # --- Fondo
    bcfg = cfg["background"]
    bp = BackgroundParams(**bcfg)

    grid, S = create_default_grid()
    sol = solve_background(S, bp)
    assert_background_ok(sol)

    # Interpolador H(z) desde la tabla del fondo
    # Nota: sol["z"] crece cuando a decrece; aseguramos monotonicidad en interpolacion
    z_arr = sol["z"]
    H_arr = sol["H"]
    H_of_z = make_interpolators(z_arr, H_arr)

    # --- Datos demo
    datasets = {
        "hz": load_demo_csv(cfg["data"]["hz"], "hz"),
        "sne": load_demo_csv(cfg["data"]["sne"], "sne"),
        "bao": load_demo_csv(cfg["data"]["bao"], "bao"),
    }

    # Modelo observable minimo:
    # - H(z) viene del fondo (MVP)
    # - mu(z) usa distancias integrando 1/H
    # - DV/rd: proxy simple con H(z) (solo para demo; sustituye por DV real)
    def H_model(z):
        return np.asarray(H_of_z(z), float)

    def mu_model(z):
        return distance_modulus(np.asarray(z, float), H_model(z))

    def dvrd_model(z):
        z = np.asarray(z, float)
        # Proxy demo: 1/H(z) reescalado (no fisico; evita dependencias externas)
        return 100.0 / np.maximum(H_model(z), 1e-9)

    # --- Priors MVP
    pri = yaml.safe_load(open("src/mcmc/config/priors.yaml", "r", encoding="utf-8"))

    def logprior(theta):
        H0, rho0, zt, eps = theta
        if not (pri["H0"][0] <= H0 <= pri["H0"][1]):
            return -np.inf
        if not (pri["rho0"][0] <= rho0 <= pri["rho0"][1]):
            return -np.inf
        if not (pri["z_trans"][0] <= zt <= pri["z_trans"][1]):
            return -np.inf
        if not (pri["eps"][0] <= eps <= pri["eps"][1]):
            return -np.inf
        return 0.0

    # En el MVP, solo ajustamos H0 y parametros rho_id, pero no reinyectamos rho_id aun en H(z).
    # Esto es deliberado: deja el pipeline emcee estable. El siguiente paso es acoplar rho_id -> Friedmann.
    def logprob(theta):
        lp = logprior(theta)
        if not np.isfinite(lp):
            return -np.inf

        H0, rho0, zt, eps = theta

        # actualiza H0 del fondo en caliente (sin reintegrar; MVP)
        # Para el modelo completo: reintegrar solve_background con H0 y recomputar H_of_z.
        # Aqui solo aplicamos escala lineal a H(z) para mantener estabilidad.
        scale = H0 / bp.H0

        def H_model_local(z):
            return scale * H_model(z)

        def mu_model_local(z):
            return distance_modulus(np.asarray(z, float), H_model_local(z))

        def dvrd_model_local(z):
            z = np.asarray(z, float)
            return 100.0 / np.maximum(H_model_local(z), 1e-9)

        _ = RhoIDParams(rho0=rho0, z_trans=zt, eps=eps)  # placeholder: conectado en release siguiente

        mloc = {"H(z)": H_model_local, "mu(z)": mu_model_local, "DVrd(z)": dvrd_model_local}
        ll = loglike_total(datasets, mloc)
        return lp + ll

    # --- run emcee
    nsteps = args.nsteps if args.nsteps is not None else int(cfg["run"]["nsteps"])
    nwalkers = args.nwalkers if args.nwalkers is not None else int(cfg["run"]["nwalkers"])
    seed = int(cfg["run"]["seed"])

    x0 = np.array([bp.H0, cfg["rho_id"]["rho0"], cfg["rho_id"]["z_trans"], cfg["rho_id"]["eps"]], float)
    sampler = run_emcee(logprob, x0, nwalkers=nwalkers, nsteps=nsteps, seed=seed)

    chain = sampler.get_chain()  # (nsteps, nwalkers, ndim)
    summary = summarize_chain(chain, burn=max(10, nsteps // 10), thin=2)

    print("Posterior (p16, p50, p84):")
    names = ["H0", "rho0", "z_trans", "eps"]
    for i, n in enumerate(names):
        print(f"{n:8s}: {summary['p16'][i]: .4f}  {summary['p50'][i]: .4f}  {summary['p84'][i]: .4f}")

    np.save("chain.npy", chain)
    print("OK: guardado chain.npy")


if __name__ == "__main__":
    main()
