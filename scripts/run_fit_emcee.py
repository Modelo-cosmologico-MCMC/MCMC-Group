from __future__ import annotations

import argparse
from pathlib import Path
import yaml
import numpy as np
import pandas as pd

from mcmc.core.friedmann_effective import EffectiveParams, H_of_z
from mcmc.channels.rho_id_refined import RhoIDRefinedParams
from mcmc.observables.distances import distance_modulus
from mcmc.observables.bao_distances import dv_over_rd
from mcmc.observables.likelihoods import loglike_total
from mcmc.inference.emcee_fit import run_emcee
from mcmc.inference.postprocess import summarize_chain


def load_csv(path: str, kind: str) -> dict:
    """Carga dataset CSV segun tipo."""
    df = pd.read_csv(path)
    if kind == "hz":
        return {"z": df["z"].to_numpy(), "H": df["H"].to_numpy(), "sigma": df["sigma"].to_numpy()}
    if kind == "sne":
        return {"z": df["z"].to_numpy(), "mu": df["mu"].to_numpy(), "sigma": df["sigma"].to_numpy()}
    if kind == "bao":
        return {"z": df["z"].to_numpy(), "dv_rd": df["dv_rd"].to_numpy(), "sigma": df["sigma"].to_numpy()}
    raise ValueError(f"kind desconocido: {kind}")


def main():
    ap = argparse.ArgumentParser(
        description="Ajuste MCMC con modelo efectivo Bloque II"
    )
    ap.add_argument("--config", default="src/mcmc/config/defaults.yaml",
                    help="Archivo de configuracion YAML")
    ap.add_argument("--nsteps", type=int, default=None,
                    help="Numero de pasos MCMC")
    ap.add_argument("--nwalkers", type=int, default=None,
                    help="Numero de walkers")
    ap.add_argument("--outdir", default="results/fit_effective",
                    help="Directorio de salida")
    args = ap.parse_args()

    # Cargar configuracion
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    pri = yaml.safe_load(Path("src/mcmc/config/priors.yaml").read_text(encoding="utf-8"))

    eff = cfg["effective"]
    run = cfg["run"]
    data = cfg["data"]

    # Cargar datasets
    datasets = {
        "hz": load_csv(data["hz"], "hz"),
        "sne": load_csv(data["sne"], "sne"),
        "bao": load_csv(data["bao"], "bao"),
    }

    # Crear directorio de salida
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "config_used.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8"
    )

    # Priors uniformes
    def logprior(theta):
        H0, rho_b0, rho0, zt, eps, rd, M = theta
        if not (pri["H0"][0] <= H0 <= pri["H0"][1]):
            return -np.inf
        if not (pri["rho_b0"][0] <= rho_b0 <= pri["rho_b0"][1]):
            return -np.inf
        if not (pri["rho0"][0] <= rho0 <= pri["rho0"][1]):
            return -np.inf
        if not (pri["z_trans"][0] <= zt <= pri["z_trans"][1]):
            return -np.inf
        if not (pri["eps"][0] <= eps <= pri["eps"][1]):
            return -np.inf
        if not (pri["rd"][0] <= rd <= pri["rd"][1]):
            return -np.inf
        if not (pri["M"][0] <= M <= pri["M"][1]):
            return -np.inf
        return 0.0

    def logprob(theta):
        lp = logprior(theta)
        if not np.isfinite(lp):
            return -np.inf

        H0, rho_b0, rho0, zt, eps, rd, M = theta

        # Construir parametros del modelo efectivo
        rid = RhoIDRefinedParams(rho0=float(rho0), z_trans=float(zt), eps=float(eps))
        p = EffectiveParams(H0=float(H0), rho_b0=float(rho_b0), rho_id=rid)

        # Modelo observable (Bloque II efectivo)
        def H_model(z):
            return H_of_z(np.asarray(z, float), p)

        def mu_model(z):
            z = np.asarray(z, float)
            return distance_modulus(z, H_model(z), M=float(M))

        def dvrd_model(z):
            z = np.asarray(z, float)
            return dv_over_rd(z, H_model(z), rd=float(rd))

        model = {"H(z)": H_model, "mu(z)": mu_model, "DVrd(z)": dvrd_model}

        try:
            ll = loglike_total(datasets, model)
        except Exception:
            return -np.inf

        if not np.isfinite(ll):
            return -np.inf

        return lp + ll

    # Inicializacion desde defaults
    x0 = np.array([
        eff["H0"],
        eff["rho_b0"],
        eff["rho_id"]["rho0"],
        eff["rho_id"]["z_trans"],
        eff["rho_id"]["eps"],
        eff["rd"],
        eff["M"],
    ], dtype=float)

    nsteps = args.nsteps if args.nsteps is not None else int(run["nsteps"])
    nwalkers = args.nwalkers if args.nwalkers is not None else int(run["nwalkers"])
    seed = int(run["seed"])

    print(f"=== MCMC Fit (Bloque II Efectivo) ===")
    print(f"Parametros: H0, rho_b0, rho0, z_trans, eps, rd, M")
    print(f"nwalkers={nwalkers}, nsteps={nsteps}, seed={seed}")
    print(f"Valores iniciales: {x0}")
    print()

    # Ejecutar emcee
    sampler = run_emcee(logprob, x0, nwalkers=nwalkers, nsteps=nsteps, seed=seed)

    chain = sampler.get_chain()  # (nsteps, nwalkers, ndim)
    logp = sampler.get_log_prob()

    # Guardar resultados
    np.save(outdir / "chain.npy", chain)
    np.save(outdir / "logp.npy", logp)

    # Resumen
    summary = summarize_chain(chain, burn=max(10, nsteps // 10), thin=2)
    names = ["H0", "rho_b0", "rho0", "z_trans", "eps", "rd", "M"]

    lines = ["Posterior (p16, p50, p84):"]
    lines.append("-" * 50)
    for i, n in enumerate(names):
        lines.append(
            f"{n:10s}: {summary['p16'][i]: .6f}  {summary['p50'][i]: .6f}  {summary['p84'][i]: .6f}"
        )
    lines.append("-" * 50)

    text = "\n".join(lines)
    (outdir / "summary.txt").write_text(text + "\n", encoding="utf-8")

    print(text)
    print()
    print(f"OK: Resultados guardados en {outdir}")


if __name__ == "__main__":
    main()
