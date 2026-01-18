from __future__ import annotations

import argparse
from pathlib import Path
import yaml
import numpy as np

from mcmc.data.io import load_dataset
from mcmc.models.builder import build_model_from_config
from mcmc.inference.emcee_fit import run_emcee
from mcmc.inference.postprocess import summarize_chain
from mcmc.observables.likelihoods import loglike_total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="src/mcmc/config/defaults.yaml")
    ap.add_argument("--nsteps", type=int, default=None)
    ap.add_argument("--nwalkers", type=int, default=None)
    ap.add_argument("--outdir", default="results/fit_run")
    ap.add_argument("--evaluate-only", action="store_true", help="Only compute loglike, no emcee.")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    pri = yaml.safe_load(Path("src/mcmc/config/priors.yaml").read_text(encoding="utf-8"))

    data = cfg["data"]
    run = cfg["run"]
    backend = cfg.get("model", {}).get("backend", "effective")

    ds_hz = load_dataset("hz", data["hz"], name="hz", cov_path=data.get("hz_cov"))
    ds_sne = load_dataset("sne", data["sne"], name="sne", cov_path=data.get("sne_cov"))
    ds_bao = load_dataset("bao", data["bao"], name="bao", cov_path=data.get("bao_cov"))

    datasets = {
        "hz": ds_hz.as_legacy_dict(),
        "sne": ds_sne.as_legacy_dict(),
        "bao": ds_bao.as_legacy_dict(),
    }

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    # --- Evaluate-only path (e.g. backend=block1)
    if args.evaluate_only or backend == "block1":
        model = build_model_from_config(cfg)
        ll = loglike_total(datasets, model)
        (outdir / "summary.txt").write_text(f"backend={backend}\nloglike={ll:.8f}\n", encoding="utf-8")
        print(f"backend={backend} loglike={ll:.8f}")
        return

    # --- emcee path (effective backend)
    eff = cfg["effective"]
    rid = eff["rho_id"]

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

        # construimos un cfg "overlay" para build_model_from_config
        cfg_local = dict(cfg)
        cfg_local["model"] = {"backend": "effective"}
        cfg_local["effective"] = dict(eff)
        cfg_local["effective"]["H0"] = float(H0)
        cfg_local["effective"]["rho_b0"] = float(rho_b0)
        cfg_local["effective"]["rd"] = float(rd)
        cfg_local["effective"]["M"] = float(M)
        cfg_local["effective"]["rho_id"] = {
            "rho0": float(rho0),
            "z_trans": float(zt),
            "eps": float(eps),
        }

        model = build_model_from_config(cfg_local)
        ll = loglike_total(datasets, model)
        return lp + ll

    x0 = np.array([
        eff["H0"],
        eff["rho_b0"],
        rid["rho0"],
        rid["z_trans"],
        rid["eps"],
        eff["rd"],
        eff["M"],
    ], dtype=float)

    nsteps = args.nsteps if args.nsteps is not None else int(run["nsteps"])
    nwalkers = args.nwalkers if args.nwalkers is not None else int(run["nwalkers"])
    seed = int(run["seed"])

    sampler = run_emcee(logprob, x0, nwalkers=nwalkers, nsteps=nsteps, seed=seed)

    chain = sampler.get_chain()
    logp = sampler.get_log_prob()

    np.save(outdir / "chain.npy", chain)
    np.save(outdir / "logp.npy", logp)

    summary = summarize_chain(chain, burn=max(10, nsteps // 10), thin=2)
    names = ["H0", "rho_b0", "rho0", "z_trans", "eps", "rd", "M"]

    lines = ["Posterior (p16, p50, p84):"]
    for i, n in enumerate(names):
        lines.append(f"{n:8s}: {summary['p16'][i]: .6f}  {summary['p50'][i]: .6f}  {summary['p84'][i]: .6f}")
    text = "\n".join(lines)
    (outdir / "summary.txt").write_text(text + "\n", encoding="utf-8")
    print(text)
    print(f"OK: saved results to {outdir}")


if __name__ == "__main__":
    main()
