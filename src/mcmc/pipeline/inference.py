from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from mcmc.data.io import load_dataset
from mcmc.models.builder import build_model_from_config
from mcmc.observables.likelihoods import loglike_total
from mcmc.inference.emcee_fit import run_emcee
from mcmc.inference.postprocess import summarize_chain


@dataclass(frozen=True)
class RunOutputs:
    outdir: Path
    loglike: float
    chain_path: Path | None
    logp_path: Path | None
    summary_path: Path


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_yaml(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def run_from_config(config_path: str | Path) -> RunOutputs:
    """Run inference pipeline from YAML config (evaluate or fit mode)."""
    cfg = load_yaml(config_path)

    run_cfg = cfg.get("run", {})
    run_id = str(run_cfg.get("run_id", "run"))
    out_root = Path(str(run_cfg.get("outdir", "outputs")))
    outdir = out_root / run_id
    _ensure_dir(outdir)

    # Persist config used
    (outdir / "config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    # Load datasets
    data = cfg.get("data", {})
    ds_hz = load_dataset("hz", data.get("hz", "data/demo/hz.csv"), name="hz", cov_path=data.get("hz_cov"))
    ds_sne = load_dataset("sne", data.get("sne", "data/demo/sne.csv"), name="sne", cov_path=data.get("sne_cov"))
    ds_bao = load_dataset("bao", data.get("bao", "data/demo/bao.csv"), name="bao", cov_path=data.get("bao_cov"))

    datasets = {
        "hz": ds_hz.as_legacy_dict(),
        "sne": ds_sne.as_legacy_dict(),
        "bao": ds_bao.as_legacy_dict(),
    }

    # Build model (backend)
    model = build_model_from_config(cfg)

    # Evaluate loglike
    ll = float(loglike_total(datasets, model))
    (outdir / "loglike.txt").write_text(f"loglike={ll:.12f}\n", encoding="utf-8")

    mode = str(run_cfg.get("mode", "evaluate")).lower()
    backend = str(cfg.get("model", {}).get("backend", "effective"))

    # Evaluate-only mode
    if mode == "evaluate":
        summary_path = outdir / "summary.txt"
        summary_path.write_text(f"mode=evaluate\nbackend={backend}\nloglike={ll:.12f}\n", encoding="utf-8")
        return RunOutputs(outdir=outdir, loglike=ll, chain_path=None, logp_path=None, summary_path=summary_path)

    # Fit mode (only for effective backend)
    if backend != "effective":
        summary_path = outdir / "summary.txt"
        summary_path.write_text(
            f"mode=fit\nbackend={backend}\nERROR: fit mode currently implemented for backend=effective only.\n"
            f"Use mode=evaluate for block1/unified.\nloglike={ll:.12f}\n",
            encoding="utf-8",
        )
        return RunOutputs(outdir=outdir, loglike=ll, chain_path=None, logp_path=None, summary_path=summary_path)

    # Load priors
    pri = load_yaml("src/mcmc/config/priors.yaml")
    eff = cfg["effective"]
    rid = eff["rho_id"]

    def logprior(theta: np.ndarray) -> float:
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

    def logprob(theta: np.ndarray) -> float:
        lp = logprior(theta)
        if not np.isfinite(lp):
            return -np.inf

        H0, rho_b0, rho0, zt, eps, rd, M = theta

        cfg_local = dict(cfg)
        cfg_local["model"] = {"backend": "effective"}
        cfg_local["effective"] = dict(eff)
        cfg_local["effective"]["H0"] = float(H0)
        cfg_local["effective"]["rho_b0"] = float(rho_b0)
        cfg_local["effective"]["rd"] = float(rd)
        cfg_local["effective"]["M"] = float(M)
        cfg_local["effective"]["rho_id"] = {"rho0": float(rho0), "z_trans": float(zt), "eps": float(eps)}

        m = build_model_from_config(cfg_local)
        return lp + float(loglike_total(datasets, m))

    x0 = np.array(
        [eff["H0"], eff["rho_b0"], rid["rho0"], rid["z_trans"], rid["eps"], eff["rd"], eff["M"]],
        dtype=float,
    )

    nwalkers = int(run_cfg.get("nwalkers", 32))
    nsteps = int(run_cfg.get("nsteps", 500))
    seed = int(run_cfg.get("seed", 42))

    sampler = run_emcee(logprob, x0, nwalkers=nwalkers, nsteps=nsteps, seed=seed)

    chain = sampler.get_chain()
    logp = sampler.get_log_prob()

    chain_path = outdir / "chain.npy"
    logp_path = outdir / "logp.npy"
    np.save(chain_path, chain)
    np.save(logp_path, logp)

    burn = max(10, int(nsteps * float(run_cfg.get("burn_frac", 0.10))))
    thin = max(1, int(run_cfg.get("thin", 2)))
    summary = summarize_chain(chain, burn=burn, thin=thin)
    names = ["H0", "rho_b0", "rho0", "z_trans", "eps", "rd", "M"]

    lines = [f"mode=fit\nbackend=effective\nloglike_start={ll:.12f}", "Posterior (p16, p50, p84):"]
    for i, n in enumerate(names):
        lines.append(f"{n:8s}: {summary['p16'][i]: .6f}  {summary['p50'][i]: .6f}  {summary['p84'][i]: .6f}")

    summary_path = outdir / "summary.txt"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return RunOutputs(outdir=outdir, loglike=ll, chain_path=chain_path, logp_path=logp_path, summary_path=summary_path)
