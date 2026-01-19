"""Staged pipeline: Two-phase execution respecting ontological regimes.

Stage 1 (Pre-BB): S ∈ [0, S_BB] - Primordial regime
Stage 2 (Post-BB): S > S_BB (z ≥ 0) - Observable cosmology

Each stage produces its own output files with clear regime labeling.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

import numpy as np
import yaml

from mcmc.core.ontology import THRESHOLDS
from mcmc.core.solve_prebb import solve_prebb, PreBBParams, PreBBResult, get_prebb_boundary_conditions
from mcmc.core.solve_postbb import solve_postbb, PostBBParams, PostBBResult, evaluate_postbb_at_z
from mcmc.blocks.block0 import Block0Params
from mcmc.blocks.block0.pregeometric_field import PreGeometricFieldParams
from mcmc.blocks.block1 import Block1Params, CronosParams
from mcmc.core.chronos import ChronosParams as CoreChronosParams


@dataclass
class StagedOutputs:
    """Outputs from staged pipeline."""
    outdir: Path
    prebb_path: Path | None
    postbb_path: Path | None
    boundary_path: Path | None
    summary_path: Path


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _as_float(d: dict, k: str, default: float) -> float:
    v = d.get(k, default)
    return float(v)


def _serialize_arrays(obj: Any) -> Any:
    """Convert numpy arrays to lists for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _serialize_arrays(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize_arrays(v) for v in obj]
    return obj


def run_staged_pipeline(config_path: str | Path) -> StagedOutputs:
    """Run the staged pipeline from YAML config.

    Executes both pre-BB and post-BB stages with clear output separation.

    Args:
        config_path: Path to YAML configuration file.

    Returns:
        StagedOutputs with paths to all output files.
    """
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))

    run_cfg = cfg.get("run", {})
    run_id = str(run_cfg.get("run_id", "staged_run"))
    out_root = Path(str(run_cfg.get("outdir", "outputs")))
    outdir = out_root / run_id
    _ensure_dir(outdir)

    # Save config
    (outdir / "config_used.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8"
    )

    mode = str(run_cfg.get("mode", "ontological")).lower()

    prebb_path = None
    postbb_path = None
    boundary_path = None

    summary_lines = [
        "MCMC Staged Pipeline Output",
        f"Run ID: {run_id}",
        f"Mode: {mode}",
        f"S_BB (Big Bang): {THRESHOLDS.S_BB}",
        "",
    ]

    # Stage 1: Pre-BB (ontological mode or full)
    if mode in ("ontological", "full"):
        prebb_result = _run_stage_prebb(cfg, outdir)
        prebb_path = outdir / "stage1_prebb.json"

        # Extract boundary conditions for handoff
        bc = get_prebb_boundary_conditions(prebb_result)
        boundary_path = outdir / "boundary_conditions.json"
        boundary_path.write_text(
            json.dumps(_serialize_arrays(bc), indent=2), encoding="utf-8"
        )

        summary_lines.extend([
            "=== Stage 1: Pre-BB (S ≤ S_BB) ===",
            f"S range: [{prebb_result.S.min():.4f}, {prebb_result.S.max():.4f}]",
            f"t(S_BB): {prebb_result.t_BB:.6e} (anchor: 0)",
            f"a_rel(S_BB): {prebb_result.a_rel_BB:.6f} (ontological, NOT FRW)",
            f"Mp_pre: {prebb_result.Mp_pre:.6f}",
            f"Ep_pre: {prebb_result.Ep_pre:.6f}",
            "",
        ])

    # Stage 2: Post-BB (evaluate, fit, or full)
    if mode in ("evaluate", "fit", "full"):
        postbb_result = _run_stage_postbb(cfg, outdir)
        postbb_path = outdir / "stage2_postbb.json"

        # Evaluate at sample redshifts
        z_sample = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        obs = evaluate_postbb_at_z(postbb_result, z_sample)

        summary_lines.extend([
            "=== Stage 2: Post-BB (S > S_BB, z ≥ 0) ===",
            f"H0: {postbb_result.H0:.2f} km/s/Mpc",
            f"Age of universe (t0): {postbb_result.t0:.2f} Gyr",
            f"rd: {postbb_result.rd:.2f} Mpc",
            "",
            "Sample observables:",
            "  z     |  H(z)   | mu(z)   | DV/rd(z)",
            "  ------|---------|---------|----------",
        ])

        for i, z in enumerate(z_sample):
            h = obs["H"][i]
            mu = obs["mu"][i] if z > 0 else float("nan")
            dv = obs["DVrd"][i] if z > 0 else float("nan")
            summary_lines.append(f"  {z:.2f}  | {h:7.2f} | {mu:7.2f} | {dv:8.4f}")

        summary_lines.append("")

    summary_path = outdir / "summary.txt"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    return StagedOutputs(
        outdir=outdir,
        prebb_path=prebb_path,
        postbb_path=postbb_path,
        boundary_path=boundary_path,
        summary_path=summary_path,
    )


def _run_stage_prebb(cfg: dict, outdir: Path) -> PreBBResult:
    """Run Stage 1: Pre-BB solver."""
    # Build parameters from config
    b0_cfg = cfg.get("block0", {}) or {}
    # Support both "block1" and legacy "ontological_block1" keys
    b1_cfg = cfg.get("block1", {}) or cfg.get("ontological_block1", {}) or {}

    # Block 0 params
    field_cfg = b0_cfg.get("field", {}) or {}
    field_params = PreGeometricFieldParams(
        dS=_as_float(field_cfg, "dS", 1e-3),
        k0=_as_float(field_cfg, "k0", 8.0),
        center=_as_float(field_cfg, "center", 0.005),
        width=_as_float(field_cfg, "width", 0.002),
        floor=_as_float(field_cfg, "floor", 0.5),
    )
    block0 = Block0Params(
        Mp0=_as_float(b0_cfg, "Mp0", 0.999),
        Ep0=_as_float(b0_cfg, "Ep0", 0.001),
        eps=_as_float(b0_cfg, "eps", 0.001),
        S_min=_as_float(b0_cfg, "S_min", 0.001),
        S_max=_as_float(b0_cfg, "S_max", 0.009),
        dS=_as_float(b0_cfg, "dS", 1e-3),
        field_params=field_params,
    )

    # Block 1 params
    cron_cfg = b1_cfg.get("cronos", {}) or {}
    cronos = CronosParams(
        C_early=_as_float(cron_cfg, "C_early", 2.2),
        C_mid=_as_float(cron_cfg, "C_mid", 1.7),
        C_late=_as_float(cron_cfg, "C_late", 1.05),
        S1=_as_float(cron_cfg, "S1", 0.010),
        S2=_as_float(cron_cfg, "S2", 0.100),
        S3=_as_float(cron_cfg, "S3", 1.000),
        S4=_as_float(cron_cfg, "S4", 1.001),
        T0=_as_float(cron_cfg, "T0", 1.0),
        T_peak_amp=_as_float(cron_cfg, "T_peak_amp", 0.15),
        T_peak_w=_as_float(cron_cfg, "T_peak_w", 0.010),
        phi_env_amp=_as_float(cron_cfg, "phi_env_amp", 0.7),
        phi_env_tau=_as_float(cron_cfg, "phi_env_tau", 0.6),
        phi_bump_amp=_as_float(cron_cfg, "phi_bump_amp", 0.8),
        phi_bump_w=_as_float(cron_cfg, "phi_bump_w", 0.020),
    )
    block1 = Block1Params(
        S_min=_as_float(b1_cfg, "S_min", 0.010),
        S_max=_as_float(b1_cfg, "S_max", THRESHOLDS.S_BB),
        dS=_as_float(b1_cfg, "dS", 1e-3),
        H0=_as_float(b1_cfg, "H0", 67.4),
        sign=_as_float(b1_cfg, "sign", +1.0),
        cronos=cronos,
    )

    # Core Chronos params for time mapping
    core_chronos = CoreChronosParams()

    params = PreBBParams(block0=block0, block1=block1, chronos=core_chronos)
    result = solve_prebb(params)

    # Save pre-BB output
    # NOTE: a_rel is ONTOLOGICAL relative scale factor (NOT FRW)
    # a_rel(S_BB) = 1 by normalization at Big Bang threshold
    output = {
        "regime": "pre-BB",
        "S_range": [float(result.S.min()), float(result.S.max())],
        "t_BB": float(result.t_BB),
        "a_rel_BB": float(result.a_rel_BB),
        "note": "a_rel is ontological, NOT FRW scale factor",
        "block0": {
            "Mp_pre": result.Mp_pre,
            "Ep_pre": result.Ep_pre,
            "phi_pre": result.phi_pre,
            "kpre_mean": result.kpre_mean,
        },
        "block1": {
            "S": result.S.tolist(),
            "t": result.t.tolist(),
            "a_rel": result.a_rel.tolist(),
            "z_onto": result.z_onto.tolist(),
            "H_ref": result.H_ref.tolist(),
        },
    }

    (outdir / "stage1_prebb.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )

    return result


def _run_stage_postbb(cfg: dict, outdir: Path) -> PostBBResult:
    """Run Stage 2: Post-BB solver."""
    # Build parameters from config
    eff_cfg = cfg.get("effective", {}) or {}
    rid_cfg = eff_cfg.get("rho_id", {}) or {}

    params = PostBBParams(
        H0=_as_float(eff_cfg, "H0", 67.4),
        rho_b0=_as_float(eff_cfg, "rho_b0", 0.30),
        rho0=_as_float(rid_cfg, "rho0", 0.70),
        z_trans=_as_float(rid_cfg, "z_trans", 1.0),
        eps=_as_float(rid_cfg, "eps", 0.05),
        rd=_as_float(eff_cfg, "rd", 147.0),
        M=_as_float(eff_cfg, "M", -19.3),
    )

    result = solve_postbb(params)

    # Evaluate at a grid for output
    z_grid = np.linspace(0, 3, 100)
    obs = evaluate_postbb_at_z(result, z_grid)

    # Save post-BB output
    output = {
        "regime": "post-BB",
        "H0": result.H0,
        "t0": result.t0,
        "rd": result.rd,
        "M": result.M,
        "rho_id": {
            "rho0": result.rho_id_params.rho0,
            "z_trans": result.rho_id_params.z_trans,
            "eps": result.rho_id_params.eps,
        },
        "observables": {
            "z": obs["z"].tolist(),
            "H": obs["H"].tolist(),
            "mu": obs["mu"].tolist(),
            "DVrd": obs["DVrd"].tolist(),
        },
    }

    (outdir / "stage2_postbb.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )

    return result
