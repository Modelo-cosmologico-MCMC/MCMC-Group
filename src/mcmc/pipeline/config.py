from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml

from mcmc.blocks.block0 import Block0Params
from mcmc.blocks.block0.pregeometric_field import PreGeometricFieldParams
from mcmc.blocks.block1 import Block1Params, CronosParams
from mcmc.blocks.block2 import Block2Params


@dataclass(frozen=True)
class RunConfig:
    run_id: str
    out_dir: str
    block0: Block0Params
    block1: Block1Params
    block2: Block2Params


def _as_float(d: Dict[str, Any], k: str, default: float) -> float:
    v = d.get(k, default)
    return float(v)


def load_config(path: str | Path) -> RunConfig:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))

    run_id = str(payload.get("run_id", "run"))
    out_dir = str(payload.get("out_dir", "outputs"))

    # Block 0
    b0 = payload.get("block0", {}) or {}
    field_cfg = b0.get("field", {}) or {}
    field_params = PreGeometricFieldParams(
        dS=_as_float(field_cfg, "dS", 1e-3),
        k0=_as_float(field_cfg, "k0", 8.0),
        center=_as_float(field_cfg, "center", 0.005),
        width=_as_float(field_cfg, "width", 0.002),
        floor=_as_float(field_cfg, "floor", 0.5),
    )
    block0 = Block0Params(
        Mp0=_as_float(b0, "Mp0", 0.999),
        Ep0=_as_float(b0, "Ep0", 0.001),
        eps=_as_float(b0, "eps", 0.001),
        S_min=_as_float(b0, "S_min", 0.001),
        S_max=_as_float(b0, "S_max", 0.009),
        dS=_as_float(b0, "dS", 1e-3),
        field_params=field_params,
    )

    # Block 1
    b1 = payload.get("block1", {}) or {}
    cron = b1.get("cronos", {}) or {}
    cronos = CronosParams(
        C_early=_as_float(cron, "C_early", 2.2),
        C_mid=_as_float(cron, "C_mid", 1.7),
        C_late=_as_float(cron, "C_late", 1.05),
        S1=_as_float(cron, "S1", 0.010),
        S2=_as_float(cron, "S2", 0.100),
        S3=_as_float(cron, "S3", 1.000),
        S4=_as_float(cron, "S4", 1.001),
        T0=_as_float(cron, "T0", 1.0),
        T_peak_amp=_as_float(cron, "T_peak_amp", 0.15),
        T_peak_w=_as_float(cron, "T_peak_w", 0.010),
        phi_env_amp=_as_float(cron, "phi_env_amp", 0.7),
        phi_env_tau=_as_float(cron, "phi_env_tau", 0.6),
        phi_bump_amp=_as_float(cron, "phi_bump_amp", 0.8),
        phi_bump_w=_as_float(cron, "phi_bump_w", 0.020),
    )
    block1 = Block1Params(
        S_min=_as_float(b1, "S_min", 0.010),
        S_max=_as_float(b1, "S_max", 1.001),
        dS=_as_float(b1, "dS", 1e-3),
        H0=_as_float(b1, "H0", 67.4),
        sign=_as_float(b1, "sign", +1.0),
        cronos=cronos,
    )

    # Block 2
    b2 = payload.get("block2", {}) or {}
    block2 = Block2Params(
        Mp0=_as_float(b2, "Mp0", 0.99),
        Mpeq=_as_float(b2, "Mpeq", 0.50),
        gamma_S=_as_float(b2, "gamma_S", 1.0),
        kappa=_as_float(b2, "kappa", 0.02),
        alpha_base=_as_float(b2, "alpha_base", 2.0 / 3.0),
        delta_alpha=_as_float(b2, "delta_alpha", 0.20),
        w0=_as_float(b2, "w0", -1.0),
        delta_w=_as_float(b2, "delta_w", 0.02),
        S1=_as_float(b2, "S1", 0.010),
        S4=_as_float(b2, "S4", 1.001),
    )

    return RunConfig(run_id=run_id, out_dir=out_dir, block0=block0, block1=block1, block2=block2)
