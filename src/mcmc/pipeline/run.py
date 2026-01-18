from __future__ import annotations

from pathlib import Path
import json

from mcmc.blocks.block0 import run_block0, export_block0_conditions
from mcmc.blocks.block1 import run_block1, CronosModel
from mcmc.blocks.block2 import run_block2, build_H_eff
from .config import RunConfig


def run_pipeline(cfg: RunConfig) -> Path:
    out_dir = Path(cfg.out_dir) / cfg.run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Bloque 0
    b0 = run_block0(cfg.block0)
    export_block0_conditions(out_dir / "block0_initial_conditions.json", b0)

    # Bloque 1 (núcleo)
    b1 = run_block1(cfg.block1)

    # phi_ten(S) desde CronosModel (coherencia ontológica: N=exp(phi_ten))
    cron = CronosModel(cfg.block1.cronos)
    phi = [cron.phi_ten(Si) for Si in b1["S"]]

    # Bloque 2
    b2 = run_block2(b1["S"], phi, cfg.block2)
    H_eff = build_H_eff(b1["a"], b2["alpha_eff"], float(b1["H0"]))

    payload = {
        "block0": {
            "Mp_pre": b0["Mp_pre"],
            "Ep_pre": b0["Ep_pre"],
            "phi_pre": b0["phi_pre"],
            "kpre_mean": b0["kpre_mean"],
        },
        "block1": {
            "S": b1["S"],
            "a": b1["a"],
            "z": b1["z"],
            "t_rel": b1["t_rel"],
            "H_ref": b1["H"],
            "H0": b1["H0"],
        },
        "block2": {
            "Mp_eff": b2["Mp_eff"],
            "Ep_eff": b2["Ep_eff"],
            "alpha_eff": b2["alpha_eff"],
            "w_eff": b2["w_eff"],
            "H_eff": H_eff,
        },
    }

    out_path = out_dir / "pipeline_output.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path
