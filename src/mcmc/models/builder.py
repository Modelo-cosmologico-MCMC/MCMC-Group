from __future__ import annotations

from mcmc.models.effective_model import EffectiveModelConfig, build_effective_model
from mcmc.models.block1_model import Block1ModelConfig, build_block1_model


def build_model_from_config(cfg: dict):
    model_cfg = cfg.get("model", {})
    backend = model_cfg.get("backend", "effective")

    if backend == "effective":
        eff = cfg["effective"]
        rid = eff["rho_id"]
        m = EffectiveModelConfig(
            H0=float(eff["H0"]),
            rho_b0=float(eff["rho_b0"]),
            rho0=float(rid["rho0"]),
            z_trans=float(rid["z_trans"]),
            eps=float(rid["eps"]),
            rd=float(eff["rd"]),
            M=float(eff["M"]),
        )
        return build_effective_model(m)

    if backend == "block1":
        b = cfg.get("block1", {})
        m = Block1ModelConfig(
            rd=float(b.get("rd", 147.0)),
            M=float(b.get("M", -19.3)),
            H0=float(b.get("H0", 67.4)),
        )
        return build_block1_model(m)

    raise ValueError(f"Unknown backend: {backend} (use 'effective' or 'block1')")
