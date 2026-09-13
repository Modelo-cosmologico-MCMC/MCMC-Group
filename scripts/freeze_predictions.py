#!/usr/bin/env python
"""Congela la prediccion de fondo del MCMC ANTES de nuevos datos.

Motivacion (informe semanal, punto 5): las proximas muestras SNe+BAO
(DESI DR3, Rubin) reduciran fuertemente el espacio permitido. Ajustar
rho_id(z) retrospectivamente destruiria el poder predictivo. Este script
escribe una prediccion versionada e inmutable de H(z), distancias y
w_eff(z) con los parametros fiduciales actuales, sellada con sha256.

Uso:
    python scripts/freeze_predictions.py --version v1
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path

import numpy as np
import yaml

from mcmc.cobaya_interface.background import BackgroundCalculator, BackgroundParams


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/run_base.yaml")
    ap.add_argument("--version", default="v1")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--zmax", type=float, default=3.0)
    ap.add_argument("--npoints", type=int, default=61)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    eff = cfg["effective"]
    rid = eff["rho_id"]
    params = BackgroundParams(
        H0=float(eff["H0"]),
        rho_b0=float(eff["rho_b0"]),
        rho0=float(rid["rho0"]),
        z_trans=float(rid["z_trans"]),
        eps=float(rid["eps"]),
        rd=float(eff["rd"]),
        M=float(eff["M"]),
    )

    calc = BackgroundCalculator(params)
    z = np.linspace(0.0, args.zmax, args.npoints)
    table = calc.table(z)

    payload = {
        "kind": "mcmc_frozen_background_prediction",
        "version": args.version,
        "frozen_on": date.today().isoformat(),
        "source_config": args.config,
        "params": params.__dict__,
        "grid": {k: np.asarray(v).tolist() for k, v in table.items()},
        "notes": (
            "Prediccion previa a DESI DR3 / Rubin. No modificar: cualquier "
            "cambio de parametros requiere una nueva version con changelog."
        ),
    }
    body = json.dumps(payload, indent=2, sort_keys=True)
    payload["sha256_of_grid"] = hashlib.sha256(body.encode("utf-8")).hexdigest()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / f"frozen_prediction_{args.version}.json"
    if out_path.exists():
        raise SystemExit(
            f"ERROR: {out_path} ya existe. Las predicciones congeladas son "
            f"inmutables: usa --version con un identificador nuevo."
        )
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
    print(f"Prediccion congelada: {out_path}")
    print(f"sha256: {payload['sha256_of_grid']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
