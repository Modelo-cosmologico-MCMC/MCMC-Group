#!/usr/bin/env python
"""Primer ajuste sintetico (Prioridad 2 del plan de validacion).

Flujo:
  1. Genera BAO + SNe + H(z) sinteticos con verdad conocida y semilla fija.
  2. Ajusta MAP (Nelder-Mead) y cadena emcee con el mismo pipeline.
  3. Comprueba el round-trip del mapeo entropico S <-> z.
  4. Escribe informe JSON con verdad, posterior, pulls y veredicto.

Uso:
    python scripts/run_synthetic_recovery.py \
        --outdir results/synthetic_recovery --nsteps 1500 --nwalkers 32
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mcmc.cobaya_interface.background import BackgroundParams
from mcmc.validation.synthetic import SyntheticSpec, generate_synthetic_datasets
from mcmc.validation.recovery import run_recovery, s_map_roundtrip_check


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default="results/synthetic_recovery")
    ap.add_argument("--datadir", default="data/synthetic")
    ap.add_argument("--seed", type=int, default=20260801)
    ap.add_argument("--nwalkers", type=int, default=32)
    ap.add_argument("--nsteps", type=int, default=1500)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    truth = BackgroundParams(
        H0=67.4, rho_b0=0.30, rho0=0.70, z_trans=1.0, eps=0.05, rd=147.0, M=-19.3
    )
    spec = SyntheticSpec(truth=truth, seed=args.seed)

    print(f"[1/3] Generando datos sinteticos en {args.datadir} (seed={args.seed})")
    synth = generate_synthetic_datasets(spec, outdir=args.datadir)
    datasets = {k: synth[k] for k in ("hz", "sne", "bao")}

    print(f"[2/3] Ajuste MAP + emcee ({args.nwalkers} walkers x {args.nsteps} pasos)")
    result = run_recovery(
        datasets, truth, nwalkers=args.nwalkers, nsteps=args.nsteps, seed=args.seed
    )

    print("[3/3] Comprobando mapeo entropico S <-> z")
    s_check = s_map_roundtrip_check()

    report = {
        "spec": {
            "seed": args.seed,
            "files": synth["hashes"],
        },
        "recovery": result.__dict__,
        "s_map_roundtrip": s_check,
        "passed": result.all_recovered() and s_check["passed"],
    }
    report_path = outdir / "recovery_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")

    print(f"\nInforme: {report_path}")
    print(f"{'param':8s} {'verdad':>10s} {'p50':>10s} {'pull':>8s}  2sigma")
    for name in result.truth:
        print(
            f"{name:8s} {result.truth[name]:10.4f} {result.p50[name]:10.4f} "
            f"{result.pull[name]:8.2f}  {'OK' if result.within_2sigma[name] else 'FUERA'}"
        )
    print(f"\nS<->z round-trip: max_err={s_check['max_roundtrip_error']:.2e} "
          f"({'OK' if s_check['passed'] else 'FALLO'})")
    print(f"\nVEREDICTO GLOBAL: {'RECUPERADO' if report['passed'] else 'NO RECUPERADO'}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
