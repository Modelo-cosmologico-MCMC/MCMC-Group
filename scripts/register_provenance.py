#!/usr/bin/env python
"""Registra y verifica la procedencia (sha256, version, fuente) de datos.

Uso:
    # Registrar un fichero externo
    python scripts/register_provenance.py add data/real/bao_desi_dr2.csv \
        --source "https://data.desi.lbl.gov/..." --version "DR2" \
        --notes "Covarianza aparte en bao_desi_dr2_cov.npz"

    # Verificar todo el registro
    python scripts/register_provenance.py verify
"""
from __future__ import annotations

import argparse
from datetime import date

from mcmc.data.provenance import record_file, verify_registry, DEFAULT_REGISTRY


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_add = sub.add_parser("add", help="Registrar un fichero")
    ap_add.add_argument("path")
    ap_add.add_argument("--source", default="")
    ap_add.add_argument("--version", default="")
    ap_add.add_argument("--retrieved", default=date.today().isoformat())
    ap_add.add_argument("--notes", default="")

    sub.add_parser("verify", help="Verificar hashes de todo el registro")

    args = ap.parse_args()

    if args.cmd == "add":
        rec = record_file(
            args.path,
            source=args.source,
            version=args.version,
            retrieved=args.retrieved,
            notes=args.notes,
            registry_path=args.registry,
        )
        print(f"Registrado: {rec.path}")
        print(f"  sha256: {rec.sha256}")
        print(f"  version: {rec.version or '(sin version)'}")
        return 0

    result = verify_registry(args.registry)
    for path in result["ok"]:
        print(f"OK       {path}")
    for path in result["changed"]:
        print(f"CAMBIADO {path}")
    for path in result["missing"]:
        print(f"FALTA    {path}")
    print(f"\n{'VERIFICADO' if result['passed'] else 'FALLO DE VERIFICACION'}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
