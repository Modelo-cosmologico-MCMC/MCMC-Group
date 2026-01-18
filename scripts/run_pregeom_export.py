from __future__ import annotations

import argparse
import json
from pathlib import Path

from mcmc.pregeom.s0_state import PreGeomParams, compute_initial_conditions


def main() -> None:
    """
    Genera condiciones iniciales del Bloque 0 (pre-geometrico).

    Exporta un JSON con los parametros iniciales para el Bloque I.
    """
    ap = argparse.ArgumentParser(
        description="Genera condiciones iniciales pre-geometricas (Bloque 0)"
    )
    ap.add_argument("--eps", type=float, default=1e-2,
                    help="Imperfeccion primordial (default: 0.01)")
    ap.add_argument("--phi0", type=float, default=0.0,
                    help="Condicion inicial campo tensional")
    ap.add_argument("--k0", type=float, default=1.0,
                    help="Rigidez pre-geometrica")
    ap.add_argument("--S_start", type=float, default=0.010,
                    help="Punto de entrada al Bloque I")
    ap.add_argument("--out", type=str, default="results/initial_conditions.json",
                    help="Ruta de salida JSON")
    args = ap.parse_args()

    p = PreGeomParams(
        eps=args.eps,
        phi0=args.phi0,
        k0=args.k0,
        S_start=args.S_start
    )
    ic = compute_initial_conditions(p)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(ic.to_dict(), indent=2), encoding="utf-8")

    print(f"Bloque 0 -> Condiciones iniciales:")
    print(f"  Mp_pre  = {ic.Mp_pre:.6f}")
    print(f"  Ep_pre  = {ic.Ep_pre:.6f}")
    print(f"  phi_pre = {ic.phi_pre:.6f}")
    print(f"  k_pre   = {ic.k_pre:.6f}")
    print(f"  S_start = {ic.S_start:.6f}")
    print(f"Guardado en: {out}")


if __name__ == "__main__":
    main()
