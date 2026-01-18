from __future__ import annotations

import argparse
import pandas as pd
import matplotlib.pyplot as plt

from mcmc.core.s_grid import create_default_grid
from mcmc.core.background import BackgroundParams, solve_background
from mcmc.core.checks import assert_background_ok


def main():
    ap = argparse.ArgumentParser(
        description="Genera tabla de fondo cosmologico usando Ley de Cronos refinada"
    )
    ap.add_argument("--plot", action="store_true", help="Muestra graficas basicas")
    ap.add_argument("--out", default="background_table.csv", help="Ruta de salida CSV")
    args = ap.parse_args()

    grid, S = create_default_grid()
    p = BackgroundParams()

    # Pasar sellos ontologicos a solve_background
    sol = solve_background(
        S, p,
        S1=grid.seals.S1,
        S2=grid.seals.S2,
        S3=grid.seals.S3,
        S4=grid.seals.S4,
    )
    assert_background_ok(sol)

    # Guardar tabla con columnas extendidas
    df = pd.DataFrame({
        "S": sol["S"],
        "a": sol["a"],
        "z": sol["z"],
        "t_rel": sol["t_rel"],
        "H": sol["H"],
        "C": sol["C"],
        "T": sol["T"],
        "N": sol["N"],
        "Phi_ten": sol["Phi_ten"],
    })
    df.to_csv(args.out, index=False)
    print(f"OK: guardado {args.out} ({len(df)} filas).")

    # Mostrar resumen
    print(f"\nResumen:")
    print(f"  S range: [{S[0]:.4f}, {S[-1]:.4f}]")
    print(f"  z range: [{sol['z'][-1]:.4f}, {sol['z'][0]:.2f}]")
    print(f"  H(z=0) = {sol['H'][-1]:.2f} km/s/Mpc")
    print(f"  C range: [{sol['C'].min():.3f}, {sol['C'].max():.3f}]")

    if args.plot:
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))

        # H(z)
        axes[0, 0].plot(sol["z"], sol["H"])
        axes[0, 0].set_xlabel("z")
        axes[0, 0].set_ylabel("H(z) [km/s/Mpc]")
        axes[0, 0].set_title("Hubble H(z)")
        axes[0, 0].invert_xaxis()

        # a(S)
        axes[0, 1].plot(sol["S"], sol["a"])
        axes[0, 1].set_xlabel("S")
        axes[0, 1].set_ylabel("a(S)")
        axes[0, 1].set_title("Factor de escala a(S)")

        # C(S)
        axes[1, 0].plot(sol["S"], sol["C"])
        axes[1, 0].set_xlabel("S")
        axes[1, 0].set_ylabel("C(S)")
        axes[1, 0].set_title("Funcion de expansion C(S)")
        axes[1, 0].axvline(grid.seals.S2, color='r', linestyle='--', alpha=0.5, label='S2')
        axes[1, 0].axvline(grid.seals.S3, color='g', linestyle='--', alpha=0.5, label='S3')
        axes[1, 0].legend()

        # Phi_ten(S) y N(S)
        ax2 = axes[1, 1].twinx()
        axes[1, 1].plot(sol["S"], sol["Phi_ten"], 'b-', label='Phi_ten')
        ax2.plot(sol["S"], sol["N"], 'r-', label='N(S)')
        axes[1, 1].set_xlabel("S")
        axes[1, 1].set_ylabel("Phi_ten(S)", color='b')
        ax2.set_ylabel("N(S)", color='r')
        axes[1, 1].set_title("Campo tensional y lapse")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
