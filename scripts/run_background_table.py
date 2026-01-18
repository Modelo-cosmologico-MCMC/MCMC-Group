from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mcmc.core.s_grid import create_default_grid
from mcmc.core.background import BackgroundParams, solve_background
from mcmc.core.checks import assert_background_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true", help="Muestra graficas basicas")
    ap.add_argument("--out", default="background_table.csv", help="Ruta de salida CSV")
    args = ap.parse_args()

    grid, S = create_default_grid()
    p = BackgroundParams()
    sol = solve_background(S, p)
    assert_background_ok(sol)

    df = pd.DataFrame({
        "S": sol["S"],
        "a": sol["a"],
        "z": sol["z"],
        "t_rel": sol["t_rel"],
        "H": sol["H"],
    })
    df.to_csv(args.out, index=False)
    print(f"OK: guardado {args.out} ({len(df)} filas).")

    if args.plot:
        plt.figure()
        plt.plot(df["z"], df["H"])
        plt.xlabel("z")
        plt.ylabel("H(z) [km/s/Mpc]")
        plt.title("Fondo MCMC (MVP)")
        plt.gca().invert_xaxis()
        plt.show()

        plt.figure()
        plt.plot(df["S"], df["a"])
        plt.xlabel("S")
        plt.ylabel("a(S)")
        plt.title("a(S)")
        plt.show()


if __name__ == "__main__":
    main()
