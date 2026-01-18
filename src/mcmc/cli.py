from __future__ import annotations

import argparse

from mcmc.pipeline import load_config, run_pipeline


def main() -> None:
    ap = argparse.ArgumentParser(prog="mcmc")
    sub = ap.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="Ejecuta pipeline Bloque0→1→2")
    run.add_argument("--config", required=True, help="Path to YAML config file")

    args = ap.parse_args()

    if args.cmd == "run":
        cfg = load_config(args.config)
        out = run_pipeline(cfg)
        print(str(out))


if __name__ == "__main__":
    main()
