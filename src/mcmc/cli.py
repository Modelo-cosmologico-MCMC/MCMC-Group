from __future__ import annotations

import argparse

from mcmc.pipeline import load_config, run_pipeline, run_from_config


def main() -> None:
    ap = argparse.ArgumentParser(prog="mcmc", description="MCMC Cosmological Model CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # Ontological pipeline (Block 0→1→2)
    run = sub.add_parser("run", help="Run ontological pipeline Block0→1→2")
    run.add_argument("--config", required=True, help="Path to YAML config file")

    # Inference pipeline (evaluate/fit)
    fit = sub.add_parser("fit", help="Run inference pipeline (evaluate or emcee fit)")
    fit.add_argument("--config", required=True, help="Path to YAML config file")

    args = ap.parse_args()

    if args.cmd == "run":
        cfg = load_config(args.config)
        out = run_pipeline(cfg)
        print(f"Ontological pipeline output: {out}")

    elif args.cmd == "fit":
        out = run_from_config(args.config)
        print(f"Inference output: {out.outdir}")
        print(f"Loglike: {out.loglike:.8f}")


if __name__ == "__main__":
    main()
