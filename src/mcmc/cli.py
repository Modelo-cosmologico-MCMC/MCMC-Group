"""MCMC CLI: Command-line interface for cosmological model pipeline.

Commands:
    mcmc staged --config <path>   Run staged pipeline (pre-BB + post-BB)
    mcmc fit --config <path>      Run inference pipeline (evaluate/emcee)
    mcmc run --config <path>      [DEPRECATED] Legacy pipeline
"""
from __future__ import annotations

import argparse
import sys

from mcmc.pipeline import (
    load_config,
    run_pipeline,
    run_from_config,
    run_staged_pipeline,
)


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="mcmc",
        description="MCMC Cosmological Model CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  mcmc staged --config configs/run_base.yaml    # Staged pipeline (recommended)
  mcmc fit --config configs/run_base.yaml       # Inference (evaluate/fit)
  mcmc run --config configs/run_base.yaml       # Legacy pipeline (deprecated)
        """,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    # Staged pipeline (RECOMMENDED - pre-BB + post-BB)
    staged = sub.add_parser(
        "staged",
        help="Run staged pipeline: Stage 1 (pre-BB) + Stage 2 (post-BB)",
    )
    staged.add_argument("--config", required=True, help="Path to YAML config file")

    # Inference pipeline (evaluate/fit)
    fit = sub.add_parser(
        "fit",
        help="Run inference pipeline (evaluate loglike or emcee fit)",
    )
    fit.add_argument("--config", required=True, help="Path to YAML config file")

    # Legacy pipeline (DEPRECATED)
    run = sub.add_parser(
        "run",
        help="[DEPRECATED] Legacy ontological pipeline - use 'staged' instead",
    )
    run.add_argument("--config", required=True, help="Path to YAML config file")

    args = ap.parse_args()

    if args.cmd == "staged":
        result = run_staged_pipeline(args.config)
        print("Staged pipeline complete.")
        print(f"  Output dir: {result.outdir}")
        print(f"  Summary: {result.summary_path}")
        if result.prebb_path:
            print(f"  Pre-BB: {result.prebb_path}")
        if result.postbb_path:
            print(f"  Post-BB: {result.postbb_path}")
        if result.boundary_path:
            print(f"  Boundary conditions: {result.boundary_path}")

    elif args.cmd == "fit":
        out = run_from_config(args.config)
        print("Inference complete.")
        print(f"  Output dir: {out.outdir}")
        print(f"  Log-likelihood: {out.loglike:.8f}")
        if out.chain_path:
            print(f"  Chain: {out.chain_path}")

    elif args.cmd == "run":
        print(
            "WARNING: 'mcmc run' is deprecated. Use 'mcmc staged' for the new "
            "two-phase pipeline that respects pre-BB/post-BB separation.",
            file=sys.stderr,
        )
        cfg = load_config(args.config)
        out = run_pipeline(cfg)
        print(f"Legacy pipeline output: {out}")


if __name__ == "__main__":
    main()
