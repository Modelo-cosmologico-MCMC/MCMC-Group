#!/usr/bin/env python
"""Unified pipeline wrapper for MCMC cosmological model.

Usage:
    python scripts/wrappers/run_pipeline.py --config configs/run_base.yaml --mode evaluate
    python scripts/wrappers/run_pipeline.py --config configs/run_base.yaml --mode fit
    python scripts/wrappers/run_pipeline.py --config configs/run_base.yaml --mode ontological
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to path for standalone execution
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from mcmc.pipeline import load_config, run_pipeline, run_from_config


def main() -> int:
    ap = argparse.ArgumentParser(
        prog="run_pipeline",
        description="Unified MCMC pipeline wrapper",
    )
    ap.add_argument(
        "--config",
        required=True,
        help="Path to YAML config file",
    )
    ap.add_argument(
        "--mode",
        choices=["evaluate", "fit", "ontological"],
        default="evaluate",
        help="Pipeline mode: evaluate (loglike only), fit (emcee), ontological (Block0→1→2)",
    )
    args = ap.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}", file=sys.stderr)
        return 1

    if args.mode == "ontological":
        # Run ontological pipeline (Block 0 → 1 → 2)
        cfg = load_config(config_path)
        out = run_pipeline(cfg)
        print("=" * 60)
        print("Ontological Pipeline Complete")
        print("=" * 60)
        print(f"Run ID: {out.run_id}")
        print(f"Output: {out.out_dir}")
        print()
        print("Block 0 (Pre-geometric):")
        print(f"  S range: [{out.block0.S_grid[0]:.4f}, {out.block0.S_grid[-1]:.4f}]")
        print(f"  Mp_pre final: {out.block0.Mp_pre[-1]:.6f}")
        print(f"  Ep_pre final: {out.block0.Ep_pre[-1]:.6f}")
        print(f"  Unit sum: {out.block0.Mp_pre[-1] + out.block0.Ep_pre[-1]:.10f}")
        print()
        print("Block 1 (Core Ontological):")
        print(f"  S range: [{out.block1.S_grid[0]:.4f}, {out.block1.S_grid[-1]:.4f}]")
        print(f"  a(S4): {out.block1.a[-1]:.10f}")
        print(f"  z(S4): {out.block1.z[-1]:.10f}")
        print()
        print("Block 2 (Effective Cosmology):")
        print(f"  S range: [{out.block2.S_grid[0]:.4f}, {out.block2.S_grid[-1]:.4f}]")
        print(f"  Mp_eff final: {out.block2.Mp_eff[-1]:.6f}")
        print(f"  Ep_eff final: {out.block2.Ep_eff[-1]:.6f}")
        print(f"  H_eff(z=0): {out.block2.H_eff[-1]:.4f} km/s/Mpc")
        return 0

    # Run inference pipeline (evaluate or fit)
    out = run_from_config(config_path)
    print("=" * 60)
    print(f"Inference Pipeline Complete (mode={args.mode})")
    print("=" * 60)
    print(f"Output dir: {out.outdir}")
    print(f"Log-likelihood: {out.loglike:.8f}")
    print(f"Summary: {out.summary_path}")
    if out.chain_path:
        print(f"Chain: {out.chain_path}")
        print(f"LogP: {out.logp_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
