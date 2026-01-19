#!/usr/bin/env python
"""Plot observables and chain diagnostics from a pipeline run.

Usage:
    python scripts/plot_run.py --outdir outputs/run_base
    python scripts/plot_run.py --latest
    python scripts/plot_run.py --outdir outputs/run_base --zmax 3.0

Generates PNG plots in outputs/<run_id>/plots/:
    - Hz.png, mu.png, bao_dvrd.png (model + data)
    - Hz_residuals.png, mu_residuals.png, bao_residuals.png (if data exist)
    - trace_<param>.png, posterior_<param>.png (if chain.npy exists)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

# Add src to path for standalone execution
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mcmc.data.io import load_dataset
from mcmc.models.builder import build_model_from_config


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _load_used_config(outdir: Path) -> dict:
    cfg_path = outdir / "config_used.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing {cfg_path}. Run the pipeline first.")
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8"))


def _find_latest_run(base: Path = Path("outputs")) -> Path:
    """Find the most recently modified run directory."""
    if not base.exists():
        raise FileNotFoundError(f"No outputs directory found: {base}")

    candidates = [d for d in base.iterdir() if d.is_dir() and (d / "config_used.yaml").exists()]
    if not candidates:
        raise FileNotFoundError(f"No valid runs found in {base}")

    latest = max(candidates, key=lambda d: (d / "config_used.yaml").stat().st_mtime)
    return latest


def _load_dataset_safe(kind: str, path: str | None, cov_path: str | None = None):
    """Load dataset, return None if path is missing or file doesn't exist."""
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        return load_dataset(kind, path, name=kind, cov_path=cov_path)
    except Exception:
        return None


def plot_observables(cfg: dict, outdir: Path, nz: int = 400, zmax: float | None = None) -> None:
    """Plot H(z), mu(z), DV/rd(z) model curves with data overlay."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir = outdir / "plots"
    _safe_mkdir(plots_dir)

    # Load datasets (if paths exist)
    data = cfg.get("data", {})
    ds_hz = _load_dataset_safe("hz", data.get("hz"), data.get("hz_cov"))
    ds_sne = _load_dataset_safe("sne", data.get("sne"), data.get("sne_cov"))
    ds_bao = _load_dataset_safe("bao", data.get("bao"), data.get("bao_cov"))

    # Build model from config_used.yaml
    model = build_model_from_config(cfg)

    # Choose z-range
    z_candidates = []
    for ds in (ds_hz, ds_sne, ds_bao):
        if ds is not None:
            z_candidates.append(np.max(ds.z))
    z_data_max = float(np.max(z_candidates)) if z_candidates else 2.0
    z_hi = float(zmax) if zmax is not None else max(2.0, 1.2 * z_data_max)

    z_grid = np.linspace(1e-4, z_hi, nz)  # Start from small z to avoid division issues

    # --- H(z)
    H_grid = model["H(z)"](z_grid)
    plt.figure(figsize=(8, 5))
    plt.plot(z_grid, H_grid, "b-", lw=2, label="Model H(z)")
    if ds_hz is not None:
        plt.errorbar(ds_hz.z, ds_hz.y, yerr=ds_hz.sigma, fmt="o", ms=4, capsize=2, label="Data H(z)")
    plt.xlabel("z", fontsize=12)
    plt.ylabel("H(z) [km/s/Mpc]", fontsize=12)
    plt.title("Hubble function", fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "Hz.png", dpi=180)
    plt.close()
    print(f"  Saved: {plots_dir / 'Hz.png'}")

    # --- mu(z)
    mu_grid = model["mu(z)"](z_grid)
    plt.figure(figsize=(8, 5))
    plt.plot(z_grid, mu_grid, "b-", lw=2, label="Model mu(z)")
    if ds_sne is not None:
        plt.errorbar(ds_sne.z, ds_sne.y, yerr=ds_sne.sigma, fmt="o", ms=4, capsize=2, label="Data mu(z)")
    plt.xlabel("z", fontsize=12)
    plt.ylabel("mu(z) [mag]", fontsize=12)
    plt.title("Distance modulus", fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "mu.png", dpi=180)
    plt.close()
    print(f"  Saved: {plots_dir / 'mu.png'}")

    # --- DV/rd(z)
    dvrd_grid = model["DVrd(z)"](z_grid)
    plt.figure(figsize=(8, 5))
    plt.plot(z_grid, dvrd_grid, "b-", lw=2, label="Model DV/rd(z)")
    if ds_bao is not None:
        plt.errorbar(ds_bao.z, ds_bao.y, yerr=ds_bao.sigma, fmt="o", ms=4, capsize=2, label="Data DV/rd")
    plt.xlabel("z", fontsize=12)
    plt.ylabel("DV/rd(z)", fontsize=12)
    plt.title("BAO observable", fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "bao_dvrd.png", dpi=180)
    plt.close()
    print(f"  Saved: {plots_dir / 'bao_dvrd.png'}")

    # --- Residuals (if data exist)
    if ds_hz is not None:
        r = model["H(z)"](ds_hz.z) - ds_hz.y
        plt.figure(figsize=(8, 4))
        plt.errorbar(ds_hz.z, r, yerr=ds_hz.sigma, fmt="o", ms=4, capsize=2)
        plt.axhline(0.0, color="k", ls="--", lw=1)
        plt.xlabel("z", fontsize=12)
        plt.ylabel("H(z) residual [km/s/Mpc]", fontsize=12)
        plt.title("H(z) residuals (model - data)", fontsize=14)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / "Hz_residuals.png", dpi=180)
        plt.close()
        print(f"  Saved: {plots_dir / 'Hz_residuals.png'}")

    if ds_sne is not None:
        r = model["mu(z)"](ds_sne.z) - ds_sne.y
        plt.figure(figsize=(8, 4))
        plt.errorbar(ds_sne.z, r, yerr=ds_sne.sigma, fmt="o", ms=4, capsize=2)
        plt.axhline(0.0, color="k", ls="--", lw=1)
        plt.xlabel("z", fontsize=12)
        plt.ylabel("mu residual [mag]", fontsize=12)
        plt.title("mu(z) residuals (model - data)", fontsize=14)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / "mu_residuals.png", dpi=180)
        plt.close()
        print(f"  Saved: {plots_dir / 'mu_residuals.png'}")

    if ds_bao is not None:
        r = model["DVrd(z)"](ds_bao.z) - ds_bao.y
        plt.figure(figsize=(8, 4))
        plt.errorbar(ds_bao.z, r, yerr=ds_bao.sigma, fmt="o", ms=4, capsize=2)
        plt.axhline(0.0, color="k", ls="--", lw=1)
        plt.xlabel("z", fontsize=12)
        plt.ylabel("DV/rd residual", fontsize=12)
        plt.title("DV/rd residuals (model - data)", fontsize=14)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / "bao_residuals.png", dpi=180)
        plt.close()
        print(f"  Saved: {plots_dir / 'bao_residuals.png'}")


def plot_chain(outdir: Path) -> None:
    """Plot trace and posterior histograms if chain.npy exists."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    chain_path = outdir / "chain.npy"
    if not chain_path.exists():
        print("  No chain.npy found, skipping chain diagnostics.")
        return

    plots_dir = outdir / "plots"
    _safe_mkdir(plots_dir)

    chain = np.load(chain_path)  # shape: (nsteps, nwalkers, ndim)
    nsteps, nwalkers, ndim = chain.shape
    flat = chain.reshape(nsteps * nwalkers, ndim)

    # Parameter names (canonical order for effective backend)
    names = ["H0", "rho_b0", "rho0", "z_trans", "eps", "rd", "M"][:ndim]

    print(f"  Chain shape: {chain.shape} ({nsteps} steps, {nwalkers} walkers, {ndim} params)")

    # Trace plots
    for i in range(ndim):
        plt.figure(figsize=(10, 3))
        for w in range(min(nwalkers, 16)):
            plt.plot(chain[:, w, i], alpha=0.5, lw=0.5)
        plt.xlabel("Step", fontsize=12)
        plt.ylabel(names[i], fontsize=12)
        plt.title(f"Trace: {names[i]}", fontsize=14)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / f"trace_{names[i]}.png", dpi=150)
        plt.close()
        print(f"  Saved: {plots_dir / f'trace_{names[i]}.png'}")

    # Posterior histograms (burn-in: first 20%)
    burn = int(0.2 * nsteps)
    flat_burned = chain[burn:, :, :].reshape(-1, ndim)

    for i in range(ndim):
        plt.figure(figsize=(6, 4))
        plt.hist(flat_burned[:, i], bins=50, density=True, alpha=0.7)
        median = np.median(flat_burned[:, i])
        q16, q84 = np.percentile(flat_burned[:, i], [16, 84])
        plt.axvline(median, color="k", ls="-", lw=2, label=f"median={median:.4f}")
        plt.axvline(q16, color="k", ls="--", lw=1)
        plt.axvline(q84, color="k", ls="--", lw=1)
        plt.xlabel(names[i], fontsize=12)
        plt.ylabel("Density", fontsize=12)
        plt.title(f"Posterior: {names[i]} (burn-in removed)", fontsize=14)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / f"posterior_{names[i]}.png", dpi=150)
        plt.close()
        print(f"  Saved: {plots_dir / f'posterior_{names[i]}.png'}")

    # Summary statistics
    summary_path = plots_dir / "chain_summary.txt"
    with open(summary_path, "w") as f:
        f.write("Parameter Summary (after 20% burn-in)\n")
        f.write("=" * 50 + "\n")
        for i in range(ndim):
            med = np.median(flat_burned[:, i])
            q16, q84 = np.percentile(flat_burned[:, i], [16, 84])
            f.write(f"{names[i]:12s}: {med:12.6f} (+{q84-med:.6f} / -{med-q16:.6f})\n")
    print(f"  Saved: {summary_path}")


def main() -> int:
    ap = argparse.ArgumentParser(
        prog="plot_run",
        description="Generate plots from a pipeline run output",
    )
    ap.add_argument("--outdir", help="Path to outputs/<run_id> directory")
    ap.add_argument("--latest", action="store_true", help="Auto-detect latest run in outputs/")
    ap.add_argument("--nz", type=int, default=400, help="Number of z points for model curves")
    ap.add_argument("--zmax", type=float, default=None, help="Maximum z for plots")
    args = ap.parse_args()

    # Determine output directory
    if args.latest:
        try:
            outdir = _find_latest_run()
            print(f"Auto-detected latest run: {outdir}")
        except FileNotFoundError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 1
    elif args.outdir:
        outdir = Path(args.outdir)
    else:
        print("ERROR: Specify --outdir or --latest", file=sys.stderr)
        return 1

    if not outdir.exists():
        print(f"ERROR: Output directory not found: {outdir}", file=sys.stderr)
        return 1

    print("=" * 60)
    print(f"Generating plots for: {outdir}")
    print("=" * 60)

    try:
        cfg = _load_used_config(outdir)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    print(f"Backend: {cfg.get('model', {}).get('backend', 'unknown')}")
    print()

    print("Observable plots:")
    plot_observables(cfg, outdir, nz=args.nz, zmax=args.zmax)
    print()

    print("Chain diagnostics:")
    plot_chain(outdir)
    print()

    print("=" * 60)
    print(f"OK: All plots saved to {outdir / 'plots'}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
