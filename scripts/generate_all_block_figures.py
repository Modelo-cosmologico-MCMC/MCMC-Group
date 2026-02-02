#!/usr/bin/env python3
"""Generate all block simulation figures.

This unified script runs all block simulations and generates PNG visualizations
for the complete MCMC implementation.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def run_script(script_name: str, description: str) -> bool:
    """Run a simulation script and report status.

    Args:
        script_name: Name of script in scripts/ directory
        description: Human-readable description

    Returns:
        True if successful, False otherwise
    """
    script_path = Path(__file__).parent / script_name
    print(f"Running {description}...")
    print("-" * 50)

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )
        print(result.stdout)
        if result.returncode != 0:
            print(f"Error: {result.stderr}")
            return False
        return True
    except Exception as e:
        print(f"Error running {script_name}: {e}")
        return False


def main():
    """Run all block simulations."""
    print("=" * 70)
    print("MCMC Block Simulation Figures Generator")
    print("=" * 70)
    print()

    scripts = [
        ("simulate_block3.py", "Block 3 (N-body Cronos)"),
        ("simulate_block4.py", "Block 4 (Lattice-Gauge)"),
        ("simulate_block5.py", "Block 5 (Qubit Tensorial)"),
        ("simulate_auxiliary.py", "Auxiliary (Baryogenesis)"),
    ]

    results = {}
    for script, description in scripts:
        results[description] = run_script(script, description)
        print()

    # Summary
    print("=" * 70)
    print("Summary of Generated Figures")
    print("=" * 70)

    outdir = Path(__file__).parent.parent / "reports/figures/blocks"
    if outdir.exists():
        figures = sorted(outdir.glob("*.png"))
        print(f"\nGenerated {len(figures)} figures in {outdir.relative_to(Path(__file__).parent.parent)}/:\n")

        # Group by block
        block3 = [f for f in figures if f.name.startswith("block3")]
        block4 = [f for f in figures if f.name.startswith("block4")]
        block5 = [f for f in figures if f.name.startswith("block5")]
        auxiliary = [f for f in figures if f.name.startswith("auxiliary")]

        print("Block 3 (N-body Cronos):")
        for fig in block3:
            print(f"  - {fig.name}")

        print("\nBlock 4 (Lattice-Gauge):")
        for fig in block4:
            print(f"  - {fig.name}")

        print("\nBlock 5 (Qubit Tensorial):")
        for fig in block5:
            print(f"  - {fig.name}")

        print("\nAuxiliary (Baryogenesis):")
        for fig in auxiliary:
            print(f"  - {fig.name}")
    else:
        print("No figures generated.")

    # Status summary
    print("\n" + "=" * 70)
    print("Execution Status:")
    for description, success in results.items():
        status = "SUCCESS" if success else "FAILED"
        print(f"  {description}: {status}")
    print("=" * 70)


if __name__ == "__main__":
    main()
