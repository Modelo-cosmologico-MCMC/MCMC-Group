from __future__ import annotations

import argparse
from mcmc.data.registry import list_datasets, get_spec
from mcmc.data.io import load_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default=None, help="Dataset name from registry. If omitted, validates all.")
    args = ap.parse_args()

    names = [args.name] if args.name else list_datasets()
    ok = 0
    skipped = 0

    for n in names:
        spec = get_spec(n)
        try:
            ds = load_dataset(spec.kind, spec.path, name=spec.name, cov_path=spec.cov_path)
            print(f"OK   {spec.name:24s} kind={spec.kind:3s} n={len(ds.z)} path={spec.path}")
            ok += 1
        except FileNotFoundError:
            print(f"SKIP {spec.name:24s} (missing file) path={spec.path}")
            skipped += 1

    print(f"Done. ok={ok} skipped={skipped}")


if __name__ == "__main__":
    main()
