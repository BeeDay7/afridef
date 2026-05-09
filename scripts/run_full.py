"""Run the full AfriDef pipeline (all 3 layers) for one or more seeds.

Usage
-----
    # Single seed
    python scripts/run_full.py --config configs/default.yaml --seed 0 --nrows 50000

    # All 5 seeds (paper results)
    python scripts/run_full.py --config configs/default.yaml --seeds 0 1 2 3 4

    # Quick smoke test
    python scripts/run_full.py --config configs/default.yaml --nrows 10000 --no-stackelberg
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from afridef.train import main as run_seed


def parse_args():
    p = argparse.ArgumentParser(description="AfriDef full pipeline — KSM 2026")
    p.add_argument("--config",  default="configs/default.yaml",
                   help="Path to default.yaml")
    p.add_argument("--seed",    type=int,  default=None,
                   help="Single seed (overridden by --seeds)")
    p.add_argument("--seeds",   type=int,  nargs="+", default=None,
                   help="List of seeds, e.g. 0 1 2 3 4")
    p.add_argument("--nrows",   type=int,  default=0,
                   help="Sub-sample PaySim rows (0 = full dataset)")
    p.add_argument("--csv",     default=None,
                   help="Path to PaySim CSV (overrides config)")
    p.add_argument("--out-dir", default="results",
                   help="Directory for per-seed JSON output")
    p.add_argument("--no-augment",     action="store_true",
                   help="Disable Layer 1 (WGAN-GP augmentation)")
    p.add_argument("--no-trades",      action="store_true",
                   help="Disable Layer 2 (TRADES adversarial training)")
    p.add_argument("--no-stackelberg", action="store_true",
                   help="Disable Layer 3 (Stackelberg threshold adaptation)")
    return p.parse_args()


def main():
    args = parse_args()

    seeds = args.seeds if args.seeds is not None else (
        [args.seed] if args.seed is not None else [0]
    )
    nrows = args.nrows if args.nrows > 0 else None

    print(f"AfriDef run_full | seeds={seeds} | nrows={nrows or 'full'}")
    print(f"  augment={not args.no_augment} | trades={not args.no_trades} "
          f"| stackelberg={not args.no_stackelberg}")

    all_results = []
    for seed in seeds:
        print(f"\n{'='*60}\nSEED {seed}\n{'='*60}")
        result = run_seed(
            config_path=args.config,
            seed=seed,
            csv_path=args.csv,
            nrows=nrows,
            augment=not args.no_augment,
            use_trades=not args.no_trades,
            use_stackelberg=not args.no_stackelberg,
            out_dir=args.out_dir,
        )
        all_results.append(result)

    # Summary across seeds
    if len(all_results) > 1:
        valid = [r for r in all_results if not (
            isinstance(r["auroc"], float) and r["auroc"] != r["auroc"]  # nan check
        )]
        if valid:
            aurocs = [r["auroc"] for r in valid]
            aps    = [r["ap"]    for r in valid]
            f1s    = [r["f1"]    for r in valid]
            n_flip = sum(1 for r in valid if r.get("flipped", False))

            print(f"\n{'='*60}")
            print(f"SUMMARY across {len(valid)}/{len(all_results)} valid seeds")
            print("=" * 60)
            print(f"  AUROC  : {np.mean(aurocs):.4f} ± {np.std(aurocs):.4f}")
            print(f"  AP     : {np.mean(aps):.4f}")
            print(f"  F1     : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
            print(f"  Flipped: {n_flip}/{len(valid)} seeds")

            summary = {
                "auroc_mean": round(np.mean(aurocs), 4),
                "auroc_std":  round(np.std(aurocs),  4),
                "ap_mean":    round(np.mean(aps),     4),
                "f1_mean":    round(np.mean(f1s),     4),
                "n_flipped":  n_flip,
                "seeds":      all_results,
            }
            out = Path(args.out_dir) / "summary.json"
            out.write_text(json.dumps(summary, indent=2))
            print(f"  Saved → {out}")


if __name__ == "__main__":
    main()
