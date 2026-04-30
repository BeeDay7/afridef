"""Single-seed end-to-end run of the full AfriDef framework."""
from __future__ import annotations

import argparse

from afridef.train import main


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    main(args.config, args.seed)
