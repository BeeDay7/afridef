"""End-to-end orchestrator wiring Layer 1 (WGAN), Layer 2 (TRADES GNN), and
Layer 3 (Stackelberg). Stub.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def main(config_path: str, seed: int) -> None:
    cfg = yaml.safe_load(Path(config_path).read_text())
    cfg["seed"] = seed
    print(f"[afridef] running seed={seed} on {cfg['data']['primary']}")
    # TODO:
    # 1. load + temporal split
    # 2. build bipartite graph
    # 3. train WGAN-GP on real fraud, synthesise oversamples
    # 4. adversarially train GNN with TRADES on augmented data
    # 5. run Stackelberg outer loop on val
    # 6. evaluate on test (clean + FGSM/PGD + temporal drift)
    # 7. log to wandb / write results JSON
    raise NotImplementedError("TODO: orchestrator")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    main(args.config, args.seed)
