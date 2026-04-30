"""Run the baseline panel: XGBoost, vanilla GraphSAGE, GAT, autoencoder,
CTGAN-augmented XGBoost, robust-GraphSAGE-no-Stackelberg.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml


BASELINES = [
    "xgboost",
    "graphsage_vanilla",
    "gat_vanilla",
    "autoencoder_anomaly",
    "ctgan_xgboost",
    "robust_graphsage_no_stackelberg",
]


def main(config_path: str) -> None:
    cfg = yaml.safe_load(Path(config_path).read_text())
    print(f"[run_baselines] running {len(BASELINES)} baselines")
    for name in BASELINES:
        print(f"  -> {name}")
        # TODO: dispatch to baseline implementations and log metrics
    raise NotImplementedError("TODO: dispatch to baseline implementations")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    main(args.config)
