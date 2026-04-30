"""Download and prepare PaySim and IEEE-CIS Fraud datasets via Kaggle CLI.
Generates the Africa-stress synthetic resample.
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import yaml


def kaggle_download(dataset: str, dest: Path) -> None:
    """Download a Kaggle dataset to `dest`. Requires kaggle CLI auth."""
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(
        ["kaggle", "datasets", "download", "-d", dataset, "-p", str(dest), "--unzip"]
    )


def main(config_path: str) -> None:
    cfg = yaml.safe_load(Path(config_path).read_text())
    data_root = Path("data")
    data_root.mkdir(exist_ok=True)

    print("[prepare_data] downloading PaySim ...")
    kaggle_download("ealaxi/paysim1", data_root)

    print("[prepare_data] downloading IEEE-CIS ...")
    subprocess.check_call(
        ["kaggle", "competitions", "download", "-c", "ieee-fraud-detection",
         "-p", str(data_root / "ieee_cis")]
    )

    print("[prepare_data] generating Africa-stress synthetic ...")
    # TODO: wire afridef.data.africa_stress_resample and write to disk
    print("[prepare_data] DONE")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    main(args.config)
