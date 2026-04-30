# AfriDef — Adversarial-Resilient Mobile Money Fraud Detection

Companion code for the KSM 2026 paper *"Adversarial-Resilient AI for Mobile
Money Fraud Detection in Sub-Saharan Africa: A Generative-Augmented,
Game-Theoretic Defense Framework against Evolving Attackers."*

## What's in here

```
afridef/
├── README.md                 ← you are here
├── environment.yml           ← conda env spec
├── Makefile                  ← reproducible entry points
├── configs/
│   └── default.yaml          ← experiment hyperparameters
├── data/
│   └── README.md             ← how to obtain PaySim & IEEE-CIS
├── src/afridef/
│   ├── __init__.py
│   ├── data.py               ← dataset loaders & graph construction
│   ├── models.py             ← GraphSAGE, GAT, baselines
│   ├── augment.py            ← Layer 1 — conditional WGAN-GP
│   ├── adv_train.py          ← Layer 2 — TRADES adversarial training
│   ├── stackelberg.py        ← Layer 3 — Stackelberg outer loop
│   ├── attacks.py            ← FGSM, PGD on tabular & graph features
│   ├── metrics.py            ← PR-AUC, recall@FPR, drift index
│   └── train.py              ← end-to-end orchestrator
├── scripts/
│   ├── prepare_data.py       ← download/prepare PaySim & IEEE-CIS
│   ├── run_baselines.py
│   └── run_full.py
├── notebooks/                ← exploratory analysis
└── results/                  ← logged metrics, checkpoints (gitignored)
```

## Quick start

```bash
conda env create -f environment.yml
conda activate afridef
make data      # downloads PaySim, IEEE-CIS to ./data
make baseline  # trains XGBoost + vanilla GraphSAGE
make full      # runs the full three-layer framework
make all       # everything end-to-end with 5 seeds
```

## Reproducibility

- All seeds fixed in `configs/default.yaml`.
- 5-seed bootstrap CIs reported on every headline number.
- Single GPU sufficient (RTX 3090/4090-class or cloud A10/A100).
- Total compute budget: ~80 GPU-hours.

## Status

Skeleton scaffolded 27 April 2026. Targeting submission by 15 June 2026.

## Citation

If you use this code, please cite the KSM 2026 paper (BibTeX in
`paper_ksm.tex`).
