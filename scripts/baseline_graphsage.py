"""
GraphSAGE baseline on sub-sampled PaySim — vanilla and WGAN-GP-augmented.

Key design:
  - ONE unified graph built from all rows (train+val+test).
  - Edge-level fraud prediction: for each transaction edge, concat
    src-node and dst-node embeddings -> linear head -> fraud prob.
  - Temporal edge masks: train/val/test edges selected by chronological
    order; all edges used for message passing (standard static-graph baseline).
  - Node features log-normalised to keep gradients stable.
  - Optional --augment flag: runs WGAN-GP on training fraud rows first,
    appends synthetic fraud to the training set, then trains GraphSAGE.

Run from the afridef project root:
    # Vanilla baseline
    python scripts/baseline_graphsage.py --config configs/default.yaml --nrows 50000

    # Layer 1+2: WGAN-GP augmentation + GraphSAGE
    python scripts/baseline_graphsage.py --config configs/default.yaml --nrows 50000 --augment
"""
from __future__ import annotations
import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PAYSIM_TYPES = ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"]
CSV_PATH = "data/raw/PS_20174392719_1491204439457_log.csv"


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------
def build_unified_graph(
    df: pd.DataFrame,
    n_train: int,
    n_val: int,
    n_test: int,
) -> Data:
    """Build one PyG graph from ALL rows with temporal edge masks.

    Split sizes are passed in explicitly (not recomputed from len(df)) so
    that adding synthetic training rows never shifts the val/test windows.

    Node features (4-dim, computed from training edges only to avoid leakage):
        [log1p(degree), log1p(total_amount), log1p(mean_amount), fraud_rate]

    Edge features (4-dim):
        [log1p(amount), type_encoded/4, log1p(oldbalanceOrg), log1p(newbalanceOrig)]

    Edge labels: isFraud (binary, edge-level).
    Masks: train_mask / val_mask / test_mask on edges.
    """
    df = df.sort_values("step").reset_index(drop=True)
    n = len(df)

    # Clamp n_test to whatever rows remain after train+val
    n_test = min(n_test, n - n_train - n_val)

    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask   = torch.zeros(n, dtype=torch.bool)
    test_mask  = torch.zeros(n, dtype=torch.bool)
    train_mask[:n_train]              = True
    val_mask[n_train : n_train+n_val] = True
    test_mask[n_train+n_val : n_train+n_val+n_test] = True

    # Unified node vocabulary (all rows)
    orig_ids = df["nameOrig"].values
    dest_ids = df["nameDest"].values
    unique_nodes = np.unique(np.concatenate([orig_ids, dest_ids]))
    node_vocab   = {n: i for i, n in enumerate(unique_nodes)}
    n_nodes = len(unique_nodes)

    src_idx = np.array([node_vocab[v] for v in orig_ids], dtype=np.int64)
    dst_idx = np.array([node_vocab[v] for v in dest_ids], dtype=np.int64)
    edge_index = torch.tensor(np.stack([src_idx, dst_idx]), dtype=torch.long)

    # Edge features (log-normalised amounts)
    type_enc = (
        pd.Categorical(df["type"], categories=PAYSIM_TYPES)
        .codes.astype(np.float32) / 4.0          # scale to [0,1]
    )
    edge_attr = torch.tensor(
        np.stack([
            np.log1p(df["amount"].values.astype(np.float32)),
            type_enc,
            np.log1p(df["oldbalanceOrg"].values.astype(np.float32)),
            np.log1p(df["newbalanceOrig"].values.astype(np.float32)),
        ], axis=1),
        dtype=torch.float,
    )

    # Edge labels
    y_edge = torch.tensor(df["isFraud"].values, dtype=torch.float)

    # Node features — aggregate from TRAINING edges only
    tr_src = src_idx[:n_train]
    tr_dst = dst_idx[:n_train]
    tr_amt = df["amount"].values[:n_train].astype(np.float32)
    tr_fr  = df["isFraud"].values[:n_train].astype(np.float32)

    src_deg = np.bincount(tr_src, minlength=n_nodes).astype(np.float32)
    dst_deg = np.bincount(tr_dst, minlength=n_nodes).astype(np.float32)
    src_amt = np.bincount(tr_src, weights=tr_amt, minlength=n_nodes).astype(np.float32)
    dst_amt = np.bincount(tr_dst, weights=tr_amt, minlength=n_nodes).astype(np.float32)
    src_fr  = np.bincount(tr_src, weights=tr_fr,  minlength=n_nodes).astype(np.float32)

    total_deg = src_deg + dst_deg
    total_amt = src_amt + dst_amt
    mean_amt  = np.where(total_deg > 0, total_amt / np.maximum(total_deg, 1), 0.0)
    fraud_rate = np.where(src_deg > 0, src_fr  / np.maximum(src_deg,  1), 0.0)

    x = torch.tensor(
        np.stack([
            np.log1p(total_deg),
            np.log1p(total_amt),
            np.log1p(mean_amt),
            fraud_rate,
        ], axis=1).astype(np.float32),
        dtype=torch.float,
    )

    print(f"  Graph    : {n_nodes:,} nodes | {n:,} edges")
    print(f"  Fraud    : train={int(tr_fr.sum()):,} | "
          f"val={int(y_edge[val_mask].sum()):,} | "
          f"test={int(y_edge[test_mask].sum()):,}")

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y_edge=y_edge,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )


# ---------------------------------------------------------------------------
# Model — edge-level GraphSAGE
# ---------------------------------------------------------------------------
class EdgeGraphSAGE(nn.Module):
    """GraphSAGE encoder + edge-level classifier.

    For each transaction edge (u -> v):
        score = MLP( h_u || h_v || edge_feat )
    """
    def __init__(
        self,
        node_in_dim: int,
        edge_in_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.dropout = dropout

        # GNN encoder
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(node_in_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))

        # Edge classifier
        self.edge_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def encode(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x

    def forward(self, x, edge_index, edge_attr, mask=None):
        h = self.encode(x, edge_index)
        src, dst = edge_index
        if mask is not None:
            src      = src[mask]
            dst      = dst[mask]
            edge_attr = edge_attr[mask]
        edge_repr = torch.cat([h[src], h[dst], edge_attr], dim=-1)
        return self.edge_head(edge_repr).squeeze(-1)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(model: nn.Module, data: Data, mask: torch.Tensor, split: str):
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index, data.edge_attr, mask=mask)
        probs  = torch.sigmoid(logits).numpy()
        labels = data.y_edge[mask].numpy()

    n_pos = labels.sum()
    if n_pos == 0 or n_pos == len(labels):
        print(f"  [{split}] skipped — only one class present ({int(n_pos)} positives)")
        return float("nan"), float("nan"), float("nan")

    auroc = roc_auc_score(labels, probs)
    ap    = average_precision_score(labels, probs)
    preds = (probs >= 0.5).astype(int)
    f1    = f1_score(labels, preds, zero_division=0)
    return auroc, ap, f1


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def run_training(cfg: dict, df: pd.DataFrame, augment: bool = False, trades: bool = False, stackelberg: bool = False) -> None:
    seed = cfg.get("seed", 0)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    gnn = cfg["gnn"]

    # Compute fixed split sizes from the ORIGINAL df so val/test edges are
    # always the same rows regardless of how many synthetic rows we add.
    val_frac  = cfg["data"]["val_temporal_split"]
    test_frac = cfg["data"]["test_temporal_split"]
    n_orig  = len(df)
    n_test  = int(n_orig * test_frac)    # 10,000
    n_val   = int(n_orig * val_frac)     # 5,000
    n_train = n_orig - n_val - n_test    # 35,000  (real rows only)

    # ── Optional Layer 1: WGAN-GP augmentation ────────────────────────────
    if augment:
        from afridef.augment import WGANGPAugmentor
        df_sorted   = df.sort_values("step").reset_index(drop=True)
        train_slice = df_sorted.iloc[:n_train].copy()
        rest_slice  = df_sorted.iloc[n_train:].copy()

        aug = WGANGPAugmentor(cfg["wgan"], verbose=True)
        train_slice = aug.augment(train_slice,
                                  oversample_ratio=cfg["wgan"]["oversample_ratio"])

        # Synthetic rows have step=0; place them randomly within the real
        # training step window so they sort before val/test rows.
        synth_mask = train_slice["step"] == 0
        if synth_mask.any():
            real_steps = train_slice.loc[~synth_mask, "step"]
            lo, hi = int(real_steps.min()), int(real_steps.max())
            rng_s  = np.random.default_rng(seed)
            train_slice.loc[synth_mask, "step"] = rng_s.integers(
                lo, hi + 1, size=int(synth_mask.sum())
            )

        n_train_graph = len(train_slice)   # 35,000 real + 450 synthetic
        df = pd.concat([train_slice, rest_slice], ignore_index=True)
        print(f"  Augmented training set: {len(train_slice):,} rows "
              f"| fraud rate: {train_slice['isFraud'].mean()*100:.2f}%")
    else:
        n_train_graph = n_train

    print("\nBuilding unified graph ...")
    data = build_unified_graph(
        df,
        n_train=n_train_graph,
        n_val=n_val,
        n_test=n_test,
    )

    # Class weight from training edges
    tr_labels = data.y_edge[data.train_mask]
    n_pos = int(tr_labels.sum().item())
    n_neg = int((tr_labels == 0).sum().item())
    pw    = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float)
    print(f"  pos_weight: {pw.item():.1f}  ({n_pos} fraud / {n_neg} non-fraud in train)")

    model = EdgeGraphSAGE(
        node_in_dim=data.num_node_features,
        edge_in_dim=data.edge_attr.shape[1],
        hidden_dim=gnn["hidden_dim"],
        num_layers=gnn["num_layers"],
        dropout=gnn["dropout"],
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=gnn["lr"], weight_decay=gnn["weight_decay"]
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

    # ── Optional Layer 2: TRADES adversarial training ─────────────────────
    if trades:
        from afridef.adv_train import trades_loss as _trades_loss
        tr_cfg = cfg["trades"]
        print(f"\nTraining TRADES-GraphSAGE for {gnn['epochs']} epochs "
              f"(eps={tr_cfg['epsilon']}, beta={tr_cfg['beta']}, "
              f"steps={tr_cfg['pgd_steps']}) ...")
    else:
        print(f"\nTraining edge-level GraphSAGE for {gnn['epochs']} epochs ...")

    for epoch in range(1, gnn["epochs"] + 1):
        model.train()
        optimizer.zero_grad()

        if trades:
            loss = _trades_loss(
                model, data.x, data.y_edge, data.edge_index,
                eps=tr_cfg["epsilon"],
                step=tr_cfg["pgd_step_size"],
                n_steps=tr_cfg["pgd_steps"],
                beta=tr_cfg["beta"],
                edge_attr=data.edge_attr,
                mask=data.train_mask,
                pos_weight=pw,
            )
        else:
            logits = model(data.x, data.edge_index, data.edge_attr, mask=data.train_mask)
            loss   = criterion(logits, tr_labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if epoch % 10 == 0:
            auroc, ap, f1 = evaluate(model, data, data.val_mask, "val")
            print(
                f"  epoch {epoch:3d} | loss={loss.item():.4f} "
                f"| val AUROC={auroc:.4f} | AP={ap:.4f} | F1={f1:.4f}"
            )

    # ── Optional Layer 3: Stackelberg threshold adaptation ────────────────
    if stackelberg:
        from afridef.stackelberg import stackelberg_loop
        print("\nRunning Stackelberg outer loop (Layer 3) ...")
        # Use reduced iterations locally; Colab will use full cfg
        stk_cfg = dict(cfg["stackelberg"])
        stk_cfg["outer_iterations"] = min(stk_cfg["outer_iterations"], 5)
        stk_state = stackelberg_loop(model, data, stk_cfg)
        print(stk_state.summary())
        print(f"\nFinal adapted threshold: τ = {stk_state.threshold:.3f}")
        # Re-evaluate test set at adapted threshold
        model.eval()
        with torch.no_grad():
            probs_test = torch.sigmoid(
                model(data.x, data.edge_index, data.edge_attr, mask=data.test_mask)
            ).numpy()
        labels_test = data.y_edge[data.test_mask].numpy()
        from sklearn.metrics import roc_auc_score, average_precision_score, f1_score as _f1
        auroc_stk = roc_auc_score(labels_test, probs_test) if labels_test.sum() > 0 else float("nan")
        ap_stk    = average_precision_score(labels_test, probs_test) if labels_test.sum() > 0 else float("nan")
        f1_stk    = _f1(labels_test, (probs_test >= stk_state.threshold).astype(int), zero_division=0)
        print(f"  Test @ τ={stk_state.threshold:.3f}: AUROC={auroc_stk:.4f} | AP={ap_stk:.4f} | F1={f1_stk:.4f}")

    # Label for results header
    parts = []
    if augment:     parts.append("WGAN-GP")
    if trades:      parts.append("TRADES")
    if stackelberg: parts.append("Stackelberg")
    parts.append("GraphSAGE")
    layer_str = "s 1+2+3" if stackelberg else ("s 1+2" if augment and trades else " 2" if trades else " 1+2" if augment else " 2 only")
    label = " + ".join(parts) + f" (Layer{layer_str})"
    print("\n" + "=" * 55)
    print(f"TEST RESULTS — {label}")
    print("=" * 55)
    auroc, ap, f1 = evaluate(model, data, data.test_mask, "test")
    print(f"  AUROC  : {auroc:.4f}")
    print(f"  AP     : {ap:.4f}")
    print(f"  F1@0.5 : {f1:.4f}")
    print("=" * 55)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Vanilla GraphSAGE baseline")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--nrows",
        type=int,
        default=50_000,
        help="Sub-sample rows from PaySim CSV (0 = full ~6M rows)",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Apply WGAN-GP augmentation (Layer 1) before training GraphSAGE",
    )
    parser.add_argument(
        "--trades",
        action="store_true",
        help="Use TRADES adversarial training (Layer 2) instead of standard CE",
    )
    parser.add_argument(
        "--stackelberg",
        action="store_true",
        help="Run Stackelberg threshold-adaptation outer loop (Layer 3) after training",
    )
    args = parser.parse_args()

    cfg   = yaml.safe_load(Path(args.config).read_text())
    nrows = args.nrows if args.nrows > 0 else None

    print(f"Reading CSV ({nrows or 'full'} rows) ...")
    df = pd.read_csv(CSV_PATH, nrows=nrows)
    print(f"  Loaded {len(df):,} rows | fraud rate: {df['isFraud'].mean()*100:.2f}%")

    run_training(cfg, df, augment=args.augment, trades=args.trades, stackelberg=args.stackelberg)


if __name__ == "__main__":
    main()
