"""
AfriDef — EdgeGraphSAGE baseline + optional full pipeline (KSM 2026).

Pure PyTorch implementation — no torch_geometric required.

Flags
-----
--augment    : Layer 1 — WGAN-GP fraud oversampling
--trades     : Layer 2 — TRADES adversarial training
--stackelberg: Layer 3 — Stackelberg threshold adaptation

Run from the afridef project root:
    python scripts/baseline_graphsage.py --config configs/default.yaml --nrows 50000
    python scripts/baseline_graphsage.py --config configs/default.yaml --nrows 50000 --augment --trades --stackelberg
"""
from __future__ import annotations
import argparse
import os
import random
import sys
from pathlib import Path

# ── Allow `from afridef.xxx import ...` when run from project root ────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

# ── PaySim transaction types ──────────────────────────────────────────────────
PAYSIM_TYPES = ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"]


# ═══════════════════════════════════════════════════════════════════════════════
# Pure-PyTorch SAGEConv
# ═══════════════════════════════════════════════════════════════════════════════
class SAGEConv(nn.Module):
    """Mean-pooling SAGEConv.  No torch_geometric needed."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin = nn.Linear(in_dim * 2, out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index
        agg = torch.zeros_like(x)
        agg.scatter_add_(0, dst.unsqueeze(1).expand(-1, x.size(1)), x[src])
        deg = torch.bincount(dst, minlength=x.size(0)).float().clamp(min=1)
        agg = agg / deg.unsqueeze(1)
        return self.lin(torch.cat([x, agg], dim=1))


# ═══════════════════════════════════════════════════════════════════════════════
# Edge-level GraphSAGE model
# ═══════════════════════════════════════════════════════════════════════════════
class EdgeGraphSAGE(nn.Module):
    """GraphSAGE encoder + per-transaction edge classifier.

    score(u→v) = MLP( h_u ‖ h_v ‖ edge_feat )
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
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(node_in_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))

        self.edge_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for conv in self.convs[:-1]:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.encode(x, edge_index)
        s, d = edge_index
        if mask is not None:
            s, d, edge_attr = s[mask], d[mask], edge_attr[mask]
        return self.edge_head(torch.cat([h[s], h[d], edge_attr], dim=-1)).squeeze(-1)


# ═══════════════════════════════════════════════════════════════════════════════
# Data container (replaces torch_geometric.data.Data)
# ═══════════════════════════════════════════════════════════════════════════════
class GraphData:
    """Lightweight graph container."""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    @property
    def num_edges(self):
        return self.edge_index.size(1)

    @property
    def num_node_features(self):
        return self.x.size(1)


# ═══════════════════════════════════════════════════════════════════════════════
# Graph construction
# ═══════════════════════════════════════════════════════════════════════════════
def build_graph(
    df: pd.DataFrame,
    n_train: int,
    n_val: int,
    n_test: int,
    device: torch.device,
) -> GraphData:
    """Build a bipartite transaction graph with temporal edge masks.

    Node features (4-dim, computed from training edges only to avoid leakage):
        [log1p(degree), log1p(total_amount), log1p(mean_amount), fraud_rate]

    Edge features (4-dim):
        [log1p(amount), type_enc/4, log1p(oldbalanceOrg), log1p(newbalanceOrig)]
    """
    df = df.sort_values("step").reset_index(drop=True)
    n = len(df)
    n_test = min(n_test, n - n_train - n_val)

    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask   = torch.zeros(n, dtype=torch.bool)
    test_mask  = torch.zeros(n, dtype=torch.bool)
    train_mask[:n_train]                            = True
    val_mask[n_train : n_train + n_val]             = True
    test_mask[n_train + n_val : n_train + n_val + n_test] = True

    # Node vocabulary
    orig_ids = df["nameOrig"].values
    dest_ids = df["nameDest"].values
    unique   = np.unique(np.concatenate([orig_ids, dest_ids]))
    vocab    = {v: i for i, v in enumerate(unique)}
    N        = len(unique)

    src_idx = np.array([vocab[v] for v in orig_ids], dtype=np.int64)
    dst_idx = np.array([vocab[v] for v in dest_ids], dtype=np.int64)

    # Edge features
    type_enc = (
        pd.Categorical(df["type"], categories=PAYSIM_TYPES)
        .codes.astype(np.float32) / 4.0
    )
    edge_attr = torch.tensor(np.stack([
        np.log1p(df["amount"].values.astype(np.float32)),
        type_enc,
        np.log1p(df["oldbalanceOrg"].values.astype(np.float32)),
        np.log1p(df["newbalanceOrig"].values.astype(np.float32)),
    ], axis=1), dtype=torch.float, device=device)

    y_edge     = torch.tensor(df["isFraud"].values, dtype=torch.float, device=device)
    edge_index = torch.tensor(np.stack([src_idx, dst_idx]), dtype=torch.long, device=device)

    # Node features from training edges only (no leakage)
    ts = src_idx[:n_train]; td = dst_idx[:n_train]
    ta = df["amount"].values[:n_train].astype(np.float32)
    tf = df["isFraud"].values[:n_train].astype(np.float32)

    src_deg = np.bincount(ts, minlength=N).astype(np.float32)
    dst_deg = np.bincount(td, minlength=N).astype(np.float32)
    src_amt = np.bincount(ts, weights=ta, minlength=N).astype(np.float32)
    dst_amt = np.bincount(td, weights=ta, minlength=N).astype(np.float32)
    src_fr  = np.bincount(ts, weights=tf, minlength=N).astype(np.float32)

    total_deg  = src_deg + dst_deg
    total_amt  = src_amt + dst_amt
    mean_amt   = np.where(total_deg > 0, total_amt / np.maximum(total_deg, 1), 0.0)
    fraud_rate = np.where(src_deg > 0,  src_fr  / np.maximum(src_deg,   1), 0.0)

    x = torch.tensor(np.stack([
        np.log1p(total_deg),
        np.log1p(total_amt),
        np.log1p(mean_amt),
        fraud_rate,
    ], axis=1).astype(np.float32), dtype=torch.float, device=device)

    print(f"  Graph : {N:,} nodes | {n:,} edges")
    print(f"  Fraud : train={int(y_edge[train_mask].sum()):,} | "
          f"val={int(y_edge[val_mask].sum()):,} | "
          f"test={int(y_edge[test_mask].sum()):,}")

    return GraphData(
        x=x, edge_index=edge_index, edge_attr=edge_attr, y_edge=y_edge,
        train_mask=train_mask.to(device),
        val_mask=val_mask.to(device),
        test_mask=test_mask.to(device),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════════════
def evaluate(
    model: nn.Module,
    data: GraphData,
    mask: torch.Tensor,
    split: str = "",
    flip: bool = False,
) -> tuple[float, float, float]:
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index, data.edge_attr, mask=mask)
        probs  = torch.sigmoid(logits).cpu().numpy()
        labels = data.y_edge[mask].cpu().numpy()

    if flip:
        probs = 1.0 - probs

    n_pos = int(labels.sum())
    if n_pos == 0 or n_pos == len(labels):
        tag = split + " " if split else ""
        print(f"  [{tag}skipped — only one class ({n_pos} positives)]")
        return float("nan"), float("nan"), float("nan")

    auroc = roc_auc_score(labels, probs)
    ap    = average_precision_score(labels, probs)
    f1    = f1_score(labels, (probs >= 0.5).astype(int), zero_division=0)
    return auroc, ap, f1


# ═══════════════════════════════════════════════════════════════════════════════
# Stackelberg threshold (grid search, no RL dependency)
# ═══════════════════════════════════════════════════════════════════════════════
def stackelberg_threshold(
    model: nn.Module,
    data: GraphData,
    eps: float = 0.05,
    outer: int = 3,
    flip: bool = False,
) -> float:
    """Adversarial-perturbation grid search for the optimal detection threshold."""
    best_tau, best_f1 = 0.5, 0.0
    for _ in range(outer):
        x_adv = data.x + (torch.rand_like(data.x) * 2 - 1).clamp(-eps, eps) * eps
        model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(
                model(x_adv, data.edge_index, data.edge_attr, mask=data.val_mask)
            ).cpu().numpy()
        if flip:
            probs = 1.0 - probs
        labels = data.y_edge[data.val_mask].cpu().numpy()
        if labels.sum() == 0:
            continue
        for tau in np.arange(0.1, 0.91, 0.05):
            f1 = f1_score(labels, (probs >= tau).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_tau = f1, float(tau)
    return best_tau


# ═══════════════════════════════════════════════════════════════════════════════
# TRADES loss (inline — avoids import chain issues)
# ═══════════════════════════════════════════════════════════════════════════════
def pgd_attack(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    mask: torch.Tensor,
    eps: float,
    step: float,
    n_steps: int,
) -> torch.Tensor:
    x_orig = x.detach()
    x_adv  = x_orig + torch.empty_like(x_orig).uniform_(-eps, eps)
    x_adv  = x_adv.detach()
    labels = y[mask].float()
    for _ in range(n_steps):
        x_adv = x_adv.requires_grad_(True)
        logits = model(x_adv, edge_index, edge_attr, mask=mask)
        loss   = F.binary_cross_entropy_with_logits(logits, labels)
        grad   = torch.autograd.grad(loss, x_adv)[0]
        x_adv  = (x_adv.detach() + step * grad.sign())
        x_adv  = torch.clamp(x_adv, x_orig - eps, x_orig + eps).detach()
    return x_adv


def trades_loss(
    model: nn.Module,
    data: GraphData,
    pos_weight: torch.Tensor,
    beta: float,
    eps: float,
    step: float,
    n_steps: int,
) -> torch.Tensor:
    model.eval()
    x_adv = pgd_attack(
        model, data.x, data.y_edge, data.edge_index,
        data.edge_attr, data.train_mask, eps, step, n_steps,
    )
    model.train()
    labels = data.y_edge[data.train_mask].float()

    logits_nat = model(data.x,   data.edge_index, data.edge_attr, mask=data.train_mask)
    logits_adv = model(x_adv, data.edge_index, data.edge_attr, mask=data.train_mask)

    loss_nat = F.binary_cross_entropy_with_logits(logits_nat, labels, pos_weight=pos_weight)

    p_nat = torch.sigmoid(logits_nat).clamp(1e-6, 1 - 1e-6)
    p_adv = torch.sigmoid(logits_adv).clamp(1e-6, 1 - 1e-6)
    kl = (p_nat * (p_nat.log() - p_adv.log())
          + (1 - p_nat) * ((1 - p_nat).log() - (1 - p_adv).log())).mean()
    return loss_nat + beta * kl


# ═══════════════════════════════════════════════════════════════════════════════
# Training loop
# ═══════════════════════════════════════════════════════════════════════════════
def run_training(
    cfg: dict,
    df_orig: pd.DataFrame,
    device: torch.device,
    augment: bool = False,
    use_trades: bool = False,
    use_stackelberg: bool = False,
) -> dict:
    seed = cfg.get("seed", 0)
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    gnn_cfg    = cfg["gnn"]
    trades_cfg = cfg["trades"]

    n_orig  = len(df_orig)
    n_test  = int(n_orig * cfg["data"]["test_temporal_split"])
    n_val   = int(n_orig * cfg["data"]["val_temporal_split"])
    n_train = n_orig - n_val - n_test

    # ── Layer 1: WGAN-GP augmentation ─────────────────────────────────────────
    if augment:
        from afridef.augment import WGANGPAugmentor
        df_sorted   = df_orig.sort_values("step").reset_index(drop=True)
        train_slice = df_sorted.iloc[:n_train].copy()
        rest_slice  = df_sorted.iloc[n_train:].copy()

        aug = WGANGPAugmentor(cfg["wgan"], device=str(device), verbose=True)
        train_slice = aug.augment(
            train_slice, oversample_ratio=cfg["wgan"]["oversample_ratio"]
        )
        # Give synthetic rows a valid step number inside the training window
        synth_mask = train_slice["step"] == 0
        if synth_mask.any():
            real_steps = train_slice.loc[~synth_mask, "step"]
            lo, hi = int(real_steps.min()), int(real_steps.max())
            rng = np.random.default_rng(seed)
            train_slice.loc[synth_mask, "step"] = rng.integers(
                lo, hi + 1, size=int(synth_mask.sum())
            )
        n_train_graph = len(train_slice)
        df = pd.concat([train_slice, rest_slice], ignore_index=True)
        print(f"  Augmented: {len(train_slice):,} train rows | "
              f"fraud rate: {train_slice['isFraud'].mean()*100:.2f}%")
    else:
        df             = df_orig.copy()
        n_train_graph  = n_train

    # ── Build graph ───────────────────────────────────────────────────────────
    print("\nBuilding graph ...")
    data = build_graph(df, n_train_graph, n_val, n_test, device)

    tr_labels = data.y_edge[data.train_mask]
    n_pos = int(tr_labels.sum().item())
    n_neg = int((tr_labels == 0).sum().item())
    pw    = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float, device=device)
    print(f"  pos_weight={pw.item():.1f}  ({n_pos} fraud / {n_neg} non-fraud in train)")

    model = EdgeGraphSAGE(
        node_in_dim=data.x.shape[1],
        edge_in_dim=data.edge_attr.shape[1],
        hidden_dim=gnn_cfg["hidden_dim"],
        num_layers=gnn_cfg["num_layers"],
        dropout=gnn_cfg["dropout"],
    ).to(device)
    opt = torch.optim.Adam(
        model.parameters(),
        lr=gnn_cfg["lr"],
        weight_decay=gnn_cfg["weight_decay"],
    )

    method = ("TRADES" if use_trades else "BCE")
    print(f"\nTraining {gnn_cfg['epochs']} epochs ({method}) ...")

    best_val, best_state = 0.0, None
    for ep in range(1, gnn_cfg["epochs"] + 1):
        model.train()
        opt.zero_grad()

        if use_trades:
            loss = trades_loss(
                model, data, pw,
                beta=trades_cfg["beta"],
                eps=trades_cfg["epsilon"],
                step=trades_cfg["pgd_step_size"],
                n_steps=trades_cfg["pgd_steps"],
            )
        else:
            logits = model(data.x, data.edge_index, data.edge_attr, mask=data.train_mask)
            loss   = F.binary_cross_entropy_with_logits(logits, tr_labels.float(), pos_weight=pw)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if ep % 10 == 0:
            auroc, ap, f1 = evaluate(model, data, data.val_mask, "val")
            print(f"  ep {ep:3d} | loss={loss.item():.4f} | "
                  f"val AUROC={auroc:.4f} | AP={ap:.4f} | F1={f1:.4f}")
            if not np.isnan(auroc) and auroc > best_val:
                best_val = auroc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
        print(f"  Restored best val AUROC={best_val:.4f}")

    # Calibration flip heuristic
    flip = best_val < 0.5
    if flip:
        print(f"  [Calibration] val AUROC={best_val:.4f} < 0.5 — score = 1−prob")

    # ── Layer 3: Stackelberg ──────────────────────────────────────────────────
    if use_stackelberg:
        print("\nRunning Stackelberg threshold search ...")
        tau = stackelberg_threshold(model, data, eps=0.05, outer=5, flip=flip)
    else:
        tau = 0.5
    print(f"  Threshold τ = {tau:.3f}")

    # ── Test evaluation ───────────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        probs_test = torch.sigmoid(
            model(data.x, data.edge_index, data.edge_attr, mask=data.test_mask)
        ).cpu().numpy()
    if flip:
        probs_test = 1.0 - probs_test
    labels_test = data.y_edge[data.test_mask].cpu().numpy()

    parts = (["WGAN-GP"] if augment else []) + (["TRADES"] if use_trades else []) \
          + (["Stackelberg"] if use_stackelberg else []) + ["GraphSAGE"]
    label = " + ".join(parts)

    print(f"\n{'='*55}")
    print(f"TEST RESULTS — {label}")
    print("=" * 55)
    if labels_test.sum() == 0:
        print("  [WARN] No fraud in test set.")
        auroc = ap = f1 = float("nan")
    else:
        auroc = roc_auc_score(labels_test, probs_test)
        ap    = average_precision_score(labels_test, probs_test)
        f1    = f1_score(labels_test, (probs_test >= tau).astype(int), zero_division=0)
        print(f"  AUROC  : {auroc:.4f}")
        print(f"  AP     : {ap:.4f}")
        print(f"  F1@τ   : {f1:.4f}  (τ={tau:.3f})")
        print(f"  Flipped: {flip}")
    print("=" * 55)
    return {"auroc": auroc, "ap": ap, "f1": f1, "tau": tau, "flipped": flip}


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="AfriDef EdgeGraphSAGE — KSM 2026")
    parser.add_argument("--config",      default="configs/default.yaml")
    parser.add_argument("--nrows",       type=int, default=50_000,
                        help="Rows to load from PaySim CSV (0 = full dataset)")
    parser.add_argument("--csv",         default=None,
                        help="Path to PaySim CSV (overrides config)")
    parser.add_argument("--augment",     action="store_true",
                        help="Layer 1: WGAN-GP fraud augmentation")
    parser.add_argument("--trades",      action="store_true",
                        help="Layer 2: TRADES adversarial training")
    parser.add_argument("--stackelberg", action="store_true",
                        help="Layer 3: Stackelberg threshold adaptation")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())

    # Resolve PaySim CSV path
    csv_candidates = [
        args.csv,
        cfg["data"].get("paysim_path"),
        "data/raw/PS_20174392719_1491204439457_log.csv",
        "data/paysim.csv",
    ]
    csv_path = None
    for c in csv_candidates:
        if c and Path(c).exists():
            csv_path = c
            break

    if csv_path is None:
        print("\n[ERROR] PaySim CSV not found. Provide the path with --csv, e.g.:")
        print('  python scripts/baseline_graphsage.py --csv "C:/path/to/paysim.csv" --nrows 50000')
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"Config : {args.config}")
    print(f"CSV    : {csv_path}")

    nrows = args.nrows if args.nrows > 0 else None
    print(f"\nLoading PaySim ({nrows or 'full'} rows) ...")
    df = pd.read_csv(csv_path, nrows=nrows)
    print(f"  {len(df):,} rows | fraud rate: {df['isFraud'].mean()*100:.3f}%")

    run_training(
        cfg, df, device,
        augment=args.augment,
        use_trades=args.trades,
        use_stackelberg=args.stackelberg,
    )


if __name__ == "__main__":
    main()
