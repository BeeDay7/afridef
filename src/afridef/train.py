"""End-to-end orchestrator: Layer 1 (WGAN-GP) + Layer 2 (TRADES GNN)
+ Layer 3 (Stackelberg threshold adaptation).

Called by scripts/run_full.py.  For interactive use, prefer
scripts/baseline_graphsage.py which has identical logic and CLI flags.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from .models import build_model

# ── PaySim type vocabulary ────────────────────────────────────────────────────
PAYSIM_TYPES = ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"]


# ═══════════════════════════════════════════════════════════════════════════════
# Lightweight graph container (no torch_geometric dependency)
# ═══════════════════════════════════════════════════════════════════════════════
class GraphData:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)

    @property
    def num_edges(self):
        return self.edge_index.size(1)


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
    df = df.sort_values("step").reset_index(drop=True)
    n = len(df)
    n_test = min(n_test, n - n_train - n_val)

    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask   = torch.zeros(n, dtype=torch.bool)
    test_mask  = torch.zeros(n, dtype=torch.bool)
    train_mask[:n_train]                             = True
    val_mask[n_train : n_train + n_val]              = True
    test_mask[n_train + n_val : n_train + n_val + n_test] = True

    orig_ids = df["nameOrig"].values
    dest_ids = df["nameDest"].values
    unique   = np.unique(np.concatenate([orig_ids, dest_ids]))
    vocab    = {v: i for i, v in enumerate(unique)}
    N        = len(unique)

    src_idx = np.array([vocab[v] for v in orig_ids], dtype=np.int64)
    dst_idx = np.array([vocab[v] for v in dest_ids], dtype=np.int64)

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
    fraud_rate = np.where(src_deg > 0,  src_fr / np.maximum(src_deg, 1), 0.0)

    x = torch.tensor(np.stack([
        np.log1p(total_deg), np.log1p(total_amt),
        np.log1p(mean_amt),  fraud_rate,
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
# PGD + TRADES (inline to avoid circular imports)
# ═══════════════════════════════════════════════════════════════════════════════
def _pgd(model, x, y, ei, ea, mask, eps, step, n_steps):
    x0 = x.detach()
    xa = x0 + torch.empty_like(x0).uniform_(-eps, eps)
    xa = xa.detach()
    lb = y[mask].float()
    for _ in range(n_steps):
        xa = xa.requires_grad_(True)
        loss = F.binary_cross_entropy_with_logits(
            model(xa, ei, ea, mask=mask), lb)
        g = torch.autograd.grad(loss, xa)[0]
        xa = (xa.detach() + step * g.sign()).clamp(x0 - eps, x0 + eps).detach()
    return xa


def _trades(model, data, pw, cfg):
    model.eval()
    xa = _pgd(model, data.x, data.y_edge, data.edge_index, data.edge_attr,
              data.train_mask, cfg["epsilon"], cfg["pgd_step_size"], cfg["pgd_steps"])
    model.train()
    lb = data.y_edge[data.train_mask].float()
    ln = model(data.x, data.edge_index, data.edge_attr, mask=data.train_mask)
    la = model(xa,     data.edge_index, data.edge_attr, mask=data.train_mask)
    pn = torch.sigmoid(ln).clamp(1e-6, 1-1e-6)
    pa = torch.sigmoid(la).clamp(1e-6, 1-1e-6)
    kl = (pn*(pn.log()-pa.log()) + (1-pn)*((1-pn).log()-(1-pa).log())).mean()
    return F.binary_cross_entropy_with_logits(ln, lb, pos_weight=pw) + cfg["beta"] * kl


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation + calibration
# ═══════════════════════════════════════════════════════════════════════════════
def evaluate(model, data, mask, flip=False):
    model.eval()
    with torch.no_grad():
        probs  = torch.sigmoid(
            model(data.x, data.edge_index, data.edge_attr, mask=mask)
        ).cpu().numpy()
        labels = data.y_edge[mask].cpu().numpy()
    if flip:
        probs = 1.0 - probs
    n_pos = int(labels.sum())
    if n_pos == 0 or n_pos == len(labels):
        return float("nan"), float("nan"), float("nan")
    return (roc_auc_score(labels, probs),
            average_precision_score(labels, probs),
            f1_score(labels, (probs >= 0.5).astype(int), zero_division=0))


def stackelberg_threshold(model, data, eps=0.05, outer=3, flip=False):
    best_tau, best_f1 = 0.5, 0.0
    for _ in range(outer):
        xa = data.x + (torch.rand_like(data.x) * 2 - 1).clamp(-eps, eps) * eps
        model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(
                model(xa, data.edge_index, data.edge_attr, mask=data.val_mask)
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
# Main entry point
# ═══════════════════════════════════════════════════════════════════════════════
def main(
    config_path: str,
    seed: int,
    csv_path: str | None = None,
    nrows: int | None = None,
    augment: bool = True,
    use_trades: bool = True,
    use_stackelberg: bool = True,
    out_dir: str = "results",
) -> dict:
    """Run the full AfriDef pipeline for a single seed.

    Parameters
    ----------
    config_path     : path to configs/default.yaml
    seed            : random seed
    csv_path        : PaySim CSV override (uses config path if None)
    nrows           : row limit for quick tests (None = full dataset)
    augment         : Layer 1 — WGAN-GP fraud oversampling
    use_trades      : Layer 2 — TRADES adversarial training
    use_stackelberg : Layer 3 — Stackelberg threshold adaptation
    out_dir         : results directory for JSON output
    """
    cfg = yaml.safe_load(Path(config_path).read_text())
    cfg["seed"] = seed

    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gnn_cfg = cfg["gnn"]
    tr_cfg  = cfg["trades"]

    # ── Locate PaySim CSV ─────────────────────────────────────────────────────
    candidates = [
        csv_path,
        cfg["data"].get("paysim_path"),
        "data/raw/PS_20174392719_1491204439457_log.csv",
        "data/paysim.csv",
    ]
    resolved = next((c for c in candidates if c and Path(c).exists()), None)
    if resolved is None:
        raise FileNotFoundError(
            "PaySim CSV not found. Pass csv_path= or set data.paysim_path in config."
        )

    print(f"[afridef] seed={seed} | device={device}")
    print(f"  Loading PaySim ({nrows or 'full'} rows) from {resolved} ...")
    df = pd.read_csv(resolved, nrows=nrows)
    print(f"  {len(df):,} rows | fraud rate: {df['isFraud'].mean()*100:.3f}%")

    n_orig  = len(df)
    n_test  = int(n_orig * cfg["data"]["test_temporal_split"])
    n_val   = int(n_orig * cfg["data"]["val_temporal_split"])
    n_train = n_orig - n_val - n_test

    # ── Layer 1: WGAN-GP ──────────────────────────────────────────────────────
    if augment:
        from .augment import WGANGPAugmentor
        df_s  = df.sort_values("step").reset_index(drop=True)
        tr_sl = df_s.iloc[:n_train].copy()
        rs_sl = df_s.iloc[n_train:].copy()
        aug   = WGANGPAugmentor(cfg["wgan"], device=str(device), verbose=True)
        tr_sl = aug.augment(tr_sl, oversample_ratio=cfg["wgan"]["oversample_ratio"])
        synth = tr_sl["step"] == 0
        if synth.any():
            real_steps = tr_sl.loc[~synth, "step"]
            lo, hi = int(real_steps.min()), int(real_steps.max())
            rng = np.random.default_rng(seed)
            tr_sl.loc[synth, "step"] = rng.integers(lo, hi+1, size=int(synth.sum()))
        n_train_g = len(tr_sl)
        df = pd.concat([tr_sl, rs_sl], ignore_index=True)
    else:
        n_train_g = n_train

    # ── Build graph ───────────────────────────────────────────────────────────
    print("\nBuilding graph ...")
    data = build_graph(df, n_train_g, n_val, n_test, device)

    tr_labels = data.y_edge[data.train_mask]
    n_pos = int(tr_labels.sum().item())
    n_neg = int((tr_labels == 0).sum().item())
    pw    = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float, device=device)
    print(f"  pos_weight={pw.item():.1f}  ({n_pos} fraud / {n_neg} non-fraud)")

    # ── Build model ───────────────────────────────────────────────────────────
    model = build_model(
        backbone=gnn_cfg.get("backbone", "graphsage"),
        node_in_dim=data.x.shape[1],
        edge_in_dim=data.edge_attr.shape[1],
        hidden_dim=gnn_cfg["hidden_dim"],
        num_layers=gnn_cfg["num_layers"],
        dropout=gnn_cfg["dropout"],
    ).to(device)

    opt = torch.optim.Adam(
        model.parameters(), lr=gnn_cfg["lr"], weight_decay=gnn_cfg["weight_decay"]
    )

    method = "TRADES" if use_trades else "BCE"
    print(f"\nTraining {gnn_cfg['epochs']} epochs ({method}) ...")
    best_val, best_state = 0.0, None

    for ep in range(1, gnn_cfg["epochs"] + 1):
        model.train(); opt.zero_grad()
        loss = (_trades(model, data, pw, tr_cfg) if use_trades
                else F.binary_cross_entropy_with_logits(
                    model(data.x, data.edge_index, data.edge_attr,
                          mask=data.train_mask),
                    tr_labels.float(), pos_weight=pw))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if ep % 10 == 0:
            auroc, ap, f1 = evaluate(model, data, data.val_mask)
            print(f"  ep {ep:3d} | loss={loss.item():.4f} | "
                  f"val AUROC={auroc:.4f} | AP={ap:.4f} | F1={f1:.4f}")
            if not np.isnan(auroc) and auroc > best_val:
                best_val = auroc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
        print(f"  Restored best val AUROC={best_val:.4f}")

    flip = best_val < 0.5
    if flip:
        print(f"  [Calibration] val AUROC={best_val:.4f} < 0.5 — score = 1−prob")

    # ── Layer 3: Stackelberg ──────────────────────────────────────────────────
    tau = (stackelberg_threshold(model, data, eps=0.05, outer=5, flip=flip)
           if use_stackelberg else 0.5)
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

    if labels_test.sum() == 0:
        auroc = ap = f1 = float("nan")
    else:
        auroc = roc_auc_score(labels_test, probs_test)
        ap    = average_precision_score(labels_test, probs_test)
        f1    = f1_score(labels_test, (probs_test >= tau).astype(int), zero_division=0)

    print(f"\n{'='*50}")
    print(f"SEED {seed} — TEST  AUROC={auroc:.4f} | AP={ap:.4f} | F1={f1:.4f}")
    print("=" * 50)

    result = {
        "seed": seed, "auroc": float(auroc), "ap": float(ap),
        "f1": float(f1), "tau": float(tau), "flipped": flip,
        "method": {
            "augment": augment, "trades": use_trades,
            "stackelberg": use_stackelberg,
            "backbone": gnn_cfg.get("backbone", "graphsage"),
        },
    }
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(out_dir) / f"seed_{seed}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"  Saved → {out_path}")
    return result
