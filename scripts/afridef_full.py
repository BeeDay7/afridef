"""
AfriDef FULL — KSM 2026
Layer 1 : WGAN-GP  synthetic fraud augmentation  (numpy-level, no device issues)
Layer 2 : TRADES   adversarial training
Layer 3 : Stackelberg threshold adaptation

Kaggle-ready — no __file__, no argparse, no torch_geometric.
"""
import os, json, random, glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

# ── Config ────────────────────────────────────────────────────────────────────
SEEDS        = [0, 1, 2, 3, 4]
N_EPOCHS     = 100
HIDDEN_DIM   = 128
N_LAYERS     = 2
DROPOUT      = 0.3
LR           = 1e-3
WD           = 1e-5
NROWS        = 300_000
VAL_FRAC     = 0.10
TEST_FRAC    = 0.20
RESULTS_DIR  = "/kaggle/working/afridef_results_full"
PAYSIM_TYPES = ["PAYMENT","TRANSFER","CASH_OUT","DEBIT","CASH_IN"]

# WGAN-GP
WGAN_EPOCHS  = 60
WGAN_LR      = 1e-4
WGAN_GP_LAM  = 10.0
WGAN_N_CRIT  = 5
AUG_RATIO    = 2.0       # target: total fraud = AUG_RATIO × original fraud count

# TRADES
TRADES_BETA  = 6.0
TRADES_EPS   = 0.10
TRADES_STEPS = 7
TRADES_ALPHA = 0.02

# Stackelberg
STK_EPS   = 0.05
STK_OUTER = 3

os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Locate PaySim ──────────────────────────────────────────────────────────────
def find_paysim():
    for c in [
        "/kaggle/input/paysim1/PS_20174392719_1491204439457_log.csv",
        "/kaggle/working/data/raw/PS_20174392719_1491204439457_log.csv",
    ]:
        if os.path.exists(c):
            return c
    hits = glob.glob("/kaggle/**/*log.csv", recursive=True)
    if hits:
        return hits[0]
    raise FileNotFoundError("PaySim CSV not found.")

# ── LAYER 1: WGAN-GP (pure numpy/CPU output) ──────────────────────────────────
class _Gen(nn.Module):
    def __init__(self, z_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 64), nn.LeakyReLU(0.2),
            nn.Linear(64, 64),    nn.LeakyReLU(0.2),
            nn.Linear(64, out_dim),
        )
    def forward(self, z): return self.net(z)

class _Crit(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.LeakyReLU(0.2),
            nn.Linear(64, 64),     nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
        )
    def forward(self, x): return self.net(x)

def _gp(crit, real, fake):
    """Gradient penalty (CPU tensors only)."""
    alpha = torch.rand(real.size(0), 1)
    mix   = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    d     = crit(mix)
    g     = torch.autograd.grad(d, mix,
                grad_outputs=torch.ones_like(d),
                create_graph=True, retain_graph=True)[0]
    return ((g.norm(2, dim=1) - 1) ** 2).mean()

def wgan_augment_numpy(real_np, n_generate, z_dim=16, seed=0):
    """
    Train a WGAN-GP on real fraud features (numpy array, shape [N, D]).
    Returns numpy array of synthetic samples [n_generate, D].
    Everything runs on CPU — no CUDA device issues.
    """
    torch.manual_seed(seed)
    real = torch.tensor(real_np, dtype=torch.float32)  # CPU
    n, D = real.shape

    if n < 4:
        idx = np.random.randint(0, n, n_generate)
        return real_np[idx] + np.random.randn(n_generate, D) * 0.01

    G = _Gen(z_dim, D)
    C = _Crit(D)
    oG = torch.optim.Adam(G.parameters(), lr=WGAN_LR, betas=(0.0, 0.9))
    oC = torch.optim.Adam(C.parameters(), lr=WGAN_LR, betas=(0.0, 0.9))

    bs = min(64, n)
    for ep in range(WGAN_EPOCHS):
        for _ in range(WGAN_N_CRIT):
            idx  = torch.randint(0, n, (bs,))
            r    = real[idx]
            f    = G(torch.randn(bs, z_dim)).detach()
            lC   = C(f).mean() - C(r).mean() + WGAN_GP_LAM * _gp(C, r, f)
            oC.zero_grad(); lC.backward(); oC.step()
        z    = torch.randn(bs, z_dim)
        lG   = -C(G(z)).mean()
        oG.zero_grad(); lG.backward(); oG.step()

    G.eval()
    with torch.no_grad():
        synth = G(torch.randn(n_generate, z_dim)).numpy()
    print(f"  WGAN-GP: generated {n_generate} synthetic fraud samples")
    return synth

def augment_dataframe(df_train, seed=0):
    """
    Layer 1: use WGAN-GP to double the fraud rows in df_train.
    Returns augmented DataFrame (still CPU / pandas — no torch device issues).
    """
    fraud_df = df_train[df_train["isFraud"] == 1].copy()
    n_real   = len(fraud_df)
    n_synth  = max(0, int(n_real * AUG_RATIO) - n_real)
    if n_synth == 0:
        print(f"  WGAN-GP: {n_real} fraud rows — no augmentation needed")
        return df_train

    # Features to model with GAN (continuous columns)
    feat_cols = ["amount", "oldbalanceOrg", "newbalanceOrig",
                 "oldbalanceDest", "newbalanceDest"]
    real_np = np.log1p(fraud_df[feat_cols].values.astype(np.float32))

    synth_log = wgan_augment_numpy(real_np, n_synth, seed=seed)
    synth_raw = np.expm1(synth_log).clip(min=0)

    # Build synthetic rows by sampling real fraud rows and overwriting features
    np.random.seed(seed)
    base_idx   = np.random.choice(fraud_df.index, n_synth, replace=True)
    synth_rows = fraud_df.loc[base_idx].copy().reset_index(drop=True)
    for i, col in enumerate(feat_cols):
        synth_rows[col] = synth_raw[:, i]
    synth_rows["isFraud"] = 1

    aug = pd.concat([df_train, synth_rows], ignore_index=True)
    print(f"  WGAN-GP: df_train {len(df_train):,} → {len(aug):,} rows "
          f"(fraud: {n_real} → {n_real + n_synth})")
    return aug

# ── Pure-PyTorch SAGEConv ──────────────────────────────────────────────────────
class SAGEConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim * 2, out_dim)

    def forward(self, x, edge_index):
        src, dst = edge_index
        agg = torch.zeros_like(x)
        agg.scatter_add_(0, dst.unsqueeze(1).expand(-1, x.size(1)), x[src])
        deg = torch.bincount(dst, minlength=x.size(0)).float().clamp(min=1)
        return self.lin(torch.cat([x, agg / deg.unsqueeze(1)], dim=1))

# ── Edge-level GraphSAGE ───────────────────────────────────────────────────────
class EdgeGraphSAGE(nn.Module):
    def __init__(self, node_in, edge_in, hidden=128, layers=2, dropout=0.3):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList([SAGEConv(node_in, hidden)])
        for _ in range(layers - 1):
            self.convs.append(SAGEConv(hidden, hidden))
        self.head = nn.Sequential(
            nn.Linear(hidden * 2 + edge_in, hidden), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def encode(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index)

    def forward(self, x, edge_index, edge_attr, mask=None):
        h = self.encode(x, edge_index)
        s, d = edge_index
        if mask is not None:
            s, d, edge_attr = s[mask], d[mask], edge_attr[mask]
        return self.head(torch.cat([h[s], h[d], edge_attr], dim=-1)).squeeze(-1)

# ── Graph builder ──────────────────────────────────────────────────────────────
def build_graph(df, device):
    """Build graph from a (possibly augmented) DataFrame. All tensors → device."""
    df = df.sort_values("step").reset_index(drop=True)
    n  = len(df)

    orig_ids = df["nameOrig"].values
    dest_ids = df["nameDest"].values
    unique   = np.unique(np.concatenate([orig_ids, dest_ids]))
    vocab    = {v: i for i, v in enumerate(unique)}
    N        = len(unique)

    src = np.array([vocab[v] for v in orig_ids], dtype=np.int64)
    dst = np.array([vocab[v] for v in dest_ids], dtype=np.int64)

    type_enc = (pd.Categorical(df["type"], categories=PAYSIM_TYPES)
                .codes.astype(np.float32) / 4.0)
    edge_attr = torch.tensor(np.stack([
        np.log1p(df["amount"].values.astype(np.float32)),
        type_enc,
        np.log1p(df["oldbalanceOrg"].values.astype(np.float32)),
        np.log1p(df["newbalanceOrig"].values.astype(np.float32)),
    ], axis=1), dtype=torch.float).to(device)

    y  = torch.tensor(df["isFraud"].values, dtype=torch.float).to(device)
    ei = torch.tensor(np.stack([src, dst]),  dtype=torch.long ).to(device)

    # Node features (computed from all rows — augmented rows already included)
    sd = np.bincount(src, minlength=N).astype(np.float32)
    dd = np.bincount(dst, minlength=N).astype(np.float32)
    ta = df["amount"].values.astype(np.float32)
    tf = df["isFraud"].values.astype(np.float32)
    sa = np.bincount(src, weights=ta, minlength=N).astype(np.float32)
    da = np.bincount(dst, weights=ta, minlength=N).astype(np.float32)
    sf = np.bincount(src, weights=tf, minlength=N).astype(np.float32)
    td = sd + dd
    ta_total = sa + da
    mean_a   = np.where(td > 0, ta_total / np.maximum(td, 1), 0.)
    fr_rate  = np.where(sd > 0, sf / np.maximum(sd, 1), 0.)
    x = torch.tensor(np.stack([
        np.log1p(td), np.log1p(ta_total), np.log1p(mean_a), fr_rate,
    ], axis=1).astype(np.float32)).to(device)

    class Data: pass
    d = Data()
    d.x, d.edge_index, d.edge_attr, d.y = x, ei, edge_attr, y
    d.N = N
    return d

# ── LAYER 2: TRADES loss ───────────────────────────────────────────────────────
def trades_loss(model, data, mask, pos_weight):
    """TRADES: BCE(clean) + beta * KL(clean ‖ adv)."""
    x, ei, ea = data.x, data.edge_index, data.edge_attr
    device = x.device

    model.eval()
    x_adv = x.detach() + 1e-3 * torch.randn_like(x)
    for _ in range(TRADES_STEPS):
        x_adv = x_adv.detach().requires_grad_(True)
        with torch.enable_grad():
            p_adv   = torch.sigmoid(model(x_adv, ei, ea, mask=mask))
            p_clean = torch.sigmoid(model(x,     ei, ea, mask=mask)).detach()
            kl = F.kl_div(
                (p_adv  .clamp(1e-7, 1-1e-7)).log(),
                p_clean .clamp(1e-7, 1-1e-7),
                reduction="batchmean",
            )
            kl.backward()
        x_adv = x_adv + TRADES_ALPHA * x_adv.grad.sign()
        x_adv = torch.max(torch.min(x_adv, x + TRADES_EPS), x - TRADES_EPS).detach()

    model.train()
    lg_clean = model(x,     ei, ea, mask=mask)
    lg_adv   = model(x_adv, ei, ea, mask=mask)
    bce = F.binary_cross_entropy_with_logits(
        lg_clean, data.y[mask], pos_weight=pos_weight.to(device))
    p_c = torch.sigmoid(lg_clean).clamp(1e-7, 1-1e-7)
    p_a = torch.sigmoid(lg_adv  ).clamp(1e-7, 1-1e-7)
    kl  = F.kl_div(p_a.log(), p_c, reduction="batchmean")
    return bce + TRADES_BETA * kl

# ── Evaluation ─────────────────────────────────────────────────────────────────
def evaluate(model, data, mask, flip=False):
    model.eval()
    with torch.no_grad():
        p = torch.sigmoid(model(data.x, data.edge_index,
                                data.edge_attr, mask=mask)).cpu().numpy()
        y = data.y[mask].cpu().numpy()
    if flip: p = 1.0 - p
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan"), float("nan"), float("nan")
    return (roc_auc_score(y, p),
            average_precision_score(y, p),
            f1_score(y, (p >= .5).astype(int), zero_division=0))

# ── LAYER 3: Stackelberg threshold ─────────────────────────────────────────────
def stackelberg_threshold(model, data, val_mask, flip=False):
    best_tau, best_f1 = 0.5, 0.0
    for _ in range(STK_OUTER):
        x_adv = data.x + (torch.rand_like(data.x)*2-1).clamp(-STK_EPS, STK_EPS)*STK_EPS
        model.eval()
        with torch.no_grad():
            p = torch.sigmoid(model(x_adv, data.edge_index,
                                    data.edge_attr, mask=val_mask)).cpu().numpy()
        if flip: p = 1.0 - p
        y = data.y[val_mask].cpu().numpy()
        if y.sum() == 0: continue
        for tau in np.arange(0.1, 0.9, 0.05):
            f1 = f1_score(y, (p >= tau).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_tau = f1, tau
    return best_tau

# ── Per-seed runner ────────────────────────────────────────────────────────────
def run_seed(seed, df_full, device):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    # Chronological split indices
    df_s   = df_full.sort_values("step").reset_index(drop=True)
    n      = len(df_s)
    n_test = int(n * TEST_FRAC)
    n_val  = int(n * VAL_FRAC)
    n_tr   = n - n_val - n_test

    df_train = df_s.iloc[:n_tr].copy()
    df_val   = df_s.iloc[n_tr:n_tr+n_val].copy()
    df_test  = df_s.iloc[n_tr+n_val:].copy()

    # ── Layer 1: WGAN-GP — augment training DataFrame (pure numpy/CPU) ────────
    print("  [Layer 1] WGAN-GP augmentation ...")
    df_train_aug = augment_dataframe(df_train, seed=seed)

    # Build graphs (train augmented, val/test clean)
    # Merge to single graph for GNN message passing, use index masks
    df_all    = pd.concat([df_train_aug, df_val, df_test], ignore_index=True)
    n_tr_aug  = len(df_train_aug)
    n_va      = len(df_val)
    n_te      = len(df_test)
    n_all     = len(df_all)

    data = build_graph(df_all, device)

    # Masks — indices into the sorted-concat df
    train_idx = torch.zeros(n_all, dtype=torch.bool)
    val_idx   = torch.zeros(n_all, dtype=torch.bool)
    test_idx  = torch.zeros(n_all, dtype=torch.bool)
    train_idx[:n_tr_aug]                      = True
    val_idx  [n_tr_aug:n_tr_aug+n_va]        = True
    test_idx [n_tr_aug+n_va:]                = True
    data.train_mask = train_idx.to(device)
    data.val_mask   = val_idx.to(device)
    data.test_mask  = test_idx.to(device)

    print(f"  Graph: {data.N:,} nodes | {n_all:,} edges | "
          f"train fraud={int(data.y[data.train_mask].sum())} "
          f"val={int(data.y[data.val_mask].sum())} "
          f"test={int(data.y[data.test_mask].sum())}")

    n_pos = int(data.y[data.train_mask].sum())
    n_neg = int((data.y[data.train_mask] == 0).sum())
    pw    = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float)
    print(f"  pos_weight={pw.item():.1f}  (train: fraud={n_pos} non-fraud={n_neg})")

    model = EdgeGraphSAGE(
        node_in=data.x.shape[1], edge_in=data.edge_attr.shape[1],
        hidden=HIDDEN_DIM, layers=N_LAYERS, dropout=DROPOUT,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)

    # ── Layer 2: TRADES adversarial training ──────────────────────────────────
    print(f"  [Layer 2] TRADES training {N_EPOCHS} epochs (β={TRADES_BETA}) ...")
    best_val, best_state = 0.0, None
    for ep in range(1, N_EPOCHS + 1):
        opt.zero_grad()
        loss = trades_loss(model, data, data.train_mask, pw)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if ep % 10 == 0:
            auroc, ap, f1 = evaluate(model, data, data.val_mask)
            print(f"  ep {ep:3d} | loss={loss.item():.4f} | "
                  f"val AUROC={auroc:.4f} | AP={ap:.4f} | F1={f1:.4f}")
            if not np.isnan(auroc) and auroc > best_val:
                best_val  = auroc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
        print(f"  Restored best val AUROC={best_val:.4f}")

    flip = False
    if best_val < 0.5:
        flip = True
        print(f"  [CALIBRATION] Flipping (val AUROC={best_val:.4f})")

    # ── Layer 3: Stackelberg threshold ────────────────────────────────────────
    print("  [Layer 3] Stackelberg threshold adaptation ...")
    tau = stackelberg_threshold(model, data, data.val_mask, flip=flip)
    print(f"  τ={tau:.3f}")

    # Test evaluation
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(data.x, data.edge_index,
                                    data.edge_attr, mask=data.test_mask)).cpu().numpy()
    if flip: probs = 1.0 - probs
    labels = data.y[data.test_mask].cpu().numpy()

    if labels.sum() == 0:
        auroc = ap = f1 = float("nan")
        print("  [WARN] No fraud in test — NaN")
    else:
        auroc  = roc_auc_score(labels, probs)
        ap     = average_precision_score(labels, probs)
        f1_05  = f1_score(labels, (probs >= 0.5).astype(int), zero_division=0)
        f1_tau = f1_score(labels, (probs >= tau).astype(int), zero_division=0)
        print(f"  TEST τ=0.50: AUROC={auroc:.4f} AP={ap:.4f} F1={f1_05:.4f}"
              f"{'  [FLIPPED]' if flip else ''}")
        print(f"  TEST τ={tau:.3f}: AUROC={auroc:.4f} AP={ap:.4f} F1={f1_tau:.4f}")
        f1 = f1_tau

    result = {"seed": seed, "auroc": float(auroc), "ap": float(ap),
              "f1": float(f1), "tau": float(tau), "flipped": flip}
    with open(f"{RESULTS_DIR}/seed_{seed}.json", "w") as fh:
        json.dump(result, fh)
    print(f"  Seed {seed}: {result}")
    return result

# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"AfriDef FULL | WGAN-GP + TRADES(β={TRADES_BETA}) + Stackelberg | SEEDS={SEEDS}")

    csv_path = find_paysim()
    print(f"Loading {NROWS:,} rows from {csv_path} ...")
    df = pd.read_csv(csv_path, nrows=NROWS)
    print(f"  {len(df):,} rows | fraud rate: {df['isFraud'].mean()*100:.3f}%")

    all_results = []
    for seed in SEEDS:
        print(f"\n{'='*60}\nSEED {seed}\n{'='*60}")
        all_results.append(run_seed(seed, df, device))

    print(f"\n{'='*60}")
    print("FINAL SUMMARY — AfriDef FULL (WGAN-GP + TRADES + Stackelberg)")
    print("=" * 60)
    valid = [r for r in all_results if not np.isnan(r["auroc"])]
    if valid:
        mu_a = np.mean([r["auroc"] for r in valid])
        sd_a = np.std( [r["auroc"] for r in valid])
        mu_p = np.mean([r["ap"]    for r in valid])
        mu_f = np.mean([r["f1"]    for r in valid])
        sd_f = np.std( [r["f1"]    for r in valid])
        nfl  = sum(1 for r in valid if r.get("flipped"))
        print(f"  AUROC  : {mu_a:.4f} ± {sd_a:.4f}")
        print(f"  AP     : {mu_p:.4f}")
        print(f"  F1     : {mu_f:.4f} ± {sd_f:.4f}")
        print(f"  Flipped: {nfl}/{len(valid)} seeds")
        summary = {
            "auroc_mean": round(mu_a,4), "auroc_std": round(sd_a,4),
            "ap_mean":    round(mu_p,4),
            "f1_mean":    round(mu_f,4), "f1_std":    round(sd_f,4),
            "n_flipped":  nfl,
            "method":     "afridef_full_wgangp_trades_stackelberg",
            "seeds":      all_results,
        }
        with open(f"{RESULTS_DIR}/summary.json", "w") as fh:
            json.dump(summary, fh, indent=2)
        print(f"  Saved → {RESULTS_DIR}/summary.json")
        print(f"\nRESULT_JSON:" + json.dumps(summary))
    else:
        print("  No valid results.")
