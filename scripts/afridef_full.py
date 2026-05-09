"""
AfriDef FULL — KSM 2026
Layer 1 : WGAN-GP  synthetic fraud augmentation
Layer 2 : TRADES   adversarial training
Layer 3 : Stackelberg threshold adaptation

Kaggle-ready (no __file__, no argparse, no torch_geometric).
Compare RESULT_JSON against afridef_baseline.py for Table 2.
"""
import os, sys, json, random, glob
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

# ── Config ───────────────────────────────────────────────────────────────────
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
WGAN_N_CRIT  = 5          # critic steps per generator step
AUG_RATIO    = 2.0        # generate this × real-fraud synthetic samples

# TRADES
TRADES_BETA  = 6.0
TRADES_EPS   = 0.10
TRADES_STEPS = 7
TRADES_LR    = 0.02

# Stackelberg
STK_EPS      = 0.05
STK_OUTER    = 3

os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Locate PaySim ─────────────────────────────────────────────────────────────
def find_paysim():
    candidates = [
        "/kaggle/input/paysim1/PS_20174392719_1491204439457_log.csv",
        "/kaggle/working/data/raw/PS_20174392719_1491204439457_log.csv",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    hits = glob.glob("/kaggle/**/*log.csv", recursive=True)
    if hits:
        return hits[0]
    raise FileNotFoundError("PaySim CSV not found.")

# ── Pure-PyTorch SAGEConv ─────────────────────────────────────────────────────
class SAGEConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim * 2, out_dim)

    def forward(self, x, edge_index):
        src, dst = edge_index
        agg = torch.zeros_like(x)
        agg.scatter_add_(0, dst.unsqueeze(1).expand(-1, x.size(1)), x[src])
        deg = torch.bincount(dst, minlength=x.size(0)).float().clamp(min=1)
        agg = agg / deg.unsqueeze(1)
        return self.lin(torch.cat([x, agg], dim=1))

# ── Edge-level GraphSAGE ──────────────────────────────────────────────────────
class EdgeGraphSAGE(nn.Module):
    def __init__(self, node_in, edge_in, hidden=128, layers=2, dropout=0.3):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList([SAGEConv(node_in, hidden)])
        for _ in range(layers - 1):
            self.convs.append(SAGEConv(hidden, hidden))
        self.head = nn.Sequential(
            nn.Linear(hidden * 2 + edge_in, hidden),
            nn.ReLU(),
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

# ── LAYER 1: WGAN-GP ──────────────────────────────────────────────────────────
class Generator(nn.Module):
    def __init__(self, noise_dim, out_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim, hidden), nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden),   nn.LeakyReLU(0.2),
            nn.Linear(hidden, out_dim),
        )
    def forward(self, z): return self.net(z)

class Critic(nn.Module):
    def __init__(self, in_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden), nn.LeakyReLU(0.2),
            nn.Linear(hidden, 1),
        )
    def forward(self, x): return self.net(x)

def gradient_penalty(critic, real, fake, device):
    alpha = torch.rand(real.size(0), 1, device=device)
    interp = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    d_interp = critic(interp)
    grads = torch.autograd.grad(
        outputs=d_interp, inputs=interp,
        grad_outputs=torch.ones_like(d_interp),
        create_graph=True, retain_graph=True)[0]
    return ((grads.norm(2, dim=1) - 1) ** 2).mean()

def wgan_augment(real_feats, n_generate, device, noise_dim=16):
    """Train WGAN-GP on real fraud edge features, return synthetic samples."""
    real_feats = real_feats.to(device)
    feat_dim   = real_feats.shape[1]
    n_real     = real_feats.shape[0]

    if n_real < 4:
        # Too few real fraud samples to train GAN — just replicate with jitter
        idx  = torch.randint(0, n_real, (n_generate,), device=device)
        jitter = torch.randn(n_generate, feat_dim, device=device) * 0.01
        return real_feats[idx] + jitter

    G = Generator(noise_dim, feat_dim).to(device)
    C = Critic(feat_dim).to(device)
    opt_G = torch.optim.Adam(G.parameters(), lr=WGAN_LR, betas=(0.0, 0.9))
    opt_C = torch.optim.Adam(C.parameters(), lr=WGAN_LR, betas=(0.0, 0.9))

    bs = min(64, n_real)
    for ep in range(WGAN_EPOCHS):
        # Critic steps
        for _ in range(WGAN_N_CRIT):
            idx  = torch.randint(0, n_real, (bs,), device=device)
            real = real_feats[idx]
            z    = torch.randn(bs, noise_dim, device=device)
            fake = G(z).detach()
            gp   = gradient_penalty(C, real, fake, device)
            loss_C = C(fake).mean() - C(real).mean() + WGAN_GP_LAM * gp
            opt_C.zero_grad(); loss_C.backward(); opt_C.step()
        # Generator step
        z      = torch.randn(bs, noise_dim, device=device)
        fake   = G(z)
        loss_G = -C(fake).mean()
        opt_G.zero_grad(); loss_G.backward(); opt_G.step()

    G.eval()
    with torch.no_grad():
        z_all = torch.randn(n_generate, noise_dim, device=device)
        synth = G(z_all)
    print(f"  WGAN-GP: generated {n_generate} synthetic fraud edge features")
    return synth

# ── LAYER 2: TRADES ───────────────────────────────────────────────────────────
def trades_loss(model, x, y, ei, ea, mask, pos_weight, device):
    """TRADES: BCE(clean) + beta * KL(clean || adv)"""
    model.eval()
    x_adv = x.detach().clone()
    x_adv = x_adv + 0.001 * torch.randn_like(x_adv)

    for _ in range(TRADES_STEPS):
        x_adv.requires_grad_(True)
        logits_adv   = model(x_adv, ei, ea, mask=mask)
        logits_clean = model(x,     ei, ea, mask=mask).detach()
        p_clean = torch.sigmoid(logits_clean)
        p_adv   = torch.sigmoid(logits_adv)
        # KL(clean || adv)
        kl = F.kl_div(
            torch.log(p_adv.clamp(1e-7, 1 - 1e-7)),
            p_clean.clamp(1e-7, 1 - 1e-7),
            reduction="batchmean",
        )
        kl.backward()
        with torch.no_grad():
            x_adv = x_adv + TRADES_LR * x_adv.grad.sign()
            x_adv = torch.max(torch.min(x_adv, x + TRADES_EPS), x - TRADES_EPS)
            x_adv = x_adv.detach()

    model.train()
    logits_clean = model(x,     ei, ea, mask=mask)
    logits_adv   = model(x_adv, ei, ea, mask=mask)

    loss_bce = F.binary_cross_entropy_with_logits(
        logits_clean, y[mask], pos_weight=pos_weight.to(device))
    p_clean = torch.sigmoid(logits_clean).clamp(1e-7, 1 - 1e-7)
    p_adv   = torch.sigmoid(logits_adv  ).clamp(1e-7, 1 - 1e-7)
    loss_kl = F.kl_div(torch.log(p_adv), p_clean, reduction="batchmean")

    return loss_bce + TRADES_BETA * loss_kl

# ── Graph builder (with optional synthetic fraud edges) ───────────────────────
def build_graph(df, n_train, n_val, n_test, device, synth_edge_attr=None):
    df = df.sort_values("step").reset_index(drop=True)
    n  = len(df)
    n_test = min(n_test, n - n_train - n_val)

    tm = torch.zeros(n, dtype=torch.bool)
    vm = torch.zeros(n, dtype=torch.bool)
    xm = torch.zeros(n, dtype=torch.bool)
    tm[:n_train]                            = True
    vm[n_train:n_train+n_val]              = True
    xm[n_train+n_val:n_train+n_val+n_test] = True

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
    ], axis=1), dtype=torch.float, device=device)

    y  = torch.tensor(df["isFraud"].values, dtype=torch.float, device=device)
    ei = torch.tensor(np.stack([src, dst]), dtype=torch.long, device=device)

    # Node features from training split
    ts, td = src[:n_train], dst[:n_train]
    ta = df["amount"].values[:n_train].astype(np.float32)
    tf = df["isFraud"].values[:n_train].astype(np.float32)
    sd = np.bincount(ts, minlength=N).astype(np.float32)
    dd = np.bincount(td, minlength=N).astype(np.float32)
    sa = np.bincount(ts, weights=ta, minlength=N).astype(np.float32)
    da = np.bincount(td, weights=ta, minlength=N).astype(np.float32)
    sf = np.bincount(ts, weights=tf, minlength=N).astype(np.float32)
    total_d = sd + dd
    total_a = sa + da
    mean_a  = np.where(total_d > 0, total_a / np.maximum(total_d, 1), 0.)
    fr_rate = np.where(sd > 0, sf / np.maximum(sd, 1), 0.)
    x = torch.tensor(np.stack([
        np.log1p(total_d), np.log1p(total_a), np.log1p(mean_a), fr_rate,
    ], axis=1).astype(np.float32), dtype=torch.float, device=device)

    # Inject WGAN-GP synthetic fraud edges into training mask
    if synth_edge_attr is not None and len(synth_edge_attr) > 0:
        n_synth = len(synth_edge_attr)
        # Use random high-fraud-rate source/dest nodes for synthetic edges
        fraud_src_idx = np.where(fr_rate > 0)[0]
        if len(fraud_src_idx) == 0:
            fraud_src_idx = np.arange(N)
        s_synth = torch.tensor(
            np.random.choice(fraud_src_idx, n_synth), dtype=torch.long, device=device)
        d_synth = torch.tensor(
            np.random.choice(N, n_synth), dtype=torch.long, device=device)
        syn_ea  = synth_edge_attr.to(device)
        syn_y   = torch.ones(n_synth, dtype=torch.float)   # CPU — moved to device at end
        syn_tm  = torch.ones(n_synth, dtype=torch.bool)
        syn_vm  = torch.zeros(n_synth, dtype=torch.bool)
        syn_xm  = torch.zeros(n_synth, dtype=torch.bool)

        # All mask/label tensors on CPU until the final .to(device) at the end
        ei         = torch.cat([ei.cpu(), torch.stack([s_synth.cpu(), d_synth.cpu()])], dim=1)
        edge_attr  = torch.cat([edge_attr.cpu(), syn_ea.cpu()], dim=0)
        y          = torch.cat([y.cpu(),  syn_y],  dim=0)
        tm         = torch.cat([tm, syn_tm], dim=0)
        vm         = torch.cat([vm, syn_vm], dim=0)
        xm         = torch.cat([xm, syn_xm], dim=0)

    n_total = y.shape[0]
    print(f"  Graph: {N:,} nodes | {n_total:,} edges "
          f"(+{n_total-n:,} synthetic) | "
          f"train fraud={int(y[tm].sum())} val={int(y[vm].sum())} "
          f"test={int(y[xm].sum())}")

    class Data: pass
    d = Data()
    d.x          = x.to(device)
    d.edge_index = ei.to(device)
    d.edge_attr  = edge_attr.to(device)
    d.y          = y.to(device)
    d.train_mask = tm.to(device)
    d.val_mask   = vm.to(device)
    d.test_mask  = xm.to(device)
    return d

# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate(model, data, mask, flip=False):
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index, data.edge_attr, mask=mask)
        probs  = torch.sigmoid(logits).cpu().numpy()
        labels = data.y[mask].cpu().numpy()
    if flip: probs = 1.0 - probs
    n_pos = int(labels.sum())
    if n_pos == 0 or n_pos == len(labels):
        return float("nan"), float("nan"), float("nan")
    return (roc_auc_score(labels, probs),
            average_precision_score(labels, probs),
            f1_score(labels, (probs >= 0.5).astype(int), zero_division=0))

# ── LAYER 3: Stackelberg threshold ────────────────────────────────────────────
def stackelberg_threshold(model, data, eps=0.05, outer=3, flip=False):
    best_tau, best_f1 = 0.5, 0.0
    x_clean = data.x.clone()
    for _ in range(outer):
        x_adv = x_clean + (torch.rand_like(x_clean)*2-1).clamp(-eps,eps)*eps
        model.eval()
        with torch.no_grad():
            logits = model(x_adv, data.edge_index, data.edge_attr,
                           mask=data.val_mask)
            probs  = torch.sigmoid(logits).cpu().numpy()
        if flip: probs = 1.0 - probs
        labels = data.y[data.val_mask].cpu().numpy()
        if labels.sum() == 0: continue
        for tau in np.arange(0.1, 0.9, 0.05):
            f1 = f1_score(labels, (probs >= tau).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_tau = f1, tau
    return best_tau

# ── Per-seed runner ───────────────────────────────────────────────────────────
def run_seed(seed, df_orig, device):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    n_orig  = len(df_orig)
    n_test  = int(n_orig * TEST_FRAC)
    n_val   = int(n_orig * VAL_FRAC)
    n_train = n_orig - n_val - n_test

    df_sorted = df_orig.sort_values("step").reset_index(drop=True)

    # ── LAYER 1: WGAN-GP augmentation ────────────────────────────────────────
    print("  [Layer 1] WGAN-GP augmentation ...")
    type_enc_train = (pd.Categorical(
        df_sorted["type"].iloc[:n_train], categories=PAYSIM_TYPES)
        .codes.astype(np.float32) / 4.0)
    ea_train = np.stack([
        np.log1p(df_sorted["amount"].values[:n_train].astype(np.float32)),
        type_enc_train,
        np.log1p(df_sorted["oldbalanceOrg"].values[:n_train].astype(np.float32)),
        np.log1p(df_sorted["newbalanceOrig"].values[:n_train].astype(np.float32)),
    ], axis=1)
    fraud_mask_train = df_sorted["isFraud"].values[:n_train].astype(bool)
    real_fraud_ea    = torch.tensor(ea_train[fraud_mask_train], dtype=torch.float)
    n_real_fraud     = real_fraud_ea.shape[0]
    n_synth          = max(0, int(n_real_fraud * AUG_RATIO) - n_real_fraud)

    synth_ea = None
    if n_synth > 0:
        synth_ea = wgan_augment(real_fraud_ea, n_synth, device)
    else:
        print(f"  WGAN-GP: {n_real_fraud} real fraud — no augmentation needed")

    data = build_graph(df_sorted, n_train, n_val, n_test, device, synth_ea)

    tr_labels = data.y[data.train_mask]
    n_pos = int(tr_labels.sum())
    n_neg = int((tr_labels == 0).sum())
    pw    = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float, device=device)
    print(f"  pos_weight={pw.item():.1f}  (train fraud={n_pos} / non-fraud={n_neg})")

    model = EdgeGraphSAGE(
        node_in=data.x.shape[1], edge_in=data.edge_attr.shape[1],
        hidden=HIDDEN_DIM, layers=N_LAYERS, dropout=DROPOUT,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)

    # ── LAYER 2: TRADES training ──────────────────────────────────────────────
    print(f"  [Layer 2] TRADES training {N_EPOCHS} epochs (β={TRADES_BETA}) ...")
    best_val, best_state = 0.0, None
    for ep in range(1, N_EPOCHS + 1):
        opt.zero_grad()
        loss = trades_loss(model, data.x, data.y,
                           data.edge_index, data.edge_attr,
                           data.train_mask, pw, device)
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

    flip = False
    if best_val < 0.5:
        flip = True
        print(f"  [CALIBRATION] val AUROC={best_val:.4f} < 0.5 — flipping")

    # ── LAYER 3: Stackelberg threshold ────────────────────────────────────────
    print("  [Layer 3] Stackelberg threshold adaptation ...")
    tau = stackelberg_threshold(model, data, eps=STK_EPS, outer=STK_OUTER, flip=flip)
    print(f"  Stackelberg adapted threshold: τ={tau:.3f}")

    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index, data.edge_attr, mask=data.test_mask)
        probs  = torch.sigmoid(logits).cpu().numpy()
    if flip: probs = 1.0 - probs
    labels     = data.y[data.test_mask].cpu().numpy()
    n_pos_test = int(labels.sum())

    if n_pos_test == 0:
        print("  [WARN] No fraud in test set — NaN")
        auroc = ap = f1 = float("nan")
    else:
        auroc  = roc_auc_score(labels, probs)
        ap     = average_precision_score(labels, probs)
        f1_05  = f1_score(labels, (probs >= 0.5).astype(int), zero_division=0)
        f1_tau = f1_score(labels, (probs >= tau).astype(int), zero_division=0)
        print(f"  TEST @ τ=0.50 : AUROC={auroc:.4f} | AP={ap:.4f} | F1={f1_05:.4f}"
              f"{'  [FLIPPED]' if flip else ''}")
        print(f"  TEST @ τ={tau:.3f}: AUROC={auroc:.4f} | AP={ap:.4f} | F1={f1_tau:.4f}")
        f1 = f1_tau

    result = {
        "seed": seed, "auroc": float(auroc), "ap": float(ap), "f1": float(f1),
        "tau": float(tau), "flipped": flip,
    }
    with open(f"{RESULTS_DIR}/seed_{seed}.json", "w") as fh:
        json.dump(result, fh)
    print(f"  Seed {seed} done: {result}")
    return result

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"AfriDef FULL | WGAN-GP + TRADES(β={TRADES_BETA}) + Stackelberg | SEEDS={SEEDS}")

    csv_path = find_paysim()
    print(f"Loading PaySim ({NROWS:,} rows) from {csv_path} ...")
    df = pd.read_csv(csv_path, nrows=NROWS)
    print(f"  {len(df):,} rows | fraud rate: {df['isFraud'].mean()*100:.3f}%")

    all_results = []
    for seed in SEEDS:
        print(f"\n{'='*60}")
        print(f"SEED {seed}")
        print("=" * 60)
        res = run_seed(seed, df, device)
        all_results.append(res)

    print(f"\n{'='*60}")
    print("FINAL SUMMARY — AfriDef FULL (WGAN-GP + TRADES + Stackelberg)")
    print("=" * 60)
    valid = [r for r in all_results if not np.isnan(r["auroc"])]
    if valid:
        mean_auroc = np.mean([r["auroc"] for r in valid])
        std_auroc  = np.std( [r["auroc"] for r in valid])
        mean_ap    = np.mean([r["ap"]    for r in valid])
        mean_f1    = np.mean([r["f1"]    for r in valid])
        std_f1     = np.std( [r["f1"]    for r in valid])
        n_flipped  = sum(1 for r in valid if r.get("flipped", False))
        print(f"  AUROC  : {mean_auroc:.4f} ± {std_auroc:.4f}")
        print(f"  AP     : {mean_ap:.4f}")
        print(f"  F1     : {mean_f1:.4f} ± {std_f1:.4f}")
        print(f"  Flipped: {n_flipped}/{len(valid)} seeds")
        summary = {
            "auroc_mean": round(mean_auroc, 4),
            "auroc_std":  round(std_auroc,  4),
            "ap_mean":    round(mean_ap,     4),
            "f1_mean":    round(mean_f1,     4),
            "f1_std":     round(std_f1,      4),
            "n_flipped":  n_flipped,
            "method":     "afridef_full_wgangp_trades_stackelberg",
            "seeds":      all_results,
        }
        with open(f"{RESULTS_DIR}/summary.json", "w") as fh:
            json.dump(summary, fh, indent=2)
        print(f"  Saved → {RESULTS_DIR}/summary.json")
        print(f"\nRESULT_JSON:" + json.dumps(summary))
    else:
        print("  No valid results.")
