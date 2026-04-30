"""
Layer 1 — Conditional WGAN-GP minority-class augmentor (AfriDef).

Trains a Wasserstein GAN with Gradient Penalty on the tabular features of
*fraud* transactions from PaySim. Generated synthetic fraud rows are appended
to the training DataFrame before graph construction, directly addressing the
extreme class imbalance (~0.2 % fraud rate).

Design:
  - Input features: [log1p(amount), type_enc, log1p(oldbalanceOrg),
                     log1p(newbalanceOrig), log1p(oldbalanceDest),
                     log1p(newbalanceDest)]  — 6 dimensions
  - Generator  : noise_dim -> hidden_dim x2 -> out_dim (MLP + LayerNorm)
  - Critic     : out_dim   -> hidden_dim x2 -> 1       (MLP, no sigmoid)
  - Training   : n_critic critic steps per generator step; gradient penalty
                 on interpolated samples (Gulrajani et al., 2017).
  - Inverse-transform: generated samples are de-normalised back to the
    original PaySim column scales so they can be concatenated with real data.

Usage (standalone):
    from afridef.augment import WGANGPAugmentor
    aug = WGANGPAugmentor(cfg["wgan"])
    aug.fit(train_df)                          # train on fraud rows only
    synthetic_df = aug.generate(n=450)         # returns a DataFrame
    augmented_df = pd.concat([train_df, synthetic_df], ignore_index=True)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Feature schema
# ---------------------------------------------------------------------------
PAYSIM_TYPES = ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"]

# Columns used as GAN input / output
FEAT_COLS = [
    "amount", "type", "oldbalanceOrg", "newbalanceOrig",
    "oldbalanceDest", "newbalanceDest",
]
FEAT_DIM = 6          # 5 raw numeric + 1 type code → all normalised to [0,1]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _encode(df: pd.DataFrame) -> np.ndarray:
    """Encode a PaySim DataFrame slice into a normalised float array."""
    type_enc = (
        pd.Categorical(df["type"], categories=PAYSIM_TYPES)
        .codes.astype(np.float32) / 4.0          # → [0, 1]
    )
    X = np.stack([
        np.log1p(df["amount"].values.astype(np.float32)),
        type_enc,
        np.log1p(df["oldbalanceOrg"].values.astype(np.float32)),
        np.log1p(df["newbalanceOrig"].values.astype(np.float32)),
        np.log1p(df["oldbalanceDest"].values.astype(np.float32)),
        np.log1p(df["newbalanceDest"].values.astype(np.float32)),
    ], axis=1)
    return X


def _decode(X: np.ndarray, dominant_type: str = "CASH_OUT") -> pd.DataFrame:
    """Inverse-transform generated samples back to PaySim column space.

    The GAN operates in log-space; we expm1 back and round the type column
    to the nearest valid integer code.  The dominant_type fallback is used
    when a generated type code rounds to an invalid value.
    """
    type_codes = np.clip(np.round(X[:, 1] * 4).astype(int), 0, 4)
    idx_to_type = {i: t for i, t in enumerate(PAYSIM_TYPES)}
    dom_code = PAYSIM_TYPES.index(dominant_type)
    types = [idx_to_type.get(c, dominant_type) for c in type_codes]

    df = pd.DataFrame({
        "step":            0,                              # placeholder
        "type":            types,
        "amount":          np.expm1(X[:, 0]).clip(0),
        "nameOrig":        "C_SYNTH",
        "oldbalanceOrg":   np.expm1(X[:, 2]).clip(0),
        "newbalanceOrig":  np.expm1(X[:, 3]).clip(0),
        "nameDest":        "M_SYNTH",
        "oldbalanceDest":  np.expm1(X[:, 4]).clip(0),
        "newbalanceDest":  np.expm1(X[:, 5]).clip(0),
        "isFraud":         1,
        "isFlaggedFraud":  0,
    })
    return df


# ---------------------------------------------------------------------------
# Network architecture
# ---------------------------------------------------------------------------
class _Generator(nn.Module):
    def __init__(self, noise_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class _Critic(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Gradient penalty
# ---------------------------------------------------------------------------
def _gradient_penalty(
    critic: _Critic,
    real: torch.Tensor,
    fake: torch.Tensor,
    device: torch.device,
    gp_weight: float = 10.0,
) -> torch.Tensor:
    """Gulrajani et al. (2017) gradient penalty."""
    alpha = torch.rand(real.size(0), 1, device=device)
    interp = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    d_interp = critic(interp)
    grads = torch.autograd.grad(
        outputs=d_interp,
        inputs=interp,
        grad_outputs=torch.ones_like(d_interp),
        create_graph=True,
        retain_graph=True,
    )[0]
    penalty = ((grads.norm(2, dim=1) - 1) ** 2).mean()
    return gp_weight * penalty


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
class WGANGPAugmentor:
    """Fits a WGAN-GP on fraud rows and generates synthetic fraud samples.

    Parameters
    ----------
    cfg : dict
        The ``wgan`` sub-dict from configs/default.yaml.
    device : str
        ``"cpu"`` (default) or ``"cuda"``.
    verbose : bool
        Print training progress every 50 epochs.
    """

    def __init__(
        self,
        cfg: dict,
        device: str = "cpu",
        verbose: bool = True,
    ):
        self.noise_dim  = cfg["noise_dim"]
        self.hidden_dim = cfg["hidden_dim"]
        self.n_critic   = cfg["n_critic"]
        self.gp_weight  = cfg["gp_weight"]
        self.lr         = cfg["lr"]
        self.betas      = tuple(cfg["betas"])
        self.epochs     = cfg["epochs"]
        self.verbose    = verbose

        self.device = torch.device(device)
        self._fitted   = False
        self._G: _Generator | None = None
        self._dom_type: str = "CASH_OUT"

        # feature stats for optional post-hoc clipping
        self._feat_min: np.ndarray | None = None
        self._feat_max: np.ndarray | None = None

    # ------------------------------------------------------------------
    def fit(self, train_df: pd.DataFrame) -> "WGANGPAugmentor":
        """Train the GAN on fraud transactions from *train_df*.

        Only rows where ``isFraud == 1`` are used.  Raises ``ValueError``
        if fewer than 10 fraud rows are found (too few to train a GAN).
        """
        fraud_df = train_df[train_df["isFraud"] == 1].copy()
        n_fraud  = len(fraud_df)
        if n_fraud < 10:
            raise ValueError(
                f"Only {n_fraud} fraud rows found — need at least 10 to train WGAN-GP. "
                "Increase --nrows or check data."
            )

        # Determine the dominant fraud transaction type (for decode fallback)
        self._dom_type = fraud_df["type"].value_counts().index[0]

        X_real = _encode(fraud_df)
        self._feat_min = X_real.min(axis=0)
        self._feat_max = X_real.max(axis=0)

        real_t = torch.tensor(X_real, dtype=torch.float32, device=self.device)

        G = _Generator(self.noise_dim, self.hidden_dim, FEAT_DIM).to(self.device)
        C = _Critic(FEAT_DIM, self.hidden_dim).to(self.device)

        opt_G = torch.optim.Adam(G.parameters(), lr=self.lr, betas=self.betas)
        opt_C = torch.optim.Adam(C.parameters(), lr=self.lr, betas=self.betas)

        if self.verbose:
            print(f"\n[WGAN-GP] Training on {n_fraud} fraud rows "
                  f"for {self.epochs} epochs …")

        for epoch in range(1, self.epochs + 1):
            # ── Critic steps ──────────────────────────────────────────
            for _ in range(self.n_critic):
                idx  = torch.randint(0, n_fraud, (min(64, n_fraud),))
                real = real_t[idx]

                z    = torch.randn(real.size(0), self.noise_dim, device=self.device)
                fake = G(z).detach()

                gp   = _gradient_penalty(C, real, fake, self.device, self.gp_weight)
                loss_C = C(fake).mean() - C(real).mean() + gp

                opt_C.zero_grad()
                loss_C.backward()
                opt_C.step()

            # ── Generator step ────────────────────────────────────────
            z      = torch.randn(min(64, n_fraud), self.noise_dim, device=self.device)
            fake   = G(z)
            loss_G = -C(fake).mean()

            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

            if self.verbose and epoch % 50 == 0:
                print(f"  epoch {epoch:4d}/{self.epochs} | "
                      f"C={loss_C.item():.4f} | G={loss_G.item():.4f}")

        self._G       = G
        self._fitted  = True
        if self.verbose:
            print("[WGAN-GP] Training complete.\n")
        return self

    # ------------------------------------------------------------------
    def generate(self, n: int) -> pd.DataFrame:
        """Generate *n* synthetic fraud rows as a PaySim-schema DataFrame."""
        if not self._fitted or self._G is None:
            raise RuntimeError("Call fit() before generate().")
        self._G.eval()
        with torch.no_grad():
            z    = torch.randn(n, self.noise_dim, device=self.device)
            fake = self._G(z).cpu().numpy()
        return _decode(fake, dominant_type=self._dom_type)

    # ------------------------------------------------------------------
    def augment(
        self,
        train_df: pd.DataFrame,
        oversample_ratio: float = 5.0,
    ) -> pd.DataFrame:
        """Fit on *train_df* and return train_df + synthetic fraud rows.

        Parameters
        ----------
        train_df : pd.DataFrame
            Training slice (real data).
        oversample_ratio : float
            How many synthetic fraud rows per real fraud row.
        """
        self.fit(train_df)
        n_fraud = int(train_df["isFraud"].sum())
        n_synth = max(1, int(n_fraud * oversample_ratio))
        synth_df = self.generate(n_synth)
        print(f"[WGAN-GP] Appending {n_synth} synthetic fraud rows "
              f"(ratio={oversample_ratio:.1f}x, real={n_fraud}).")
        return pd.concat([train_df, synth_df], ignore_index=True)
