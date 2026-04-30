"""Layer 3: Stackelberg threshold-adaptation outer loop (AfriDef §4.3).

Game formulation (Brückner & Scheffer 2011; Stackelberg variant):
  - Defender (leader) commits to detection threshold τ ∈ [0,1].
  - Attacker (follower) trains a PPO policy π that perturbs transaction
    NODE FEATURES within L_inf(eps) to maximise evasion: P(score < τ).
  - Defender observes attacker behaviour and re-tunes τ by grid search
    on the held-out validation set under the latest attacker.
  - Repeat for outer_iterations rounds or until |Δτ| < convergence_tol.

Gymnasium environment (FraudEvasionEnv):
  - Single-step episode: sample one fraud training edge, output perturbation.
  - Observation : [src_node_features ∥ dst_node_features]  (2 × node_dim)
  - Action      : continuous vector in [-1, 1]^node_dim, scaled by eps.
  - Reward      : +1.0 if perturbed score < τ (evasion), −0.1 otherwise.

This module is self-contained; import stackelberg_loop into the training
script after the GNN has been trained.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO


# ---------------------------------------------------------------------------
# State container
# ---------------------------------------------------------------------------
@dataclass
class StackelbergState:
    threshold: float
    attacker_policy: Optional[object]
    defender_utility_history: List[float] = field(default_factory=list)
    attacker_utility_history: List[float] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"  Final threshold τ : {self.threshold:.3f}",
            f"  Defender F1 hist  : {[round(v,4) for v in self.defender_utility_history]}",
            f"  Evasion rate hist : {[round(v,3) for v in self.attacker_utility_history]}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Gymnasium environment
# ---------------------------------------------------------------------------
class FraudEvasionEnv(gym.Env):
    """Single-step fraud evasion MDP for the PPO attacker.

    Each episode:
      1. Sample a random fraud training edge.
      2. Attacker outputs a perturbation vector (action).
      3. Reward is +1 if the perturbed edge score falls below τ.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        model: nn.Module,
        data,                        # PyG Data object
        fraud_edge_idx: np.ndarray,
        eps: float,
        threshold: float,
    ):
        super().__init__()
        self.model          = model
        self.data           = data
        self.fraud_edge_idx = fraud_edge_idx
        self.eps            = eps
        self.threshold      = threshold

        n_feat = data.x.shape[1]
        self.action_space      = spaces.Box(-1.0, 1.0, (n_feat,), dtype=np.float32)
        self.observation_space = spaces.Box(
            -np.inf, np.inf, (2 * n_feat,), dtype=np.float32
        )
        self._cur_edge: int = 0

    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._cur_edge = int(np.random.choice(self.fraud_edge_idx))
        obs = self._obs(self.data.x)
        return obs, {}

    def _obs(self, x: torch.Tensor) -> np.ndarray:
        src, dst = self.data.edge_index[:, self._cur_edge].tolist()
        return np.concatenate([x[src].numpy(), x[dst].numpy()]).astype(np.float32)

    # ------------------------------------------------------------------
    def step(self, action: np.ndarray):
        src, _ = self.data.edge_index[:, self._cur_edge].tolist()

        # Perturb source node features within L_inf ball
        delta  = torch.tensor(action * self.eps, dtype=torch.float32)
        x_adv  = self.data.x.clone()
        x_adv[src] = torch.clamp(
            x_adv[src] + delta,
            self.data.x[src] - self.eps,
            self.data.x[src] + self.eps,
        )

        # Evaluate single edge
        single_mask = torch.zeros(self.data.num_edges, dtype=torch.bool)
        single_mask[self._cur_edge] = True

        self.model.eval()
        with torch.no_grad():
            logit = self.model(
                x_adv, self.data.edge_index, self.data.edge_attr,
                mask=single_mask,
            )
            prob = float(torch.sigmoid(logit).item())

        evaded = prob < self.threshold
        reward = 1.0 if evaded else -0.1
        obs    = self._obs(x_adv)

        return obs, reward, True, False, {"evaded": evaded, "prob": prob}


# ---------------------------------------------------------------------------
# Defender best response
# ---------------------------------------------------------------------------
def defender_best_response(
    model: nn.Module,
    data,
    threshold_grid: List[float],
    x_override: Optional[torch.Tensor] = None,
) -> float:
    """Return the threshold τ ∈ threshold_grid that maximises F1 on val edges.

    Falls back to train edges when val has no fraud (common on small samples).
    """
    model.eval()
    x = x_override if x_override is not None else data.x

    # Prefer val; fall back to train if val has no positives
    mask   = data.val_mask
    labels = data.y_edge[mask].numpy()
    if labels.sum() == 0:
        mask   = data.train_mask
        labels = data.y_edge[mask].numpy()

    with torch.no_grad():
        probs = torch.sigmoid(
            model(x, data.edge_index, data.edge_attr, mask=mask)
        ).numpy()

    best_tau, best_f1 = threshold_grid[0], -1.0
    for tau in threshold_grid:
        preds = (probs >= tau).astype(int)
        f1    = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1  = f1
            best_tau = tau

    return best_tau


# ---------------------------------------------------------------------------
# Attacker PPO training
# ---------------------------------------------------------------------------
def attacker_train_ppo(
    env: FraudEvasionEnv,
    n_timesteps: int,
    lr: float,
) -> PPO:
    """Train a PPO policy for n_timesteps and return it."""
    n_steps    = max(8, min(64, n_timesteps // 4))
    batch_size = max(4, min(32, n_steps // 2))
    attacker   = PPO(
        "MlpPolicy", env,
        learning_rate=lr,
        n_steps=n_steps,
        batch_size=batch_size,
        verbose=0,
    )
    attacker.learn(total_timesteps=n_timesteps)
    return attacker


# ---------------------------------------------------------------------------
# Evasion rate measurement
# ---------------------------------------------------------------------------
def measure_evasion_rate(
    model: nn.Module,
    data,
    attacker: PPO,
    fraud_idx: np.ndarray,
    threshold: float,
    eps: float,
    n_eval: int = 50,
) -> float:
    """Fraction of fraud edges the trained attacker evades at threshold τ."""
    evaded = 0
    model.eval()
    for _ in range(n_eval):
        edge_i    = int(np.random.choice(fraud_idx))
        src, dst  = data.edge_index[:, edge_i].tolist()
        obs       = np.concatenate(
            [data.x[src].numpy(), data.x[dst].numpy()]
        ).astype(np.float32)
        action, _ = attacker.predict(obs, deterministic=True)

        delta  = torch.tensor(action * eps, dtype=torch.float32)
        x_adv  = data.x.clone()
        x_adv[src] = torch.clamp(
            x_adv[src] + delta,
            data.x[src] - eps,
            data.x[src] + eps,
        )
        single        = torch.zeros(data.num_edges, dtype=torch.bool)
        single[edge_i] = True
        with torch.no_grad():
            prob = float(torch.sigmoid(
                model(x_adv, data.edge_index, data.edge_attr, mask=single)
            ).item())
        if prob < threshold:
            evaded += 1

    return evaded / n_eval


# ---------------------------------------------------------------------------
# Outer Stackelberg loop
# ---------------------------------------------------------------------------
def stackelberg_loop(
    model: nn.Module,
    data,
    cfg: dict,
) -> StackelbergState:
    """Alternating best-response Stackelberg loop.

    Parameters
    ----------
    model : trained EdgeGraphSAGE (Layers 1+2)
    data  : PyG Data with train/val/test masks and y_edge labels
    cfg   : ``stackelberg`` sub-dict from configs/default.yaml

    Returns
    -------
    StackelbergState with final τ and utility histories.
    """
    threshold_grid = cfg["defender_threshold_grid"]
    outer_iters    = cfg["outer_iterations"]
    attacker_ts    = cfg.get("attacker_episodes", 200)   # total PPO timesteps
    attacker_lr    = cfg["attacker_lr"]
    conv_tol       = cfg["convergence_tol"]
    eps            = 0.1    # L_inf budget — matches TRADES epsilon

    # Fraud training edges
    fraud_mask = data.train_mask & (data.y_edge == 1)
    fraud_idx  = fraud_mask.nonzero(as_tuple=False).squeeze(1).numpy()

    if len(fraud_idx) == 0:
        print("  [Stackelberg] No fraud edges in training set — skipping Layer 3.")
        return StackelbergState(threshold=0.5, attacker_policy=None)

    # Initial defender threshold (no attacker yet)
    threshold = defender_best_response(model, data, threshold_grid)
    print(f"  [Stackelberg] Starting τ = {threshold:.3f}  |  "
          f"{len(fraud_idx)} fraud training edges")

    state = StackelbergState(threshold=threshold, attacker_policy=None)

    for i in range(1, outer_iters + 1):

        # ── 1. Attacker best-responds ──────────────────────────────────
        env      = FraudEvasionEnv(model, data, fraud_idx, eps, threshold)
        attacker = attacker_train_ppo(env, attacker_ts, attacker_lr)
        evasion  = measure_evasion_rate(
            model, data, attacker, fraud_idx, threshold, eps
        )
        state.attacker_utility_history.append(evasion)

        # ── 2. Defender best-responds ──────────────────────────────────
        new_tau = defender_best_response(model, data, threshold_grid)

        # Compute defender F1 at new_tau on val (or train if val empty)
        eval_mask  = data.val_mask if data.y_edge[data.val_mask].sum() > 0 \
                     else data.train_mask
        eval_labels = data.y_edge[eval_mask].numpy()
        model.eval()
        with torch.no_grad():
            eval_probs = torch.sigmoid(
                model(data.x, data.edge_index, data.edge_attr, mask=eval_mask)
            ).numpy()
        def_f1 = f1_score(
            eval_labels, (eval_probs >= new_tau).astype(int), zero_division=0
        )
        state.defender_utility_history.append(def_f1)

        delta     = abs(new_tau - threshold)
        threshold = new_tau
        state.threshold       = threshold
        state.attacker_policy = attacker

        print(f"  iter {i:2d}/{outer_iters} | τ={threshold:.3f} | "
              f"defender_F1={def_f1:.4f} | evasion_rate={evasion:.3f} | "
              f"|Δτ|={delta:.4f}")

        if delta < conv_tol and i > 1:
            print(f"  [Stackelberg] Converged at iteration {i}.")
            break

    return state
