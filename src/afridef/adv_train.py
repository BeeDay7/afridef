"""Layer 2: TRADES adversarial training for the EdgeGraphSAGE classifier.

Reference: Zhang et al., "Theoretically principled trade-off between
robustness and accuracy" (ICML 2019).

TRADES objective (edge-level form):
    L = L_CE(f(x), y) + beta * KL( p(f(x)) || p(f(x_adv)) )

where x_adv is found by PGD maximising KL(p(x) || p(x_adv)).
Node features x are perturbed within an L_inf ball of radius epsilon;
edge structure and edge attributes are held fixed.

Usage:
    from afridef.adv_train import trades_loss

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = trades_loss(
            model, data.x, data.y_edge, data.edge_index,
            eps=cfg["trades"]["epsilon"],
            step=cfg["trades"]["pgd_step_size"],
            n_steps=cfg["trades"]["pgd_steps"],
            beta=cfg["trades"]["beta"],
            edge_attr=data.edge_attr,
            mask=data.train_mask,
            pos_weight=pos_weight,
        )
        loss.backward()
        optimizer.step()
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from .attacks import pgd


# ---------------------------------------------------------------------------
# TRADES loss
# ---------------------------------------------------------------------------
def trades_loss(
    model: nn.Module,
    x: torch.Tensor,
    y_edge: torch.Tensor,
    edge_index: torch.Tensor,
    eps: float,
    step: float,
    n_steps: int,
    beta: float,
    edge_attr: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
    pos_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """TRADES loss for edge-level GNN fraud detection.

    Parameters
    ----------
    model       : EdgeGraphSAGE (or any model with matching signature)
    x           : node feature matrix  [N, d]
    y_edge      : edge-level fraud labels [E]
    edge_index  : graph connectivity   [2, E]
    eps         : L_inf perturbation radius
    step        : PGD step size
    n_steps     : number of PGD steps
    beta        : TRADES robustness / accuracy trade-off coefficient
    edge_attr   : edge feature matrix [E, d_e]  (None for node-level models)
    mask        : boolean mask selecting training edges  [E]
    pos_weight  : class weight for the natural-loss BCE term

    Returns
    -------
    Scalar loss tensor.
    """
    # ── Find adversarial node features via PGD ────────────────────────────
    model.eval()
    with torch.no_grad():
        # Detach so PGD doesn't accumulate into the main graph
        pass
    x_adv = pgd(
        model, x, y_edge, edge_index,
        eps=eps, step=step, n_steps=n_steps,
        edge_attr=edge_attr, mask=mask,
    )
    model.train()

    # Active labels
    y = (y_edge[mask] if mask is not None else y_edge).float()

    # ── Natural loss ──────────────────────────────────────────────────────
    if edge_attr is not None:
        logits_nat = model(x, edge_index, edge_attr, mask=mask)
    else:
        logits_nat = model(x, edge_index)

    loss_nat = F.binary_cross_entropy_with_logits(
        logits_nat, y, pos_weight=pos_weight
    )

    # ── Adversarial KL term ───────────────────────────────────────────────
    if edge_attr is not None:
        logits_adv = model(x_adv, edge_index, edge_attr, mask=mask)
    else:
        logits_adv = model(x_adv, edge_index)

    p_nat = torch.sigmoid(logits_nat).clamp(1e-6, 1 - 1e-6)
    p_adv = torch.sigmoid(logits_adv).clamp(1e-6, 1 - 1e-6)

    # Binary KL(p_nat || p_adv)
    kl = (
        p_nat * (p_nat.log() - p_adv.log())
        + (1 - p_nat) * ((1 - p_nat).log() - (1 - p_adv).log())
    ).mean()

    return loss_nat + beta * kl


# ---------------------------------------------------------------------------
# Adversarial evaluation helper
# ---------------------------------------------------------------------------
def eval_under_attack(
    model: nn.Module,
    data,
    mask: torch.Tensor,
    eps: float,
    step: float,
    n_steps: int,
) -> torch.Tensor:
    """Return fraud probabilities on adversarially-perturbed node features.

    Used to populate the 'adv robustness' columns in Table 1.
    """
    from .attacks import pgd as _pgd
    model.eval()
    x_adv = _pgd(
        model, data.x, data.y_edge, data.edge_index,
        eps=eps, step=step, n_steps=n_steps,
        edge_attr=data.edge_attr, mask=mask,
    )
    with torch.no_grad():
        logits = model(x_adv, data.edge_index, data.edge_attr, mask=mask)
    return torch.sigmoid(logits)
