"""FGSM and PGD attacks on node features of a transaction graph.

Attacks perturb the NODE FEATURE MATRIX x within an L_inf ball of radius
eps.  Edge features and graph structure are left unchanged — consistent
with the threat model in AfriDef §3.2, where the attacker manipulates
account-level attributes (e.g., fabricated balance history) but cannot
rewire the transaction graph itself.

Both functions work with:
  - Node-level models: model(x, edge_index)
  - Edge-level models: model(x, edge_index, edge_attr, mask=mask)

Pass edge_attr=None and mask=None for node-level models (backwards compat).
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


def _forward(model, x, edge_index, edge_attr, mask):
    """Unified forward for node-level and edge-level models."""
    if edge_attr is not None:
        return model(x, edge_index, edge_attr, mask=mask)
    return model(x, edge_index)


def _labels(y_edge, mask):
    """Extract labels for the active edge subset."""
    if mask is not None:
        return y_edge[mask].float()
    return y_edge.float()


# ---------------------------------------------------------------------------
# FGSM
# ---------------------------------------------------------------------------
def fgsm(
    model,
    x: torch.Tensor,
    y_edge: torch.Tensor,
    edge_index: torch.Tensor,
    eps: float,
    edge_attr: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """One-step Fast Gradient Sign Method on node features (L_inf)."""
    x = x.clone().detach().requires_grad_(True)
    logits = _forward(model, x, edge_index, edge_attr, mask)
    loss   = F.binary_cross_entropy_with_logits(logits, _labels(y_edge, mask))
    loss.backward()
    x_adv = (x + eps * x.grad.sign()).detach()
    return x_adv


# ---------------------------------------------------------------------------
# PGD
# ---------------------------------------------------------------------------
def pgd(
    model,
    x: torch.Tensor,
    y_edge: torch.Tensor,
    edge_index: torch.Tensor,
    eps: float,
    step: float,
    n_steps: int,
    edge_attr: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Multi-step Projected Gradient Descent on node features (L_inf ball).

    Starts from a random point inside the ball (standard PGD initialisation)
    and projects back after each step.
    """
    x_orig = x.clone().detach()
    # Random start inside the L_inf ball
    x_adv = x_orig + torch.empty_like(x_orig).uniform_(-eps, eps)
    x_adv = torch.clamp(x_adv, x_orig - eps, x_orig + eps).detach()

    y = _labels(y_edge, mask)

    for _ in range(n_steps):
        x_adv = x_adv.requires_grad_(True)
        logits = _forward(model, x_adv, edge_index, edge_attr, mask)
        loss   = F.binary_cross_entropy_with_logits(logits, y)
        grad   = torch.autograd.grad(loss, x_adv)[0]
        x_adv  = x_adv.detach() + step * grad.sign()
        x_adv  = torch.clamp(x_adv, x_orig - eps, x_orig + eps).detach()

    return x_adv
