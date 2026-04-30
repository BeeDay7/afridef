"""Dataset loaders and bipartite graph construction for AfriDef.

PaySim (primary): customer (nameOrig) -- transaction --> destination (nameDest).
We treat nameDest entries with prefix 'M' as merchants and 'C' as customers;
agents are inferred from high-CICO velocity nameDest nodes.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
PAYSIM_COLUMNS = [
    "step", "type", "amount",
    "nameOrig", "oldbalanceOrg", "newbalanceOrig",
    "nameDest", "oldbalanceDest", "newbalanceDest",
    "isFraud", "isFlaggedFraud",
]

PAYSIM_TYPES = ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"]


@dataclass
class TemporalSplit:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_paysim(path: str | Path) -> pd.DataFrame:
    """Load PaySim and assert schema."""
    df = pd.read_csv(path)
    missing = set(PAYSIM_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"PaySim missing columns: {missing}")
    return df


def temporal_split(df: pd.DataFrame, val_frac: float, test_frac: float) -> TemporalSplit:
    """Chronological split by `step` (PaySim's hour column)."""
    df = df.sort_values("step").reset_index(drop=True)
    n = len(df)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    n_train = n - n_val - n_test
    return TemporalSplit(
        train=df.iloc[:n_train].copy(),
        val=df.iloc[n_train:n_train + n_val].copy(),
        test=df.iloc[n_train + n_val:].copy(),
    )


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------
def build_bipartite_graph(df: pd.DataFrame) -> Data:
    """Build a customer-destination bipartite transaction graph.

    Returns a PyG Data object. Node features: aggregated transaction stats per
    node. Edge features: per-transaction features. Labels: edge-level fraud.
    """
    raise NotImplementedError("TODO: aggregate node features, encode types, "
                              "construct edge_index/edge_attr/y tensors.")


def africa_stress_resample(df: pd.DataFrame,
                           cashout_weight: float = 2.0,
                           transfer_weight: float = 1.5,
                           seed: int = 0) -> pd.DataFrame:
    """Re-weight PaySim toward CASH_OUT and TRANSFER types as proxies for
    African-context fraud (CICO laundering, SIM-swap account takeover)."""
    rng = np.random.default_rng(seed)
    weights = pd.Series(1.0, index=df.index)
    weights[df["type"] == "CASH_OUT"] = cashout_weight
    weights[df["type"] == "TRANSFER"] = transfer_weight
    probs = weights / weights.sum()
    idx = rng.choice(df.index, size=len(df), p=probs.values, replace=True)
    return df.loc[idx].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def class_balance(df: pd.DataFrame, label_col: str = "isFraud") -> Tuple[int, int, float]:
    """Return (n_pos, n_neg, fraud_rate)."""
    n_pos = int(df[label_col].sum())
    n_neg = len(df) - n_pos
    return n_pos, n_neg, n_pos / max(len(df), 1)
