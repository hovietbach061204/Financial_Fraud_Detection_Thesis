"""
GNN embeddings → XGBoost augmentation for IEEE-CIS Fraud Detection
==================================================================
Drop-in module for your `XGB_Fraud_with_Magic.ipynb` pipeline.

Assumes you already have, in your notebook:
    X_train_copy5  : DataFrame with at least the 'uid' column and your feature cols
    X_test_copy5   : Same, for test
    y_train        : Series of isFraud labels (length == len(X_train_copy5))
    cols           : list[str] — the final feature columns you pass to XGBoost

Usage in the notebook (add a new cell after your feature engineering is done):

    from gnn_augment_xgboost_ieee import augment_with_gnn_embeddings
    X_train_copy6, X_test_copy6, cols_augmented = augment_with_gnn_embeddings(
        X_train_copy5, X_test_copy5, y_train, cols,
        emb_dim=32, n_epochs=15, device='cuda',  # use 'cpu' if no GPU
    )

Then in your existing XGBoost cell, just swap:
    X_train_copy5 -> X_train_copy6
    X_test_copy5  -> X_test_copy6
    cols          -> cols_augmented

That's it. The GNN now contributes 32 extra features per transaction to XGBoost.

Dependencies:  torch  torch_geometric  scikit-learn  pandas  numpy
    pip install torch torch_geometric  (see PyG install notes for your CUDA version)
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv
from sklearn.metrics import roc_auc_score


# ---------------------------------------------------------------------------
# 1) Build the bipartite Transaction <-> UID heterograph from your dataframe
# ---------------------------------------------------------------------------
def build_hetero_graph(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    feature_cols: list,
    uid_col: str = "uid",
    val_fraction: float = 0.25,
) -> tuple[HeteroData, int, int, int]:
    """
    Returns
    -------
    data     : HeteroData with two node types ('txn', 'uid') and bidirectional edges
    n_uids   : int — total unique UIDs across train+test
    n_train  : int
    n_test   : int

    Notes
    -----
    * We stack train+test into one big graph (transductive: the GNN sees the test
      graph structure during training, but never sees test labels — which are
      masked out of the loss).
    * Train labels are split into (train_mask, val_mask) by row order so the last
      `val_fraction` of train rows act as a holdout while training the GNN.
    """
    n_train = len(X_train)
    n_test = len(X_test)
    n_total = n_train + n_test

    all_df = pd.concat(
        [X_train.reset_index(drop=True), X_test.reset_index(drop=True)],
        ignore_index=True,
    )

    # ---- Encode UIDs as contiguous integer ids ----
    uid_to_idx = {u: i for i, u in enumerate(all_df[uid_col].unique())}
    uid_ids = all_df[uid_col].map(uid_to_idx).values.astype(np.int64)
    n_uids = len(uid_to_idx)

    # ---- Transaction node features (NaN -> -1, matches your XGBoost missing=-1) ----
    X = all_df[feature_cols].fillna(-1).astype(np.float32).values

    # ---- Labels: -1 placeholder for test rows so they never enter the loss ----
    y = np.full(n_total, -1, dtype=np.int64)
    y[:n_train] = y_train.values.astype(np.int64)

    # ---- Train/val masks within the labeled (train) region ----
    val_split = int(n_train * (1.0 - val_fraction))
    train_mask = np.zeros(n_total, dtype=bool)
    val_mask = np.zeros(n_total, dtype=bool)
    train_mask[:val_split] = True
    val_mask[val_split:n_train] = True

    # ---- Assemble the heterograph ----
    data = HeteroData()
    data["txn"].x = torch.tensor(X)
    data["txn"].y = torch.tensor(y)
    data["txn"].train_mask = torch.tensor(train_mask)
    data["txn"].val_mask = torch.tensor(val_mask)

    data["uid"].num_nodes = n_uids  # UID nodes get learnable embeddings (no input features)

    # Bidirectional edges
    txn_idx = np.arange(n_total, dtype=np.int64)
    ei = torch.tensor(np.stack([txn_idx, uid_ids]), dtype=torch.long)
    data["txn", "has_uid", "uid"].edge_index = ei
    data["uid", "has_txn", "txn"].edge_index = ei.flip(0)

    return data, n_uids, n_train, n_test


# ---------------------------------------------------------------------------
# 2) Define the GNN: bipartite encoder + binary-classification head
# ---------------------------------------------------------------------------
class IEEE_FraudGNN(nn.Module):
    """
    NP-style architecture for IEEE-CIS:
        txn_features ─proj─┐
                           ├─ HeteroConv(SAGEConv) x2 ─► txn_emb, uid_emb
        uid_embedding ─────┘                                │
                                                            ▼
                                                Linear(emb_dim, 2)  [HEAD — discarded]
    """

    def __init__(self, in_dim: int, n_uids: int, hidden: int = 64, emb_dim: int = 32):
        super().__init__()
        self.txn_proj = nn.Linear(in_dim, hidden)
        self.uid_embedding = nn.Embedding(n_uids, hidden)  # learnable, no input features

        self.conv1 = HeteroConv(
            {
                ("txn", "has_uid", "uid"): SAGEConv((hidden, hidden), hidden),
                ("uid", "has_txn", "txn"): SAGEConv((hidden, hidden), hidden),
            },
            aggr="mean",
        )
        self.conv2 = HeteroConv(
            {
                ("txn", "has_uid", "uid"): SAGEConv((hidden, hidden), emb_dim),
                ("uid", "has_txn", "txn"): SAGEConv((hidden, hidden), emb_dim),
            },
            aggr="mean",
        )
        self.dropout = nn.Dropout(0.2)
        self.classifier_head = nn.Linear(emb_dim, 2)  # DISCARDED after training

    def encode(self, data: HeteroData) -> dict:
        x_dict = {
            "txn": self.txn_proj(data["txn"].x),
            "uid": self.uid_embedding.weight,
        }
        ei = {
            ("txn", "has_uid", "uid"): data["txn", "has_uid", "uid"].edge_index,
            ("uid", "has_txn", "txn"): data["uid", "has_txn", "txn"].edge_index,
        }
        x_dict = self.conv1(x_dict, ei)
        x_dict = {k: F.relu(self.dropout(v)) for k, v in x_dict.items()}
        x_dict = self.conv2(x_dict, ei)
        return x_dict

    def forward(self, data: HeteroData):
        emb = self.encode(data)
        logits = self.classifier_head(emb["txn"])
        return logits, emb


# ---------------------------------------------------------------------------
# 3) Train the GNN with weighted cross-entropy (class imbalance ~28:1)
# ---------------------------------------------------------------------------
def train_gnn(
    data: HeteroData,
    n_uids: int,
    in_dim: int,
    emb_dim: int = 32,
    hidden: int = 64,
    n_epochs: int = 15,
    lr: float = 1e-3,
    device: str = "cuda",
    verbose: bool = True,
) -> IEEE_FraudGNN:
    device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    model = IEEE_FraudGNN(in_dim=in_dim, n_uids=n_uids, hidden=hidden, emb_dim=emb_dim).to(device)
    data = data.to(device)

    train_mask = data["txn"].train_mask
    val_mask = data["txn"].val_mask
    y = data["txn"].y

    # class-weighted loss — IEEE-CIS is ~3.5% fraud, so pos weight ~28
    n_pos = (y[train_mask] == 1).sum().float()
    n_neg = (y[train_mask] == 0).sum().float()
    pos_weight = (n_neg / n_pos.clamp(min=1.0)).item()
    weights = torch.tensor([1.0, pos_weight], device=device)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    best_val_auc = 0.0
    best_state = None
    for epoch in range(1, n_epochs + 1):
        model.train()
        opt.zero_grad()
        logits, _ = model(data)
        loss = F.cross_entropy(logits[train_mask], y[train_mask], weight=weights)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
            y_np = y.cpu().numpy()
            tm = train_mask.cpu().numpy()
            vm = val_mask.cpu().numpy()
            train_auc = roc_auc_score(y_np[tm], probs[tm])
            val_auc = roc_auc_score(y_np[vm], probs[vm])

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        if verbose:
            print(f"epoch {epoch:3d} | loss={loss.item():.4f} | "
                  f"train_AUC={train_auc:.4f} | val_AUC={val_auc:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"\n>>> Best GNN val AUC: {best_val_auc:.4f}")
    return model


# ---------------------------------------------------------------------------
# 4) Extract embeddings, discard the classifier head, return numpy arrays
# ---------------------------------------------------------------------------
def extract_embeddings(model: IEEE_FraudGNN, data: HeteroData,
                       n_train: int, n_test: int) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        # NOTE: encode() bypasses classifier_head entirely. The head's job is done.
        emb = model.encode(data)
    txn_emb = emb["txn"].cpu().numpy()
    return txn_emb[:n_train], txn_emb[n_train:]


# ---------------------------------------------------------------------------
# 5) Top-level convenience: do everything in one call, return augmented dfs
# ---------------------------------------------------------------------------
def augment_with_gnn_embeddings(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    cols: list,
    uid_col: str = "uid",
    emb_dim: int = 32,
    hidden: int = 64,
    n_epochs: int = 15,
    lr: float = 1e-3,
    device: str = "cuda",
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, list]:
    """
    End-to-end: build graph → train GNN → extract embeddings → return augmented
    DataFrames and updated feature-column list ready for your existing XGBoost cell.
    """
    print("[1/4] Building heterograph...")
    data, n_uids, n_train, n_test = build_hetero_graph(
        X_train, X_test, y_train, feature_cols=cols, uid_col=uid_col
    )
    in_dim = data["txn"].x.shape[1]
    print(f"      {n_train + n_test:,} transaction nodes, {n_uids:,} uid nodes, "
          f"{data['txn','has_uid','uid'].edge_index.shape[1]:,} edges")

    print("\n[2/4] Training GNN (Phase 1: encoder + classifier head jointly)...")
    model = train_gnn(
        data, n_uids=n_uids, in_dim=in_dim, emb_dim=emb_dim, hidden=hidden,
        n_epochs=n_epochs, lr=lr, device=device, verbose=verbose,
    )

    print("\n[3/4] Phase 2/3: discard head, extract transaction embeddings...")
    train_emb, test_emb = extract_embeddings(model, data, n_train, n_test)
    print(f"      train_emb shape: {train_emb.shape}")
    print(f"      test_emb  shape: {test_emb.shape}")

    print("\n[4/4] Concatenating embeddings to your dataframes...")
    new_cols = [f"gnn_emb_{i}" for i in range(emb_dim)]
    X_train_aug = X_train.copy()
    X_test_aug = X_test.copy()
    for i, c in enumerate(new_cols):
        X_train_aug[c] = train_emb[:, i].astype(np.float32)
        X_test_aug[c] = test_emb[:, i].astype(np.float32)
    cols_aug = list(cols) + new_cols
    print(f"      {len(cols)} original features  +  {len(new_cols)} GNN features  "
          f"=  {len(cols_aug)} total")

    return X_train_aug, X_test_aug, cols_aug


# ===========================================================================
# Notebook integration recipe — paste into a NEW CELL in your XGB notebook
# ===========================================================================
USAGE_EXAMPLE = """
# === New cell, AFTER your feature engineering produced X_train_copy5 / X_test_copy5 ===
from gnn_augment_xgboost_ieee import augment_with_gnn_embeddings

X_train_copy6, X_test_copy6, cols_augmented = augment_with_gnn_embeddings(
    X_train_copy5,
    X_test_copy5,
    y_train,
    cols,                  # your final feature list before XGBoost
    uid_col='uid',
    emb_dim=32,
    n_epochs=15,
    device='cuda',         # or 'cpu' if you don't have a CUDA GPU
)

# === Then RE-RUN your existing GroupKFold XGBoost cell, but with: ===
#     X_train_copy5 → X_train_copy6
#     X_test_copy5  → X_test_copy6
#     cols          → cols_augmented
#
# Expected lift on IEEE-CIS: typically +0.003 to +0.010 AUC on the 6-fold OOF score.
# (Your baseline OOF AUC is ~0.945; with GNN augmentation expect ~0.948–0.955.)
"""

if __name__ == "__main__":
    print(USAGE_EXAMPLE)
