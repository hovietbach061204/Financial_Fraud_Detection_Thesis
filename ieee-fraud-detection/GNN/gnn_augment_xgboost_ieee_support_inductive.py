"""
GNN embeddings → XGBoost augmentation for IEEE-CIS Fraud Detection
==================================================================
Drop-in module for your `XGB_Fraud_with_Magic.ipynb` pipeline.

Two modes:
    mode='inductive'    [DEFAULT, RECOMMENDED FOR THESIS]
        GNN is trained on the train-only sub-graph. At inference, test
        transactions are attached to the graph and the frozen encoder
        produces embeddings for them. This mirrors how a production
        fraud system actually works: at deploy time, you cannot have
        seen future transactions during training.

    mode='transductive'  [ACADEMIC DEFAULT, KNOWN MILD LEAKAGE]
        Standard practice in GNN benchmarks: build one big graph containing
        train + test rows, train GNN on train labels only. Test features
        and graph structure DO leak into the encoder weights. For IEEE-CIS
        (a temporal split) this is a real methodological concern, even
        though no test LABEL leaks. Use only as a baseline/upper-bound.

Usage in the notebook (add a new cell after feature engineering):

    from gnn_augment_xgboost_ieee_support_inductive import augment_with_gnn_embeddings
    X_train_copy6, X_test_copy6, cols_augmented = augment_with_gnn_embeddings(
        X_train_copy5, X_test_copy5, y_train, cols,
        mode='inductive',          # recommended; 'transductive' for comparison
        emb_dim=32, n_epochs=15, device='cuda',
    )

Then in your existing XGBoost cell, just swap:
    X_train_copy5 -> X_train_copy6
    X_test_copy5  -> X_test_copy6
    cols          -> cols_augmented

Reporting tip for thesis: run BOTH modes and report both AUC numbers.
The gap (inductive < transductive) is your honest measurement of how
much the transductive setting was overstating things.

Dependencies:  torch  torch_geometric  scikit-learn  pandas  numpy
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
from sklearn.preprocessing import StandardScaler


DEFAULT_GNN_FEATURE_COLS = [
    "TransactionDT", "TransactionAmt", "ProductCD",
    "card1", "card2", "card3", "card4", "card5", "card6",
    "addr1", "addr2", "dist1", "dist2",
    "P_emaildomain", "R_emaildomain",
    "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10",
    "C11", "C12", "C13", "C14",
    "D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9", "D10",
    "D11", "D12", "D13", "D14", "D15",
    "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9",
    "id_01", "id_02", "id_03", "id_04", "id_05", "id_06",
    "id_09", "id_10", "id_11", "id_12", "id_13", "id_14",
    "id_15", "id_16", "id_17", "id_19", "id_20", "id_28",
    "id_29", "id_30", "id_31", "id_32", "id_33", "id_34",
    "id_35", "id_36", "id_37", "id_38",
    "DeviceType", "DeviceInfo", "cents", "dollars", "day",
]

DEFAULT_GNN_ENTITY_COLS = [
    "uid",
    "card1_addr1",
    "card1_addr1_P_emaildomain",
    "DeviceInfo",
]


def _existing_cols(df: pd.DataFrame, cols: list) -> list:
    return [c for c in cols if c in df.columns]


def _valid_entity_mask(s: pd.Series) -> pd.Series:
    mask = s.notna()
    as_str = s.astype(str)
    return mask & ~as_str.isin(["", "-1", "-1.0", "nan", "None", "NONE"])


def _make_entity_vocab(
    txn_df: pd.DataFrame,
    entity_cols: list,
) -> dict:
    entity_to_idx = {}
    for col in _existing_cols(txn_df, entity_cols):
        s = txn_df[col]
        keys = s[_valid_entity_mask(s)].astype(str).radd(f"{col}=").unique()
        for key in keys:
            entity_to_idx[key] = len(entity_to_idx)
    return entity_to_idx


def _make_typed_entity_vocabs(
    txn_df: pd.DataFrame,
    entity_cols: list,
) -> dict[str, dict]:
    entity_vocabs = {}
    for col in _existing_cols(txn_df, entity_cols):
        s = txn_df[col]
        values = s[_valid_entity_mask(s)].astype(str).unique()
        if len(values) > 0:
            entity_vocabs[col] = {v: i for i, v in enumerate(values)}
    return entity_vocabs


def _scale_gnn_feature_frames(
    X_fit: pd.DataFrame,
    X_valid: pd.DataFrame,
    feature_cols: list,
    entity_cols: list,
) -> tuple[pd.DataFrame, pd.DataFrame, list]:
    """Fit scaler on historical fold rows and transform GNN feature columns only."""
    scale_cols = [c for c in feature_cols if c not in set(entity_cols)]
    if not scale_cols:
        return X_fit.copy(), X_valid.copy(), []

    X_fit_scaled = X_fit.copy()
    X_valid_scaled = X_valid.copy()

    fit_values = (
        X_fit[scale_cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(-1)
        .astype(np.float32)
    )
    valid_values = (
        X_valid[scale_cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(-1)
        .astype(np.float32)
    )

    scaler = StandardScaler()
    fit_scaled = scaler.fit_transform(fit_values).astype(np.float32)
    valid_scaled = scaler.transform(valid_values).astype(np.float32)
    for i, col in enumerate(scale_cols):
        X_fit_scaled[col] = fit_scaled[:, i]
        X_valid_scaled[col] = valid_scaled[:, i]
    return X_fit_scaled, X_valid_scaled, scale_cols


# ---------------------------------------------------------------------------
# Helper: build a bipartite Transaction <-> UID heterograph from any df slice
# ---------------------------------------------------------------------------
def _build_graph_from_df(
    txn_df: pd.DataFrame,
    feature_cols: list,
    uid_col: str,
    uid_to_idx: dict,
    labels: np.ndarray | None = None,
    train_mask: np.ndarray | None = None,
    val_mask: np.ndarray | None = None,
) -> HeteroData:
    """Pure builder — does NOT decide what's train/test, caller passes that in."""
    n_txn = len(txn_df)
    uid_ids = txn_df[uid_col].map(uid_to_idx).fillna(-1).astype(np.int64).values
    # If any uid is unknown (test-time inductive case before extension), drop those edges
    keep = uid_ids >= 0

    X = txn_df[feature_cols].fillna(-1).astype(np.float32).values

    data = HeteroData()
    data["txn"].x = torch.tensor(X)
    if labels is not None:
        data["txn"].y = torch.tensor(labels, dtype=torch.long)
    if train_mask is not None:
        data["txn"].train_mask = torch.tensor(train_mask, dtype=torch.bool)
    if val_mask is not None:
        data["txn"].val_mask = torch.tensor(val_mask, dtype=torch.bool)

    data["uid"].num_nodes = len(uid_to_idx)

    txn_idx = np.arange(n_txn, dtype=np.int64)[keep]
    uid_idx = uid_ids[keep]
    ei = torch.tensor(np.stack([txn_idx, uid_idx]), dtype=torch.long)
    data["txn", "has_uid", "uid"].edge_index = ei
    data["uid", "has_txn", "txn"].edge_index = ei.flip(0)
    return data


def _build_multi_entity_graph_from_df(
    txn_df: pd.DataFrame,
    feature_cols: list,
    entity_cols: list,
    entity_to_idx: dict,
    labels: np.ndarray | None = None,
    train_mask: np.ndarray | None = None,
    val_mask: np.ndarray | None = None,
) -> HeteroData:
    """Build txn <-> entity graph using multiple categorical identity columns."""
    n_txn = len(txn_df)
    feature_cols = _existing_cols(txn_df, feature_cols)
    if not feature_cols:
        raise ValueError("No GNN feature columns are present in the dataframe.")

    X = txn_df[feature_cols].fillna(-1).astype(np.float32).values

    data = HeteroData()
    data["txn"].x = torch.tensor(X)
    if labels is not None:
        data["txn"].y = torch.tensor(labels, dtype=torch.long)
    if train_mask is not None:
        data["txn"].train_mask = torch.tensor(train_mask, dtype=torch.bool)
    if val_mask is not None:
        data["txn"].val_mask = torch.tensor(val_mask, dtype=torch.bool)

    data["entity"].num_nodes = len(entity_to_idx)

    edge_txn_parts = []
    edge_entity_parts = []
    for col in _existing_cols(txn_df, entity_cols):
        s = txn_df[col]
        mask = _valid_entity_mask(s)
        if not mask.any():
            continue

        row_idx = np.flatnonzero(mask.to_numpy())
        keys = s[mask].astype(str).radd(f"{col}=")
        entity_idx = keys.map(entity_to_idx).fillna(-1).astype(np.int64).values
        keep = entity_idx >= 0
        if keep.any():
            edge_txn_parts.append(row_idx[keep])
            edge_entity_parts.append(entity_idx[keep])

    if edge_txn_parts:
        txn_idx = np.concatenate(edge_txn_parts).astype(np.int64)
        entity_idx = np.concatenate(edge_entity_parts).astype(np.int64)
        ei = torch.tensor(np.stack([txn_idx, entity_idx]), dtype=torch.long)
    else:
        ei = torch.empty((2, 0), dtype=torch.long)

    data["txn", "has_entity", "entity"].edge_index = ei
    data["entity", "has_txn", "txn"].edge_index = ei.flip(0)
    return data


def _build_typed_entity_graph_from_df(
    txn_df: pd.DataFrame,
    feature_cols: list,
    entity_vocabs: dict[str, dict],
    labels: np.ndarray | None = None,
    train_mask: np.ndarray | None = None,
    val_mask: np.ndarray | None = None,
) -> HeteroData:
    """Build txn <-> typed entity graph with separate node/relation types."""
    n_txn = len(txn_df)
    feature_cols = _existing_cols(txn_df, feature_cols)
    if not feature_cols:
        raise ValueError("No GNN feature columns are present in the dataframe.")

    X = txn_df[feature_cols].fillna(-1).astype(np.float32).values

    data = HeteroData()
    data["txn"].x = torch.tensor(X)
    if labels is not None:
        data["txn"].y = torch.tensor(labels, dtype=torch.long)
    if train_mask is not None:
        data["txn"].train_mask = torch.tensor(train_mask, dtype=torch.bool)
    if val_mask is not None:
        data["txn"].val_mask = torch.tensor(val_mask, dtype=torch.bool)

    for entity_type, vocab in entity_vocabs.items():
        data[entity_type].num_nodes = len(vocab)

        if entity_type not in txn_df.columns:
            ei = torch.empty((2, 0), dtype=torch.long)
        else:
            s = txn_df[entity_type]
            mask = _valid_entity_mask(s)
            if mask.any():
                row_idx = np.flatnonzero(mask.to_numpy())
                entity_idx = s[mask].astype(str).map(vocab).fillna(-1).astype(np.int64).values
                keep = entity_idx >= 0
                if keep.any():
                    ei = torch.tensor(
                        np.stack([row_idx[keep], entity_idx[keep]]),
                        dtype=torch.long,
                    )
                else:
                    ei = torch.empty((2, 0), dtype=torch.long)
            else:
                ei = torch.empty((2, 0), dtype=torch.long)

        data["txn", f"has_{entity_type}", entity_type].edge_index = ei
        data[entity_type, f"rev_has_{entity_type}", "txn"].edge_index = ei.flip(0)
    print(f"Graph Structure: {data}")
    return data


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


class IEEE_FraudMultiEntityGNN(nn.Module):
    """
    Transaction encoder connected to one unified entity node type.

    Entity nodes are namespaced values such as:
        uid=...
        card1=...
        card1_addr1=...
        P_emaildomain=...
    """

    def __init__(self, in_dim: int, n_entities: int, hidden: int = 64, emb_dim: int = 32):
        super().__init__()
        self.txn_proj = nn.Linear(in_dim, hidden)
        self.entity_embedding = nn.Embedding(n_entities, hidden)

        self.conv1 = HeteroConv(
            {
                ("txn", "has_entity", "entity"): SAGEConv((hidden, hidden), hidden),
                ("entity", "has_txn", "txn"): SAGEConv((hidden, hidden), hidden),
            },
            aggr="mean",
        )
        self.conv2 = HeteroConv(
            {
                ("txn", "has_entity", "entity"): SAGEConv((hidden, hidden), emb_dim),
                ("entity", "has_txn", "txn"): SAGEConv((hidden, hidden), emb_dim),
            },
            aggr="mean",
        )
        self.dropout = nn.Dropout(0.2)
        self.classifier_head = nn.Linear(emb_dim, 2)

    def encode(self, data: HeteroData) -> dict:
        x_dict = {
            "txn": self.txn_proj(data["txn"].x),
            "entity": self.entity_embedding.weight,
        }
        ei = {
            ("txn", "has_entity", "entity"): data["txn", "has_entity", "entity"].edge_index,
            ("entity", "has_txn", "txn"): data["entity", "has_txn", "txn"].edge_index,
        }
        x_dict = self.conv1(x_dict, ei)
        x_dict = {k: F.relu(self.dropout(v)) for k, v in x_dict.items()}
        x_dict = self.conv2(x_dict, ei)
        return x_dict

    def forward(self, data: HeteroData):
        emb = self.encode(data)
        logits = self.classifier_head(emb["txn"])
        return logits, emb


class IEEE_FraudTypedEntityGNN(nn.Module):
    """
    Transaction encoder with separate entity node and relation types.

    Example relations:
        txn <-> uid
        txn <-> card1_addr1
        txn <-> card1_addr1_P_emaildomain
        txn <-> DeviceInfo
    """

    def __init__(
        self,
        in_dim: int,
        entity_num_nodes: dict[str, int],
        hidden: int = 64,
        emb_dim: int = 32,
    ):
        super().__init__()
        self.entity_types = list(entity_num_nodes)
        self.txn_proj = nn.Linear(in_dim, hidden)
        self.entity_embeddings = nn.ModuleDict(
            {
                entity_type: nn.Embedding(n_nodes, hidden)
                for entity_type, n_nodes in entity_num_nodes.items()
            }
        )

        conv1 = {}
        conv2 = {}
        for entity_type in self.entity_types:
            conv1[("txn", f"has_{entity_type}", entity_type)] = SAGEConv(
                (hidden, hidden), hidden
            )
            conv1[(entity_type, f"rev_has_{entity_type}", "txn")] = SAGEConv(
                (hidden, hidden), hidden
            )
            conv2[("txn", f"has_{entity_type}", entity_type)] = SAGEConv(
                (hidden, hidden), emb_dim
            )
            conv2[(entity_type, f"rev_has_{entity_type}", "txn")] = SAGEConv(
                (hidden, hidden), emb_dim
            )

        self.conv1 = HeteroConv(conv1, aggr="mean")
        self.conv2 = HeteroConv(conv2, aggr="mean")
        self.dropout = nn.Dropout(0.2)
        self.classifier_head = nn.Linear(emb_dim, 2)

    def encode(self, data: HeteroData) -> dict:
        x_dict = {"txn": self.txn_proj(data["txn"].x)}
        for entity_type in self.entity_types:
            x_dict[entity_type] = self.entity_embeddings[entity_type].weight

        edge_index_dict = {}
        for entity_type in self.entity_types:
            edge_index_dict[("txn", f"has_{entity_type}", entity_type)] = (
                data["txn", f"has_{entity_type}", entity_type].edge_index
            )
            edge_index_dict[(entity_type, f"rev_has_{entity_type}", "txn")] = (
                data[entity_type, f"rev_has_{entity_type}", "txn"].edge_index
            )

        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {k: F.relu(self.dropout(v)) for k, v in x_dict.items()}
        x_dict = self.conv2(x_dict, edge_index_dict)
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
    grad_clip_norm: float | None = None,
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
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        opt.step()

        model.eval()
        with torch.no_grad():
            eval_logits, _ = model(data)
            probs = F.softmax(eval_logits, dim=1)[:, 1].cpu().numpy()
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

def train_gnn_without_validation(
    data: HeteroData,
    n_uids: int,
    in_dim: int,
    emb_dim: int = 32,
    hidden: int = 64,
    n_epochs: int = 15,
    lr: float = 1e-3,
    device: str = "cuda",
    verbose: bool = True,
    checkpoint_metric: str = "train_loss",
    grad_clip_norm: float | None = 5.0,
) -> IEEE_FraudGNN:
    assert checkpoint_metric in ("train_loss", "train_auc", "last")

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

    has_val = val_mask.sum().item() > 0
    best_score = np.inf if checkpoint_metric == "train_loss" and not has_val else -np.inf
    best_state = None

    for epoch in range(1, n_epochs + 1):
        model.train()
        opt.zero_grad()

        logits, _ = model(data)
        loss = F.cross_entropy(logits[train_mask], y[train_mask], weight=weights)
        loss.backward()
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        opt.step()

        model.eval()
        with torch.no_grad():
            eval_logits, _ = model(data)
            eval_loss = F.cross_entropy(
                eval_logits[train_mask], y[train_mask], weight=weights
            ).item()
            probs = F.softmax(eval_logits, dim=1)[:, 1].cpu().numpy()

            y_np = y.cpu().numpy()
            tm = train_mask.cpu().numpy()

            train_auc = roc_auc_score(y_np[tm], probs[tm])

            if has_val:
                vm = val_mask.cpu().numpy()
                val_auc = roc_auc_score(y_np[vm], probs[vm])
                score = val_auc
            else:
                val_auc = None
                if checkpoint_metric == "train_loss":
                    score = eval_loss
                elif checkpoint_metric == "train_auc":
                    score = train_auc
                else:
                    score = None

        if has_val:
            is_better = score > best_score
        elif checkpoint_metric == "train_loss":
            is_better = score < best_score
        elif checkpoint_metric == "train_auc":
            is_better = score > best_score
        else:
            is_better = False

        if is_better:
            best_score = score
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        if verbose:
            if has_val:
                print(f"epoch {epoch:3d} | loss={eval_loss:.4f} | "
                      f"train_AUC={train_auc:.4f} | val_AUC={val_auc:.4f}")
            else:
                print(f"epoch {epoch:3d} | loss={eval_loss:.4f} | "
                      f"train_AUC={train_auc:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    if has_val:
        print(f"\n>>> Best GNN val AUC: {best_score:.4f}")
    elif checkpoint_metric == "train_loss":
        print(f"\n>>> Best GNN train loss: {best_score:.4f}")
    elif checkpoint_metric == "train_auc":
        print(f"\n>>> Best GNN train AUC: {best_score:.4f}")
    else:
        print("\n>>> Finished GNN training without validation mask")

    return model


def train_multi_entity_gnn(
    data: HeteroData,
    n_entities: int,
    in_dim: int,
    emb_dim: int = 32,
    hidden: int = 64,
    n_epochs: int = 15,
    lr: float = 1e-3,
    device: str = "cuda",
    verbose: bool = True,
    checkpoint_metric: str = "train_loss",
    grad_clip_norm: float | None = 5.0,
) -> IEEE_FraudMultiEntityGNN:
    assert checkpoint_metric in ("train_loss", "train_auc", "last")

    device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    model = IEEE_FraudMultiEntityGNN(
        in_dim=in_dim, n_entities=n_entities, hidden=hidden, emb_dim=emb_dim
    ).to(device)
    data = data.to(device)

    train_mask = data["txn"].train_mask
    val_mask = data["txn"].val_mask
    y = data["txn"].y

    n_pos = (y[train_mask] == 1).sum().float()
    n_neg = (y[train_mask] == 0).sum().float()
    pos_weight = (n_neg / n_pos.clamp(min=1.0)).item()
    weights = torch.tensor([1.0, pos_weight], device=device)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    has_val = val_mask.sum().item() > 0
    best_score = np.inf if checkpoint_metric == "train_loss" and not has_val else -np.inf
    best_state = None

    for epoch in range(1, n_epochs + 1):
        model.train()
        opt.zero_grad()
        logits, _ = model(data)
        loss = F.cross_entropy(logits[train_mask], y[train_mask], weight=weights)
        loss.backward()
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        opt.step()

        model.eval()
        with torch.no_grad():
            eval_logits, _ = model(data)
            eval_loss = F.cross_entropy(
                eval_logits[train_mask], y[train_mask], weight=weights
            ).item()
            probs = F.softmax(eval_logits, dim=1)[:, 1].cpu().numpy()

            y_np = y.cpu().numpy()
            tm = train_mask.cpu().numpy()
            train_auc = roc_auc_score(y_np[tm], probs[tm])

            if has_val:
                vm = val_mask.cpu().numpy()
                val_auc = roc_auc_score(y_np[vm], probs[vm])
                score = val_auc
            else:
                val_auc = None
                if checkpoint_metric == "train_loss":
                    score = eval_loss
                elif checkpoint_metric == "train_auc":
                    score = train_auc
                else:
                    score = None

        if has_val:
            is_better = score > best_score
        elif checkpoint_metric == "train_loss":
            is_better = score < best_score
        elif checkpoint_metric == "train_auc":
            is_better = score > best_score
        else:
            is_better = False

        if is_better:
            best_score = score
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        if verbose:
            if has_val:
                print(f"epoch {epoch:3d} | loss={eval_loss:.4f} | "
                      f"train_AUC={train_auc:.4f} | val_AUC={val_auc:.4f}")
            else:
                print(f"epoch {epoch:3d} | loss={eval_loss:.4f} | "
                      f"train_AUC={train_auc:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    if has_val:
        print(f"\n>>> Best multi-entity GNN val AUC: {best_score:.4f}")
    elif checkpoint_metric == "train_loss":
        print(f"\n>>> Best multi-entity GNN train loss: {best_score:.4f}")
    elif checkpoint_metric == "train_auc":
        print(f"\n>>> Best multi-entity GNN train AUC: {best_score:.4f}")
    else:
        print("\n>>> Finished multi-entity GNN training without validation mask")

    return model


def train_typed_entity_gnn(
    data: HeteroData,
    entity_num_nodes: dict[str, int],
    in_dim: int,
    emb_dim: int = 32,
    hidden: int = 64,
    n_epochs: int = 15,
    lr: float = 1e-3,
    device: str = "cuda",
    verbose: bool = True,
    checkpoint_metric: str = "train_loss",
    grad_clip_norm: float | None = 5.0,
) -> IEEE_FraudTypedEntityGNN:
    assert checkpoint_metric in ("train_loss", "train_auc", "last")

    device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    model = IEEE_FraudTypedEntityGNN(
        in_dim=in_dim,
        entity_num_nodes=entity_num_nodes,
        hidden=hidden,
        emb_dim=emb_dim,
    ).to(device)
    data = data.to(device)

    train_mask = data["txn"].train_mask
    val_mask = data["txn"].val_mask
    y = data["txn"].y

    n_pos = (y[train_mask] == 1).sum().float()
    n_neg = (y[train_mask] == 0).sum().float()
    pos_weight = (n_neg / n_pos.clamp(min=1.0)).item()
    weights = torch.tensor([1.0, pos_weight], device=device)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    has_val = val_mask.sum().item() > 0
    best_score = np.inf if checkpoint_metric == "train_loss" and not has_val else -np.inf
    best_state = None

    for epoch in range(1, n_epochs + 1):
        model.train()
        opt.zero_grad()
        logits, _ = model(data)
        loss = F.cross_entropy(logits[train_mask], y[train_mask], weight=weights)
        loss.backward()
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        opt.step()

        model.eval()
        with torch.no_grad():
            eval_logits, _ = model(data)
            eval_loss = F.cross_entropy(
                eval_logits[train_mask], y[train_mask], weight=weights
            ).item()
            probs = F.softmax(eval_logits, dim=1)[:, 1].cpu().numpy()

            y_np = y.cpu().numpy()
            tm = train_mask.cpu().numpy()
            train_auc = roc_auc_score(y_np[tm], probs[tm])

            if has_val:
                vm = val_mask.cpu().numpy()
                val_auc = roc_auc_score(y_np[vm], probs[vm])
                score = val_auc
            else:
                val_auc = None
                if checkpoint_metric == "train_loss":
                    score = eval_loss
                elif checkpoint_metric == "train_auc":
                    score = train_auc
                else:
                    score = None

        if has_val:
            is_better = score > best_score
        elif checkpoint_metric == "train_loss":
            is_better = score < best_score
        elif checkpoint_metric == "train_auc":
            is_better = score > best_score
        else:
            is_better = False

        if is_better:
            best_score = score
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        if verbose:
            if has_val:
                print(f"epoch {epoch:3d} | loss={eval_loss:.4f} | "
                      f"train_AUC={train_auc:.4f} | val_AUC={val_auc:.4f}")
            else:
                print(f"epoch {epoch:3d} | loss={eval_loss:.4f} | "
                      f"train_AUC={train_auc:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    if has_val:
        print(f"\n>>> Best typed-entity GNN val AUC: {best_score:.4f}")
    elif checkpoint_metric == "train_loss":
        print(f"\n>>> Best typed-entity GNN train loss: {best_score:.4f}")
    elif checkpoint_metric == "train_auc":
        print(f"\n>>> Best typed-entity GNN train AUC: {best_score:.4f}")
    else:
        print("\n>>> Finished typed-entity GNN training without validation mask")

    return model


# ---------------------------------------------------------------------------
# 4) Extract embeddings, discard the classifier head, return numpy arrays
# ---------------------------------------------------------------------------
def extract_embeddings(model: nn.Module, data: HeteroData,
                       n_train: int, n_test: int) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        # NOTE: encode() bypasses classifier_head entirely. The head's job is done.
        emb = model.encode(data)
    txn_emb = emb["txn"].cpu().numpy()
    return txn_emb[:n_train], txn_emb[n_train:]


def augment_xgboost_fold_with_gnn_embeddings(
    X_all: pd.DataFrame,
    y_all: pd.Series,
    cols: list,
    idxT: np.ndarray,
    idxV: np.ndarray,
    uid_col: str = "uid",
    emb_dim: int = 64,
    hidden: int = 64,
    n_epochs: int = 50,
    lr: float = 3e-4,
    device: str = "cuda",
    verbose: bool = True,
    checkpoint_metric: str = "train_loss",
    grad_clip_norm: float | None = 5.0,
    use_gnn_validation: bool = True,
    use_multi_entity_graph: bool = True,
    gnn_feature_cols: list | None = None,
    entity_cols: list | None = None,
    scale_gnn_features: bool = False,
    prefix: str = "gnn_emb_",
) -> tuple[pd.DataFrame, pd.DataFrame, list]:
    """
    Build fold-specific GNN features for rolling XGBoost validation.

    XGBoost still receives the full `cols` feature set. The GNN can receive a
    smaller `gnn_feature_cols` set and, by default, a richer multi-entity graph.

    If use_gnn_validation=True, the GNN graph contains idxT + idxV rows, trains
    on idxT labels, validates on idxV labels, and selects the best GNN checkpoint
    by validation AUC.

    If use_gnn_validation=False, the GNN is trained only on idxT rows. After
    training, idxV rows are attached to the graph without labels, and embeddings
    are extracted for idxT + idxV.

    This matches:
        train months < M -> validate month M
    without training the GNN loss on the withheld validation month.
    """
    idxT = np.asarray(idxT)
    idxV = np.asarray(idxV)

    X_fit = X_all.iloc[idxT].reset_index(drop=True)
    y_fit = y_all.iloc[idxT].reset_index(drop=True)
    X_valid = X_all.iloc[idxV].reset_index(drop=True)
    y_valid = y_all.iloc[idxV].reset_index(drop=True)

    gnn_feature_cols_used = _existing_cols(
        X_all, DEFAULT_GNN_FEATURE_COLS if gnn_feature_cols is None else gnn_feature_cols
    )
    if not gnn_feature_cols_used:
        gnn_feature_cols_used = _existing_cols(X_all, cols)

    entity_cols_used = _existing_cols(
        X_all, DEFAULT_GNN_ENTITY_COLS if entity_cols is None else entity_cols
    )

    print(
        f"[GNN fold] Using {len(gnn_feature_cols_used):,} txn feature columns "
        f"for GNN; XGBoost keeps {len(cols):,} columns."
    )

    if scale_gnn_features:
        X_fit_gnn, X_valid_gnn, scaled_cols = _scale_gnn_feature_frames(
            X_fit,
            X_valid,
            gnn_feature_cols_used,
            entity_cols_used,
        )
        print(
            f"[GNN fold] StandardScaler fit on idxT and applied to "
            f"{len(scaled_cols):,} GNN feature columns."
        )
    else:
        X_fit_gnn = X_fit
        X_valid_gnn = X_valid

    if use_multi_entity_graph:
        if not entity_cols_used:
            raise ValueError("use_multi_entity_graph=True but no entity columns are present.")

        print(f"[GNN fold] Typed entity columns: {entity_cols_used}")

        if use_gnn_validation:
            gnn_df = pd.concat([X_fit_gnn, X_valid_gnn], ignore_index=True)
            gnn_y = np.r_[
                y_fit.values.astype(np.int64),
                y_valid.values.astype(np.int64),
            ]
            entity_vocabs = _make_typed_entity_vocabs(gnn_df, entity_cols_used)
            if not entity_vocabs:
                raise ValueError("No typed entity vocabularies could be built for this fold.")
            entity_num_nodes = {k: len(v) for k, v in entity_vocabs.items()}

            train_mask = np.zeros(len(gnn_df), dtype=bool)
            val_mask = np.zeros(len(gnn_df), dtype=bool)
            train_mask[:len(X_fit)] = True
            val_mask[len(X_fit):] = True

            print("[GNN fold] Building train+valid typed-entity graph...")
            gnn_data = _build_typed_entity_graph_from_df(
                gnn_df,
                feature_cols=gnn_feature_cols_used,
                entity_vocabs=entity_vocabs,
                labels=gnn_y,
                train_mask=train_mask,
                val_mask=val_mask,
            )

            edge_count = sum(
                gnn_data["txn", f"has_{entity_type}", entity_type].edge_index.shape[1]
                for entity_type in entity_vocabs
            )
            print(
                f"[GNN fold] Training on {len(X_fit):,} rows, validating on "
                f"{len(X_valid):,} rows, typed entity nodes: "
                f"{sum(entity_num_nodes.values()):,}, edges: {edge_count:,}..."
            )
            model = train_typed_entity_gnn(
                gnn_data,
                entity_num_nodes=entity_num_nodes,
                in_dim=gnn_data["txn"].x.shape[1],
                emb_dim=emb_dim,
                hidden=hidden,
                n_epochs=n_epochs,
                lr=lr,
                device=device,
                verbose=verbose,
                checkpoint_metric=checkpoint_metric,
                grad_clip_norm=grad_clip_norm,
            )

            gnn_data = gnn_data.to(next(model.parameters()).device)
            tr_emb, va_emb = extract_embeddings(model, gnn_data, len(X_fit), len(X_valid))

        else:
            train_entity_vocabs = _make_typed_entity_vocabs(X_fit_gnn, entity_cols_used)
            if not train_entity_vocabs:
                raise ValueError("No typed entity vocabularies could be built for this fold.")
            train_entity_num_nodes = {k: len(v) for k, v in train_entity_vocabs.items()}
            train_mask = np.ones(len(X_fit), dtype=bool)
            val_mask = np.zeros(len(X_fit), dtype=bool)

            print("[GNN fold] Building train-only typed-entity graph...")
            train_data = _build_typed_entity_graph_from_df(
                X_fit_gnn,
                feature_cols=gnn_feature_cols_used,
                entity_vocabs=train_entity_vocabs,
                labels=y_fit.values.astype(np.int64),
                train_mask=train_mask,
                val_mask=val_mask,
            )

            train_edge_count = sum(
                train_data["txn", f"has_{entity_type}", entity_type].edge_index.shape[1]
                for entity_type in train_entity_vocabs
            )
            print(
                f"[GNN fold] Training on {len(X_fit):,} rows, "
                f"typed entity nodes: {sum(train_entity_num_nodes.values()):,}, "
                f"edges: {train_edge_count:,}..."
            )
            model = train_typed_entity_gnn(
                train_data,
                entity_num_nodes=train_entity_num_nodes,
                in_dim=train_data["txn"].x.shape[1],
                emb_dim=emb_dim,
                hidden=hidden,
                n_epochs=n_epochs,
                lr=lr,
                device=device,
                verbose=verbose,
                checkpoint_metric=checkpoint_metric,
                grad_clip_norm=grad_clip_norm,
            )

            full_entity_vocabs = {k: dict(v) for k, v in train_entity_vocabs.items()}
            valid_entity_vocabs = _make_typed_entity_vocabs(X_valid_gnn, entity_cols_used)
            oov_by_type = {}
            for entity_type, valid_vocab in valid_entity_vocabs.items():
                if entity_type not in full_entity_vocabs:
                    continue
                before = len(full_entity_vocabs[entity_type])
                for value in valid_vocab:
                    if value not in full_entity_vocabs[entity_type]:
                        full_entity_vocabs[entity_type][value] = len(full_entity_vocabs[entity_type])
                oov_by_type[entity_type] = len(full_entity_vocabs[entity_type]) - before

            n_oov = sum(oov_by_type.values())
            print(
                f"[GNN fold] Extracting embeddings on train+valid typed-entity graph; "
                f"validation-only entity nodes: {n_oov:,}"
            )

            if n_oov:
                with torch.no_grad():
                    for entity_type, n_type_oov in oov_by_type.items():
                        if n_type_oov <= 0:
                            continue
                        old_emb = model.entity_embeddings[entity_type].weight.data
                        mean_emb = old_emb.mean(dim=0, keepdim=True)
                        new_table = torch.cat(
                            [old_emb, mean_emb.repeat(n_type_oov, 1)], dim=0
                        )
                        model.entity_embeddings[entity_type] = nn.Embedding.from_pretrained(
                            new_table, freeze=False
                        ).to(old_emb.device)

            infer_df = pd.concat([X_fit_gnn, X_valid_gnn], ignore_index=True)
            infer_data = _build_typed_entity_graph_from_df(
                infer_df,
                feature_cols=gnn_feature_cols_used,
                entity_vocabs=full_entity_vocabs,
            ).to(next(model.parameters()).device)

            tr_emb, va_emb = extract_embeddings(model, infer_data, len(X_fit), len(X_valid))

    elif use_gnn_validation:
        gnn_df = pd.concat([X_fit_gnn, X_valid_gnn], ignore_index=True)
        gnn_y = np.r_[
            y_fit.values.astype(np.int64),
            y_valid.values.astype(np.int64),
        ]
        gnn_uid_vocab = {u: i for i, u in enumerate(gnn_df[uid_col].unique())}

        train_mask = np.zeros(len(gnn_df), dtype=bool)
        val_mask = np.zeros(len(gnn_df), dtype=bool)
        train_mask[:len(X_fit)] = True
        val_mask[len(X_fit):] = True

        print("[GNN fold] Building train+valid UID graph with validation mask...")
        gnn_data = _build_graph_from_df(
            gnn_df,
            feature_cols=gnn_feature_cols_used,
            uid_col=uid_col,
            uid_to_idx=gnn_uid_vocab,
            labels=gnn_y,
            train_mask=train_mask,
            val_mask=val_mask,
        )

        print(
            f"[GNN fold] Training on {len(X_fit):,} rows, validating on "
            f"{len(X_valid):,} rows, UID nodes: {len(gnn_uid_vocab):,}..."
        )
        model = train_gnn(
            gnn_data,
            n_uids=len(gnn_uid_vocab),
            in_dim=gnn_data["txn"].x.shape[1],
            emb_dim=emb_dim,
            hidden=hidden,
            n_epochs=n_epochs,
            lr=lr,
            device=device,
            verbose=verbose,
            grad_clip_norm=grad_clip_norm,
        )

        gnn_data = gnn_data.to(next(model.parameters()).device)
        tr_emb, va_emb = extract_embeddings(model, gnn_data, len(X_fit), len(X_valid))

    else:
        train_uid_vocab = {u: i for i, u in enumerate(X_fit_gnn[uid_col].unique())}

        train_mask = np.ones(len(X_fit), dtype=bool)
        val_mask = np.zeros(len(X_fit), dtype=bool)

        print("[GNN fold] Building train-only UID graph...")
        train_data = _build_graph_from_df(
            X_fit_gnn,
            feature_cols=gnn_feature_cols_used,
            uid_col=uid_col,
            uid_to_idx=train_uid_vocab,
            labels=y_fit.values.astype(np.int64),
            train_mask=train_mask,
            val_mask=val_mask,
        )

        print(
            f"[GNN fold] Training on {len(X_fit):,} rows and "
            f"{len(train_uid_vocab):,} UID nodes..."
        )
        model = train_gnn_without_validation(
            train_data,
            n_uids=len(train_uid_vocab),
            in_dim=train_data["txn"].x.shape[1],
            emb_dim=emb_dim,
            hidden=hidden,
            n_epochs=n_epochs,
            lr=lr,
            device=device,
            verbose=verbose,
            checkpoint_metric=checkpoint_metric,
            grad_clip_norm=grad_clip_norm,
        )

        full_uid_vocab = dict(train_uid_vocab)
        valid_only_uids = [u for u in X_valid_gnn[uid_col].unique() if u not in full_uid_vocab]
        for u in valid_only_uids:
            full_uid_vocab[u] = len(full_uid_vocab)

        n_oov = len(full_uid_vocab) - len(train_uid_vocab)
        print(
            f"[GNN fold] Extracting embeddings on train+valid UID graph; "
            f"validation-only UIDs: {n_oov:,}"
        )

        if n_oov:
            with torch.no_grad():
                old_emb = model.uid_embedding.weight.data
                mean_emb = old_emb.mean(dim=0, keepdim=True)
                new_table = torch.cat([old_emb, mean_emb.repeat(n_oov, 1)], dim=0)
                model.uid_embedding = nn.Embedding.from_pretrained(
                    new_table, freeze=False
                ).to(old_emb.device)

        infer_df = pd.concat([X_fit_gnn, X_valid_gnn], ignore_index=True)
        infer_data = _build_graph_from_df(
            infer_df,
            feature_cols=gnn_feature_cols_used,
            uid_col=uid_col,
            uid_to_idx=full_uid_vocab,
        ).to(next(model.parameters()).device)

        tr_emb, va_emb = extract_embeddings(model, infer_data, len(X_fit), len(X_valid))

    new_cols = [f"{prefix}{i}" for i in range(emb_dim)]
    X_tr_aug = X_all.iloc[idxT].copy()
    X_va_aug = X_all.iloc[idxV].copy()
    for i, c in enumerate(new_cols):
        X_tr_aug[c] = tr_emb[:, i].astype(np.float32)
        X_va_aug[c] = va_emb[:, i].astype(np.float32)

    return X_tr_aug, X_va_aug, list(cols) + new_cols


# ---------------------------------------------------------------------------
# 5) Top-level convenience: do everything in one call, return augmented dfs
# ---------------------------------------------------------------------------
def augment_with_gnn_embeddings(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    cols: list,
    uid_col: str = "uid",
    mode: str = "inductive",         # 'inductive' (recommended) or 'transductive'
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

    Modes
    -----
    'inductive'    (RECOMMENDED for thesis honesty):
        - Training graph contains ONLY train transactions and the UIDs they touch.
        - Inference graph extends to include test transactions and any test-only UIDs.
        - Test-only UID embeddings are initialized to the mean of the trained UID
          embeddings (a neutral prior), so the encoder runs on test cleanly.
        - This is what a production fraud system would actually do.

    'transductive' (academic standard, mild structural leakage):
        - Training graph contains train + test transactions and all UIDs.
        - Only train labels enter the loss. No label leakage.
        - But test features and graph structure influence the encoder's weights
          during training. For a temporal split like IEEE-CIS this slightly
          overstates real-world performance.
    """
    assert mode in ("inductive", "transductive"), f"unknown mode={mode}"
    print(f"=== GNN augmentation, mode='{mode}' ===\n")

    if mode == "transductive":
        # ---- transductive: one big graph, train on train labels only ----
        print("[1/4] Building train+test heterograph (transductive)...")
        data, n_uids, n_train, n_test = build_hetero_graph(
            X_train, X_test, y_train, feature_cols=cols, uid_col=uid_col
        )
        in_dim = data["txn"].x.shape[1]
        print(f"      {n_train + n_test:,} txn nodes, {n_uids:,} uid nodes, "
              f"{data['txn','has_uid','uid'].edge_index.shape[1]:,} edges")

        print("\n[2/4] Training GNN on combined graph...")
        model = train_gnn(data, n_uids=n_uids, in_dim=in_dim, emb_dim=emb_dim,
                          hidden=hidden, n_epochs=n_epochs, lr=lr, device=device,
                          verbose=verbose)

        print("\n[3/4] Extracting embeddings (encoder only)...")
        train_emb, test_emb = extract_embeddings(model, data, n_train, n_test)

    else:
        # ---- inductive: train on train-only graph, then inference on extended graph ----
        n_train = len(X_train)
        n_test = len(X_test)

        # 1) UID vocabulary from TRAIN ONLY
        train_uid_vocab = {u: i for i, u in enumerate(X_train[uid_col].unique())}
        n_uids_train = len(train_uid_vocab)

        # train/val split inside train
        val_split = int(n_train * 0.75)
        train_mask = np.zeros(n_train, dtype=bool); train_mask[:val_split] = True
        val_mask   = np.zeros(n_train, dtype=bool); val_mask[val_split:] = True

        print("[1/4] Building TRAIN-ONLY heterograph (inductive training graph)...")
        train_data = _build_graph_from_df(
            X_train, feature_cols=cols, uid_col=uid_col,
            uid_to_idx=train_uid_vocab,
            labels=y_train.values.astype(np.int64),
            train_mask=train_mask, val_mask=val_mask,
        )
        in_dim = train_data["txn"].x.shape[1]
        print(f"      train txn nodes: {n_train:,}  |  train UID nodes: {n_uids_train:,}")

        print("\n[2/4] Training GNN on train-only graph (no test rows visible)...")
        model = train_gnn(train_data, n_uids=n_uids_train, in_dim=in_dim,
                          emb_dim=emb_dim, hidden=hidden, n_epochs=n_epochs,
                          lr=lr, device=device, verbose=verbose)

        # 2) EXTEND vocabulary with test-only UIDs
        test_only_uids = [u for u in X_test[uid_col].unique() if u not in train_uid_vocab]
        full_uid_vocab = dict(train_uid_vocab)
        for u in test_only_uids:
            full_uid_vocab[u] = len(full_uid_vocab)
        n_uids_full = len(full_uid_vocab)
        n_oov = n_uids_full - n_uids_train
        print(f"\n[3/4] Inference: extend graph with test rows. "
              f"Test-only UIDs (OOV): {n_oov:,} / {len(X_test[uid_col].unique()):,}")

        # 3) Grow the UID embedding table: keep trained rows, init new rows with mean
        with torch.no_grad():
            old_emb = model.uid_embedding.weight.data
            mean_emb = old_emb.mean(dim=0, keepdim=True)
            new_table = torch.cat(
                [old_emb, mean_emb.repeat(n_oov, 1)], dim=0
            )
            model.uid_embedding = nn.Embedding.from_pretrained(
                new_table, freeze=False
            ).to(old_emb.device)

        # 4) Build the FULL graph for inference (train + test txns)
        all_df = pd.concat([X_train.reset_index(drop=True),
                            X_test.reset_index(drop=True)], ignore_index=True)
        full_data = _build_graph_from_df(
            all_df, feature_cols=cols, uid_col=uid_col,
            uid_to_idx=full_uid_vocab,
        ).to(next(model.parameters()).device)

        # 5) Extract embeddings using the frozen encoder
        train_emb, test_emb = extract_embeddings(model, full_data, n_train, n_test)

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
          f"=  {len(cols_aug)} total\n")

    return X_train_aug, X_test_aug, cols_aug


# ===========================================================================
# Notebook integration recipe — paste into a NEW CELL in your XGB notebook
# ===========================================================================
USAGE_EXAMPLE = """
# === New cell, AFTER feature engineering produced X_train_copy5 / X_test_copy5 ===
from gnn_augment_xgboost_ieee import augment_with_gnn_embeddings

# RECOMMENDED: inductive (production-realistic, defensible in thesis)
X_train_copy6, X_test_copy6, cols_augmented = augment_with_gnn_embeddings(
    X_train_copy5, X_test_copy5, y_train, cols,
    mode='inductive',          # <-- use this for your final reported numbers
    uid_col='uid', emb_dim=32, n_epochs=15, device='cuda',
)

# OPTIONAL: also run transductive for a side-by-side comparison in your thesis
X_train_copy6_T, X_test_copy6_T, cols_augmented_T = augment_with_gnn_embeddings(
    X_train_copy5, X_test_copy5, y_train, cols,
    mode='transductive',       # <-- known mild leakage, report as upper bound
    uid_col='uid', emb_dim=32, n_epochs=15, device='cuda',
)

# === Then RE-RUN your existing GroupKFold XGBoost cell, swapping: ===
#     X_train_copy5 → X_train_copy6        (or X_train_copy6_T)
#     X_test_copy5  → X_test_copy6         (or X_test_copy6_T)
#     cols          → cols_augmented       (or cols_augmented_T)
#
# Reporting tip: present three OOF AUC numbers in your thesis table —
#   (a) baseline XGBoost           e.g. 0.9450
#   (b) +inductive GNN embeddings  e.g. 0.9478   <-- honest gain
#   (c) +transductive GNN          e.g. 0.9495   <-- upper bound / leakage budget
# The (c)-(b) gap is your quantitative measurement of how much transductive
# training was inflating the score by peeking at test structure.
"""

if __name__ == "__main__":
    print(USAGE_EXAMPLE)
