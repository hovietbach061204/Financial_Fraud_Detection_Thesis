"""
GTAN-style (attribute-driven, HOMOGENEOUS transaction graph) GAT embeddings -> XGBoost
======================================================================================
A PyTorch-Geometric re-implementation of AI4Risk/antifraud's **GTAN**
("Semi-supervised Credit Card Fraud Detection via Attribute-driven Graph
Representation", Xiang et al., AAAI 2023), adapted to the IEEE-CIS dataset and wired
to augment your existing XGBoost pipeline — the same public API shape as
`gnn_augment_xgboost_ieee_support_inductive.py`.

WHAT IS COPIED FROM GTAN
------------------------
1) GRAPH CONSTRUCTION (homogeneous, transactions-as-nodes).
   GTAN's S-FFSD loader connects transactions that share a value in any of
   ["Source", "Target", "Location", "Type"]: for each such column it groups rows,
   sorts each group by Time, and for every row adds edges to itself and the next two
   later rows in the group (`edge_per_trans = 3`). All four relations are merged into
   ONE directed homogeneous graph (edge types are NOT stored). We replicate this loop
   exactly; only the grouping columns change to IEEE-CIS identity columns
   (card1, card1_addr1, card1_addr1_P_emaildomain, DeviceInfo) sorted by TransactionDT.

   Because edges point earlier -> later, a node only ever AGGREGATES FROM ITS OWN PAST
   (and itself). That is temporally causal: no future transaction can leak into a
   node's representation through the graph structure.

2) MODEL (GTAN-GNN).
   - A gated graph-Transformer conv stack. GTAN's `TransformerConv` is the UniMP
     operator (multi-head Q/K/V dot-product attention over neighbors + a *gated* skip
     connection + LayerNorm + PReLU). PyG's `TransformerConv(..., beta=True,
     root_weight=True)` implements exactly that gated residual, so we use it and add
     LayerNorm + PReLU + dropout between layers to mirror GTAN.
   - The GTAN label-propagation trick (a.k.a. UniMP masked-label trick): neighbour
     labels are embedded and added as a residual to the node features BEFORE message
     passing, while the node being predicted has its own label masked to "unknown".
     This is the single most important ingredient that separates GTAN from a plain GAT.

WHAT IS DIFFERENT FROM THE ORIGINAL REPO (and why)
--------------------------------------------------
- Framework: PyG instead of DGL. Your whole pipeline (and Apple-Silicon/MPS) is PyG;
  DGL on MPS is painful. The math is identical.
- Validation: the repo uses a random StratifiedKFold (transductive, mild temporal
  leakage). We additionally offer an **inductive** mode (train-only graph, test rows
  attached only at inference) so you can report an honest production-style number, just
  like your existing gnn_augment module.
- Categorical attribute embeddings (GTAN's `TransEmbedding`) are OPTIONAL here and OFF
  by default: your feature list is already mostly frequency-encoded / numeric, and the
  IEEE identity columns (card1_addr1_*) have huge cardinality that would blow up an
  nn.Embedding table. Enable `cat_features=[...]` if you want them.

USAGE (new cell after feature engineering produced X_train_copy5 / X_test_copy5)
-------------------------------------------------------------------------------
    from gtan_augment_xgboost_ieee import augment_with_gtan_embeddings

    X_train_g, X_test_g, cols_g = augment_with_gtan_embeddings(
        X_train_copy5, X_test_copy5, y_train, cols,
        mode='inductive',              # 'inductive' (recommended) or 'transductive'
        time_col='TransactionDT',      # used ONLY to order edges, never as a feature
        hidden_dim=16, heads=4, n_layers=2, n_epochs=30, device='cuda',
    )
    # then swap X_train_copy5->X_train_g, X_test_copy5->X_test_g, cols->cols_g in XGBoost.

Dependencies: torch  torch_geometric  scikit-learn  pandas  numpy
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv
from torch_geometric.utils import add_remaining_self_loops
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Defaults — your full node-feature list, and the GTAN edge-grouping columns.
# ---------------------------------------------------------------------------
DEFAULT_GTAN_FEATURE_COLS = [
    'TransactionAmt', 'ProductCD_FE', 'card1', 'card2', 'card3', 'card5', 'card6_FE',
    'addr1', 'addr2', 'dist1', 'dist2', 'P_emaildomain_FE', 'R_emaildomain_FE',
    'C1', 'C2', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14',
    'D1', 'D2', 'D3', 'D4', 'D5', 'D10', 'D11', 'D15',
    'M1', 'M2', 'M3', 'M4_FE', 'M6', 'M7', 'M8', 'M9',
    'V1', 'V3', 'V4', 'V6', 'V8', 'V11', 'V13', 'V14', 'V17', 'V20', 'V23', 'V26',
    'V27', 'V30', 'V36', 'V37', 'V40', 'V41', 'V44', 'V47', 'V48', 'V54', 'V56', 'V59',
    'V62', 'V65', 'V67', 'V68', 'V70', 'V76', 'V78', 'V80', 'V82', 'V86', 'V88', 'V89',
    'V91', 'V107', 'V108', 'V111', 'V115', 'V117', 'V120', 'V121', 'V123', 'V124',
    'V127', 'V129', 'V130', 'V136', 'V138', 'V139', 'V142', 'V147', 'V156', 'V160',
    'V162', 'V165', 'V166', 'V169', 'V171', 'V173', 'V175', 'V176', 'V178', 'V180',
    'V182', 'V185', 'V187', 'V188', 'V198', 'V203', 'V205', 'V207', 'V209', 'V210',
    'V215', 'V218', 'V220', 'V221', 'V223', 'V224', 'V226', 'V228', 'V229', 'V234',
    'V235', 'V238', 'V240', 'V250', 'V252', 'V253', 'V257', 'V258', 'V260', 'V261',
    'V264', 'V266', 'V267', 'V271', 'V274', 'V277', 'V281', 'V283', 'V284', 'V285',
    'V286', 'V289', 'V291', 'V294', 'V296', 'V297', 'V301', 'V303', 'V305', 'V307',
    'V309', 'V310', 'V314', 'V320',
    'id_01', 'id_02', 'id_03', 'id_04', 'id_05', 'id_06', 'id_09', 'id_10', 'id_11',
    'id_12', 'id_13', 'id_15_FE', 'id_16', 'id_17', 'id_18', 'id_19', 'id_20', 'id_28',
    'id_29', 'id_31_FE', 'id_35', 'id_36', 'id_37', 'id_38',
    'DeviceType', 'DeviceInfo_FE', 'cents', 'dollars',
    'addr1_FE', 'card1_FE', 'card2_FE', 'card3_FE',
    'card1_addr1', 'card1_addr1_P_emaildomain', 'card1_addr1_FE',
    'card1_addr1_P_emaildomain_FE',
    'TransactionAmt_card1_mean', 'TransactionAmt_card1_std',
    'TransactionAmt_card1_addr1_mean', 'TransactionAmt_card1_addr1_std',
    'TransactionAmt_card1_addr1_P_emaildomain_mean',
    'TransactionAmt_card1_addr1_P_emaildomain_std',
    'D9_card1_mean', 'D9_card1_std', 'D9_card1_addr1_mean', 'D9_card1_addr1_std',
    'D9_card1_addr1_P_emaildomain_mean', 'D9_card1_addr1_P_emaildomain_std',
    'D11_card1_mean', 'D11_card1_std', 'D11_card1_addr1_mean', 'D11_card1_addr1_std',
    'D11_card1_addr1_P_emaildomain_mean', 'D11_card1_addr1_P_emaildomain_std',
    'is_december', 'is_holiday', 'uid_FE', 'delta_seconds_prev', 'uid_count_so_far',
    'uid_prev_amt', 'uid_amt_diff_prev', 'uid_amt_ratio_prev', 'uid_amt_cummean',
    'uid_amt_cummax', 'DT_hour_sin', 'DT_hour_cos', 'DT_day_week_sin', 'DT_day_week_cos',
    'DT_day_month_sin', 'DT_day_month_cos', 'DT_week_month_sin', 'DT_week_month_cos',
]

# IEEE-CIS analogue of GTAN's ["Source","Target","Location","Type"]: link transactions
# that share a card / card+addr / card+addr+email / device.  These are RAW identity
# columns (not the *_FE frequency encodings) so grouping is by true identity.
DEFAULT_EDGE_GROUP_COLS = [
    "uid",
    "card1_addr1",
    "card1_addr1_P_emaildomain",
    "DeviceInfo",
]

DEFAULT_TIME_COL = "TransactionDT"
EDGE_PER_TRANS = 3          # GTAN default: self + next 2 later transactions per group


# ===========================================================================
# 1) GRAPH CONSTRUCTION  (faithful port of GTAN's S-FFSD edge loop)
# ===========================================================================
def _valid_entity_mask(s: pd.Series) -> pd.Series:
    """Treat NaN / -1 / empty / 'nan' sentinels as "no entity" -> no edges."""
    as_str = s.astype(str)
    return s.notna() & ~as_str.isin(["", "-1", "-1.0", "nan", "None", "NONE"])


def build_gtan_edge_index(
    df: pd.DataFrame,
    group_cols: list,
    time_col: str,
    edge_per_trans: int = EDGE_PER_TRANS,
    num_nodes: int | None = None,
) -> torch.Tensor:
    """
    Replicate GTAN's homogeneous edge construction.

    For each column in `group_cols`:
        group rows by the column value, sort each group by `time_col`, and for every
        row i add edges  src=row[i] -> dst=row[i+j]  for j in 0..edge_per_trans-1.
    All columns' edges are concatenated into a single directed homogeneous graph
    (edge types are dropped, exactly as in the repo). Self-loops are then ensured for
    every node so isolated transactions still see themselves.

    `df` must be 0..N-1 indexed (positional == node id). Returns edge_index (2, E).
    """
    n = len(df)
    if num_nodes is None:
        num_nodes = n
    pos = np.arange(n)
    time_vals = df[time_col].values if time_col in df.columns else pos

    src_all: list[int] = []
    dst_all: list[int] = []

    for col in group_cols:
        if col not in df.columns:
            continue
        s = df[col]
        valid = _valid_entity_mask(s).to_numpy()
        if not valid.any():
            continue
        sub = pd.DataFrame({
            "node": pos[valid],
            "key": s[valid].astype(str).to_numpy(),
            "t": np.asarray(time_vals)[valid],
        })
        for _, g in sub.groupby("key", sort=False):
            g = g.sort_values("t", kind="mergesort")          # stable, ties keep order
            idxs = g["node"].to_numpy()
            L = len(idxs)
            for i in range(L):
                for j in range(edge_per_trans):
                    if i + j < L:
                        src_all.append(int(idxs[i]))           # earlier
                        dst_all.append(int(idxs[i + j]))       # later (>= earlier)

    if src_all:
        edge_index = torch.tensor([src_all, dst_all], dtype=torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    # Ensure every node has at least a self-loop (TransformerConv needs in-edges to be
    # meaningful; isolated nodes otherwise only get the root/skip term).
    edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=num_nodes)
    return edge_index


# ===========================================================================
# 2) MODEL  (GTAN-GNN: gated graph-Transformer + label-embedding residual)
# ===========================================================================
class GTAN(nn.Module):
    """
    PyG port of GTAN's GraphAttnModel.

    in_feats     : number of numeric node features
    hidden_dim   : per-head hidden width (GTAN uses hid_dim//4 with 4 heads)
    heads        : number of attention heads (concatenated)
    n_layers     : number of gated TransformerConv layers
    n_classes    : 2 (legit / fraud); class index `n_classes` (=2) means "unknown label"
    """

    def __init__(self, in_feats, hidden_dim=64, heads=4, n_layers=2,
                 n_classes=2, dropout=0.2):
        super().__init__()
        self.in_feats = in_feats
        self.n_classes = n_classes
        width = hidden_dim * heads

        # ----- GTAN label-embedding trick -----
        # index 0,1 = known labels; index n_classes (=2) = "unknown" -> zero embedding.
        self.label_emb = nn.Embedding(n_classes + 1, in_feats, padding_idx=n_classes)
        self.feat_lin = nn.Linear(in_feats, width)
        self.label_lin = nn.Linear(in_feats, width)
        self.label_proc = nn.Sequential(
            nn.BatchNorm1d(width), nn.PReLU(), nn.Dropout(dropout),
            nn.Linear(width, in_feats),
        )
        self.input_drop = nn.Dropout(dropout)

        # ----- gated graph-Transformer stack -----
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        dim = in_feats
        for _ in range(n_layers):
            self.convs.append(
                TransformerConv(dim, hidden_dim, heads=heads, concat=True,
                                beta=True, dropout=dropout)   # beta=True => gated skip
            )
            self.norms.append(nn.LayerNorm(width))
            dim = width
        self.act = nn.PReLU()
        self.drop = nn.Dropout(dropout)
        self.emb_dim = dim                                    # penultimate embedding size

        # ----- classification head (discarded when extracting embeddings) -----
        self.head = nn.Sequential(
            nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.PReLU(), nn.Dropout(dropout),
            nn.Linear(dim, n_classes),
        )

    def encode(self, x, edge_index, y_input):
        """Return the penultimate node embeddings (used to augment XGBoost)."""
        le = self.input_drop(self.label_emb(y_input))        # (N, in_feats)
        comb = self.feat_lin(x) + self.label_lin(le)         # (N, width)
        comb = self.label_proc(comb)                         # (N, in_feats)
        h = x + comb                                         # residual (GTAN)
        for conv, norm in zip(self.convs, self.norms):
            h = conv(h, edge_index)
            h = self.drop(self.act(norm(h)))
        return h

    def forward(self, x, edge_index, y_input):
        h = self.encode(x, edge_index, y_input)
        return self.head(h), h


# ===========================================================================
# 3) TRAINING  (full-graph, class-weighted, with UniMP/GTAN label masking)
# ===========================================================================
def _make_y_input(y, known_idx, n_classes, device):
    """Build the label-input vector: `known_idx` reveal true labels, the rest unknown."""
    y_input = torch.full((y.shape[0],), fill_value=n_classes, dtype=torch.long, device=device)
    y_input[known_idx] = y[known_idx]
    return y_input


def train_gtan(
    x, edge_index, y, train_idx, val_idx,
    in_feats, hidden_dim=16, heads=4, n_layers=2, n_classes=2,
    dropout=0.2, n_epochs=30, lr=3e-4, weight_decay=1e-5,
    label_rate=0.5, pos_weight=None, grad_clip=2.0,
    device="cpu", verbose=True,
    batch_size=None, num_neighbors=None,
):
    """
    Train the GTAN model.

    batch_size : None -> FULL-GRAPH training (whole graph each step; simplest; needs the
                         entire graph + activations to fit in memory).
                 int  -> MINI-BATCH training with PyG `NeighborLoader`: each step samples a
                         batch of seed nodes and their k-hop neighbourhood into a subgraph,
                         so peak memory is bounded by the batch, not the whole graph. Use
                         for very large graphs / limited GPU.
    num_neighbors : per-hop fan-out for NeighborLoader (length == n_layers). Default
                    [-1]*n_layers == FULL neighborhood (faithful to GTAN's
                    MultiLayerFullNeighborSampler). Cap it (e.g. [15, 10]) to trade a bit of
                    fidelity for speed/memory.
    label_rate : (full-graph only) fraction of TRAIN labels revealed to the label-embedding
                 each step; the rest are masked and supervised on (the GTAN/UniMP masked-
                 label scheme). In mini-batch mode the seed nodes' own labels are masked
                 each batch instead, which is the original GTAN behavior.
    pos_weight : weight on the fraud class in cross-entropy. If None, sqrt(n_neg / n_pos).
    """
    dev = torch.device(device if (torch.cuda.is_available() or device in ("cpu", "mps", "cuda"))
                       else "cpu")
    y_cpu = y.detach().to("cpu")
    ti_cpu = train_idx.detach().to("cpu")
    if pos_weight is None:
        n_pos = float((y_cpu[ti_cpu] == 1).sum().clamp(min=1))
        n_neg = float((y_cpu[ti_cpu] == 0).sum().clamp(min=1))
        pos_weight = float(np.sqrt(n_neg / n_pos))
    weight = torch.tensor([1.0, pos_weight], device=dev)

    model = GTAN(in_feats, hidden_dim, heads, n_layers, n_classes, dropout).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    if batch_size is None:
        return _train_full_graph(model, opt, x, edge_index, y, train_idx, val_idx, n_classes,
                                 weight, n_epochs, label_rate, grad_clip, dev, verbose)
    if num_neighbors is None:
        num_neighbors = [-1] * n_layers
    return _train_minibatch(model, opt, x, edge_index, y, train_idx, val_idx, n_classes,
                            weight, n_epochs, grad_clip, dev, verbose, batch_size, num_neighbors)


def _train_full_graph(model, opt, x, edge_index, y, train_idx, val_idx, n_classes,
                      weight, n_epochs, label_rate, grad_clip, dev, verbose):
    x = x.to(dev); edge_index = edge_index.to(dev); y = y.to(dev)
    train_idx = train_idx.to(dev); val_idx = val_idx.to(dev)
    n_train = train_idx.shape[0]
    n_reveal = int(label_rate * n_train)
    y_input_eval = _make_y_input(y, train_idx, n_classes, dev)        # eval: all train revealed

    best_auc, best_state = -1.0, None
    for epoch in range(1, n_epochs + 1):
        model.train()
        perm = train_idx[torch.randperm(n_train, device=dev)]
        reveal_idx = perm[:n_reveal]
        supervise_idx = perm[n_reveal:] if n_reveal < n_train else perm
        y_input = _make_y_input(y, reveal_idx, n_classes, dev)

        opt.zero_grad()
        logits, _ = model(x, edge_index, y_input)
        loss = F.cross_entropy(logits[supervise_idx], y[supervise_idx], weight=weight)
        loss.backward()
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()

        model.eval()
        with torch.no_grad():
            logits, _ = model(x, edge_index, y_input_eval)
            prob = F.softmax(logits, dim=1)[:, 1]
            val_auc = roc_auc_score(y[val_idx].cpu().numpy(), prob[val_idx].cpu().numpy())
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if verbose:
            print(f"epoch {epoch:3d} | loss={loss.item():.4f} | val_AUC={val_auc:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f">>> Best GTAN val AUC: {best_auc:.4f}")
    return model


def _train_minibatch(model, opt, x, edge_index, y, train_idx, val_idx, n_classes,
                     weight, n_epochs, grad_clip, dev, verbose, batch_size, num_neighbors):
    from torch_geometric.loader import NeighborLoader
    from torch_geometric.data import Data

    x = x.detach().cpu().float()
    y = y.detach().cpu().long()
    edge_index = edge_index.detach().cpu()
    train_idx = train_idx.detach().cpu()
    val_idx = val_idx.detach().cpu()

    data = Data(x=x, edge_index=edge_index, y=y)
    # global known-label vector: reveal ONLY train labels (val/test -> unknown). Carried as
    # a node attribute so NeighborLoader slices it per subgraph as `batch.yik`.
    data.yik = _make_y_input(y, train_idx, n_classes, torch.device("cpu"))

    train_loader = NeighborLoader(data, num_neighbors=num_neighbors, input_nodes=train_idx,
                                  batch_size=batch_size, shuffle=True)
    val_loader = NeighborLoader(data, num_neighbors=num_neighbors, input_nodes=val_idx,
                                batch_size=batch_size, shuffle=False)

    best_auc, best_state = -1.0, None
    for epoch in range(1, n_epochs + 1):
        model.train()
        tot, nb = 0.0, 0
        for batch in train_loader:
            bs = batch.batch_size
            batch = batch.to(dev)
            yin = batch.yik.clone()
            yin[:bs] = n_classes                          # mask SEED nodes' own labels (GTAN)
            opt.zero_grad()
            logits, _ = model(batch.x, batch.edge_index, yin)
            seed_logits, seed_y = logits[:bs], batch.y[:bs]
            m = seed_y != n_classes
            loss = F.cross_entropy(seed_logits[m], seed_y[m], weight=weight)
            loss.backward()
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            tot += float(loss.item()); nb += 1

        model.eval()
        vp = np.zeros(y.shape[0], dtype=np.float32)
        with torch.no_grad():
            for batch in val_loader:
                bs = batch.batch_size
                batch = batch.to(dev)
                logits, _ = model(batch.x, batch.edge_index, batch.yik)  # val seeds already unknown
                p = F.softmax(logits[:bs], dim=1)[:, 1].cpu().numpy()
                vp[batch.n_id[:bs].cpu().numpy()] = p
        val_auc = roc_auc_score(y[val_idx].numpy(), vp[val_idx.numpy()])
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if verbose:
            print(f"epoch {epoch:3d} | loss={tot/max(nb,1):.4f} | val_AUC={val_auc:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f">>> Best GTAN val AUC: {best_auc:.4f}")
    return model


def gtan_predict(model, x, edge_index, y_input_known, n_classes, dev,
                 batch_size=None, num_neighbors=None):
    """
    Run the trained model over ALL nodes and return (prob[N], emb[N, D]) using the known
    train labels (no seed masking -- this is the embedding/inference pass). Full-graph if
    `batch_size` is None, else a NeighborLoader pass with the same fan-out as training.
    """
    model.eval()
    N = x.shape[0]
    if batch_size is None:
        with torch.no_grad():
            logits, h = model(x.to(dev), edge_index.to(dev), y_input_known.to(dev))
            return (F.softmax(logits, dim=1)[:, 1].cpu().numpy(), h.cpu().numpy())

    from torch_geometric.loader import NeighborLoader
    from torch_geometric.data import Data
    data = Data(x=x.detach().cpu().float(), edge_index=edge_index.detach().cpu())
    data.yik = y_input_known.detach().cpu().long()
    loader = NeighborLoader(data, num_neighbors=(num_neighbors or [-1]),
                            input_nodes=None, batch_size=batch_size, shuffle=False)
    prob = np.zeros(N, dtype=np.float32)
    emb = None
    with torch.no_grad():
        for batch in loader:
            bs = batch.batch_size
            batch = batch.to(dev)
            logits, h = model(batch.x, batch.edge_index, batch.yik)
            ids = batch.n_id[:bs].cpu().numpy()
            prob[ids] = F.softmax(logits[:bs], dim=1)[:, 1].cpu().numpy()
            hb = h[:bs].cpu().numpy()
            if emb is None:
                emb = np.zeros((N, hb.shape[1]), dtype=np.float32)
            emb[ids] = hb
    return prob, emb


def extract_oof_embeddings(model, x, edge_index, y_t, train_idx, val_idx,
                           n_classes, dev, k_folds=5, seed=42,
                           batch_size=None, num_neighbors=None, verbose=True):
    """
    Leak-free embeddings for downstream models (the fix for the train/val label-leakage
    mismatch).

    Problem: a single inference pass that reveals ALL train labels bakes each TRAIN node's
    OWN label into its embedding (via the label-embedding residual), while VAL nodes have
    their own label masked. Train and val embeddings then live in different distributions,
    so XGBoost overfits the leaked train signal and collapses on val.

    Fix (out-of-fold): split train nodes into `k_folds`. For each fold, MASK that fold's own
    labels (set to "unknown") while REVEALING all other train labels, run inference, and keep
    the embeddings ONLY for the held-out fold. Every train node thus gets an embedding where
    its own label was hidden but its (train) neighbours' labels were visible -- exactly the
    condition val nodes already satisfy. Val embeddings come from one extra pass that reveals
    ALL train labels (val's own label is unknown anyway). Result: train & val embeddings share
    the same "own-label masked, train-neighbour labels visible" condition -> no leakage, while
    GTAN's neighbour-label propagation is preserved.

    Returns (prob[N], emb[N, D]) with train rows out-of-fold and val rows from the
    all-train-revealed pass. (prob for val == the honest val scoring; train prob is OOF.)
    """
    N = x.shape[0]
    D = model.emb_dim
    prob = np.zeros(N, dtype=np.float32)
    emb = np.zeros((N, D), dtype=np.float32)

    ti = train_idx.detach().cpu().numpy()
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(len(ti)), k_folds)

    # --- TRAIN nodes: K-fold, the held-out fold's OWN labels masked ---
    for k, fold in enumerate(folds):
        held = ti[fold]                                    # train nodes hidden this pass
        revealed = np.setdiff1d(ti, held)                  # other train labels stay visible
        yin = torch.full((N,), n_classes, dtype=torch.long)
        rev_t = torch.as_tensor(revealed, dtype=torch.long)
        yin[rev_t] = y_t[rev_t]
        p_k, e_k = gtan_predict(model, x, edge_index, yin, n_classes, dev,
                                batch_size=batch_size, num_neighbors=num_neighbors)
        prob[held] = p_k[held]
        emb[held] = e_k[held]
        if verbose:
            print(f"    OOF fold {k + 1}/{k_folds}: extracted {len(held):,} train nodes")

    # --- VAL nodes: one pass with ALL train labels revealed (val own label already unknown) ---
    yin = _make_y_input(y_t, train_idx, n_classes, torch.device("cpu"))
    p_v, e_v = gtan_predict(model, x, edge_index, yin, n_classes, dev,
                            batch_size=batch_size, num_neighbors=num_neighbors)
    vi = val_idx.detach().cpu().numpy()
    prob[vi] = p_v[vi]
    emb[vi] = e_v[vi]
    return prob, emb


# ===========================================================================
# 4) FEATURE PREP + EMBEDDING EXTRACTION
# ===========================================================================
def _existing(df, cols):
    return [c for c in cols if c in df.columns]


def _to_feature_matrix(df_fit, df_other, feature_cols, scale=True):
    """Numeric matrix; StandardScaler fit on df_fit only, applied to both."""
    cols = _existing(df_fit, feature_cols)
    fit = df_fit[cols].apply(pd.to_numeric, errors="coerce").fillna(-1).astype(np.float32)
    oth = df_other[cols].apply(pd.to_numeric, errors="coerce").fillna(-1).astype(np.float32)
    if scale:
        sc = StandardScaler()
        fit_v = sc.fit_transform(fit.values).astype(np.float32)
        oth_v = sc.transform(oth.values).astype(np.float32)
    else:
        fit_v, oth_v = fit.values, oth.values
    return fit_v, oth_v, cols


@torch.no_grad()
def _extract(model, x, edge_index, y_input, n_first, device):
    model.eval()
    h = model.encode(x.to(device), edge_index.to(device), y_input.to(device)).cpu().numpy()
    return h[:n_first], h[n_first:]


# ===========================================================================
# 5) TOP-LEVEL: augment XGBoost dataframes with GTAN embeddings
# ===========================================================================
def augment_with_gtan_embeddings(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    cols: list,
    mode: str = "inductive",                 # 'inductive' (recommended) or 'transductive'
    feature_cols: list | None = None,        # node features (default: DEFAULT_GTAN_FEATURE_COLS)
    group_cols: list | None = None,          # edge grouping (default: DEFAULT_EDGE_GROUP_COLS)
    time_col: str = DEFAULT_TIME_COL,        # used ONLY to order edges
    edge_per_trans: int = EDGE_PER_TRANS,
    hidden_dim: int = 16,
    heads: int = 4,
    n_layers: int = 2,
    dropout: float = 0.2,
    n_epochs: int = 30,
    lr: float = 3e-4,
    label_rate: float = 0.5,
    pos_weight: float | None = None,
    val_fraction: float = 0.25,
    device: str = "cuda",
    scale_features: bool = True,
    prefix: str = "gtan_emb_",
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, list]:
    """
    Build a GTAN-style homogeneous transaction graph, train the GTAN-GAT, extract node
    embeddings, and append them to X_train / X_test for your existing XGBoost cell.

    Returns (X_train_aug, X_test_aug, cols_aug).

    Modes
    -----
    'inductive'   : edges + training use TRAIN rows only; TEST rows are attached only at
                    inference (their label input is "unknown"). Production-realistic.
    'transductive': edges built over TRAIN+TEST; only TRAIN labels enter the loss
                    (TEST labels are "unknown"). Standard GNN-benchmark setting; mild
                    structural leakage on a temporal split.
    """
    assert mode in ("inductive", "transductive"), f"unknown mode={mode}"
    feature_cols = feature_cols or DEFAULT_GTAN_FEATURE_COLS
    group_cols = group_cols or DEFAULT_EDGE_GROUP_COLS
    n_train, n_test = len(X_train), len(X_test)
    y_np = y_train.values.astype(np.int64)
    print(f"=== GTAN augmentation, mode='{mode}' ===")
    print(f"    features={len(_existing(X_train, feature_cols))}, "
          f"edge groups={_existing(X_train, group_cols)}, time_col='{time_col}'")

    # time-ordered train/val split for checkpointing (last `val_fraction` of train rows)
    val_split = int(n_train * (1.0 - val_fraction))
    train_idx_np = np.arange(0, val_split)
    val_idx_np = np.arange(val_split, n_train)

    if mode == "transductive":
        all_df = pd.concat([X_train.reset_index(drop=True),
                            X_test.reset_index(drop=True)], ignore_index=True)
        edge_index = build_gtan_edge_index(all_df, group_cols, time_col, edge_per_trans)
        Xtr, Xte, used = _to_feature_matrix(X_train, X_test, feature_cols, scale_features)
        x = torch.tensor(np.concatenate([Xtr, Xte], axis=0))
        y = torch.tensor(np.concatenate([y_np, np.full(n_test, 2)]), dtype=torch.long)  # test=unknown
        in_feats = x.shape[1]
        print(f"    graph: {x.shape[0]:,} nodes, {edge_index.shape[1]:,} edges, in_feats={in_feats}")

        model = train_gtan(
            x, edge_index, y,
            torch.tensor(train_idx_np), torch.tensor(val_idx_np),
            in_feats, hidden_dim, heads, n_layers, 2, dropout,
            n_epochs, lr, label_rate=label_rate, pos_weight=pos_weight,
            device=device, verbose=verbose,
        )
        dev = next(model.parameters()).device
        y_input = _make_y_input(y.to(dev),
                                torch.arange(n_train, device=dev), 2, dev)  # reveal all train
        train_emb, test_emb = _extract(model, x, edge_index, y_input, n_train, dev)

    else:  # inductive
        # ---- train on TRAIN-only graph ----
        edge_tr = build_gtan_edge_index(X_train.reset_index(drop=True),
                                        group_cols, time_col, edge_per_trans,
                                        num_nodes=n_train)
        Xtr, Xte, used = _to_feature_matrix(X_train, X_test, feature_cols, scale_features)
        x_tr = torch.tensor(Xtr)
        y_tr = torch.tensor(y_np, dtype=torch.long)
        in_feats = x_tr.shape[1]
        print(f"    train graph: {n_train:,} nodes, {edge_tr.shape[1]:,} edges, in_feats={in_feats}")

        model = train_gtan(
            x_tr, edge_tr, y_tr,
            torch.tensor(train_idx_np), torch.tensor(val_idx_np),
            in_feats, hidden_dim, heads, n_layers, 2, dropout,
            n_epochs, lr, label_rate=label_rate, pos_weight=pos_weight,
            device=device, verbose=verbose,
        )
        dev = next(model.parameters()).device

        # ---- inference on FULL graph (train+test); test labels = unknown ----
        all_df = pd.concat([X_train.reset_index(drop=True),
                            X_test.reset_index(drop=True)], ignore_index=True)
        edge_full = build_gtan_edge_index(all_df, group_cols, time_col, edge_per_trans)
        x_full = torch.tensor(np.concatenate([Xtr, Xte], axis=0))
        y_full = torch.tensor(np.concatenate([y_np, np.full(n_test, 2)]), dtype=torch.long)
        y_input = _make_y_input(y_full.to(dev),
                                torch.arange(n_train, device=dev), 2, dev)
        print(f"    inference graph: {x_full.shape[0]:,} nodes, {edge_full.shape[1]:,} edges")
        train_emb, test_emb = _extract(model, x_full, edge_full, y_input, n_train, dev)

    # ---- append embeddings ----
    new_cols = [f"{prefix}{i}" for i in range(train_emb.shape[1])]
    X_train_aug = X_train.copy()
    X_test_aug = X_test.copy()
    for i, c in enumerate(new_cols):
        X_train_aug[c] = train_emb[:, i].astype(np.float32)
        X_test_aug[c] = test_emb[:, i].astype(np.float32)
    cols_aug = list(cols) + new_cols
    print(f"    {len(cols)} original + {len(new_cols)} GTAN features = {len(cols_aug)} total")
    return X_train_aug, X_test_aug, cols_aug


# ===========================================================================
# 6) CACHE-DRIVEN train / VAL workflow  (uid-disjoint v4_per_uid_timestep cache)
# ===========================================================================
# The Kaggle test set has no labels, so we cannot score on it. Instead we use the
# uid-disjoint labelled split materialised in the sequence cache (X_tr/X_va/Y_tr/Y_va).
#
# The cache stores per-uid SEQUENCE WINDOWS, shape (n_windows, max_len, n_features),
# RIGHT-padded (the real timesteps are positions [0:L]; the padding is at the end).
# GTAN is a node-level model, so we FLATTEN each window back to its real transactions:
# one graph node per real transaction. Features + labels come from the .npy cache;
# edges come from the dataframe's RAW identity columns (card1, card1_addr1,
# card1_addr1_P_emaildomain, DeviceInfo) + TransactionDT, looked up by TransactionID
# (== orig_*_flat). This recovers the full ~3.4%-fraud transaction graph.

def load_uid_disjoint_cache(cache_dir: str):
    """
    Load the v4_per_uid_timestep cache and FLATTEN right-padded windows to one row per
    real transaction.

    Returns dict with, for split in {'tr','va'}:
        X_<split>   : (N, n_features) float32  node features (already scaled)
        y_<split>   : (N,) int64               node labels {0,1}
        tid_<split> : (N,) int64               TransactionIDs (== orig_<split>_flat)
    """
    import os
    out = {}
    for sp in ("tr", "va"):
        X = np.load(os.path.join(cache_dir, f"X_{sp}.npy"))             # (n, T, F) float16
        Y = np.load(os.path.join(cache_dir, f"Y_{sp}.npy"))            # (n, T)
        L = np.load(os.path.join(cache_dir, f"L_{sp}.npy"))           # (n,)
        flat = np.load(os.path.join(cache_dir, f"orig_{sp}_flat.npy")).astype(np.int64)
        T = X.shape[1]
        mask = np.arange(T)[None, :] < L[:, None]                     # real = first L (right-pad)
        Xf = np.asarray(X)[mask].astype(np.float32)                   # (sum L, F)
        Yf = np.asarray(Y)[mask].astype(np.int64)
        if Xf.shape[0] != flat.shape[0]:
            raise ValueError(f"[{sp}] flatten count {Xf.shape[0]} != orig_flat {flat.shape[0]}; "
                             "cache padding convention may differ.")
        out[f"X_{sp}"], out[f"y_{sp}"], out[f"tid_{sp}"] = Xf, Yf, flat
    return out


def augment_with_gtan_embeddings_from_cache(
    df: pd.DataFrame,
    cache_dir: str,
    group_cols: list | None = None,
    time_col: str = DEFAULT_TIME_COL,
    edge_per_trans: int = EDGE_PER_TRANS,
    id_col: str | None = None,                 # TransactionID column; None => use df.index
    hidden_dim: int = 16,
    heads: int = 4,
    n_layers: int = 2,
    dropout: float = 0.2,
    n_epochs: int = 30,
    lr: float = 3e-4,
    label_rate: float = 0.5,
    pos_weight: float | None = None,
    device: str = "mps",
    batch_size: int | None = None,
    num_neighbors: list | None = None,
    oof_folds: int = 5,
    verbose: bool = True,
    prefix: str = "gtan_cache_",
):
    """
    Train GTAN on the uid-disjoint TRAIN split and validate on the labeled VAL split.

    batch_size : None -> full-graph training; int -> NeighborLoader minibatch training
                 (recommended for very large graphs / limited GPU memory).
    num_neighbors : per-hop fan-out (length n_layers); None + batch_size -> full neighbours.

    Nodes  : every real transaction in the cache (train windows + val windows, flattened).
    Train  : train-split transactions; ONLY their labels are revealed to the model.
    Val    : val-split transactions; their labels are hidden during message passing and
             used only to score AUC / AP  (this is the equivalent of "infer on X_va,
             check on Y_va" you asked for — train_gtan validates on these val nodes).

    `df` must contain `group_cols` + `time_col`, indexed by TransactionID (or pass
    `id_col`). It only needs to cover the train+val transactions (e.g. X_train_copy4).

    Returns dict:
        model, val_auc, val_ap,
        emb_train (DataFrame indexed by TransactionID), emb_val (same),
        val_prob, val_y, node_tids, n_train, n_val
    """
    group_cols = group_cols or DEFAULT_EDGE_GROUP_COLS

    # ---- 1) flatten cache -> per-transaction nodes (train first, then val) ----
    c = load_uid_disjoint_cache(cache_dir)
    n_train, n_val = c["X_tr"].shape[0], c["X_va"].shape[0]
    X = np.concatenate([c["X_tr"], c["X_va"]], axis=0)                # (N, F) already scaled
    y = np.concatenate([c["y_tr"], c["y_va"]]).astype(np.int64)
    tids = np.concatenate([c["tid_tr"], c["tid_va"]]).astype(np.int64)
    in_feats = X.shape[1]
    print("=== GTAN (uid-disjoint TRAIN/VAL from cache) ===")
    print(f"    train nodes={n_train:,}  val nodes={n_val:,}  in_feats={in_feats}")
    print(f"    fraud rate  train={c['y_tr'].mean():.4f}  val={c['y_va'].mean():.4f}")

    # ---- 2) edges from the dataframe (RAW identity cols + time), in node order ----
    d = df if id_col is None else df.set_index(id_col)
    probe = [t for t in tids[:2000] if t not in d.index]
    if probe:
        raise KeyError(f"{len(probe)}/2000 sampled TransactionIDs are not in the dataframe "
                       f"index. Pass the dataframe that contains these rows, or set id_col "
                       f"to the TransactionID column.")
    cols_needed = _existing(d, list(group_cols) + [time_col])
    node_df = d.loc[tids, cols_needed].reset_index(drop=True)         # positional idx == node id
    edge_index = build_gtan_edge_index(node_df, group_cols, time_col, edge_per_trans,
                                       num_nodes=len(tids))
    print(f"    graph: {len(tids):,} nodes, {edge_index.shape[1]:,} edges "
          f"(group_cols={_existing(d, group_cols)})")

    # ---- 3) train: reveal ONLY train labels; validate on val ----
    if batch_size is not None and num_neighbors is None:
        num_neighbors = [-1] * n_layers          # full-neighbour sampling, n_layers hops
        print(f"    minibatch: batch_size={batch_size}, num_neighbors={num_neighbors}")
    x = torch.tensor(X)
    y_t = torch.tensor(y, dtype=torch.long)
    train_idx = torch.arange(0, n_train)
    val_idx = torch.arange(n_train, n_train + n_val)
    model = train_gtan(
        x, edge_index, y_t, train_idx, val_idx,
        in_feats, hidden_dim, heads, n_layers, 2, dropout,
        n_epochs, lr, label_rate=label_rate, pos_weight=pos_weight,
        device=device, verbose=verbose,
        batch_size=batch_size, num_neighbors=num_neighbors,
    )
    dev = next(model.parameters()).device

    # ---- 4) final VAL scoring + LEAK-FREE node embeddings ----
    # TRAIN embeddings are out-of-fold (each node's own label masked); VAL embeddings use all
    # train labels revealed. Removes the train/val label-leakage that poisons downstream XGBoost.
    prob, emb = extract_oof_embeddings(model, x, edge_index, y_t, train_idx, val_idx, 2, dev,
                                       k_folds=oof_folds, seed=42,
                                       batch_size=batch_size, num_neighbors=num_neighbors,
                                       verbose=verbose)
    val_auc = roc_auc_score(y[n_train:], prob[n_train:])
    val_ap = average_precision_score(y[n_train:], prob[n_train:])
    print(f">>> VAL AUC = {val_auc:.4f} | VAL AP = {val_ap:.4f}")

    # embeddings keyed by TransactionID -> merge into your XGBoost frames with df.join(...)
    emb_cols = [f"{prefix}{i}" for i in range(emb.shape[1])]
    idx_name = id_col or (df.index.name if id_col is None else None) or "TransactionID"
    emb_train = pd.DataFrame(emb[:n_train], columns=emb_cols, index=tids[:n_train])
    emb_val = pd.DataFrame(emb[n_train:], columns=emb_cols, index=tids[n_train:])
    emb_train.index.name = emb_val.index.name = idx_name

    return {
        "model": model,
        "val_auc": float(val_auc),
        "val_ap": float(val_ap),
        "emb_train": emb_train,
        "emb_val": emb_val,
        "val_prob": prob[n_train:],
        "val_y": y[n_train:],
        "node_tids": tids,
        "n_train": n_train,
        "n_val": n_val,
    }


# ===========================================================================
# 7) DATAFRAME train/VAL workflow  (uid-disjoint split computed from the df,
#    features taken from the dataframe -- same return shape as _from_cache)
# ===========================================================================
# Like `augment_with_gtan_embeddings`, node FEATURES come from the dataframe
# (DEFAULT_GTAN_FEATURE_COLS, StandardScaler fit on TRAIN ONLY). But instead of
# producing test embeddings (test has no labels), it splits the LABELLED train set
# uid-disjointly -- identical seed/frac to building_sequence_UID_disjoint, so the
# val set matches the cache split -- trains, scores VAL (AUC/AP), and returns the
# SAME dict shape as `augment_with_gtan_embeddings_from_cache`.

def _split_uids_by_row_count(uid_series, frac_train=0.8, seed=42):
    """uid-disjoint split by cumulative row count.

    Byte-for-byte mirror of building_sequence_UID_disjoint.split_uids_by_row_count,
    so the same (seed, frac_train) reproduces the SAME train/val uid partition the
    .npy cache was built with -> the two GTAN runs are directly comparable.
    """
    rng = np.random.default_rng(seed)
    uid_counts = uid_series.value_counts(dropna=False)
    uids_arr = uid_counts.index.to_numpy()
    counts_arr = uid_counts.values.astype(np.int64)
    perm = rng.permutation(len(uids_arr))
    uids_shuf = uids_arr[perm]
    counts_shuf = counts_arr[perm]
    total = int(counts_shuf.sum())
    target = int(frac_train * total)
    cumsum = np.cumsum(counts_shuf)
    cut = int(np.searchsorted(cumsum, target)) + 1
    return set(uids_shuf[:cut].tolist()), set(uids_shuf[cut:].tolist())


def augment_with_gtan_embeddings_trainval(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    feature_cols: list | None = None,        # node features (default: DEFAULT_GTAN_FEATURE_COLS)
    group_cols: list | None = None,          # edge grouping (default: DEFAULT_EDGE_GROUP_COLS)
    time_col: str = DEFAULT_TIME_COL,        # used ONLY to order edges
    uid_col: str = "uid",
    seed: int = 42,
    frac_train: float = 0.8,
    edge_per_trans: int = EDGE_PER_TRANS,
    hidden_dim: int = 16,
    heads: int = 4,
    n_layers: int = 2,
    dropout: float = 0.2,
    n_epochs: int = 30,
    lr: float = 3e-4,
    label_rate: float = 0.5,
    pos_weight: float | None = None,
    device: str = "cuda",
    scale_features: bool = True,
    batch_size: int | None = None,
    num_neighbors: list | None = None,
    oof_folds: int = 5,
    prefix: str = "gtan_df_",
    verbose: bool = True,
):
    """
    uid-disjoint TRAIN/VAL GTAN on dataframe features. Same return dict as
    `augment_with_gtan_embeddings_from_cache`.

    Nodes  : every labelled train transaction (split into train-uids + val-uids).
    Train  : train-uid transactions; ONLY their labels are revealed to the model.
    Val    : val-uid transactions; labels hidden during message passing, used only
             to score AUC / AP.

    `X_train` must be indexed by TransactionID and contain `feature_cols`,
    `group_cols`, `time_col`, and `uid_col`. `y_train` is a {0,1} Series aligned by
    TransactionID (it is reindexed to X_train internally).

    batch_size : None -> full-graph; int -> NeighborLoader minibatch (recommended at
                 hidden_dim=64 to avoid OOM on the ~train-sized graph).

    Returns dict: model, val_auc, val_ap, emb_train, emb_val (DataFrames indexed by
    TransactionID), val_prob, val_y, node_tids, n_train, n_val.
    """
    feature_cols = feature_cols or DEFAULT_GTAN_FEATURE_COLS
    group_cols = group_cols or DEFAULT_EDGE_GROUP_COLS
    if isinstance(y_train, pd.DataFrame):
        y_train = y_train.iloc[:, 0]
    y = y_train.reindex(X_train.index)

    # ---- 1) uid-disjoint split (train-uids first, then val-uids) ----
    train_uids, val_uids = _split_uids_by_row_count(X_train[uid_col], frac_train, seed)
    df_tr = X_train.loc[X_train[uid_col].isin(train_uids)]
    df_va = X_train.loc[X_train[uid_col].isin(val_uids)]
    n_train, n_val = len(df_tr), len(df_va)
    print("=== GTAN (uid-disjoint TRAIN/VAL from dataframe) ===")
    print(f"    train nodes={n_train:,}  val nodes={n_val:,}  in_feats="
          f"{len(_existing(X_train, feature_cols))}")
    print(f"    fraud rate  train={y.loc[df_tr.index].mean():.4f}  "
          f"val={y.loc[df_va.index].mean():.4f}")

    # ---- 2) features (scaler fit on TRAIN only), labels + tids in node order ----
    Xtr, Xva, used = _to_feature_matrix(df_tr, df_va, feature_cols, scale_features)
    x = torch.tensor(np.concatenate([Xtr, Xva], axis=0))
    in_feats = x.shape[1]
    y_np = np.concatenate([y.loc[df_tr.index].to_numpy(np.int64),
                           y.loc[df_va.index].to_numpy(np.int64)])
    y_t = torch.tensor(y_np, dtype=torch.long)
    tids = np.concatenate([df_tr.index.to_numpy(), df_va.index.to_numpy()])

    # ---- 3) edges over train+val nodes (in node order) ----
    cols_needed = _existing(X_train, list(group_cols) + [time_col])
    node_df = pd.concat([df_tr, df_va])[cols_needed].reset_index(drop=True)
    edge_index = build_gtan_edge_index(node_df, group_cols, time_col, edge_per_trans,
                                       num_nodes=len(tids))
    print(f"    graph: {len(tids):,} nodes, {edge_index.shape[1]:,} edges "
          f"(group_cols={_existing(X_train, group_cols)})")

    # ---- 4) train: reveal ONLY train labels; validate on val ----
    if batch_size is not None and num_neighbors is None:
        num_neighbors = [-1] * n_layers
        print(f"    minibatch: batch_size={batch_size}, num_neighbors={num_neighbors}")
    train_idx = torch.arange(0, n_train)
    val_idx = torch.arange(n_train, n_train + n_val)
    model = train_gtan(
        x, edge_index, y_t, train_idx, val_idx,
        in_feats, hidden_dim, heads, n_layers, 2, dropout,
        n_epochs, lr, label_rate=label_rate, pos_weight=pos_weight,
        device=device, verbose=verbose,
        batch_size=batch_size, num_neighbors=num_neighbors,
    )
    dev = next(model.parameters()).device

    # ---- 5) final VAL scoring + LEAK-FREE node embeddings ----
    # TRAIN embeddings are out-of-fold (each node's own label masked); VAL embeddings use all
    # train labels revealed. Removes the train/val label-leakage that poisons downstream XGBoost.
    prob, emb = extract_oof_embeddings(model, x, edge_index, y_t, train_idx, val_idx, 2, dev,
                                       k_folds=oof_folds, seed=seed,
                                       batch_size=batch_size, num_neighbors=num_neighbors,
                                       verbose=verbose)
    val_auc = roc_auc_score(y_np[n_train:], prob[n_train:])
    val_ap = average_precision_score(y_np[n_train:], prob[n_train:])
    print(f">>> VAL AUC = {val_auc:.4f} | VAL AP = {val_ap:.4f}")

    emb_cols = [f"{prefix}{i}" for i in range(emb.shape[1])]
    idx_name = X_train.index.name or "TransactionID"
    emb_train = pd.DataFrame(emb[:n_train], columns=emb_cols, index=tids[:n_train])
    emb_val = pd.DataFrame(emb[n_train:], columns=emb_cols, index=tids[n_train:])
    emb_train.index.name = emb_val.index.name = idx_name

    return {
        "model": model,
        "val_auc": float(val_auc),
        "val_ap": float(val_ap),
        "emb_train": emb_train,
        "emb_val": emb_val,
        "val_prob": prob[n_train:],
        "val_y": y_np[n_train:],
        "node_tids": tids,
        "n_train": n_train,
        "n_val": n_val,
    }


# ===========================================================================
# Smoke test (needs torch_geometric installed)
# ===========================================================================
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    N = 2000
    df = pd.DataFrame({
        "TransactionDT": np.sort(rng.integers(0, 10 ** 6, N)),
        "card1": rng.integers(0, 50, N),
        "card1_addr1": rng.integers(0, 80, N),
        "card1_addr1_P_emaildomain": rng.integers(0, 120, N),
        "DeviceInfo": rng.integers(0, 30, N),
    })
    for c in DEFAULT_GTAN_FEATURE_COLS:
        if c not in df.columns:
            df[c] = rng.standard_normal(N).astype(np.float32)
    y = pd.Series((rng.random(N) < 0.035).astype(int))
    Xtr, Xte = df.iloc[:1500].reset_index(drop=True), df.iloc[1500:].reset_index(drop=True)
    ytr = y.iloc[:1500].reset_index(drop=True)
    cols = list(df.columns)
    Xtr_a, Xte_a, cols_a = augment_with_gtan_embeddings(
        Xtr, Xte, ytr, cols, mode="inductive",
        n_epochs=3, hidden_dim=8, heads=2, device="cpu", verbose=True,
    )
    print("train aug shape:", Xtr_a.shape, "| test aug shape:", Xte_a.shape)
