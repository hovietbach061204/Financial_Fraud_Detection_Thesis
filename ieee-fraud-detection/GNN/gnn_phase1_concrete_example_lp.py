"""
Phase 1 GNN training — Link Prediction (edge classification) variant
====================================================================
The companion to gnn_phase1_concrete_example_np.py, but using the LP topology
from preprocess_TabFormer_lp.py:

    NODES     : User and Merchant only  (no Transaction node!)
    EDGES     : each transaction is ONE edge from a user to a merchant,
                carrying both its features (edge_attr) and its fraud label
    ENCODER   : SAGEConv stack that learns User and Merchant embeddings
    HEAD      : eats  [user_emb || merchant_emb || edge_attr]  ->  logits
    INFERENCE : XGBoost on  [user_emb || merchant_emb || edge_attr]

Same 6 transactions, same labels, same raw numbers as the NP example —
so you can compare the two architectures directly.

Dependencies:  torch, torch_geometric, scikit-learn, xgboost, numpy
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv
from sklearn.metrics import f1_score
from sklearn.metrics.pairwise import cosine_similarity
import xgboost as xgb

torch.manual_seed(42)
np.random.seed(42)


# ============================================================
# Step 0 — Bipartite User <-> Merchant graph, transactions ARE the edges
# ============================================================
# Graph topology (read each row as one transaction = one edge):
#
#   edge T0: U0 ──> M0     (legit)
#   edge T1: U1 ──> M1     (FRAUD)
#   edge T2: U2 ──> M0     (legit)
#   edge T3: U3 ──> M2     (FRAUD)
#   edge T4: U0 ──> M1     (legit)
#   edge T5: U1 ──> M2     (FRAUD)
#
# Notice there is NO 'transaction' node type in this graph. The transaction
# features (amount, time, used_chip, is_foreign) live on the EDGE itself.

data = HeteroData()

# User and merchant NODE features — same as before
data["user"].x = torch.tensor(
    [[0.5, 1.2], [0.8, 2.1], [0.3, 0.9], [0.7, 1.5]], dtype=torch.float
)
data["merchant"].x = torch.tensor(
    [[1.5, 0.4], [2.1, 0.8], [0.9, 0.3]], dtype=torch.float
)

# EDGES = transactions
edge_index = torch.tensor(
    [[0, 1, 2, 3, 0, 1],   # src users
     [0, 1, 0, 2, 1, 2]],  # dst merchants
    dtype=torch.long,
)
data["user", "txn", "merchant"].edge_index = edge_index

# EDGE ATTRIBUTES = the transaction features (formerly transaction node features)
data["user", "txn", "merchant"].edge_attr = torch.tensor(
    [
        [12.5,    0.30, 1.0, 0.20],   # T0 - legit
        [9999.0,  0.95, 0.0, 0.90],   # T1 - FRAUD
        [45.0,    0.40, 1.0, 0.10],   # T2 - legit
        [8500.0,  0.92, 0.0, 0.85],   # T3 - FRAUD
        [22.0,    0.20, 1.0, 0.15],   # T4 - legit
        [7200.0,  0.97, 0.0, 0.95],   # T5 - FRAUD
    ],
    dtype=torch.float,
)
# EDGE LABEL = fraud / legit
data["user", "txn", "merchant"].y = torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.long)

# Reverse edge type so messages can flow merchant -> user as well as user -> merchant
data["merchant", "rev_txn", "user"].edge_index = edge_index.flip(0)


# ============================================================
# Step 1 — Define the GNN: bipartite encoder + edge-classification head
# ============================================================
HIDDEN = 8
EMBED_DIM = 8
EDGE_FEAT_DIM = 4   # amount, time, chip, foreign


class FraudGNN_LP(nn.Module):
    """
    Encoder learns embeddings for USER and MERCHANT nodes (transactions are NOT nodes).
    Classifier head reads one user emb + one merchant emb + the edge's own attributes,
    and predicts whether that specific edge (transaction) is fraud.

       user_x ─proj─┐
                    ├─SAGEConv (hop1)─SAGEConv (hop2)─► u_emb, m_emb
       merch_x ─proj─┘                                    │
                                                          ▼
                            edge_attr ─────────► concat ─► Linear ─► 2 logits
                                                  (24-d)            (the HEAD)
    """

    def __init__(self):
        super().__init__()
        # Project node features into common HIDDEN width
        self.proj_user = nn.Linear(2, HIDDEN)
        self.proj_merchant = nn.Linear(2, HIDDEN)

        # Hop 1 — bipartite message passing both directions
        self.conv1 = HeteroConv(
            {
                ("user", "txn", "merchant"): SAGEConv((HIDDEN, HIDDEN), HIDDEN),
                ("merchant", "rev_txn", "user"): SAGEConv((HIDDEN, HIDDEN), HIDDEN),
            },
            aggr="mean",
        )
        # Hop 2 — same again, output is EMBED_DIM
        self.conv2 = HeteroConv(
            {
                ("user", "txn", "merchant"): SAGEConv((HIDDEN, HIDDEN), EMBED_DIM),
                ("merchant", "rev_txn", "user"): SAGEConv((HIDDEN, HIDDEN), EMBED_DIM),
            },
            aggr="mean",
        )

        # EDGE CLASSIFIER HEAD — eats user_emb (8) + merchant_emb (8) + edge_attr (4) = 20
        # THIS HEAD WILL BE DISCARDED AFTER TRAINING
        self.classifier_head = nn.Sequential(
            nn.Linear(EMBED_DIM + EMBED_DIM + EDGE_FEAT_DIM, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
        )

    def encode(self, data):
        """Run encoder ONLY. Returns dict: 'user' -> emb, 'merchant' -> emb."""
        x_dict = {
            "user": self.proj_user(data["user"].x),
            "merchant": self.proj_merchant(data["merchant"].x),
        }
        edge_index_dict = {
            ("user", "txn", "merchant"): data["user", "txn", "merchant"].edge_index,
            ("merchant", "rev_txn", "user"): data["merchant", "rev_txn", "user"].edge_index,
        }
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {k: F.relu(v) for k, v in x_dict.items()}
        x_dict = self.conv2(x_dict, edge_index_dict)
        return x_dict  # {'user': (4,8), 'merchant': (3,8)}

    def forward(self, data):
        """Full forward used during training: encoder -> per-edge logits."""
        emb = self.encode(data)
        ei = data["user", "txn", "merchant"].edge_index    # (2, 6)
        ea = data["user", "txn", "merchant"].edge_attr     # (6, 4)

        src, dst = ei[0], ei[1]
        u_emb = emb["user"][src]            # (6, 8) — embedding of the user on each edge
        m_emb = emb["merchant"][dst]        # (6, 8) — embedding of the merchant on each edge

        # For every edge, build [user_emb || merchant_emb || edge_attr]
        edge_features = torch.cat([u_emb, m_emb, ea], dim=-1)  # (6, 20)
        logits = self.classifier_head(edge_features)            # (6, 2)
        return logits, emb


# ============================================================
# Step 2 — PHASE 1: supervised end-to-end training (edge-level loss)
# ============================================================
print("=" * 64)
print(" PHASE 1: train encoder + edge-classifier head together")
print("=" * 64)
print(" Note: loss is over EDGES (transactions), not nodes.")
print(" Gradient flows: edge logits -> head -> edge_attr & both embeddings,")
print(" then through embeddings -> SAGEConv layers -> projection layers.")
print()

model = FraudGNN_LP()
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
y_true = data["user", "txn", "merchant"].y

for epoch in range(31):
    model.train()
    optimizer.zero_grad()
    logits, _ = model(data)
    loss = F.cross_entropy(logits, y_true)
    loss.backward()
    optimizer.step()

    if epoch % 5 == 0:
        model.eval()
        with torch.no_grad():
            preds = logits.argmax(dim=1).numpy()
            f1 = f1_score(y_true.numpy(), preds, zero_division=0)
        print(f"  epoch {epoch:2d} | loss={loss.item():.4f} | F1={f1:.3f}")


# ============================================================
# Step 3 — PHASE 2: discard the (bigger) classifier head
# ============================================================
print("\n" + "=" * 64)
print(" PHASE 2: discard the edge-classifier head, keep encoder")
print("=" * 64)
full_state = model.state_dict()
head_keys = [k for k in full_state if k.startswith("classifier_head")]
encoder_state = {k: v for k, v in full_state.items() if k not in head_keys}
print(f"  Full model       : {len(full_state)} tensors")
print(f"  Encoder kept     : {len(encoder_state)} tensors")
print(f"  Head discarded   : {len(head_keys)} tensors  ({head_keys})")
print("\n  The discarded head was bigger here — it had to merge u_emb, m_emb, and")
print("  edge_attr through an MLP. Even more reason to replace it with XGBoost,")
print("  which handles such heterogeneous-feature concatenations naturally.")


# ============================================================
# Step 4 — PHASE 3: extract user & merchant embeddings
# ============================================================
print("\n" + "=" * 64)
print(" PHASE 3: extract USER and MERCHANT embeddings (no head)")
print("=" * 64)

model.eval()
with torch.no_grad():
    emb = model.encode(data)
u_emb = emb["user"].numpy()         # (4, 8)
m_emb = emb["merchant"].numpy()     # (3, 8)

print(f"\n  user embeddings shape:     {u_emb.shape}   (4 users x 8 dims)")
print(f"  merchant embeddings shape: {m_emb.shape}   (3 merchants x 8 dims)\n")

for i in range(4):
    print(f"  U{i} emb = {np.round(u_emb[i], 2)}")
print()
for j in range(3):
    print(f"  M{j} emb = {np.round(m_emb[j], 2)}")

# Compare with NP: there, every transaction had its own embedding.
# Here, the model learns embeddings per USER and per MERCHANT, and the
# transaction-level identity comes from the (user, merchant, edge_attr) triple.

print("\n  User-to-user cosine similarity:")
print(np.round(cosine_similarity(u_emb), 2))
print("    U0 and U1 should be similar (both touch fraud edges)")
print("    U2 should be closer to U0 (both touch only legit edges)")


# ============================================================
# Step 5 — PHASE 4: build per-transaction features and train XGBoost
# ============================================================
print("\n" + "=" * 64)
print(" PHASE 4: assemble per-edge XGBoost features and train")
print("=" * 64)

ei = data["user", "txn", "merchant"].edge_index.numpy()  # (2, 6)
ea = data["user", "txn", "merchant"].edge_attr.numpy()   # (6, 4)
src, dst = ei[0], ei[1]

# For each transaction (edge), gather: [user_emb || merchant_emb || edge_attr]
X_combined = np.hstack([u_emb[src], m_emb[dst], ea])    # (6, 8+8+4) = (6, 20)

print(f"\n  XGBoost input shape per transaction: {X_combined.shape}")
print( "      8 user embedding dims")
print( "    + 8 merchant embedding dims")
print( "    + 4 raw transaction (edge) features")
print( "    = 20 dims total")

clf = xgb.XGBClassifier(
    n_estimators=20, max_depth=3, learning_rate=0.3,
    eval_metric="logloss"
)
clf.fit(X_combined, y_true.numpy())

probs = clf.predict_proba(X_combined)[:, 1]
print("\n  Final fraud probabilities (LP variant):")
for i, (p, y) in enumerate(zip(probs, y_true)):
    flag = "FRAUD" if y.item() == 1 else "legit"
    print(f"    T{i}: U{src[i]} -> M{dst[i]}  ({flag:5s})  P(fraud) = {p:.3f}")


# ============================================================
# Step 6 — Side-by-side architectural summary
# ============================================================
print("\n" + "=" * 64)
print(" NP vs LP — what actually changed")
print("=" * 64)
print("""
                              NP variant                 LP variant
  ----------------------------------------------------------------------------
  Graph topology              tri-partite                bipartite
                              User -> Txn -> Merchant     User <-> Merchant
  Node types                  3 (user, txn, merchant)    2 (user, merchant)
  Where transactions live     as NODES                   as EDGES (edge_attr)
  Where fraud label lives     transaction node label     edge label
  Encoder outputs             txn embeddings             user + merchant embeddings
  Classifier head input       txn_emb (8)                u_emb || m_emb || ea (20)
  Number of head params       ~18  (Linear(8, 2))        ~370 (small MLP)
  XGBoost input per txn       raw(4) + txn_emb(8) = 12   u_emb(8) + m_emb(8) + ea(4) = 20

  Conceptual question         "Is this transaction        "Is this user-merchant
                              fraudulent?"                 interaction fraudulent?"
""")
