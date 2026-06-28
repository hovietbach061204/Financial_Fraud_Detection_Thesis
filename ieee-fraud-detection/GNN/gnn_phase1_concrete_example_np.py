"""
Phase 1 GNN training — concrete illustration
============================================
A minimal, self-contained example showing exactly what happens inside the
NVIDIA financial-fraud-training container:

    Phase 1: train GNN ENCODER + CLASSIFIER HEAD together on fraud labels
    Phase 2: surgically remove the classifier head, keep only the encoder
    Phase 3: run encoder on data -> extract per-node embeddings
    Phase 4: train XGBoost on [raw_features || gnn_embeddings]

This is a tri-partite heterograph (User -> Transaction -> Merchant) with
6 transactions (3 fraud, 3 legit), 4 users, 3 merchants — small enough that
you can trace every tensor by hand if you want.

Dependencies:  torch, torch_geometric, scikit-learn, xgboost, numpy
    pip install torch torch_geometric scikit-learn xgboost numpy
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
# Step 0 — Build a tiny tri-partite heterograph
# ============================================================
# Graph topology:
#
#   U0 ──> T0 ──> M0           Labels:
#   U1 ──> T1 ──> M1           T0=legit  T3=fraud
#   U2 ──> T2 ──> M0           T1=fraud  T4=legit
#   U3 ──> T3 ──> M2           T2=legit  T5=fraud
#   U0 ──> T4 ──> M1
#   U1 ──> T5 ──> M2
#
# Notice U0 and U1 each appear in two transactions — one legit, one fraud.
# A GNN that only looks at the transaction row in isolation can't use the
# fact that "U1 also did T5 (fraud)" to help predict T1. A GNN can.

data = HeteroData()

# Transaction features: [amount, time_normalized, used_chip, is_foreign]
data["transaction"].x = torch.tensor(
    [
        [12.5,    0.30, 1.0, 0.20],   # T0 - legit (small amount, daytime, chip)
        [9999.0,  0.95, 0.0, 0.90],   # T1 - FRAUD (huge, late night, no chip, foreign)
        [45.0,    0.40, 1.0, 0.10],   # T2 - legit
        [8500.0,  0.92, 0.0, 0.85],   # T3 - FRAUD
        [22.0,    0.20, 1.0, 0.15],   # T4 - legit
        [7200.0,  0.97, 0.0, 0.95],   # T5 - FRAUD
    ],
    dtype=torch.float,
)
data["transaction"].y = torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.long)

# User features: [avg_monthly_spend, account_age_years]
data["user"].x = torch.tensor(
    [[0.5, 1.2], [0.8, 2.1], [0.3, 0.9], [0.7, 1.5]], dtype=torch.float
)

# Merchant features: [risk_score, category_encoded]
data["merchant"].x = torch.tensor(
    [[1.5, 0.4], [2.1, 0.8], [0.9, 0.3]], dtype=torch.float
)

# Edges in COO format: [src_idx_row, dst_idx_row]
data["user", "makes", "transaction"].edge_index = torch.tensor(
    [[0, 1, 2, 3, 0, 1],   # users
     [0, 1, 2, 3, 4, 5]],  # transactions
    dtype=torch.long,
)
data["transaction", "at", "merchant"].edge_index = torch.tensor(
    [[0, 1, 2, 3, 4, 5],   # transactions
     [0, 1, 0, 2, 1, 2]],  # merchants
    dtype=torch.long,
)
# Reverse edges so messages can flow both ways during convolution
data["transaction", "rev_makes", "user"].edge_index = (
    data["user", "makes", "transaction"].edge_index.flip(0)
)
data["merchant", "rev_at", "transaction"].edge_index = (
    data["transaction", "at", "merchant"].edge_index.flip(0)
)


# ============================================================
# Step 1 — Define the GNN: encoder + classifier head
# ============================================================
HIDDEN = 8       # hidden width inside the encoder
EMBED_DIM = 8    # final embedding size that XGBoost will consume


class FraudGNN(nn.Module):
    """
    ┌────────── ENCODER (kept after training) ──────────┐  ┌── HEAD (discarded) ──┐
    raw_features -> proj -> SAGEConv -> SAGEConv -> emb  ->  Linear -> 2 logits
    """

    def __init__(self):
        super().__init__()
        # Project each node type into a common HIDDEN dim
        self.proj_txn = nn.Linear(4, HIDDEN)
        self.proj_user = nn.Linear(2, HIDDEN)
        self.proj_merchant = nn.Linear(2, HIDDEN)

        # Hop 1: aggregate one-hop neighbors along each relation
        self.conv1 = HeteroConv(
            {
                ("user", "makes", "transaction"): SAGEConv((HIDDEN, HIDDEN), HIDDEN),
                ("transaction", "rev_makes", "user"): SAGEConv((HIDDEN, HIDDEN), HIDDEN),
                ("transaction", "at", "merchant"): SAGEConv((HIDDEN, HIDDEN), HIDDEN),
                ("merchant", "rev_at", "transaction"): SAGEConv((HIDDEN, HIDDEN), HIDDEN),
            },
            aggr="mean",
        )

        # Hop 2: aggregate two-hop neighbors (e.g. transactions that share a user)
        self.conv2 = HeteroConv(
            {
                ("user", "makes", "transaction"): SAGEConv((HIDDEN, HIDDEN), EMBED_DIM),
                ("transaction", "rev_makes", "user"): SAGEConv((HIDDEN, HIDDEN), EMBED_DIM),
                ("transaction", "at", "merchant"): SAGEConv((HIDDEN, HIDDEN), EMBED_DIM),
                ("merchant", "rev_at", "transaction"): SAGEConv((HIDDEN, HIDDEN), EMBED_DIM),
            },
            aggr="mean",
        )

        # CLASSIFIER HEAD — only used during training; deleted after Phase 1
        self.classifier_head = nn.Linear(EMBED_DIM, 2)

    def encode(self, data):
        """Run encoder ONLY. Returns dict: node_type -> embedding tensor."""
        x_dict = {
            "transaction": self.proj_txn(data["transaction"].x),
            "user": self.proj_user(data["user"].x),
            "merchant": self.proj_merchant(data["merchant"].x),
        }
        edge_index_dict = {
            ("user", "makes", "transaction"): data["user", "makes", "transaction"].edge_index,
            ("transaction", "rev_makes", "user"): data["transaction", "rev_makes", "user"].edge_index,
            ("transaction", "at", "merchant"): data["transaction", "at", "merchant"].edge_index,
            ("merchant", "rev_at", "transaction"): data["merchant", "rev_at", "transaction"].edge_index,
        }
        # Hop 1
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {k: F.relu(v) for k, v in x_dict.items()}
        # Hop 2
        x_dict = self.conv2(x_dict, edge_index_dict)
        return x_dict

    def forward(self, data):
        """Full forward pass used DURING TRAINING ONLY."""
        emb = self.encode(data)
        logits = self.classifier_head(emb["transaction"])  # head used here
        return logits, emb


# ============================================================
# Step 2 — PHASE 1: supervised end-to-end training
# ============================================================
print("=" * 64)
print(" PHASE 1: train encoder + classifier head together (supervised)")
print("=" * 64)

model = FraudGNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
y_true = data["transaction"].y

for epoch in range(31):
    model.train()
    optimizer.zero_grad()
    logits, _ = model(data)                       # encoder -> head -> logits
    loss = F.cross_entropy(logits, y_true)        # gradient flows through HEAD AND ENCODER
    loss.backward()
    optimizer.step()

    if epoch % 5 == 0:
        model.eval()
        with torch.no_grad():
            preds = logits.argmax(dim=1).numpy()
            f1 = f1_score(y_true.numpy(), preds, zero_division=0)
        print(f"  epoch {epoch:2d} | loss={loss.item():.4f} | F1={f1:.3f}")

print("\n  ↑ During Phase 1, the loss is computed against the FRAUD LABELS.")
print("    Gradients flow backward through the head AND the entire encoder,")
print("    so encoder weights get shaped to produce fraud-discriminative embeddings.")


# ============================================================
# Step 3 — PHASE 2: 'discard the classifier head'
# ============================================================
print("\n" + "=" * 64)
print(" PHASE 2: surgically remove the classifier head, keep encoder")
print("=" * 64)

full_state = model.state_dict()
encoder_state = {k: v for k, v in full_state.items() if not k.startswith("classifier_head")}
head_state = {k: v for k, v in full_state.items() if k.startswith("classifier_head")}

print(f"  Full model had  {len(full_state)} parameter tensors")
print(f"  Encoder keeps   {len(encoder_state)} tensors")
print(f"  Head discarded  {len(head_state)} tensors:  {list(head_state.keys())}")
print("\n  The discarded head is ~18 numbers (8 weights x 2 + 2 biases).")
print("  Those numbers' job was to translate embeddings -> 2 logits.")
print("  XGBoost will take over that 'embeddings -> prediction' job in Phase 4.")


# ============================================================
# Step 4 — PHASE 3: extract embeddings (NO head, encoder only)
# ============================================================
print("\n" + "=" * 64)
print(" PHASE 3: run encoder.encode() -> 8-dim embedding per transaction")
print("=" * 64)

model.eval()
with torch.no_grad():
    emb_dict = model.encode(data)                 # head is bypassed entirely
txn_emb = emb_dict["transaction"].numpy()         # shape (6, 8)

print(f"\n  Embedding tensor shape: {txn_emb.shape}  (6 transactions x 8 dims)\n")
labels = ["legit", "FRAUD", "legit", "FRAUD", "legit", "FRAUD"]
for i in range(6):
    print(f"  T{i} ({labels[i]:5s}) emb = {np.round(txn_emb[i], 2)}")

print("\n  Cosine similarity between transaction embeddings:")
sim = cosine_similarity(txn_emb)
header = "        " + "   ".join(f"T{i}" for i in range(6))
print(header)
for i in range(6):
    row = "  ".join(f"{v:+.2f}" for v in sim[i])
    print(f"  T{i}: {row}    ({labels[i]})")

print("\n  Notice: the 3 fraud rows (T1, T3, T5) cluster with each other,")
print("  and the 3 legit rows (T0, T2, T4) cluster with each other.")
print("  That clustering is what the supervised Phase-1 training created.")
print("  An untrained GNN would give roughly random similarities.")


# ============================================================
# Step 5 — PHASE 4: feed embeddings to XGBoost (deployed pipeline)
# ============================================================
print("\n" + "=" * 64)
print(" PHASE 4: train XGBoost on [raw_features || gnn_embeddings]")
print("=" * 64)

raw_features = data["transaction"].x.numpy()                # shape (6, 4)
X_combined = np.hstack([raw_features, txn_emb])             # shape (6, 12)

print(f"\n  XGBoost input matrix shape: {X_combined.shape}")
print( "    = 4 raw tabular features  +  8 GNN embedding dims")

clf = xgb.XGBClassifier(
    n_estimators=20,
    max_depth=3,
    learning_rate=0.3,
    min_child_weight=0,
    eval_metric="logloss",
)
clf.fit(X_combined, y_true.numpy())

probs = clf.predict_proba(X_combined)[:, 1]
print("\n  Final fraud probabilities (this is what Triton serves at inference):")
for i, (p, y) in enumerate(zip(probs, y_true)):
    flag = "FRAUD" if y.item() == 1 else "legit"
    print(f"    T{i} ({flag:5s}): P(fraud) = {p:.3f}")

print("\n" + "=" * 64)
print(" SUMMARY OF WHAT JUST HAPPENED")
print("=" * 64)
print("""
  TRAINING (one-off, done inside the financial-fraud-training container):
      1. GNN encoder + head trained together with cross-entropy on fraud labels
      2. Head is thrown away; encoder weights are saved (state_dict_gnn_model.pth)
      3. Encoder is run on training data to produce embeddings
      4. XGBoost trained on [raw_features || embeddings]  (embedding_based_xgboost.json)

  INFERENCE (every request to Triton, repeated forever):
      1. New transaction arrives, along with its user/merchant subgraph
      2. GNN encoder runs    -> 8-dim embedding   (NO head, no logits)
      3. XGBoost predicts    -> fraud probability  (the only number returned)

  The GNN never emits a fraud probability after deployment.
  Its role is permanently 'feature extractor'.
  But those features were shaped by supervised fraud-label training in Phase 1.
""")

# ============================================================
# Step 6 — PHASE 5: Comparing between Head and XGBoost
# ============================================================

print("\n" + "=" * 64)
print(" PHASE 5: Comparing between Head and XGBoost")
print("=" * 64)

# What the discarded head would have predicted
with torch.no_grad():
    head_logits, _ = model(data)
    head_probs = torch.softmax(head_logits, dim=1)[:, 1].numpy()

# What XGBoost predicts
xgb_probs = clf.predict_proba(X_combined)[:, 1]

# Compare F1 at threshold 0.5
from sklearn.metrics import f1_score
print("Head F1:    ", f1_score(y_true, (head_probs > 0.5).astype(int)))
print("XGBoost F1: ", f1_score(y_true, (xgb_probs > 0.5).astype(int)))