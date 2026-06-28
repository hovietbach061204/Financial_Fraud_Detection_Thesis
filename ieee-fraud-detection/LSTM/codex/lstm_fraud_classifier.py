"""
LSTM sequence classifier for IEEE-CIS fraud detection.

Idea
----
For each UID (your card1_addr1_<day-D1> identifier), order the transactions in
time. Treat each transaction as the LAST step of a length-K window of that
user's recent activity, and predict isFraud for that last step. The model
therefore sees a small biography of the user before deciding.

Why this is the right "time series" framing for the IEEE problem
----------------------------------------------------------------
Prophet / NeuralProphet are univariate forecasting models — they answer
"what will y(t+1) be?", not "is THIS transaction fraud?". The IEEE task is
per-row binary classification with ~430 features, so a sequence classifier
is the natural time-aware analogue of your XGBoost. The LSTM gets explicit
access to the prior K transactions of the same user, which is precisely the
information your encode_AG aggregations were summarising in a lossy way.

Inputs expected
---------------
You should run the upstream cells of Time_Series_Fraud_with_Magic.ipynb up to
the point where X_train_copy4 / X_test_copy4 exist (i.e. uid is built and
add_time_features has been applied), then save them once with:

    X_train_copy4.to_parquet('X_train_copy4.parquet')
    X_test_copy4.to_parquet('X_test_copy4.parquet')
    y_train.to_frame().to_parquet('y_train.parquet')

This script then reads them and trains the LSTM end-to-end.

Usage
-----
    python lstm_fraud_classifier.py \
        --data-dir /Volumes/SandiskSSD/Developer/AI_Document/Financial_Fraud_Detection_Thesis/data/ieee-fraud-detection \
        --window 5 --epochs 8 --batch-size 1024
"""

from __future__ import annotations

import argparse
import gc
import os
import time
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


# --------------------------------------------------------------------------- #
# 1. Data loading
# --------------------------------------------------------------------------- #
def load_checkpoints(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Read the preprocessed train/test/y dataframes saved from the notebook."""
    X_train = pd.read_parquet(os.path.join(data_dir, "X_train_copy4.parquet"))
    X_test = pd.read_parquet(os.path.join(data_dir, "X_test_copy4.parquet"))
    y_train_df = pd.read_parquet(os.path.join(data_dir, "y_train.parquet"))
    y_train = y_train_df.iloc[:, 0].astype("int8")
    return X_train, X_test, y_train


# --------------------------------------------------------------------------- #
# 2. Feature selection
# --------------------------------------------------------------------------- #
def select_feature_columns(df: pd.DataFrame) -> List[str]:
    """Drop columns that should not be used as model inputs."""
    cols = list(df.columns)
    drop = set()

    # Same exclusions you use in the XGBoost cells, plus housekeeping cols
    drop.update(["TransactionDT", "D6", "D7", "D8", "D9", "D12", "D13", "D14"])
    drop.update(["uid", "day", "DT", "isFraud", "oof"])
    drop.update(["C3", "M5", "id_08", "id_33"])
    drop.update(["card4", "id_07", "id_14", "id_21", "id_30", "id_32", "id_34"])
    drop.update([f"id_{x}" for x in range(22, 28)])
    # Time bucket columns are kept inside the sequence — the model can use them.

    return [c for c in cols if c not in drop]


# --------------------------------------------------------------------------- #
# 3. Build (uid, DT)-ordered sliding windows
# --------------------------------------------------------------------------- #
def build_windows(
    df: pd.DataFrame,
    feature_cols: List[str],
    window: int,
    uid_col: str = "uid",
    time_col: str = "DT",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns
    -------
    X : np.ndarray, shape (n_rows, window, n_features), float32
        Each row is the last `window` transactions of the same UID, padded at
        the front with zeros if the user has fewer than `window` prior rows.
        The LAST step of each window is the row itself.
    row_index : np.ndarray, shape (n_rows,), int64
        The position in `df` that each window corresponds to. Use this to
        align predictions back to TransactionID.
    """
    df_sorted = df.sort_values([uid_col, time_col], kind="mergesort")
    feat_block = df_sorted[feature_cols].to_numpy(dtype=np.float32)
    uids = df_sorted[uid_col].to_numpy()
    orig_pos = df_sorted.index.to_numpy()  # TransactionID values

    n = len(df_sorted)
    f = len(feature_cols)
    X = np.zeros((n, window, f), dtype=np.float32)

    # iterate per-uid using boundaries
    boundaries = np.flatnonzero(np.concatenate([[True], uids[1:] != uids[:-1]]))
    boundaries = np.append(boundaries, n)

    for b_start, b_end in zip(boundaries[:-1], boundaries[1:]):
        block = feat_block[b_start:b_end]            # (g, f)
        g = block.shape[0]
        for t in range(g):
            start = max(0, t - window + 1)
            seq = block[start:t + 1]                 # (<=window, f)
            X[b_start + t, -seq.shape[0]:] = seq

    return X, orig_pos


# --------------------------------------------------------------------------- #
# 4. Torch dataset / model
# --------------------------------------------------------------------------- #
class WindowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray | None = None):
        self.X = torch.from_numpy(X)
        self.y = None if y is None else torch.from_numpy(y.astype(np.float32))

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, i):
        if self.y is None:
            return self.X[i]
        return self.X[i], self.y[i]


class LSTMClassifier(nn.Module):
    """A small bidirectional LSTM with a sigmoid head."""

    def __init__(self, n_features: int, hidden: int = 128, layers: int = 2,
                 dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
            bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        # x: (B, T, F)
        out, _ = self.lstm(x)             # (B, T, 2H)
        last = out[:, -1, :]              # take the last timestep
        return self.head(last).squeeze(-1)


# --------------------------------------------------------------------------- #
# 5. Train one fold
# --------------------------------------------------------------------------- #
def train_fold(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_va: np.ndarray, y_va: np.ndarray,
    epochs: int, batch_size: int, lr: float,
    device: torch.device,
) -> Tuple[np.ndarray, float]:
    n_features = X_tr.shape[2]

    model = LSTMClassifier(n_features=n_features).to(device)
    pos_weight = torch.tensor([(y_tr == 0).sum() / max((y_tr == 1).sum(), 1)],
                              dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    train_loader = DataLoader(WindowDataset(X_tr, y_tr),
                              batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=device.type == "cuda")
    val_loader = DataLoader(WindowDataset(X_va, y_va),
                            batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=device.type == "cuda")

    best_auc = -1.0
    best_preds = None

    for epoch in range(epochs):
        model.train()
        t0 = time.time()
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optim.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

        # validate
        model.eval()
        preds = []
        with torch.no_grad():
            for xb, _ in val_loader:
                xb = xb.to(device, non_blocking=True)
                preds.append(torch.sigmoid(model(xb)).cpu().numpy())
        preds = np.concatenate(preds)
        auc = roc_auc_score(y_va, preds)
        print(f"  epoch {epoch + 1:>2}/{epochs}  AUC={auc:.4f}  ({time.time()-t0:.1f}s)")

        if auc > best_auc:
            best_auc = auc
            best_preds = preds

    return best_preds, best_auc


# --------------------------------------------------------------------------- #
# 6. Full pipeline
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True,
                        help="Folder with X_train_copy4.parquet etc.")
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n-folds", type=int, default=6)
    parser.add_argument("--output", default="oof_lstm.csv")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else
                          ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Using device: {device}")

    print("Loading checkpoints...")
    X_train, X_test, y_train = load_checkpoints(args.data_dir)
    feature_cols = select_feature_columns(X_train)
    print(f"  rows={len(X_train):,}, features={len(feature_cols)}")

    # Fill NaNs and standardize using TRAIN-ONLY statistics to avoid leakage
    print("Imputing + scaling...")
    X_train_f = X_train[feature_cols].fillna(-1).astype(np.float32)
    X_test_f = X_test[feature_cols].fillna(-1).astype(np.float32)
    scaler = StandardScaler()
    X_train_f[feature_cols] = scaler.fit_transform(X_train_f[feature_cols])
    X_test_f[feature_cols] = scaler.transform(X_test_f[feature_cols])

    # Reattach uid + DT for window construction
    X_train_f["uid"] = X_train["uid"].values
    X_train_f["DT"] = X_train["DT"].values
    X_test_f["uid"] = X_test["uid"].values
    X_test_f["DT"] = X_test["DT"].values

    print(f"Building length-{args.window} windows...")
    X_train_seq, train_idx_order = build_windows(X_train_f, feature_cols,
                                                 window=args.window)
    X_test_seq, test_idx_order = build_windows(X_test_f, feature_cols,
                                               window=args.window)

    # y aligned to the (uid, DT) sort order
    y_aligned = y_train.loc[train_idx_order].to_numpy(dtype=np.int8)
    dt_m_aligned = X_train.loc[train_idx_order, "DT_M"].to_numpy()

    del X_train_f, X_test_f
    gc.collect()

    print("Starting GroupKFold by DT_M...")
    oof = np.zeros(len(X_train_seq), dtype=np.float32)
    skf = GroupKFold(n_splits=args.n_folds)
    fold_aucs = []
    for fold, (idxT, idxV) in enumerate(skf.split(X_train_seq, y_aligned,
                                                  groups=dt_m_aligned)):
        held_month = dt_m_aligned[idxV][0]
        print(f"\n=== Fold {fold}, holding out month {held_month} "
              f"(train={len(idxT):,}, val={len(idxV):,}) ===")
        preds, auc = train_fold(
            X_train_seq[idxT], y_aligned[idxT],
            X_train_seq[idxV], y_aligned[idxV],
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
            device=device,
        )
        oof[idxV] = preds
        fold_aucs.append(auc)

    overall = roc_auc_score(y_aligned, oof)
    print(f"\n=== LSTM OOF AUC = {overall:.4f}  "
          f"(folds: {[round(a, 4) for a in fold_aucs]}) ===")

    # Map back to TransactionID order and save
    out = pd.DataFrame({
        "TransactionID": train_idx_order,
        "oof_lstm": oof,
    }).set_index("TransactionID").reindex(X_train.index).reset_index()
    out.to_csv(args.output, index=False)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
