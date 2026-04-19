#!/usr/bin/env python3
"""
Adversarial validation on IEEE-CIS Fraud Detection:
- Class 0: train_transaction rows
- Class 1: test_transaction rows

Outputs:
1) ROC curve for adversarial classifier
2) Average LightGBM feature importance (horizontal bar chart)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import StratifiedKFold


def transform_d_columns(df: pd.DataFrame) -> pd.DataFrame:
    if "TransactionDT" not in df.columns:
        return df

    d_cols = [c for c in df.columns if c.startswith("D") and c[1:].isdigit()]
    transaction_day = df["TransactionDT"] / (24 * 60 * 60)
    for col in d_cols:
        df[col] = transaction_day - df[col]
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/ieee-fraud-detection"),
        help="Directory containing train_transaction.csv and test_transaction.csv",
    )
    parser.add_argument(
        "--nrows",
        type=int,
        default=None,
        help="Optional row limit per file for faster experiments",
    )
    parser.add_argument(
        "--top-k-features",
        type=int,
        default=50,
        help="Number of features to show in importance chart",
    )
    args = parser.parse_args()

    train_path = args.data_dir / "train_transaction.csv"
    test_path = args.data_dir / "test_transaction.csv"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Missing CSVs. Expected:\n- {train_path}\n- {test_path}"
        )

    # Use the "first 53 columns" idea from the Kaggle UID discussion.
    all_cols = pd.read_csv(train_path, nrows=0).columns.tolist()
    candidate_cols = [c for c in all_cols if c != "isFraud"]
    feature_cols = candidate_cols[:53]

    train_df = pd.read_csv(train_path, usecols=feature_cols, nrows=args.nrows)
    test_df = pd.read_csv(test_path, usecols=feature_cols, nrows=args.nrows)

    train_df["is_test"] = 0
    test_df["is_test"] = 1
    df = pd.concat([train_df, test_df], axis=0, ignore_index=True)

    y = df["is_test"].astype("int8")
    X = df.drop(columns=["is_test"])

    X = transform_d_columns(X)

    # Encode categoricals consistently over full combined set.
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in cat_cols:
        X[col], _ = pd.factorize(X[col].astype(str), sort=True)

    # Fill missing values after all transforms.
    X = X.fillna(-999)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_pred = np.zeros(len(X), dtype="float32")
    importances = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        model = lgb.LGBMClassifier(
            objective="binary",
            boosting_type="gbdt",
            n_estimators=600,
            learning_rate=0.03,
            num_leaves=64,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42 + fold,
            n_jobs=-1,
        )

        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(stopping_rounds=80, verbose=False)],
        )

        oof_pred[va_idx] = model.predict_proba(X_va)[:, 1]
        importances.append(model.feature_importances_)

    fpr, tpr, _ = roc_curve(y, oof_pred)
    roc_auc = auc(fpr, tpr)
    print(f"Adversarial ROC-AUC: {roc_auc:.6f}")

    # Plot ROC
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, lw=2, label=f"Adversarial model (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random baseline")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Adversarial Validation ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()

    # Plot average feature importance (similar to your screenshot).
    imp = np.mean(np.vstack(importances), axis=0)
    imp_df = (
        pd.DataFrame({"feature": X.columns, "importance": imp})
        .sort_values("importance", ascending=False)
        .head(args.top_k_features)
    )
    imp_df = imp_df.iloc[::-1]

    plt.figure(figsize=(12, 10))
    sns.barplot(data=imp_df, x="importance", y="feature", palette="Spectral")
    plt.title("LightGBM Features (avg over folds)")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
