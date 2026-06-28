"""
Reusable time-aware modeling experiments for the IEEE-CIS fraud data.

This module keeps the Kaggle target as transaction-level fraud probability
while making the temporal assumptions explicit:

* TransactionID remains the stable row key.
* DT is created from TransactionDT and kept as a feature-engineering column.
* Strict validation uses expanding time folds and fits encoders/aggregations
  on the fold's past rows only.
* Kaggle-style validation keeps the original transductive train+test feature
  engineering so scores remain comparable with leaderboard-style notebooks.

Example commands
----------------
Inspect date ranges:
    python time_series_fraud_experiments.py inspect --data-dir .

Run strict XGBoost UID benchmark:
    python time_series_fraud_experiments.py xgb --data-dir . --protocol strict

Run Kaggle-style transductive XGBoost UID benchmark:
    python time_series_fraud_experiments.py xgb --data-dir . --protocol kaggle

Build Prophet or NeuralProphet aggregate features:
    python time_series_fraud_experiments.py prophet --data-dir . --library prophet

Train a transaction-sequence LSTM, if PyTorch is installed:
    python time_series_fraud_experiments.py lstm --data-dir . --window 5
"""

from __future__ import annotations

import argparse
import csv
import gc
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


START_DATE = pd.Timestamp("2017-11-30")
MISSING_VALUE = -999.0

V_KEEP_NUMBERS = [
    1, 3, 4, 6, 8, 11,
    13, 14, 17, 20, 23, 26, 27, 30,
    36, 37, 40, 41, 44, 47, 48,
    54, 56, 59, 62, 65, 67, 68, 70,
    76, 78, 80, 82, 86, 88, 89, 91,
    107, 108, 111, 115, 117, 120, 121, 123,
    124, 127, 129, 130, 136,
    138, 139, 142, 147, 156, 162,
    165, 160, 166,
    178, 176, 173, 182,
    187, 203, 205, 207, 215,
    169, 171, 175, 180, 185, 188, 198, 210, 209,
    218, 223, 224, 226, 228, 229, 235,
    240, 258, 257, 253, 252, 260, 261,
    264, 266, 267, 274, 277,
    220, 221, 234, 238, 250, 271,
    294, 284, 285, 286, 291, 297,
    303, 305, 307, 309, 310, 320,
    281, 283, 289, 296, 301, 314,
]
V_KEEP = [f"V{x}" for x in V_KEEP_NUMBERS]

FIRST_TRANSACTION_COLUMNS = [
    "TransactionID", "TransactionDT", "TransactionAmt", "ProductCD",
    "card1", "card2", "card3", "card4", "card5", "card6",
    "addr1", "addr2", "dist1", "dist2", "P_emaildomain", "R_emaildomain",
]
C_COLUMNS = [f"C{i}" for i in range(1, 15)]
D_COLUMNS = [f"D{i}" for i in range(1, 16)]
M_COLUMNS = [f"M{i}" for i in range(1, 10)]

TIME_SERIES_COLUMNS = [
    "DT_M", "DT_W", "DT_D", "DT_hour", "DT_day_week",
    "DT_day_month", "DT_week_month", "is_december", "is_holiday",
]

FE_COLUMNS = [
    "addr1", "card1", "card2", "card3", "P_emaildomain",
    "card1_addr1", "card1_addr1_P_emaildomain", "uid",
]

UID_AGG_VALUE_COLUMNS = ["TransactionAmt", "D4", "D9", "D10", "D15"]
UID_NUNIQUE_VALUE_COLUMNS = ["P_emaildomain", "dist1", "id_02", "cents"]
V_NUNIQUE_COLUMNS = ["V314", "V127", "V136", "V309", "V307", "V320"]

DROP_FROM_MODEL = {
    "TransactionDT", "DT", "uid_raw", "day", "oof", "isFraud",
    "D6", "D7", "D8", "D9", "D12", "D13", "D14",
    "C3", "M5", "id_08", "id_33",
    "card4", "id_07", "id_14", "id_21", "id_30", "id_32", "id_34",
}
DROP_FROM_MODEL.update({f"id_{x}" for x in range(22, 28)})


@dataclass(frozen=True)
class LoadedData:
    train: pd.DataFrame
    test: pd.DataFrame
    y: pd.Series


@dataclass(frozen=True)
class FoldResult:
    fold: int
    train_periods: list[int]
    valid_period: int
    auc: float
    best_iteration: int | None


class SimpleStandardScaler:
    """Small scaler to avoid a hard sklearn dependency in the LSTM path."""

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, x: pd.DataFrame | np.ndarray) -> "SimpleStandardScaler":
        arr = np.asarray(x, dtype="float32")
        self.mean_ = np.nanmean(arr, axis=0)
        scale = np.nanstd(arr, axis=0)
        scale[scale == 0] = 1.0
        self.scale_ = scale
        return self

    def transform(self, x: pd.DataFrame | np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("SimpleStandardScaler must be fitted before transform.")
        arr = np.asarray(x, dtype="float32")
        return (arr - self.mean_) / self.scale_

    def fit_transform(self, x: pd.DataFrame | np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)


def binary_roc_auc_score(y_true: Iterable[int], y_score: Iterable[float]) -> float:
    """ROC-AUC for binary labels using average ranks for ties."""
    y = np.asarray(y_true, dtype=np.int8)
    scores = np.asarray(y_score, dtype=np.float64)
    mask = ~np.isnan(scores)
    y = y[mask]
    scores = scores[mask]
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    if pos == 0 or neg == 0:
        raise ValueError("ROC-AUC is undefined when a fold has one class.")

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    i = 0
    while i < len(scores):
        j = i + 1
        while j < len(scores) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        ranks[order[i:j]] = avg_rank
        i = j

    pos_rank_sum = ranks[y == 1].sum()
    return float((pos_rank_sum - pos * (pos + 1) / 2.0) / (pos * neg))


def mean_absolute(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    y = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(np.abs(y - pred)))


def root_mean_squared(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    y = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.sqrt(np.mean((y - pred) ** 2)))


def _read_header(path: Path) -> list[str]:
    with path.open(newline="") as handle:
        return next(csv.reader(handle))


def _available_usecols(path: Path, wanted: Iterable[str]) -> list[str]:
    existing = set(_read_header(path))
    return [c for c in wanted if c in existing]


def _normalize_identity_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {c: c.replace("-", "_") for c in df.columns if c.startswith("id-")}
    return df.rename(columns=rename)


def _downcast_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    for col in df.select_dtypes(include=["int64"]).columns:
        if col != "TransactionID":
            df[col] = pd.to_numeric(df[col], downcast="integer")
    return df


def load_raw_data(
    data_dir: str | Path,
    include_v: bool = True,
    nrows: int | None = None,
) -> LoadedData:
    """Load and merge transaction/identity data using TransactionID as index."""
    data_path = Path(data_dir)
    train_transaction_path = data_path / "train_transaction.csv"
    test_transaction_path = data_path / "test_transaction.csv"
    train_identity_path = data_path / "train_identity.csv"
    test_identity_path = data_path / "test_identity.csv"

    train_wanted = (
        FIRST_TRANSACTION_COLUMNS + ["isFraud"] + C_COLUMNS + D_COLUMNS + M_COLUMNS
    )
    test_wanted = FIRST_TRANSACTION_COLUMNS + C_COLUMNS + D_COLUMNS + M_COLUMNS
    if include_v:
        train_wanted += V_KEEP
        test_wanted += V_KEEP

    train_usecols = _available_usecols(train_transaction_path, train_wanted)
    test_usecols = _available_usecols(test_transaction_path, test_wanted)

    train = pd.read_csv(train_transaction_path, usecols=train_usecols, nrows=nrows)
    test = pd.read_csv(test_transaction_path, usecols=test_usecols, nrows=nrows)

    y = train.pop("isFraud").astype("int8")
    train = train.set_index("TransactionID", drop=True)
    test = test.set_index("TransactionID", drop=True)
    y.index = train.index

    train_id = pd.read_csv(train_identity_path, nrows=nrows)
    test_id = pd.read_csv(test_identity_path, nrows=nrows)
    train_id = _normalize_identity_columns(train_id).set_index("TransactionID")
    test_id = _normalize_identity_columns(test_id).set_index("TransactionID")

    train = train.merge(train_id, how="left", left_index=True, right_index=True)
    test = test.merge(test_id, how="left", left_index=True, right_index=True)

    train = _downcast_numeric(train)
    test = _downcast_numeric(test)
    return LoadedData(train=train, test=test, y=y)


def add_dt(df: pd.DataFrame) -> pd.DataFrame:
    """Create calendar/time columns from TransactionDT without changing index."""
    out = df.copy()
    dt = START_DATE + pd.to_timedelta(out["TransactionDT"], unit="s")
    iso_week = dt.dt.isocalendar().week.astype("int16")
    out["DT"] = dt
    out["DT_M"] = (((dt.dt.year - START_DATE.year) * 12) + dt.dt.month).astype("int16")
    out["DT_W"] = (((dt.dt.year - START_DATE.year) * 52) + iso_week).astype("int16")
    out["DT_D"] = np.floor(out["TransactionDT"] / 86400).astype("int16")
    out["DT_hour"] = dt.dt.hour.astype("int8")
    out["DT_day_week"] = dt.dt.dayofweek.astype("int8")
    out["DT_day_month"] = dt.dt.day.astype("int8")
    out["DT_week_month"] = (((dt.dt.day - 1) // 7) + 1).astype("int8")
    out["is_december"] = (dt.dt.month == 12).astype("int8")
    out["is_holiday"] = _is_us_federal_holiday(dt).astype("int8")
    return out


def _is_us_federal_holiday(dt: pd.Series) -> pd.Series:
    try:
        from pandas.tseries.holiday import USFederalHolidayCalendar
    except Exception:
        return pd.Series(False, index=dt.index)

    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(dt.min().normalize(), dt.max().normalize())
    return dt.dt.normalize().isin(holidays)


def add_rowwise_features(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add deterministic row-wise features shared by all model tracks."""
    train = add_dt(train)
    test = add_dt(test)

    for df in (train, test):
        df["cents"] = (df["TransactionAmt"] - np.floor(df["TransactionAmt"])).astype("float32")
        df["dollars"] = np.floor(df["TransactionAmt"]).astype("float32")
        df["day"] = np.floor(df["TransactionDT"] / 86400).astype("float32")
        df["card1_addr1"] = df["card1"].astype(str) + "_" + df["addr1"].astype(str)
        df["card1_addr1_P_emaildomain"] = (
            df["card1_addr1"].astype(str) + "_" + df["P_emaildomain"].astype(str)
        )
        df["uid_raw"] = (
            df["card1_addr1"].astype(str)
            + "_"
            + np.floor(df["day"] - df["D1"]).astype(str)
        )
        if {"D1", "D15"}.issubset(df.columns):
            df["outsider15"] = (np.abs(df["D1"] - df["D15"]) > 3).astype("int8")

    both_uid = pd.concat([train["uid_raw"], test["uid_raw"]], axis=0)
    uid_codes, _ = pd.factorize(both_uid, sort=True)
    train["uid"] = uid_codes[: len(train)].astype("int32")
    test["uid"] = uid_codes[len(train):].astype("int32")

    for time_col in TIME_SERIES_COLUMNS:
        train[f"uid_{time_col}"] = train["uid"].astype(str) + "_" + train[time_col].astype(str)
        test[f"uid_{time_col}"] = test["uid"].astype(str) + "_" + test[time_col].astype(str)

    return train, test


def fit_label_encoders(df: pd.DataFrame) -> dict[str, dict[Any, int]]:
    encoders: dict[str, dict[Any, int]] = {}
    object_cols = df.select_dtypes(include=["object", "category", "string"]).columns
    for col in object_cols:
        vals = pd.Series(df[col].dropna().unique())
        encoders[col] = {v: i for i, v in enumerate(vals)}
    return encoders


def apply_label_encoders(
    df: pd.DataFrame,
    encoders: dict[str, dict[Any, int]],
) -> pd.DataFrame:
    out = df.copy()
    for col, mapping in encoders.items():
        if col in out.columns:
            out[col] = out[col].map(mapping).fillna(-1).astype("int32")
    return out


def fit_frequency_maps(df: pd.DataFrame, columns: Iterable[str]) -> dict[str, dict[Any, int]]:
    maps: dict[str, dict[Any, int]] = {}
    for col in columns:
        if col in df.columns:
            maps[col] = df[col].value_counts(dropna=True).to_dict()
    return maps


def apply_frequency_maps(
    df: pd.DataFrame,
    maps: dict[str, dict[Any, int]],
) -> pd.DataFrame:
    out = df.copy()
    for col, mapping in maps.items():
        out[f"{col}_FE"] = out[col].map(mapping).fillna(0).astype("float32")
    return out


@dataclass(frozen=True)
class AggSpec:
    group_cols: list[str]
    mean_std_cols: list[str]
    mean_cols: list[str]
    nunique_cols: list[str]
    std_cols: list[str]


def build_agg_spec(feature_set: str, include_v: bool, columns: Iterable[str]) -> AggSpec:
    present = set(columns)
    group_cols: list[str] = []

    if feature_set in {"uid", "all"}:
        group_cols.append("uid")
    if feature_set in {"time", "all"}:
        group_cols.extend([c for c in TIME_SERIES_COLUMNS if c in present])
    if feature_set in {"uid-time", "all"}:
        group_cols.extend([f"uid_{c}" for c in TIME_SERIES_COLUMNS if f"uid_{c}" in present])

    mean_std_cols = [c for c in UID_AGG_VALUE_COLUMNS if c in present]
    mean_cols = [c for c in C_COLUMNS if c != "C3" and c in present]
    mean_cols += [c for c in M_COLUMNS if c in present]
    nunique_cols = [c for c in UID_NUNIQUE_VALUE_COLUMNS if c in present]
    if include_v:
        nunique_cols += [c for c in V_NUNIQUE_COLUMNS if c in present]
    std_cols = ["C14"] if "C14" in present else []

    return AggSpec(
        group_cols=group_cols,
        mean_std_cols=mean_std_cols,
        mean_cols=mean_cols,
        nunique_cols=nunique_cols,
        std_cols=std_cols,
    )


def fit_agg_maps(df: pd.DataFrame, spec: AggSpec) -> dict[str, tuple[str, dict[Any, float]]]:
    maps: dict[str, tuple[str, dict[Any, float]]] = {}
    for group_col in spec.group_cols:
        if group_col not in df.columns:
            continue
        for value_col in spec.mean_std_cols:
            _fit_one_agg(df, maps, group_col, value_col, "mean")
            _fit_one_agg(df, maps, group_col, value_col, "std")
        for value_col in spec.mean_cols:
            _fit_one_agg(df, maps, group_col, value_col, "mean")
        for value_col in spec.std_cols:
            _fit_one_agg(df, maps, group_col, value_col, "std")
        for value_col in spec.nunique_cols:
            _fit_one_agg(df, maps, group_col, value_col, "nunique")
    return maps


def _fit_one_agg(
    df: pd.DataFrame,
    maps: dict[str, tuple[str, dict[Any, float]]],
    group_col: str,
    value_col: str,
    agg: str,
) -> None:
    if value_col not in df.columns:
        return
    values = df[[group_col, value_col]].copy()
    if value_col.startswith("D"):
        values.loc[values[value_col] == MISSING_VALUE, value_col] = np.nan
    name = _agg_name(value_col, group_col, agg)
    maps[name] = (
        group_col,
        values.groupby(group_col, dropna=False)[value_col].agg(agg).to_dict(),
    )


def _agg_name(value_col: str, group_col: str, agg: str) -> str:
    suffix = "ct" if agg == "nunique" else agg
    return f"{value_col}_{group_col}_{suffix}"


def apply_agg_maps(
    df: pd.DataFrame,
    maps: dict[str, tuple[str, dict[Any, float]]],
) -> pd.DataFrame:
    out = df.copy()
    for name, (group_col, mapping) in maps.items():
        if group_col not in out.columns:
            continue
        mapped = out[group_col].map(mapping).astype("float32")
        out[name] = mapped
        if mapped.isna().any():
            out[f"{name}_isna"] = mapped.isna().astype("int8")
    return out


def build_protocol_features(
    fit_df: pd.DataFrame,
    transform_df: pd.DataFrame,
    feature_set: str,
    include_v: bool,
    transductive_scope: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Fit encoders/aggregates on fit_df unless transductive_scope is provided.

    Strict CV passes only the fold's training rows as fit_df. Kaggle-style
    validation passes train+test as transductive_scope to mimic the notebook.
    """
    encoder_fit = transductive_scope if transductive_scope is not None else fit_df
    encoders = fit_label_encoders(encoder_fit)

    fit_enc = apply_label_encoders(fit_df, encoders)
    transform_enc = apply_label_encoders(transform_df, encoders)
    if transductive_scope is not None:
        agg_fit = apply_label_encoders(transductive_scope, encoders)
    else:
        agg_fit = fit_enc

    fe_maps = fit_frequency_maps(agg_fit, FE_COLUMNS)
    fit_feat = apply_frequency_maps(fit_enc, fe_maps)
    transform_feat = apply_frequency_maps(transform_enc, fe_maps)

    spec = build_agg_spec(feature_set, include_v=include_v, columns=agg_fit.columns)
    agg_maps = fit_agg_maps(agg_fit, spec)
    fit_feat = apply_agg_maps(fit_feat, agg_maps)
    transform_feat = apply_agg_maps(transform_feat, agg_maps)

    feature_cols = select_model_columns(fit_feat, include_v=include_v)
    feature_cols = [c for c in feature_cols if c in transform_feat.columns]
    fit_feat, transform_feat = align_and_fill(fit_feat, transform_feat, feature_cols)
    return fit_feat, transform_feat, feature_cols


def select_model_columns(df: pd.DataFrame, include_v: bool) -> list[str]:
    cols = []
    for col in df.columns:
        if col in DROP_FROM_MODEL:
            continue
        if col.startswith("uid_DT_") or col in {f"uid_{c}" for c in TIME_SERIES_COLUMNS}:
            continue
        if col == "uid":
            continue
        if not include_v and col.startswith("V"):
            continue
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            continue
        cols.append(col)
    return cols


def align_and_fill(
    left: pd.DataFrame,
    right: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    missing_left = [c for c in feature_cols if c not in left.columns]
    missing_right = [c for c in feature_cols if c not in right.columns]
    if missing_left or missing_right:
        raise ValueError(
            f"Feature alignment failed. Missing left={missing_left}, right={missing_right}"
        )

    left = left.copy()
    right = right.copy()
    for df in (left, right):
        for col in feature_cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(MISSING_VALUE)
            else:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(MISSING_VALUE)
        df[feature_cols] = df[feature_cols].astype("float32")
    return left, right


def expanding_time_splits(
    df: pd.DataFrame,
    period_col: str = "DT_M",
    min_train_periods: int = 3,
    max_folds: int | None = None,
) -> list[tuple[np.ndarray, np.ndarray, list[int], int]]:
    periods = sorted(pd.Series(df[period_col].dropna().unique()).astype(int).tolist())
    if len(periods) <= min_train_periods:
        raise ValueError(
            f"Need more than {min_train_periods} {period_col} values for strict CV; "
            f"found {periods}."
        )

    folds = []
    validation_periods = periods[min_train_periods:]
    if max_folds is not None:
        validation_periods = validation_periods[-max_folds:]

    period_values = df[period_col].astype(int)
    for valid_period in validation_periods:
        train_periods = [p for p in periods if p < valid_period]
        idx_train = np.flatnonzero(period_values.isin(train_periods).to_numpy())
        idx_valid = np.flatnonzero((period_values == valid_period).to_numpy())
        folds.append((idx_train, idx_valid, train_periods, valid_period))
    return folds


def grouped_month_splits(
    df: pd.DataFrame,
    n_splits: int = 6,
    period_col: str = "DT_M",
) -> list[tuple[np.ndarray, np.ndarray, list[int], int]]:
    groups = df[period_col].astype(int).to_numpy()
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise ValueError(f"Need at least 2 {period_col} values for grouped month CV.")
    actual_splits = min(n_splits, len(unique_groups))

    group_sizes = {group: int((groups == group).sum()) for group in unique_groups}
    fold_groups: list[list[int]] = [[] for _ in range(actual_splits)]
    fold_sizes = np.zeros(actual_splits, dtype=np.int64)
    for group in sorted(unique_groups, key=lambda g: group_sizes[g], reverse=True):
        fold_id = int(np.argmin(fold_sizes))
        fold_groups[fold_id].append(int(group))
        fold_sizes[fold_id] += group_sizes[group]

    splits = []
    for fold_group in fold_groups:
        valid_mask = np.isin(groups, fold_group)
        idx_valid = np.flatnonzero(valid_mask)
        idx_train = np.flatnonzero(~valid_mask)
        valid_period = int(fold_group[0])
        train_periods = sorted(np.unique(groups[idx_train]).astype(int).tolist())
        splits.append((idx_train, idx_valid, train_periods, valid_period))
    return splits


def make_xgb_classifier(args: argparse.Namespace, y_train: pd.Series):
    try:
        import xgboost as xgb
    except Exception as exc:
        raise SystemExit(
            "xgboost could not be imported. Install or repair the local stack "
            "with compatible numpy/pandas/scipy/scikit-learn/xgboost versions."
        ) from exc

    scale_pos_weight = 1.0
    if args.scale_pos_weight:
        neg = int((y_train == 0).sum())
        pos = int((y_train == 1).sum())
        scale_pos_weight = neg / max(pos, 1)

    return xgb.XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        missing=MISSING_VALUE,
        eval_metric="auc",
        tree_method=args.tree_method,
        device=args.device,
        early_stopping_rounds=args.early_stopping_rounds,
        random_state=args.seed,
        scale_pos_weight=scale_pos_weight,
    )


def run_xgb(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    del rng

    loaded = load_raw_data(args.data_dir, include_v=args.include_v, nrows=args.nrows)
    train_base, test_base = add_rowwise_features(loaded.train, loaded.test)
    train_base = train_base.sort_values("DT", kind="mergesort")
    loaded_y = loaded.y.loc[train_base.index]
    test_base = test_base.sort_values("DT", kind="mergesort")

    print_range_summary(train_base, test_base, loaded_y)

    if args.protocol == "strict":
        folds = expanding_time_splits(
            train_base,
            min_train_periods=args.min_train_periods,
            max_folds=args.max_folds,
        )
    else:
        folds = grouped_month_splits(train_base, n_splits=args.n_splits)

    oof = pd.Series(np.nan, index=train_base.index, dtype="float32")
    test_preds = np.zeros(len(test_base), dtype="float32") if args.predict_test else None
    fold_results: list[FoldResult] = []

    if args.protocol == "kaggle":
        print("Building Kaggle-style transductive features on train+test...")
        scope = pd.concat([train_base, test_base], axis=0)
        train_feat, test_feat, feature_cols = build_protocol_features(
            train_base,
            test_base,
            feature_set=args.feature_set,
            include_v=args.include_v,
            transductive_scope=scope,
        )
        print(f"Feature count: {len(feature_cols):,}")
    else:
        train_feat = test_feat = None
        feature_cols = []

    for fold, (idx_train, idx_valid, train_periods, valid_period) in enumerate(folds):
        t0 = time.time()
        print(
            f"\nFold {fold}: train periods={train_periods}, "
            f"valid period={valid_period}, "
            f"train rows={len(idx_train):,}, valid rows={len(idx_valid):,}"
        )

        if args.protocol == "strict":
            fold_train_base = train_base.iloc[idx_train]
            fold_valid_base = train_base.iloc[idx_valid]
            fold_train_feat, fold_valid_feat, feature_cols = build_protocol_features(
                fold_train_base,
                fold_valid_base,
                feature_set=args.feature_set,
                include_v=args.include_v,
            )
            if args.predict_test:
                _, fold_test_feat, _ = build_protocol_features(
                    fold_train_base,
                    test_base,
                    feature_set=args.feature_set,
                    include_v=args.include_v,
                )
        else:
            assert train_feat is not None and test_feat is not None
            fold_train_feat = train_feat.iloc[idx_train]
            fold_valid_feat = train_feat.iloc[idx_valid]
            fold_test_feat = test_feat

        y_fold_train = loaded_y.iloc[idx_train]
        y_fold_valid = loaded_y.iloc[idx_valid]
        clf = make_xgb_classifier(args, y_fold_train)
        clf.fit(
            fold_train_feat[feature_cols],
            y_fold_train,
            eval_set=[(fold_valid_feat[feature_cols], y_fold_valid)],
            verbose=args.verbose_eval,
        )
        valid_pred = clf.predict_proba(fold_valid_feat[feature_cols])[:, 1]
        auc = binary_roc_auc_score(y_fold_valid, valid_pred)
        oof.iloc[idx_valid] = valid_pred.astype("float32")

        if args.predict_test and test_preds is not None:
            test_preds += clf.predict_proba(fold_test_feat[feature_cols])[:, 1] / len(folds)

        best_iteration = getattr(clf, "best_iteration", None)
        fold_results.append(
            FoldResult(
                fold=fold,
                train_periods=train_periods,
                valid_period=valid_period,
                auc=float(auc),
                best_iteration=None if best_iteration is None else int(best_iteration),
            )
        )
        print(
            f"Fold {fold} AUC={auc:.6f}, best_iteration={best_iteration}, "
            f"elapsed={time.time() - t0:.1f}s"
        )
        del clf
        gc.collect()

    scored = oof.notna()
    overall_auc = binary_roc_auc_score(loaded_y.loc[scored.index[scored]], oof.loc[scored])
    print("\nFold summary:")
    for result in fold_results:
        print(
            f"  fold={result.fold}, valid_period={result.valid_period}, "
            f"auc={result.auc:.6f}, best_iteration={result.best_iteration}"
        )
    print(f"{args.protocol.upper()} XGB OOF AUC = {overall_auc:.12f}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = f"xgb_{args.protocol}_{args.feature_set}_{'v' if args.include_v else 'no_v'}"
    oof_path = output_dir / f"oof_{tag}.csv"
    pd.DataFrame({"TransactionID": oof.index, "oof": oof.values}).to_csv(oof_path, index=False)
    print(f"Wrote {oof_path}")

    if args.predict_test and test_preds is not None:
        pred_path = output_dir / f"test_{tag}.csv"
        pd.DataFrame(
            {"TransactionID": test_base.index, "isFraud": test_preds}
        ).to_csv(pred_path, index=False)
        print(f"Wrote {pred_path}")


def print_range_summary(train: pd.DataFrame, test: pd.DataFrame, y: pd.Series) -> None:
    train_days = train["TransactionDT"] / 86400
    test_days = test["TransactionDT"] / 86400
    print(
        f"Train rows={len(train):,}, day range={train_days.min():.3f}-{train_days.max():.3f}, "
        f"fraud rate={float(y.mean()):.5f}"
    )
    print(
        f"Test rows={len(test):,}, day range={test_days.min():.3f}-{test_days.max():.3f}"
    )
    print(
        f"Train DT monotonic after sort: {train['DT'].is_monotonic_increasing}; "
        f"test DT monotonic after sort: {test['DT'].is_monotonic_increasing}"
    )


def run_inspect(args: argparse.Namespace) -> None:
    loaded = load_raw_data(args.data_dir, include_v=args.include_v, nrows=args.nrows)
    train_base, test_base = add_rowwise_features(loaded.train, loaded.test)
    train_base = train_base.sort_values("DT", kind="mergesort")
    test_base = test_base.sort_values("DT", kind="mergesort")
    y = loaded.y.loc[train_base.index]
    print_range_summary(train_base, test_base, y)
    print("\nTrain month counts:")
    print(train_base["DT_M"].value_counts().sort_index().to_string())
    print("\nTest month counts:")
    print(test_base["DT_M"].value_counts().sort_index().to_string())
    duplicate_uid_codes = train_base["uid"].duplicated().sum()
    print(f"\nUID count train={train_base['uid'].nunique():,}, duplicate rows={duplicate_uid_codes:,}")


def aggregate_fraud_rate(
    train: pd.DataFrame,
    y: pd.Series,
    bucket: str = "D",
) -> pd.DataFrame:
    df = train[["TransactionDT", "TransactionAmt"]].copy()
    df["isFraud"] = y.astype("int8")
    df["ds"] = START_DATE + pd.to_timedelta(df["TransactionDT"], unit="s")
    grouped = df.set_index("ds").resample(bucket)
    out = grouped.agg(
        y=("isFraud", "mean"),
        fraud_count=("isFraud", "sum"),
        tx_count=("isFraud", "size"),
        amt_mean=("TransactionAmt", "mean"),
    ).reset_index()
    out = out[out["tx_count"] > 0].reset_index(drop=True)
    out.attrs["freq"] = bucket
    return out


def infer_series_freq(series: pd.DataFrame) -> str:
    if series.attrs.get("freq"):
        return str(series.attrs["freq"])
    if len(series) < 3:
        return "D"
    inferred = pd.infer_freq(series["ds"])
    if inferred:
        return inferred
    diffs = series["ds"].diff().dropna()
    if diffs.empty:
        return "D"
    return pd.tseries.frequencies.to_offset(diffs.mode().iloc[0]).freqstr


def fit_prophet_model(series: pd.DataFrame, library: str, periods: int) -> tuple[Any, pd.DataFrame]:
    freq = infer_series_freq(series)
    if library == "prophet":
        try:
            from prophet import Prophet
        except Exception as exc:
            raise SystemExit("prophet is required: pip install prophet") from exc

        model = Prophet(
            yearly_seasonality=False,
            weekly_seasonality=True,
            daily_seasonality=freq.lower() in {"h", "1h"},
            changepoint_prior_scale=0.1,
        )
        model.add_country_holidays(country_name="US")
        model.fit(series[["ds", "y"]])
        future = model.make_future_dataframe(periods=periods, freq=freq)
        forecast = model.predict(future)
        keep = [
            "ds", "yhat", "trend", "weekly", "daily",
            "holidays", "yhat_lower", "yhat_upper",
        ]
        keep = [c for c in keep if c in forecast.columns]
        return model, forecast[keep]

    try:
        from neuralprophet import NeuralProphet
    except Exception as exc:
        raise SystemExit("neuralprophet is required: pip install neuralprophet") from exc

    model = NeuralProphet(
        yearly_seasonality=False,
        weekly_seasonality=True,
        daily_seasonality=freq.lower() in {"h", "1h"},
        n_lags=min(24, max(1, len(series) // 4)),
        n_forecasts=1,
        learning_rate=0.01,
    )
    model.fit(series[["ds", "y"]], freq=freq)
    future = model.make_future_dataframe(series[["ds", "y"]], periods=periods)
    forecast = model.predict(future).rename(columns={"yhat1": "yhat"})
    keep = [c for c in ["ds", "yhat", "trend", "season_weekly", "season_daily"] if c in forecast]
    return model, forecast[keep]


def validate_aggregate_forecast(
    series: pd.DataFrame,
    library: str,
    horizon: int,
) -> dict[str, float]:
    if horizon <= 0 or len(series) <= horizon + 2:
        return {}
    train_series = series.iloc[:-horizon].copy()
    valid_series = series.iloc[-horizon:].copy()
    _, forecast = fit_prophet_model(train_series, library=library, periods=horizon)
    scored = valid_series[["ds", "y"]].merge(forecast[["ds", "yhat"]], on="ds", how="left")
    scored = scored.dropna(subset=["yhat"])
    if scored.empty:
        return {}
    rmse = root_mean_squared(scored["y"], scored["yhat"])
    mae = mean_absolute(scored["y"], scored["yhat"])
    return {"mae": float(mae), "rmse": float(rmse), "rows": float(len(scored))}


def merge_forecast_features(
    transactions: pd.DataFrame,
    forecast: pd.DataFrame,
    bucket: str,
    prefix: str,
) -> pd.DataFrame:
    df = transactions[["TransactionDT"]].copy()
    df["DT"] = START_DATE + pd.to_timedelta(df["TransactionDT"], unit="s")
    df["bucket"] = df["DT"].dt.floor(bucket)
    fcst = forecast.rename(columns={"ds": "bucket"}).copy()
    value_cols = [c for c in fcst.columns if c != "bucket"]
    fcst = fcst[["bucket"] + value_cols]
    fcst = fcst.rename(columns={c: f"{prefix}{c}" for c in value_cols})
    out = df[["bucket"]].merge(fcst, on="bucket", how="left")
    out.index = transactions.index
    return out.drop(columns=["bucket"])


def run_prophet(args: argparse.Namespace) -> None:
    loaded = load_raw_data(args.data_dir, include_v=False, nrows=args.nrows)
    train_base, test_base = add_rowwise_features(loaded.train, loaded.test)
    train_base = train_base.sort_values("DT", kind="mergesort")
    y = loaded.y.loc[train_base.index]
    test_base = test_base.sort_values("DT", kind="mergesort")

    series = aggregate_fraud_rate(train_base, y, bucket=args.bucket)
    print(
        f"Aggregate series rows={len(series):,}, bucket={args.bucket}, "
        f"mean fraud rate={series['y'].mean():.5f}"
    )

    metrics = validate_aggregate_forecast(series, library=args.library, horizon=args.valid_horizon)
    if metrics:
        print(
            f"Holdout aggregate forecast: rows={int(metrics['rows'])}, "
            f"MAE={metrics['mae']:.6f}, RMSE={metrics['rmse']:.6f}"
        )

    print(f"Fitting final {args.library} model and forecasting {args.periods_ahead} buckets...")
    model, forecast = fit_prophet_model(series, library=args.library, periods=args.periods_ahead)
    del model

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    forecast_path = output_dir / f"{args.library}_forecast_{args.bucket}.csv"
    forecast.to_csv(forecast_path, index=False)
    print(f"Wrote {forecast_path}")

    prefix = f"{args.library}_"
    train_features = merge_forecast_features(train_base, forecast, bucket=args.bucket, prefix=prefix)
    test_features = merge_forecast_features(test_base, forecast, bucket=args.bucket, prefix=prefix)
    train_features.insert(0, "TransactionID", train_features.index)
    test_features.insert(0, "TransactionID", test_features.index)
    train_path = output_dir / f"{args.library}_transaction_features_train_{args.bucket}.csv"
    test_path = output_dir / f"{args.library}_transaction_features_test_{args.bucket}.csv"
    train_features.to_csv(train_path, index=False)
    test_features.to_csv(test_path, index=False)
    print(f"Wrote {train_path}")
    print(f"Wrote {test_path}")


def select_lstm_columns(df: pd.DataFrame, include_v: bool) -> list[str]:
    drop = set(DROP_FROM_MODEL)
    drop.update({"uid"})
    cols = []
    for col in df.columns:
        if col in drop:
            continue
        if not include_v and col.startswith("V"):
            continue
        if col.startswith("uid_DT_") or col in {f"uid_{c}" for c in TIME_SERIES_COLUMNS}:
            continue
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            continue
        cols.append(col)
    return cols


def build_sequence_windows(
    history_df: pd.DataFrame,
    prediction_df: pd.DataFrame,
    feature_cols: list[str],
    window: int,
    uid_col: str = "uid",
    time_col: str = "DT",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build windows for prediction_df using history_df plus earlier prediction rows.

    No labels are used. For validation, this simulates chronological scoring:
    earlier validation transactions can contribute feature history for later
    validation transactions because their row features are known at prediction
    time.
    """
    hist = history_df[[uid_col, time_col] + feature_cols].copy()
    hist["_predict"] = False
    hist["_orig_index"] = hist.index
    pred = prediction_df[[uid_col, time_col] + feature_cols].copy()
    pred["_predict"] = True
    pred["_orig_index"] = pred.index
    combined = pd.concat([hist, pred], axis=0)
    combined = combined.sort_values([uid_col, time_col, "_predict"], kind="mergesort")

    pred_count = len(prediction_df)
    n_features = len(feature_cols)
    X = np.zeros((pred_count, window, n_features), dtype="float32")
    row_index = np.zeros(pred_count, dtype=prediction_df.index.dtype)
    write_pos = 0

    for _, group in combined.groupby(uid_col, sort=False):
        values = group[feature_cols].to_numpy(dtype="float32")
        is_pred = group["_predict"].to_numpy(dtype=bool)
        indexes = group["_orig_index"].to_numpy()
        for pos in range(len(group)):
            if not is_pred[pos]:
                continue
            start = max(0, pos - window + 1)
            seq = values[start: pos + 1]
            X[write_pos, -len(seq):] = seq
            row_index[write_pos] = indexes[pos]
            write_pos += 1

    return X[:write_pos], row_index[:write_pos]


def run_lstm(args: argparse.Namespace) -> None:
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, Dataset
    except Exception as exc:
        raise SystemExit(
            "The LSTM command requires PyTorch. Install it first, for example: "
            "pip install torch"
        ) from exc

    class WindowDataset(Dataset):
        def __init__(self, X: np.ndarray, y: np.ndarray | None = None):
            self.X = torch.from_numpy(X)
            self.y = None if y is None else torch.from_numpy(y.astype("float32"))

        def __len__(self) -> int:
            return len(self.X)

        def __getitem__(self, idx: int):
            if self.y is None:
                return self.X[idx]
            return self.X[idx], self.y[idx]

    class LSTMClassifier(nn.Module):
        def __init__(self, n_features: int, hidden: int, layers: int, dropout: float):
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
            out, _ = self.lstm(x)
            return self.head(out[:, -1, :]).squeeze(-1)

    def train_one_fold(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_valid: np.ndarray,
        y_valid: np.ndarray,
        device: Any,
    ) -> tuple[np.ndarray, float]:
        model = LSTMClassifier(
            n_features=X_train.shape[2],
            hidden=args.hidden,
            layers=args.layers,
            dropout=args.dropout,
        ).to(device)
        pos_weight = torch.tensor(
            [(y_train == 0).sum() / max((y_train == 1).sum(), 1)],
            dtype=torch.float32,
            device=device,
        )
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

        train_loader = DataLoader(
            WindowDataset(X_train, y_train),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
        )
        valid_loader = DataLoader(
            WindowDataset(X_valid, y_valid),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
        )

        best_auc = -np.inf
        best_pred = np.zeros(len(y_valid), dtype="float32")
        patience_left = args.patience
        for epoch in range(args.epochs):
            model.train()
            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            model.eval()
            preds = []
            with torch.no_grad():
                for xb, _ in valid_loader:
                    xb = xb.to(device)
                    preds.append(torch.sigmoid(model(xb)).cpu().numpy())
            pred = np.concatenate(preds)
            auc = binary_roc_auc_score(y_valid, pred)
            print(f"    epoch={epoch + 1}, auc={auc:.6f}")
            if auc > best_auc:
                best_auc = auc
                best_pred = pred.astype("float32")
                patience_left = args.patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break
        return best_pred, float(best_auc)

    loaded = load_raw_data(args.data_dir, include_v=args.include_v, nrows=args.nrows)
    train_base, test_base = add_rowwise_features(loaded.train, loaded.test)
    del test_base
    train_base = train_base.sort_values("DT", kind="mergesort")
    y = loaded.y.loc[train_base.index]

    folds = expanding_time_splits(
        train_base,
        min_train_periods=args.min_train_periods,
        max_folds=args.max_folds,
    )
    oof = pd.Series(np.nan, index=train_base.index, dtype="float32")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device={device}")

    for fold, (idx_train, idx_valid, train_periods, valid_period) in enumerate(folds):
        print(
            f"\nFold {fold}: train periods={train_periods}, valid period={valid_period}, "
            f"train rows={len(idx_train):,}, valid rows={len(idx_valid):,}"
        )
        fold_train = train_base.iloc[idx_train]
        fold_valid = train_base.iloc[idx_valid]
        encoders = fit_label_encoders(fold_train)
        fold_train = apply_label_encoders(fold_train, encoders)
        fold_valid = apply_label_encoders(fold_valid, encoders)
        feature_cols = select_lstm_columns(fold_train, include_v=args.include_v)

        scaler = SimpleStandardScaler()
        fold_train[feature_cols] = scaler.fit_transform(
            fold_train[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(MISSING_VALUE)
        ).astype("float32")
        fold_valid[feature_cols] = scaler.transform(
            fold_valid[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(MISSING_VALUE)
        ).astype("float32")

        X_train, train_order = build_sequence_windows(
            history_df=fold_train.iloc[0:0],
            prediction_df=fold_train,
            feature_cols=feature_cols,
            window=args.window,
        )
        X_valid, valid_order = build_sequence_windows(
            history_df=fold_train,
            prediction_df=fold_valid,
            feature_cols=feature_cols,
            window=args.window,
        )
        y_train = y.loc[train_order].to_numpy(dtype="int8")
        y_valid = y.loc[valid_order].to_numpy(dtype="int8")
        pred, auc = train_one_fold(X_train, y_train, X_valid, y_valid, device)
        oof.loc[valid_order] = pred
        print(f"Fold {fold} LSTM AUC={auc:.6f}")
        gc.collect()

    scored = oof.notna()
    overall = binary_roc_auc_score(y.loc[scored.index[scored]], oof.loc[scored])
    print(f"\nSTRICT LSTM OOF AUC = {overall:.12f}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"oof_lstm_window{args.window}_{'v' if args.include_v else 'no_v'}.csv"
    pd.DataFrame({"TransactionID": oof.index, "oof_lstm": oof.values}).to_csv(path, index=False)
    print(f"Wrote {path}")


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data-dir",
        default=".",
        help="Directory containing train/test transaction and identity CSV files.",
    )
    parser.add_argument("--nrows", type=int, default=None, help="Read only the first N rows for smoke tests.")
    parser.add_argument("--include-v", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-dir", default="model_outputs")
    parser.add_argument("--seed", type=int, default=42)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="Inspect time ranges and UID construction.")
    add_common_args(inspect_parser)
    inspect_parser.set_defaults(func=run_inspect)

    xgb_parser = subparsers.add_parser("xgb", help="Run XGBoost transaction classifier.")
    add_common_args(xgb_parser)
    xgb_parser.add_argument("--protocol", choices=["strict", "kaggle"], default="strict")
    xgb_parser.add_argument("--feature-set", choices=["uid", "time", "uid-time", "all"], default="uid")
    xgb_parser.add_argument("--n-splits", type=int, default=6)
    xgb_parser.add_argument("--min-train-periods", type=int, default=3)
    xgb_parser.add_argument("--max-folds", type=int, default=None)
    xgb_parser.add_argument("--predict-test", action="store_true")
    xgb_parser.add_argument("--n-estimators", type=int, default=2000)
    xgb_parser.add_argument("--max-depth", type=int, default=12)
    xgb_parser.add_argument("--learning-rate", type=float, default=0.02)
    xgb_parser.add_argument("--subsample", type=float, default=0.8)
    xgb_parser.add_argument("--colsample-bytree", type=float, default=0.4)
    xgb_parser.add_argument("--tree-method", default="hist")
    xgb_parser.add_argument("--device", default="cpu")
    xgb_parser.add_argument("--early-stopping-rounds", type=int, default=100)
    xgb_parser.add_argument("--scale-pos-weight", action="store_true")
    xgb_parser.add_argument("--verbose-eval", type=int, default=100)
    xgb_parser.set_defaults(func=run_xgb)

    prophet_parser = subparsers.add_parser("prophet", help="Run aggregate Prophet/NeuralProphet forecast.")
    add_common_args(prophet_parser)
    prophet_parser.add_argument("--library", choices=["prophet", "neuralprophet"], default="prophet")
    prophet_parser.add_argument("--bucket", default="D", help="Pandas frequency, e.g. D or h.")
    prophet_parser.add_argument("--valid-horizon", type=int, default=30)
    prophet_parser.add_argument("--periods-ahead", type=int, default=220)
    prophet_parser.set_defaults(func=run_prophet)

    lstm_parser = subparsers.add_parser("lstm", help="Run strict time-forward LSTM sequence classifier.")
    add_common_args(lstm_parser)
    lstm_parser.add_argument("--window", type=int, default=5)
    lstm_parser.add_argument("--epochs", type=int, default=8)
    lstm_parser.add_argument("--batch-size", type=int, default=1024)
    lstm_parser.add_argument("--lr", type=float, default=1e-3)
    lstm_parser.add_argument("--hidden", type=int, default=128)
    lstm_parser.add_argument("--layers", type=int, default=2)
    lstm_parser.add_argument("--dropout", type=float, default=0.3)
    lstm_parser.add_argument("--patience", type=int, default=2)
    lstm_parser.add_argument("--min-train-periods", type=int, default=3)
    lstm_parser.add_argument("--max-folds", type=int, default=None)
    lstm_parser.set_defaults(func=run_lstm)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
