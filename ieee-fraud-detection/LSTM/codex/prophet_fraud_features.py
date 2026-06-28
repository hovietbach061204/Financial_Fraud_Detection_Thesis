"""
Prophet / NeuralProphet auxiliary features for IEEE-CIS fraud detection.

Why this script exists
----------------------
Prophet and NeuralProphet are univariate forecasting models. They forecast a
continuous signal y(t) over time; they are NOT classifiers and cannot
directly produce a per-transaction fraud probability with anything like the
performance of XGBoost. So we don't try to make them do that.

What they CAN do well is decompose a temporal signal into trend, weekly
seasonality, daily seasonality, holiday effects, etc. We exploit this:

  1. Aggregate isFraud rate (and transaction volume) by hour.
  2. Fit Prophet on the resulting hourly series.
  3. Pull out yhat, trend, weekly, daily as a feature table indexed by hour.
  4. Merge those features back onto every transaction by its DT_hour bucket.
  5. Hand the augmented dataframe to XGBoost.

Honest expectation: this gives a small lift at best, because XGBoost already
infers most calendar effects from your DT_hour / DT_day_week / is_holiday
columns. The interesting thesis angle is showing the decomposition itself
(plot trend + weekly + daily) and discussing whether the seasonal patterns
predicted by Prophet match what you observe. NeuralProphet adds AR lags and
optional NN regressors but the same logic applies.

Usage
-----
    pip install prophet neuralprophet
    python prophet_fraud_features.py \
        --data-dir /Volumes/SandiskSSD/Developer/AI_Document/Financial_Fraud_Detection_Thesis/data/ieee-fraud-detection \
        --bucket H \
        --use neuralprophet
"""

from __future__ import annotations

import argparse
import datetime
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)


START_DATE = datetime.datetime.strptime("2017-11-30", "%Y-%m-%d")


# --------------------------------------------------------------------------- #
# 1. Build aggregated fraud-rate series
# --------------------------------------------------------------------------- #
def aggregate_fraud_rate(
    train: pd.DataFrame,
    bucket: str = "H",
) -> pd.DataFrame:
    """
    Aggregate isFraud rate at a chosen frequency.

    Parameters
    ----------
    train : DataFrame with columns 'TransactionDT' and 'isFraud'
    bucket : pandas offset alias, e.g. 'H' (hour), 'D' (day)

    Returns
    -------
    DataFrame with columns ['ds', 'y', 'volume'].
    """
    df = train[["TransactionDT", "isFraud"]].copy()
    df["ds"] = START_DATE + pd.to_timedelta(df["TransactionDT"], unit="s")
    grp = df.groupby(pd.Grouper(key="ds", freq=bucket))["isFraud"]
    out = pd.DataFrame({
        "ds": grp.mean().index,
        "y": grp.mean().values,
        "volume": grp.size().values,
    })
    # Drop empty buckets (no transactions)
    out = out[out["volume"] > 0].reset_index(drop=True)
    return out


# --------------------------------------------------------------------------- #
# 2. Prophet path
# --------------------------------------------------------------------------- #
def fit_prophet(series: pd.DataFrame, periods_ahead: int):
    from prophet import Prophet

    m = Prophet(
        yearly_seasonality=False,   # only ~6 months of data
        weekly_seasonality=True,
        daily_seasonality=True,
        changepoint_prior_scale=0.1,
        interval_width=0.8,
    )
    m.add_country_holidays(country_name="US")
    m.fit(series[["ds", "y"]])

    future = m.make_future_dataframe(periods=periods_ahead, freq=infer_freq(series))
    fcst = m.predict(future)
    keep = ["ds", "yhat", "trend", "weekly", "daily", "yhat_lower", "yhat_upper"]
    keep = [c for c in keep if c in fcst.columns]
    return m, fcst[keep]


# --------------------------------------------------------------------------- #
# 3. NeuralProphet path
# --------------------------------------------------------------------------- #
def fit_neuralprophet(series: pd.DataFrame, periods_ahead: int):
    from neuralprophet import NeuralProphet

    m = NeuralProphet(
        yearly_seasonality=False,
        weekly_seasonality=True,
        daily_seasonality=True,
        n_lags=24,                  # AR over the last 24 buckets
        n_forecasts=1,
        learning_rate=0.01,
    )
    m.fit(series[["ds", "y"]], freq=infer_freq(series))
    future = m.make_future_dataframe(series[["ds", "y"]], periods=periods_ahead)
    fcst = m.predict(future)
    fcst = fcst.rename(columns={"yhat1": "yhat"})
    keep = [c for c in ["ds", "yhat", "trend", "season_weekly", "season_daily"]
            if c in fcst.columns]
    return m, fcst[keep]


def infer_freq(series: pd.DataFrame) -> str:
    diffs = series["ds"].diff().dropna().mode()
    delta = diffs.iloc[0]
    return pd.tseries.frequencies.to_offset(delta).freqstr


# --------------------------------------------------------------------------- #
# 4. Merge forecast components back onto transaction-level data
# --------------------------------------------------------------------------- #
def merge_features_onto_transactions(
    transactions: pd.DataFrame,
    forecast: pd.DataFrame,
    bucket: str,
    prefix: str = "tsf_",
) -> pd.DataFrame:
    """
    Attach forecast columns to each transaction by aligning DT to the same
    bucket the forecast was made at.
    """
    df = transactions.copy()
    if "DT" not in df.columns:
        df["DT"] = START_DATE + pd.to_timedelta(df["TransactionDT"], unit="s")
    df["bucket"] = pd.to_datetime(df["DT"]).dt.floor(bucket)
    fcst = forecast.rename(columns={"ds": "bucket"}).copy()
    fcst.columns = ["bucket"] + [prefix + c for c in fcst.columns if c != "bucket"]
    return df.merge(fcst, on="bucket", how="left").drop(columns=["bucket"])


# --------------------------------------------------------------------------- #
# 5. Driver
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--bucket", default="h",
                        help="Aggregation frequency. 'h' for hourly, 'D' for daily.")
    parser.add_argument("--use", choices=["prophet", "neuralprophet"],
                        default="prophet")
    parser.add_argument("--periods-ahead", type=int, default=24 * 90,
                        help="How many future buckets to forecast. "
                             "Should cover the test period.")
    parser.add_argument("--out-train",
                        default="prophet_features_train.parquet")
    parser.add_argument("--out-test",
                        default="prophet_features_test.parquet")
    parser.add_argument("--plot", action="store_true",
                        help="Save a decomposition plot.")
    args = parser.parse_args()

    print("Loading raw transaction CSVs...")
    train = pd.read_csv(
        os.path.join(args.data_dir, "../../csv_imported_file/train_transaction.csv"),
        usecols=["TransactionID", "TransactionDT", "isFraud"],
    )
    test = pd.read_csv(
        os.path.join(args.data_dir, "../../csv_imported_file/test_transaction.csv"),
        usecols=["TransactionID", "TransactionDT"],
    )

    print(f"Aggregating fraud rate at frequency '{args.bucket}'...")
    series = aggregate_fraud_rate(train, bucket=args.bucket)
    print(f"  series length = {len(series):,}, mean fraud rate = "
          f"{series['y'].mean():.4f}")

    print(f"Fitting {args.use}...")
    if args.use == "prophet":
        model, fcst = fit_prophet(series, periods_ahead=args.periods_ahead)
    else:
        model, fcst = fit_neuralprophet(series, periods_ahead=args.periods_ahead)
    print(f"  forecast rows = {len(fcst):,}, columns = {list(fcst.columns)}")

    if args.plot:
        try:
            import matplotlib.pyplot as plt
            fig = model.plot_components(fcst) if args.use == "prophet" \
                else model.plot_components(fcst)
            fig.savefig("prophet_components.png", bbox_inches="tight", dpi=120)
            print("  wrote prophet_components.png")
        except Exception as e:
            print(f"  (plot skipped: {e})")

    print("Merging forecast components onto transaction rows...")
    train_aug = merge_features_onto_transactions(train, fcst, bucket=args.bucket)
    test_aug = merge_features_onto_transactions(test, fcst, bucket=args.bucket)

    train_aug.to_parquet(args.out_train, index=False)
    test_aug.to_parquet(args.out_test, index=False)
    print(f"Wrote {args.out_train} ({len(train_aug):,} rows) and "
          f"{args.out_test} ({len(test_aug):,} rows)")

    print("\nNext step: in your notebook, merge these on TransactionID:")
    print("    extra = pd.read_parquet('prophet_features_train.parquet')")
    print("    X_train_copy8 = X_train_copy8.merge(")
    print("        extra.set_index('TransactionID')[[c for c in extra.columns "
          "if c.startswith('tsf_')]],")
    print("        left_index=True, right_index=True, how='left')")


if __name__ == "__main__":
    main()
