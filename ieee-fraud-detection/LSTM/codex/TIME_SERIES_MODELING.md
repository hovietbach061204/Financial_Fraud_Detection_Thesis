# Time-Series Fraud Modeling Runner

This folder now includes `time_series_fraud_experiments.py`, a reusable runner
for the plan discussed in the notebook review.

## Indexing Rule

Keep `TransactionID` as the dataframe index or row key for transaction-level
fraud prediction. The runner creates `DT` from `TransactionDT`, sorts by it,
and uses `DT` as a normal column for calendar features, folds, rolling/sequence
logic, and aggregate forecasting. Set `DT` as an index only temporarily for
`resample()` calls such as Prophet or NeuralProphet aggregate series creation.

## Commands

From this directory:

```bash
python time_series_fraud_experiments.py inspect --data-dir .
```

Run strict time-forward XGBoost with UID aggregation:

```bash
python time_series_fraud_experiments.py xgb \
  --data-dir . \
  --protocol strict \
  --feature-set uid \
  --include-v \
  --predict-test
```

Run Kaggle-style transductive XGBoost for comparison with the original
notebook approach:

```bash
python time_series_fraud_experiments.py xgb \
  --data-dir . \
  --protocol kaggle \
  --feature-set uid \
  --include-v \
  --predict-test
```

Try the time-derived aggregation branches:

```bash
python time_series_fraud_experiments.py xgb --data-dir . --protocol strict --feature-set time
python time_series_fraud_experiments.py xgb --data-dir . --protocol strict --feature-set uid-time
python time_series_fraud_experiments.py xgb --data-dir . --protocol strict --feature-set all
```

Create aggregate Prophet or NeuralProphet features:

```bash
python time_series_fraud_experiments.py prophet --data-dir . --library prophet --bucket D
python time_series_fraud_experiments.py prophet --data-dir . --library neuralprophet --bucket D
```

Train a strict time-forward LSTM sequence classifier:

```bash
python time_series_fraud_experiments.py lstm \
  --data-dir . \
  --window 5 \
  --epochs 8 \
  --batch-size 1024 \
  --no-include-v
```

## Outputs

The runner writes outputs to `model_outputs/` by default:

- `oof_xgb_<protocol>_<feature-set>_<v|no_v>.csv`
- `test_xgb_<protocol>_<feature-set>_<v|no_v>.csv` when `--predict-test` is used
- `prophet_forecast_<bucket>.csv` or `neuralprophet_forecast_<bucket>.csv`
- transaction-level Prophet/NeuralProphet feature CSVs that can be merged back
  into the notebook on `TransactionID`
- `oof_lstm_window<window>_<v|no_v>.csv`

## Validation Semantics

`--protocol strict` uses expanding time folds. For each validation month,
encoders, frequency counts, and aggregate features are fitted only on earlier
months. This is the thesis-valid estimate.

`--protocol kaggle` builds frequency and aggregate features from train+test,
then uses grouped month CV. This intentionally preserves the transductive
feature engineering style used by the public Kaggle notebooks, so report it as
leaderboard-comparable but not as a pure forecasting validation.

## Dependencies

The base feature preparation requires `numpy` and `pandas`.

XGBoost requires compatible versions of:

```bash
python -m pip install -r requirements-time-series.txt
```

For a smaller install, use only the part you need. LSTM requires PyTorch:

```bash
pip install torch
```

Prophet and NeuralProphet are optional and installed separately:

```bash
pip install prophet neuralprophet
```

If the notebook environment raises import errors from `pandas`, `scipy`, or
`sklearn`, repair that environment first. A broken scientific Python stack will
also prevent `xgboost` from importing.
