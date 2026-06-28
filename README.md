# Financial Fraud Detection Thesis

Research code and experiments for a bachelor thesis on transaction fraud detection using the [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) dataset.

The project compares a strong XGBoost baseline with three deep-learning directions:

- Long Short-Term Memory (LSTM) models for per-UID transaction sequences
- 1D CNN models with residual blocks and self-attention
- Graph Neural Networks (GNNs), including GTAN-style transaction graphs and GNN embeddings for XGBoost

The central methodological question is not only which model scores best, but also how validation design, entity recurrence, temporal leakage, class imbalance, and high-cardinality tabular features affect the result.

## Research focus

The experiments use two complementary validation settings:

1. **Rolling temporal validation** trains on earlier transactions and evaluates on later months. This estimates future fraud detection when cards, devices, addresses, and other entities may reappear.
2. **UID-disjoint validation** separates user-like identifiers between training and validation. This is a stricter cold-start test because validation identities are unseen during training.

Models are primarily evaluated with ROC-AUC. Some graph experiments additionally report average precision or F1 score.

## Current findings

The current experiments do not show a clear deep-learning improvement over XGBoost. This is a meaningful result rather than a failed endpoint: IEEE-CIS is highly tabular, sparse, imbalanced, and dominated by missing-value patterns and high-cardinality identity, card, address, email, device, count, and time-delta features. Gradient-boosted trees exploit these signals effectively, while neural models require careful representation learning, leakage-safe aggregation, and well-constructed temporal or graph structure.

UID-disjoint validation is also substantially stricter than rolling validation because it removes recurring identity signals. Scores from the two protocols therefore answer different research questions and should not be compared as if they measured the same deployment setting.

## Repository structure

| Path | Purpose |
| --- | --- |
| `data_inspection/` | Exploratory analysis notebooks for IEEE-CIS and additional fraud datasets |
| `ieee-fraud-detection/XGBoost/` | XGBoost baselines and time-aware feature engineering |
| `ieee-fraud-detection/LSTM/` | UID sequence construction, PyTorch LSTM experiments, and time-series utilities |
| `ieee-fraud-detection/CNN/` | CNN, residual-attention, and CNN-LSTM hybrid models |
| `ieee-fraud-detection/GNN/` | Transaction graph experiments, GTAN variants, and GNN-to-XGBoost augmentation |
| `ieee-fraud-detection/kaggle_notebook_results/` | Saved notebook runs and experimental outputs suitable for review |
| `ieee-fraud-detection/requirements-time-series.txt` | Core dependencies for the reusable time-series runner |

## Data

The datasets and generated arrays are intentionally not committed. Download the IEEE-CIS competition files from Kaggle and use this local layout:

```text
data/
└── ieee-fraud-detection/
    ├── train_transaction.csv
    ├── train_identity.csv
    ├── test_transaction.csv
    └── test_identity.csv
```

The repository also ignores exported CSV, pickle, parquet, NumPy, and sequence-cache artifacts. This keeps large or derived data out of Git history.

## Environment setup

Create an isolated Python environment from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r ieee-fraud-detection/requirements-time-series.txt
python -m pip install jupyterlab
```

GNN experiments additionally require PyTorch Geometric. Install the build compatible with your PyTorch version and CPU, CUDA, or Apple Silicon environment:

```bash
python -m pip install torch-geometric
```

## Quick start

Inspect the IEEE-CIS date ranges and available input files:

```bash
python ieee-fraud-detection/LSTM/codex/time_series_fraud_experiments.py \
  inspect \
  --data-dir data/ieee-fraud-detection
```

Run the leakage-safe, expanding-window XGBoost benchmark:

```bash
python ieee-fraud-detection/LSTM/codex/time_series_fraud_experiments.py \
  xgb \
  --data-dir data/ieee-fraud-detection \
  --protocol strict \
  --feature-set uid \
  --include-v
```

Run the Kaggle-style transductive comparison:

```bash
python ieee-fraud-detection/LSTM/codex/time_series_fraud_experiments.py \
  xgb \
  --data-dir data/ieee-fraud-detection \
  --protocol kaggle \
  --feature-set uid \
  --include-v
```

Train a strict time-forward LSTM sequence classifier:

```bash
python ieee-fraud-detection/LSTM/codex/time_series_fraud_experiments.py \
  lstm \
  --data-dir data/ieee-fraud-detection \
  --window 5 \
  --epochs 8 \
  --batch-size 1024 \
  --no-include-v
```

For the full command reference, see [`TIME_SERIES_MODELING.md`](ieee-fraud-detection/LSTM/codex/TIME_SERIES_MODELING.md). Most architecture experiments are notebook-based and can be opened with:

```bash
jupyter lab
```

## Model directions

### XGBoost

The baseline uses tabular feature engineering, UID and time aggregations, and either strict expanding-time folds or Kaggle-style grouped month validation. It provides the main reference for interpreting the neural approaches.

### LSTM

Transactions are sorted by UID and time, then converted into padded recent-history windows. Each sample predicts fraud for the latest transaction while exposing the model to the entity's previous activity.

### CNN + ResNet + Attention

The CNN experiments investigate both temporal convolution and per-row feature-axis convolution. Residual blocks, masked pooling, self-attention, optional CLS representations, and static transaction towers are explored as alternatives to a plain recurrent encoder.

### GNN and GTAN

Transactions are modeled as nodes or edges connected through shared entities such as cards, addresses, email domains, devices, users, or merchants. Experiments cover inductive and transductive graph construction, causal temporal edges, GTAN-style label-aware message passing, and extracted graph embeddings appended to XGBoost features.

## Reproducibility notes

- Use `--protocol strict` for thesis-valid time-forward estimates.
- Treat `--protocol kaggle` as a leaderboard-comparable, transductive experiment rather than a pure forecasting evaluation.
- Fit encoders, frequency counts, and aggregate features only on the training portion of each strict fold.
- Keep `TransactionID` as the stable row key and use the derived datetime column for ordering and feature engineering.
- Record the split strategy with every reported metric; rolling and UID-disjoint results are not directly interchangeable.

## Project status

This is an active research repository. Notebooks include exploratory and comparative experiments, so some workflows require running upstream feature-engineering cells before standalone model modules can be used. Large datasets, caches, model outputs, and thesis source documents remain local and are excluded from version control.
