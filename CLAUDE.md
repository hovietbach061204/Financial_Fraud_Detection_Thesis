# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research thesis project on **Financial Fraud Detection** using machine learning. The work focuses on exploring and comparing multiple real-world fraud datasets, with an emphasis on EDA, feature engineering, and model benchmarking.

## Development Environment

All work is done in **Jupyter notebooks** using Python. There is no build system, test runner, or package manager config — the environment is managed externally (likely conda or pip).

Core libraries in use: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`.

To launch Jupyter: `jupyter notebook` or `jupyter lab` from the project root.

## Repository Structure

- `data_inspection/` — One notebook per dataset for EDA and initial analysis
- `data/` — Raw datasets (CSV files, large; not all committed to git)
- `data/ieee-fraud-detection/` — IEEE-CIS Kaggle competition data + community notebooks
- `documents/` — Research papers/PDfs referenced in the thesis

The `main.py` at root is a PyCharm placeholder and not part of the project.

## Datasets

| Notebook | Dataset | Target | Size |
|---|---|---|---|
| `AIML_dataset.ipynb` | PaySim (mobile money simulation) | `isFraud` | 6.3M rows, 11 cols |
| `credit_card.ipynb` | European credit card transactions (anonymized via PCA as V1–V28) | `Class` | 284K rows, 31 cols |
| `ieee_fraud_detection.ipynb` | IEEE-CIS Kaggle competition | `isFraud` | 590K train + 507K test, 394 cols |
| `fraud_ecom.ipynb` | E-commerce fraud (user/device/IP features) | `class` | 151K rows, 11 cols |
| `merchants.ipynb` | Merchant category reference table | N/A | 800 rows |
| `simulated_credit_card_sparkov.ipynb` | Sparkov simulated credit card fraud | `is_fraud` | train+test split |
| `paySim.ipynb` | PaySim mobile transactions (alternate version) | `isFraud` | — |

All datasets use absolute paths pointing to `/Volumes/SandiskSSD/Developer/AI_Document/Financial_Fraud_Detection_Thesis/data/`. Update these if the drive mount point changes.

## Key Domain Notes

- **Class imbalance** is severe across all datasets (fraud rate: 0.13% in PaySim, 3.5% in IEEE-CIS, ~9% in e-commerce). Evaluation must use precision/recall/F1, not accuracy.
- **IEEE-CIS**: The challenge is modeling unseen clients (not unseen time). `TransactionDT` is seconds from a reference point. Client identity is approximated by combining `card1–card6` fields into a `uid`.
- **Credit card dataset**: Features V1–V28 are PCA-transformed; only `Time` and `Amount` are raw.
- **PaySim**: Fraud only occurs in `TRANSFER` and `CASH_OUT` transaction types.
- The `isFlaggedFraud` column in PaySim is a weak rule-based system with near-zero recall (0.19%) — it is a baseline to beat, not a feature to use directly.