# ADS504 Final Project: Bank Marketing

**Course:** ADS504 — Machine Learning and Deep Learning for Data Science  
**Institution:** University of San Diego  
**Dataset:** [Bank Marketing (UCI)](https://archive.ics.uci.edu/dataset/222/bank+marketing)

## Overview

This project uses the UCI Bank Marketing dataset to predict whether a bank client will subscribe to a term deposit after a telemarketing contact. The pipeline covers exploratory analysis, preprocessing, model comparison/tuning, and campaign-style evaluation so outreach can be prioritized toward clients most likely to convert.

## Problem Definition

Portuguese banks contact clients by phone to sell term deposits, but most contacts do not convert (~12% subscribe). Accuracy alone is misleading under this imbalance. We train classification models on client and campaign features to:

1. Rank clients by subscription likelihood (ROC-AUC / PR-AUC)
2. Choose decision thresholds that balance precision, recall, and simple campaign economics
3. Identify the strongest drivers of conversion for interpretation

## Repository Structure

```
data/
  raw/                 # Original UCI files (bank-full.csv, etc.)
  processed/           # Generated artifacts (gitignored; created by notebooks)
notebooks/
  01_eda.ipynb         # Exploratory data analysis
  02_preprocess.ipynb  # Cleaning, feature engineering, train/val/test split
  03_modeling.ipynb    # Model comparison, tuning, threshold selection
  04_evaluate.ipynb    # Held-out evaluation, lift, cost/profit thresholds
docs/                  # Optional presentation materials
```

## Pipeline

Run notebooks in order from the `notebooks/` directory (paths are relative to that folder).

| Notebook | Purpose |
|---|---|
| `01_eda.ipynb` | Class imbalance, distributions, categorical signals, duration leakage |
| `02_preprocess.ipynb` | Drop duration, engineer features, encode/scale (fit on train only), save splits |
| `03_modeling.ipynb` | Compare Logistic Regression / Random Forest / XGBoost; tune XGBoost; save model |
| `04_evaluate.ipynb` | Test metrics, calibration, lift/gains, cost/profit thresholds, prediction export |

### Modeling choices

- **Leakage control:** `duration` is dropped (only known after the call)
- **Imbalance handling:** balanced class weights (LR/RF), `scale_pos_weight` (XGBoost), and validation-based thresholding
- **Selected model:** tuned XGBoost (best ranking metrics), refit on train+validation
- **Evaluation:** held-out test metrics plus campaign lift and threshold policy analysis

### Key results (held-out test)

Approximate performance for the final tuned XGBoost model:

| Metric | Value |
|---|---|
| ROC-AUC | ~0.81 |
| PR-AUC | ~0.47 |
| F1 | ~0.50 |
| Recall | ~0.54 |
| Precision | ~0.46 |

Predicted-positive conversion is substantially higher than the ~12% base rate, supporting prioritized outreach.

## Setup

### Requirements

- Python 3.10+
- Packages: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `joblib`, `xgboost`, `jupyter`

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install numpy pandas scikit-learn matplotlib seaborn joblib xgboost jupyter
```

### Data

Place the UCI files under `data/raw/`. The main modeling input is:

```text
data/raw/bank-full.csv
```

Processed outputs are written to `data/processed/` when you run preprocessing and modeling. Those files are gitignored, so regenerate them locally:

```bash
cd notebooks
jupyter nbconvert --to notebook --execute 01_eda.ipynb --inplace
jupyter nbconvert --to notebook --execute 02_preprocess.ipynb --inplace
jupyter nbconvert --to notebook --execute 03_modeling.ipynb --inplace
jupyter nbconvert --to notebook --execute 04_evaluate.ipynb --inplace
```

Or open each notebook and Run All.

### Google Colab

Upload the repo (or clone it), set notebook paths to your Drive/`data` location if needed, and run the four notebooks in order.

## Collaborators

- Michael Rosalia — mrosalia@sandiego.edu
- Malik Tisani — mtisani@sandiego.edu
- Jordyn Gonzales — jordyngonzales@sandiego.edu

## License

Dataset: Creative Commons Attribution 4.0 International (CC BY 4.0) — [UCI Bank Marketing](https://doi.org/10.24432/C5K306)
