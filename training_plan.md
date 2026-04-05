# ESG-Based Stock Direction Prediction — Project Guide

A practical guide for building a deep learning model that predicts short-term stock price direction (up/down) using ESG scores, and writing an academic paper about it.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Environment Setup](#2-environment-setup)
3. [Data Pre-Processing](#3-data-pre-processing)
4. [Model Training](#4-model-training)
5. [Evaluation](#5-evaluation)
6. [Writing the Academic Paper](#6-writing-the-academic-paper)

---

## 1. Project Overview

**Goal:** Predict whether a stock's price will go up or down over the next 1–5 days, using ESG scores combined with historical price-based features.

**Your data:** ~700 US-listed stocks with environment, social, and governance scores and letter grades (from Finnhub, circa 2022).

**Model:** LSTM (Long Short-Term Memory) with an attention mechanism — the standard choice for time-series financial prediction in academic literature.

**Academic contribution:** Demonstrating whether ESG information adds predictive power over a price-only baseline model.

---

## 2. Environment Setup

### Install required libraries

```bash
pip install pandas numpy scikit-learn torch yfinance shap matplotlib seaborn jupyterlab
```

### Recommended project folder structure

```
esg_stock_prediction/
│
├── data/
│   ├── raw/
│   │   ├── esg_scores.csv          # Your uploaded CSV
│   │   └── prices/                 # Downloaded price files per ticker
│   ├── processed/
│   │   ├── features.csv            # Merged + engineered features
│   │   └── sequences.npz           # NumPy arrays ready for model
│
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_evaluation.ipynb
│
├── src/
│   ├── dataset.py                  # PyTorch Dataset class
│   ├── model.py                    # LSTM + attention architecture
│   └── train.py                    # Training loop
│
└── results/
    ├── figures/                    # Plots for the paper
    └── metrics.json                # Saved evaluation results
```

---

## 3. Data Pre-Processing

This is the most important phase. Poor pre-processing is the #1 cause of bad or misleading results.

### Step 1 — Load and clean the ESG CSV

```python
import pandas as pd

esg = pd.read_csv("data/raw/esg_scores.csv")

# Keep only the columns you need
esg = esg[["ticker", "name", "industry", "exchange",
           "environment_score", "social_score", "governance_score", "total_score",
           "environment_grade", "social_grade", "governance_grade", "total_grade"]]

# Drop rows with missing scores
esg = esg.dropna(subset=["environment_score", "social_score", "governance_score"])

print(esg.shape)         # Check how many stocks remain
print(esg.dtypes)        # Confirm column types
```

### Step 2 — Encode letter grades as numbers

The grades (B, BB, BBB, A, AA, AAA) carry ordinal information — convert them to integers.

```python
grade_map = {"B": 1, "BB": 2, "BBB": 3, "A": 4, "AA": 5, "AAA": 6}

for col in ["environment_grade", "social_grade", "governance_grade", "total_grade"]:
    esg[col + "_encoded"] = esg[col].map(grade_map)
```

### Step 3 — Normalize ESG scores to 0–1

This prevents ESG scores from dominating over price-based features during training.

```python
from sklearn.preprocessing import MinMaxScaler

scaler_esg = MinMaxScaler()
score_cols = ["environment_score", "social_score", "governance_score", "total_score"]
esg[score_cols] = scaler_esg.fit_transform(esg[score_cols])
```

### Step 4 — Download historical stock prices

Use `yfinance` to fetch daily OHLCV data for each ticker in your ESG dataset.

> **Note:** Some tickers may fail (delisted, renamed, etc.). The `try/except` block handles this gracefully.

```python
import yfinance as yf
import os

os.makedirs("data/raw/prices", exist_ok=True)

failed_tickers = []

for ticker in esg["ticker"].str.upper():
    try:
        df = yf.download(ticker, start="2020-01-01", end="2023-12-31", progress=False)
        if len(df) > 100:  # Only keep tickers with enough history
            df.to_csv(f"data/raw/prices/{ticker}.csv")
    except Exception as e:
        failed_tickers.append(ticker)
        print(f"Failed: {ticker} — {e}")

print(f"Downloaded prices for {len(esg) - len(failed_tickers)} tickers")
print(f"Failed tickers: {failed_tickers}")
```

### Step 5 — Create the prediction label

The label is 1 if the stock goes up over N days, 0 if it goes down.

```python
FORECAST_HORIZON = 5  # Predict 5 days ahead

def create_labels(price_df, horizon=FORECAST_HORIZON):
    """Forward return over N days. 1 = up, 0 = down or flat."""
    price_df["future_close"] = price_df["Close"].shift(-horizon)
    price_df["label"] = (price_df["future_close"] > price_df["Close"]).astype(int)
    price_df = price_df.dropna(subset=["future_close"])
    return price_df
```

### Step 6 — Engineer technical features from price data

```python
def add_technical_features(df):
    """Add momentum and volatility indicators."""

    # Returns
    df["return_1d"]  = df["Close"].pct_change(1)
    df["return_5d"]  = df["Close"].pct_change(5)
    df["return_20d"] = df["Close"].pct_change(20)

    # Moving averages
    df["ma_10"] = df["Close"].rolling(10).mean()
    df["ma_30"] = df["Close"].rolling(30).mean()
    df["ma_ratio"] = df["ma_10"] / df["ma_30"]  # Trend signal

    # Volatility
    df["volatility_20d"] = df["return_1d"].rolling(20).std()

    # RSI (14-day)
    delta = df["Close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi"] = 100 - (100 / (1 + gain / loss))

    # Volume change
    df["volume_change"] = df["Volume"].pct_change(1)

    return df
```

### Step 7 — Merge ESG features with price data

```python
all_stocks = []

for ticker in esg["ticker"].str.upper():
    price_path = f"data/raw/prices/{ticker}.csv"
    if not os.path.exists(price_path):
        continue

    prices = pd.read_csv(price_path, index_col=0, parse_dates=True)
    prices = add_technical_features(prices)
    prices = create_labels(prices)

    # Add static ESG features to every row
    esg_row = esg[esg["ticker"].str.upper() == ticker].iloc[0]
    for col in score_cols + ["environment_grade_encoded", "social_grade_encoded",
                              "governance_grade_encoded"]:
        prices[col] = esg_row[col]

    prices["ticker"] = ticker
    all_stocks.append(prices)

master_df = pd.concat(all_stocks).dropna()
master_df.to_csv("data/processed/features.csv")
print(f"Master dataset shape: {master_df.shape}")
```

### Step 8 — Build time-series sequences for the LSTM

LSTMs need sequences — a window of past timesteps as input.

```python
import numpy as np

LOOKBACK = 30  # Use 30 days of history to predict the next

FEATURE_COLS = [
    "return_1d", "return_5d", "return_20d",
    "ma_ratio", "volatility_20d", "rsi", "volume_change",
    "environment_score", "social_score", "governance_score", "total_score",
    "environment_grade_encoded", "social_grade_encoded", "governance_grade_encoded"
]

def build_sequences(df, lookback=LOOKBACK):
    X, y = [], []
    vals = df[FEATURE_COLS].values
    labels = df["label"].values

    for i in range(lookback, len(vals)):
        X.append(vals[i - lookback:i])
        y.append(labels[i])

    return np.array(X), np.array(y)

# Build per ticker, then concatenate
X_list, y_list = [], []
for ticker, group in master_df.groupby("ticker"):
    group = group.sort_index()
    X_t, y_t = build_sequences(group)
    X_list.append(X_t)
    y_list.append(y_t)

X_all = np.concatenate(X_list)
y_all = np.concatenate(y_list)

# IMPORTANT: Time-based split — never shuffle!
split = int(len(X_all) * 0.8)
X_train, X_test = X_all[:split], X_all[split:]
y_train, y_test = y_all[:split], y_all[split:]

np.savez("data/processed/sequences.npz",
         X_train=X_train, y_train=y_train,
         X_test=X_test, y_test=y_test)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")
```

> **Critical note on data splitting:** Always split by time, not randomly. If you shuffle the data, future information leaks into the training set and your accuracy will be artificially inflated — this is a very common mistake in student projects.

---

## 4. Model Training

### Step 1 — Define the LSTM + Attention model

```python
import torch
import torch.nn as nn

class AttentionLayer(nn.Module):
    """Computes a weighted sum over LSTM hidden states."""
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, lstm_out):
        # lstm_out shape: (batch, seq_len, hidden_size)
        weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = (weights * lstm_out).sum(dim=1)
        return context, weights


class ESGStockLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.attention = AttentionLayer(hidden_size)
        self.dropout   = nn.Dropout(dropout)
        self.fc        = nn.Linear(hidden_size, 1)
        self.sigmoid   = nn.Sigmoid()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)           # (batch, seq, hidden)
        context, attn_weights = self.attention(lstm_out)
        out = self.dropout(context)
        out = self.fc(out)
        return self.sigmoid(out).squeeze(-1), attn_weights
```

### Step 2 — Create a PyTorch Dataset

```python
from torch.utils.data import Dataset, DataLoader

class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


data = np.load("data/processed/sequences.npz")

train_loader = DataLoader(StockDataset(data["X_train"], data["y_train"]),
                          batch_size=64, shuffle=False)  # No shuffle for time-series!
test_loader  = DataLoader(StockDataset(data["X_test"],  data["y_test"]),
                          batch_size=64, shuffle=False)
```

### Step 3 — Training loop

```python
INPUT_SIZE  = len(FEATURE_COLS)   # 14 features
HIDDEN_SIZE = 64
NUM_LAYERS  = 2
DROPOUT     = 0.3
EPOCHS      = 50
LR          = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model     = ESGStockLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT).to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# Early stopping setup
best_val_loss = float("inf")
patience      = 7
patience_counter = 0

train_losses, val_losses = [], []

for epoch in range(EPOCHS):
    # --- Training ---
    model.train()
    epoch_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        preds, _ = model(X_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    # --- Validation ---
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            preds, _ = model(X_batch)
            val_loss += criterion(preds, y_batch).item()

    avg_train = epoch_loss / len(train_loader)
    avg_val   = val_loss / len(test_loader)
    train_losses.append(avg_train)
    val_losses.append(avg_val)

    print(f"Epoch {epoch+1:3d}/{EPOCHS} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")

    # Early stopping check
    if avg_val < best_val_loss:
        best_val_loss = avg_val
        torch.save(model.state_dict(), "results/best_model.pt")
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

print("Training complete. Best model saved to results/best_model.pt")
```

### Step 4 — Also train a baseline (price-only) model

This is essential for your paper. The baseline model uses only technical features, no ESG.

```python
PRICE_ONLY_COLS = [
    "return_1d", "return_5d", "return_20d",
    "ma_ratio", "volatility_20d", "rsi", "volume_change"
]

# Rebuild sequences using only price features, then train an identical LSTM.
# Compare results with the full ESG model in your results chapter.
```

---

## 5. Evaluation

### Classification metrics

```python
from sklearn.metrics import (accuracy_score, f1_score,
                              roc_auc_score, confusion_matrix,
                              classification_report)
import matplotlib.pyplot as plt
import seaborn as sns

# Load best model
model.load_state_dict(torch.load("results/best_model.pt"))
model.eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        preds, _ = model(X_batch.to(device))
        all_preds.extend((preds.cpu() > 0.5).int().tolist())
        all_labels.extend(y_batch.int().tolist())

print(classification_report(all_labels, all_preds, target_names=["Down", "Up"]))
print(f"AUC-ROC: {roc_auc_score(all_labels, all_preds):.4f}")

# Confusion matrix plot
cm = confusion_matrix(all_labels, all_preds)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Down", "Up"], yticklabels=["Down", "Up"])
plt.title("Confusion Matrix — ESG+Price Model")
plt.savefig("results/figures/confusion_matrix.png", dpi=150, bbox_inches="tight")
```

### SHAP values — which features mattered most?

This is one of the most important sections for your paper.

```python
import shap

# SHAP works with a wrapper function for PyTorch
def model_predict(X_np):
    model.eval()
    with torch.no_grad():
        t = torch.FloatTensor(X_np).to(device)
        # Flatten sequence for SHAP (use last timestep)
        preds, _ = model(t)
        return preds.cpu().numpy()

# Use a small background sample
background = data["X_train"][:200]
explainer  = shap.KernelExplainer(
    lambda x: model_predict(x.reshape(-1, LOOKBACK, INPUT_SIZE)),
    background.reshape(200, -1)
)

sample  = data["X_test"][:100]
shap_vals = explainer.shap_values(sample.reshape(100, -1))

shap.summary_plot(shap_vals, sample.reshape(100, -1),
                  feature_names=FEATURE_COLS,
                  show=False)
plt.savefig("results/figures/shap_summary.png", dpi=150, bbox_inches="tight")
```

### Attention weight visualization

```python
# Pick one sample and visualize what the model attended to over 30 timesteps
sample_x = torch.FloatTensor(data["X_test"][0:1]).to(device)
with torch.no_grad():
    _, attn = model(sample_x)

attn_np = attn.squeeze().cpu().numpy()  # Shape: (30, 1) → (30,)

plt.figure(figsize=(10, 3))
plt.bar(range(LOOKBACK), attn_np.flatten())
plt.xlabel("Day (t-30 → t-1)")
plt.ylabel("Attention weight")
plt.title("Attention weights — which past days drove the prediction?")
plt.savefig("results/figures/attention_weights.png", dpi=150, bbox_inches="tight")
```

---

## 6. Writing the Academic Paper

### Suggested paper structure

```
1. Abstract          (~250 words)
2. Introduction      (~500–800 words)
3. Literature Review (~800–1200 words)
4. Data              (~500 words)
5. Methodology       (~800–1000 words)
6. Results           (~600–800 words)
7. Discussion        (~400–600 words)
8. Conclusion        (~300 words)
9. References
```

---

### Section-by-section writing guide

#### 1. Abstract
Write this last. Summarize: the problem, your method, your main result, and the implication. One paragraph, ~250 words.

#### 2. Introduction
- Open with why ESG investing is growing (cite global ESG AUM figures).
- State the gap: most ML stock prediction studies ignore ESG data.
- State your research question: *"Does ESG information improve short-term stock direction prediction in a deep learning framework?"*
- Briefly state what you did and what you found.
- List your contributions (e.g., ESG feature encoding scheme, attention-LSTM architecture, ESG vs baseline ablation study).

#### 3. Literature Review
Cover three bodies of work and connect them:
- **ESG and stock returns** — empirical finance papers showing ESG-performance relationships.
- **Deep learning for stock prediction** — LSTM, GRU, and Transformer papers on equity markets.
- **Combining alternative data with price data** — papers using sentiment, news, or non-traditional features for prediction.

Key papers to search for on Google Scholar:
- "LSTM stock price prediction" — many papers from 2018–2023
- "ESG investing stock returns meta-analysis"
- "Machine learning ESG equity prediction"
- Friede et al. (2015) — large meta-analysis on ESG and financial performance

#### 4. Data
Describe:
- Your ESG data source (Finnhub), date range, number of stocks, and which exchanges.
- How you obtained price data (yfinance, Yahoo Finance API).
- Summary statistics table — mean/std of ESG scores, class balance (up vs down labels), number of sequences.
- A table showing grade distribution (how many stocks are rated B, BB, BBB, etc.).

#### 5. Methodology
Cover four sub-sections:
1. **Feature engineering** — how you encoded grades, normalized scores, and computed technical indicators.
2. **Sequence construction** — lookback window choice and justification.
3. **Model architecture** — diagram of the LSTM + attention layers, number of parameters.
4. **Training protocol** — loss function, optimizer, early stopping, data split strategy, and why you used a time-based split.

> Tip: Include the architecture diagram as a figure. Draw it using draw.io or a simple Python visualization.

#### 6. Results
Present results in tables and figures:
- **Table 1:** Accuracy, F1, AUC-ROC for (a) price-only baseline and (b) ESG+price model.
- **Figure 1:** Training vs validation loss curves.
- **Figure 2:** Confusion matrices side-by-side.
- **Figure 3:** SHAP summary plot (feature importances).
- **Figure 4:** Attention weight bar chart for a sample prediction.

#### 7. Discussion
Address:
- Did ESG improve prediction? By how much?
- Which ESG component (E, S, or G) was most important per SHAP?
- Why might high-ESG stocks be more predictable? (Less tail risk? More institutional coverage?)
- Limitations: static ESG scores (not time-varying), US-only data, single time period.

#### 8. Conclusion
- Restate the research question and your answer.
- Summarize the key findings in 2–3 sentences.
- Propose future work: time-varying ESG scores, multi-market extension, longer forecast horizons.

---

### Reproducibility checklist (reviewers will check this)

- [ ] Random seeds fixed: `torch.manual_seed(42)`, `np.random.seed(42)`
- [ ] All hyperparameters listed in a table in the paper
- [ ] Train/test split dates explicitly stated
- [ ] Code made available (GitHub link or supplementary material)
- [ ] Data sources and access dates cited

---

### Common mistakes to avoid

| Mistake | Why it matters |
|---|---|
| Shuffling time-series data before splitting | Causes data leakage — results look great but are fake |
| Only reporting accuracy | Accuracy is misleading on imbalanced datasets; always report F1 and AUC-ROC |
| Not training a baseline | Without a price-only model, you can't prove ESG adds value |
| Using future prices as features | Another form of leakage — check your feature lags carefully |
| Ignoring class imbalance | If 60% of days are "up", a model that always says "up" gets 60% accuracy for free |

---

*Good luck with your thesis. The project is well-scoped and your dataset is a solid foundation.*