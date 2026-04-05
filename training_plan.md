# ESG-Based Stock Direction Prediction — Complete Beginner's Project Guide

> **Who this guide is for:** Complete beginners. Every step is explained in plain language.
> No prior experience in machine learning, Python, or finance is assumed.
> Follow each phase in order — do not skip ahead.

---

## Table of Contents

1. [What We Are Building and Why](#1-what-we-are-building-and-why)
2. [Key Concepts You Need to Understand First](#2-key-concepts-you-need-to-understand-first)
3. [Environment Setup](#3-environment-setup)
4. [Data Diagnostics — Understanding What You Have](#4-data-diagnostics--understanding-what-you-have)
5. [Data Pre-Processing](#5-data-pre-processing)
6. [Model Training](#6-model-training)
7. [Evaluation](#7-evaluation)
8. [Writing the Academic Paper](#8-writing-the-academic-paper)
9. [Common Mistakes to Avoid](#9-common-mistakes-to-avoid)

---

## 1. What We Are Building and Why

### The Goal

We want to build a machine learning model that looks at a company's **ESG scores** (how environmentally responsible, socially responsible, and well-governed a company is) alongside its **recent stock price history**, and predicts: **will this stock's price be higher or lower 5 days from now?**

This is a binary classification problem. The model outputs one of two answers:
- `1` = the stock will go **up**
- `0` = the stock will go **down** (or stay flat)

### Why This Matters Academically

ESG investing has grown enormously — trillions of dollars are now managed with ESG criteria. But most deep learning stock prediction research only uses price and volume data. Your contribution is asking: **does ESG information make predictions better?**

You will prove (or disprove) this by training two models:
1. A **baseline model** — uses only price-based features
2. Your **ESG model** — uses price-based features + ESG scores

If the ESG model performs better, you have evidence that ESG data carries predictive signal.

### About Your Data

Your CSV file contains ESG scores for approximately 700 US-listed stocks, scored around 2022. It includes:
- `environment_score`, `social_score`, `governance_score`, `total_score` — numerical scores
- `environment_grade`, `social_grade`, `governance_grade`, `total_grade` — letter grades (B, BB, BBB, A, AA, AAA)
- `ticker` — the stock symbol (e.g., AAPL for Apple)
- `industry`, `exchange` — sector and market information

**Important limitation to know upfront:** Your ESG scores are a single snapshot in time (2022). They do not change over time in your dataset. This is fine for a thesis, but you must acknowledge it as a limitation in your paper. We explain how to do this in Phase 8.

### Why 700 Stocks Is Enough

Your model does not train on 700 rows. It trains on **sequences**.

For each stock, you will download 3 years of daily price data (~750 trading days). From that, you create one sequence per day using a 30-day lookback window. So each stock generates roughly 720 sequences. Across ~500 usable stocks (some will fail to download), that is approximately **360,000 training sequences** — more than enough for an LSTM.

```
700 stocks in CSV
→ ~500 usable after download failures and quality filtering
× ~750 trading days per stock
× 1 sequence per day (after 30-day warmup period)
≈ 360,000 sequences for training
```

---

## 2. Key Concepts You Need to Understand First

Read this section carefully before writing any code. Understanding these ideas will make everything else much easier.

### What Is a Neural Network?

A neural network is a mathematical function that learns patterns from examples. You show it thousands of examples (inputs + correct answers), and it gradually adjusts its internal settings (called **weights**) until it gets good at predicting the correct answer for new, unseen inputs.

Think of it like a student studying flashcards. The more flashcards they study, and the more times they review them, the better they get at answering questions they have never seen before.

### What Is an LSTM?

LSTM stands for **Long Short-Term Memory**. It is a special type of neural network designed for **sequences** — data where order matters.

Stock prices are a sequence. What happened yesterday affects today. What happened 10 days ago might still be relevant. A regular neural network treats each input independently. An LSTM remembers what it saw earlier in the sequence and uses that memory when processing later steps.

Think of it like reading a book. You do not forget the beginning of a chapter when you reach the end. An LSTM works the same way — it carries a "memory" through each step of the sequence.

### What Is Attention?

Attention is an add-on to the LSTM that lets the model decide **which days in the past were most important** for making its prediction. Instead of treating all 30 days equally, the attention layer learns to give more weight to some days and less weight to others.

This is very useful for your academic paper because you can visualize these weights and say: "The model paid the most attention to days 3 and 7 before the prediction date." This makes your model interpretable, not just a black box.

### What Is a Feature?

A feature is one piece of information that you feed into the model. Examples of features in your project:
- The stock's 1-day return (how much did it move yesterday?)
- The RSI (a technical indicator measuring whether a stock is overbought or oversold)
- The ESG environment score

The model learns which features are most useful for prediction.

### What Is Data Leakage? (Very Important)

Data leakage happens when your model accidentally gets access to information from the future during training. This makes results look much better than they really are — and it is the most common and most serious mistake in student machine learning projects.

**Example of leakage:** If you shuffle your data randomly before splitting it into train and test sets, some future data ends up in the training set. The model learns from the future, which is impossible in real life.

**The rule:** Always split time-series data by time. Train on earlier dates, test on later dates. Never shuffle.

### What Is Overfitting?

Overfitting happens when the model memorizes the training data instead of learning general patterns. It performs very well on training data but poorly on new data it has never seen.

**Analogy:** A student who memorizes all the practice exam questions word-for-word but cannot answer any question that is phrased differently has overfit the practice exams.

We prevent overfitting using:
- **Dropout** — randomly disabling some neurons during training to prevent memorization
- **Early stopping** — stopping training when performance on the test set stops improving

### What Is Class Imbalance?

If 65% of your labels are "up" and 35% are "down", the dataset is imbalanced. A model can achieve 65% accuracy by simply always predicting "up" — without learning anything meaningful.

We check for this in Phase 4 and correct for it in Phase 6.

---

## 3. Environment Setup

### Step 1 — Install Python

If you do not have Python installed, download it from [python.org](https://www.python.org/downloads/). Install version 3.10 or higher.

During installation on Windows, check the box that says **"Add Python to PATH"**. This is important.

### Step 2 — Install a Code Editor

Download and install [Visual Studio Code](https://code.visualstudio.com/) — it is free and beginner-friendly.

Alternatively, you can use **Jupyter Lab** (installed in Step 3), which lets you run code one block at a time in your browser. This is recommended for beginners.

### Step 3 — Install Required Libraries

Open your terminal (Mac/Linux) or Command Prompt (Windows) and run this command:

```bash
pip install pandas numpy scikit-learn torch yfinance shap matplotlib seaborn jupyterlab tqdm
```

**What each library does:**

| Library | Purpose |
|---|---|
| `pandas` | Loading and manipulating data tables (like Excel in Python) |
| `numpy` | Fast numerical calculations and array handling |
| `scikit-learn` | Data normalization, metrics, and helper tools |
| `torch` | PyTorch — the deep learning framework we use to build and train the LSTM |
| `yfinance` | Downloads free historical stock price data from Yahoo Finance |
| `shap` | Explains which features the model found most important |
| `matplotlib` | Creates graphs and charts |
| `seaborn` | Makes nicer-looking statistical charts |
| `jupyterlab` | A browser-based coding environment — run code block by block |
| `tqdm` | Shows progress bars so you can see how long downloads are taking |

### Step 4 — Start Jupyter Lab

In your terminal, run:

```bash
jupyter lab
```

A browser window will open. This is where you will write and run all your code.

### Step 5 — Create the Project Folder Structure

Create these folders on your computer. You can do this manually or run this Python code in a Jupyter cell:

```python
import os

folders = [
    "data/raw/prices",
    "data/processed",
    "notebooks",
    "src",
    "results/figures"
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)
    print(f"Created: {folder}")

print("\nAll folders created successfully.")
```

Then copy your `data.csv` file into the `data/raw/` folder and rename it `esg_scores.csv`.

Your project should look like this:

```
esg_stock_prediction/
│
├── data/
│   ├── raw/
│   │   ├── esg_scores.csv          ← your uploaded CSV goes here
│   │   └── prices/                 ← downloaded price files will go here
│   └── processed/
│       ├── features.csv            ← merged dataset (created in Phase 5)
│       └── sequences.npz           ← model-ready arrays (created in Phase 5)
│
├── notebooks/                      ← your Jupyter notebooks go here
│
├── src/                            ← reusable Python code goes here
│
└── results/
    └── figures/                    ← all charts and plots go here
```

### Step 6 — Fix Your Random Seeds

Put this at the top of every notebook. It makes your results reproducible — running the code twice gives the same result.

```python
import random
import numpy as np
import torch

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print("Random seeds set. Results will be reproducible.")
```

---

## 4. Data Diagnostics — Understanding What You Have

**Do not skip this phase.** Before writing any model code, you need to fully understand your data. Surprises discovered here — like severe class imbalance or a dominant industry — can change your entire approach.

Create a new notebook called `01_data_diagnostics.ipynb` and run each step below.

### Step 1 — Load the ESG Data

```python
import pandas as pd

# Load the CSV
esg = pd.read_csv("data/raw/esg_scores.csv")

# Basic overview
print("Shape (rows, columns):", esg.shape)
print("\nColumn names:")
print(esg.columns.tolist())
print("\nFirst 3 rows:")
esg.head(3)
```

You should see approximately 700 rows and 21 columns.

### Step 2 — Check for Missing Values

```python
print("Missing values per column:")
print(esg.isnull().sum())

print("\nMissing percentage:")
print((esg.isnull().sum() / len(esg) * 100).round(2))
```

**What to look for:** If `environment_score`, `social_score`, or `governance_score` have many missing values, those rows will need to be dropped. A small number of missing values (under 5%) is acceptable.

### Step 3 — Understand the ESG Score Distribution

```python
import matplotlib.pyplot as plt

score_cols = ["environment_score", "social_score", "governance_score", "total_score"]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for i, col in enumerate(score_cols):
    axes[i].hist(esg[col].dropna(), bins=30, color="steelblue", edgecolor="white")
    axes[i].set_title(col)
    axes[i].set_xlabel("Score")
    axes[i].set_ylabel("Number of stocks")

plt.suptitle("Distribution of ESG Scores Across All Stocks", fontsize=14)
plt.tight_layout()
plt.savefig("results/figures/esg_score_distributions.png", dpi=150)
plt.show()

print("\nSummary statistics:")
print(esg[score_cols].describe())
```

**What to look for:** Are scores clustered in a narrow range? Are there extreme outliers? This tells you whether normalization will be straightforward or whether outliers need special handling.

### Step 4 — Check Grade Distribution

```python
grade_cols = ["environment_grade", "social_grade", "governance_grade", "total_grade"]

for col in grade_cols:
    print(f"\n{col}:")
    print(esg[col].value_counts().sort_index())
```

**What to look for:** If most stocks have grade "B" and very few have "AA" or "AAA", the higher grades carry less statistical weight. This is normal — note it in your paper.

### Step 5 — Check Industry Concentration

This is important. If your dataset is dominated by one industry, your model may learn industry patterns rather than ESG signal.

```python
industry_counts = esg["industry"].value_counts()

print("Top 15 industries in your dataset:")
print(industry_counts.head(15))

plt.figure(figsize=(12, 5))
industry_counts.head(15).plot(kind="bar", color="steelblue")
plt.title("Number of stocks per industry (top 15)")
plt.xlabel("Industry")
plt.ylabel("Count")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("results/figures/industry_distribution.png", dpi=150)
plt.show()
```

**What to look for:** If one industry makes up more than 30% of your dataset, mention it as a potential bias in your paper. You do not need to fix it, just acknowledge it.

### Step 6 — Check Exchange Distribution

```python
print("Exchange breakdown:")
print(esg["exchange"].value_counts())
```

Your data includes both NYSE and NASDAQ stocks. This is fine — both are major US exchanges and widely covered in academic literature.

### Step 7 — Test Download a Single Ticker

Before downloading all 700 tickers, test with one to make sure it works.

```python
import yfinance as yf

test_ticker = "AAPL"
test_data = yf.download(test_ticker, start="2020-01-01", end="2023-12-31", progress=False)

print(f"Downloaded {len(test_data)} rows for {test_ticker}")
print("\nColumns available:")
print(test_data.columns.tolist())
print("\nFirst 3 rows:")
print(test_data.head(3))
```

You should see columns: `Open`, `High`, `Low`, `Close`, `Adj Close`, `Volume`. If this works, you are ready to download all tickers.

### Step 8 — Download All Ticker Prices and Check Survival Rate

This step downloads historical price data for every ticker in your ESG dataset. It may take 20–40 minutes. The progress bar will show you how far along you are.

```python
import yfinance as yf
import os
from tqdm import tqdm

os.makedirs("data/raw/prices", exist_ok=True)

tickers = esg["ticker"].str.upper().tolist()

failed_tickers     = []
too_short_tickers  = []
successful_tickers = []

for ticker in tqdm(tickers, desc="Downloading price data"):
    try:
        df = yf.download(ticker, start="2020-01-01", end="2023-12-31",
                         progress=False, auto_adjust=True)

        if len(df) < 200:
            # Less than 200 trading days is not enough for reliable sequences
            too_short_tickers.append(ticker)
            continue

        df.to_csv(f"data/raw/prices/{ticker}.csv")
        successful_tickers.append(ticker)

    except Exception as e:
        failed_tickers.append(ticker)

print(f"\n--- Download Summary ---")
print(f"Successfully downloaded : {len(successful_tickers)} tickers")
print(f"Too short (< 200 days)  : {len(too_short_tickers)} tickers")
print(f"Failed entirely         : {len(failed_tickers)} tickers")
```

**What to expect:** Roughly 400–550 tickers will succeed. Failures happen because stocks were delisted, merged, or had their ticker changed since 2022. Record these numbers for your paper.

### Step 9 — Preview Class Balance

Before building the full dataset, preview the label balance using one stock.

```python
FORECAST_HORIZON = 5  # We predict 5 days ahead

aapl = pd.read_csv("data/raw/prices/AAPL.csv", index_col=0, parse_dates=True)
aapl["future_close"] = aapl["Close"].shift(-FORECAST_HORIZON)
aapl["label"] = (aapl["future_close"] > aapl["Close"]).astype(int)
aapl = aapl.dropna(subset=["future_close"])

counts = aapl["label"].value_counts()
print("Label distribution for AAPL:")
print(f"  Up (1)  : {counts.get(1, 0)} days ({counts.get(1, 0)/len(aapl)*100:.1f}%)")
print(f"  Down (0): {counts.get(0, 0)} days ({counts.get(0, 0)/len(aapl)*100:.1f}%)")
```

**What to look for:** If the split is more than 60/40, you have a class imbalance problem. We handle this in Phase 6 using a weighted loss function.

### Diagnostics Summary Checklist

After completing this phase, write the answers to these questions in a plain text file called `data_notes.txt`. You will need them when writing your paper.

- [ ] How many stocks are in your ESG dataset?
- [ ] How many tickers successfully downloaded price data?
- [ ] What is the most common industry in your dataset?
- [ ] What is the approximate up/down label split?
- [ ] Are there any columns with significant missing values?

---

## 5. Data Pre-Processing

Create a new notebook called `02_preprocessing.ipynb`.

**Why pre-processing matters:** A model is only as good as the data it learns from. Dirty, unnormalized, or poorly structured data produces unreliable results — no matter how sophisticated the model is.

### Step 1 — Load and Clean the ESG Data

```python
import pandas as pd
import numpy as np
import os

# Load the ESG data
esg = pd.read_csv("data/raw/esg_scores.csv")

# Keep only the columns we need
esg = esg[[
    "ticker", "name", "industry", "exchange",
    "environment_score", "social_score", "governance_score", "total_score",
    "environment_grade", "social_grade", "governance_grade", "total_grade"
]]

# Standardize ticker to uppercase with no spaces
esg["ticker"] = esg["ticker"].str.upper().str.strip()

# Drop rows where any score is missing
esg = esg.dropna(subset=["environment_score", "social_score",
                          "governance_score", "total_score"])

print(f"ESG dataset after cleaning: {esg.shape[0]} stocks")
```

### Step 2 — Encode Letter Grades as Numbers

The grades (B, BB, BBB, A, AA, AAA) have a natural order — AA is better than BB. We convert them to numbers so the model can use them mathematically.

```python
# Define the mapping from grade to number
# Higher number = better ESG grade
grade_map = {
    "B"   : 1,
    "BB"  : 2,
    "BBB" : 3,
    "A"   : 4,
    "AA"  : 5,
    "AAA" : 6
}

grade_cols = ["environment_grade", "social_grade", "governance_grade", "total_grade"]

for col in grade_cols:
    esg[col + "_encoded"] = esg[col].map(grade_map)

    # Check if any grades did not map (unexpected values in the data)
    unmapped = esg[col + "_encoded"].isnull().sum()
    if unmapped > 0:
        print(f"Warning: {unmapped} rows in '{col}' could not be mapped.")
        print(f"  Unexpected values found: {esg.loc[esg[col + '_encoded'].isnull(), col].unique()}")

print("\nGrade encoding complete.")
print(esg[["ticker", "environment_grade", "environment_grade_encoded"]].head(5))
```

### Step 3 — Normalize ESG Scores to a 0–1 Range

Raw ESG scores (e.g., 200–650) are much larger numbers than the price-based features we will compute (which will be small decimals like 0.02 for a 2% return). If we do not normalize, the large ESG numbers will dominate the model unfairly.

Normalization rescales everything to the same range (0 to 1).

```python
from sklearn.preprocessing import MinMaxScaler

score_cols = ["environment_score", "social_score", "governance_score", "total_score"]

# Fit the scaler on the ESG data and transform it
scaler_esg = MinMaxScaler()
esg[score_cols] = scaler_esg.fit_transform(esg[score_cols])

print("ESG scores after normalization (all values should be between 0 and 1):")
print(esg[score_cols].describe().round(3))
```

### Step 4 — Create the Prediction Label

The label tells the model what the correct answer is. We define it as: will the closing price be higher in 5 days compared to today?

```python
FORECAST_HORIZON = 5  # Days ahead we are predicting

def create_labels(price_df, horizon=FORECAST_HORIZON):
    """
    For each trading day, look HORIZON days into the future.
    If future price > today's price → label = 1 (Up)
    If future price <= today's price → label = 0 (Down or flat)

    shift(-5) moves the Close column 5 rows upward, so on the row
    representing "today", future_close shows the price 5 days from now.
    """
    price_df = price_df.copy()
    price_df["future_close"] = price_df["Close"].shift(-horizon)
    price_df["label"] = (price_df["future_close"] > price_df["Close"]).astype(int)

    # Remove the last HORIZON rows — they have no future price yet
    price_df = price_df.dropna(subset=["future_close"])
    return price_df
```

### Step 5 — Compute Technical Price Features

These features summarize the stock's recent behavior using standard indicators from finance. Every feature uses only past data — no future information is used here.

```python
def add_technical_features(df):
    """
    Compute standard technical indicators from OHLCV price data.
    """
    df = df.copy()

    # --- Returns: How much did the price change? ---
    # pct_change(N) = (today - N days ago) / N days ago
    df["return_1d"]  = df["Close"].pct_change(1)
    df["return_5d"]  = df["Close"].pct_change(5)
    df["return_20d"] = df["Close"].pct_change(20)

    # --- Moving Averages: Trend direction ---
    # A 10-day average above a 30-day average = short-term trend is up (bullish signal)
    # A 10-day average below a 30-day average = short-term trend is down (bearish signal)
    df["ma_10"]    = df["Close"].rolling(window=10).mean()
    df["ma_30"]    = df["Close"].rolling(window=30).mean()
    df["ma_ratio"] = df["ma_10"] / df["ma_30"]

    # --- Volatility: How uncertain/risky is this stock right now? ---
    df["volatility_20d"] = df["return_1d"].rolling(window=20).std()

    # --- RSI (Relative Strength Index): Overbought or oversold? ---
    # RSI > 70 = stock may be overbought (possibly due for a fall)
    # RSI < 30 = stock may be oversold (possibly due for a rise)
    delta = df["Close"].diff()
    gain  = delta.clip(lower=0).rolling(window=14).mean()
    loss  = (-delta.clip(upper=0)).rolling(window=14).mean()
    rs    = gain / (loss + 1e-8)  # Add tiny number to prevent division by zero
    df["rsi"] = 100 - (100 / (1 + rs))

    # --- Volume Change: Is trading activity unusually high or low? ---
    df["volume_change"] = df["Volume"].pct_change(1)

    # --- Price relative to 52-week high and low ---
    df["price_vs_52w_high"] = df["Close"] / df["Close"].rolling(252).max()
    df["price_vs_52w_low"]  = df["Close"] / df["Close"].rolling(252).min()

    return df
```

### Step 6 — Merge ESG Data with Price Data

This combines your two data sources into one master dataset. Each row in the final dataset represents one trading day for one stock, with both price features and ESG features attached.

```python
# Define which columns are ESG features
ESG_FEATURE_COLS = [
    "environment_score", "social_score", "governance_score", "total_score",
    "environment_grade_encoded", "social_grade_encoded",
    "governance_grade_encoded", "total_grade_encoded"
]

all_stocks = []
skipped    = []

# Get list of tickers that were successfully downloaded
available_tickers = [
    f.replace(".csv", "")
    for f in os.listdir("data/raw/prices")
    if f.endswith(".csv")
]

print(f"Price files available: {len(available_tickers)}")

for ticker in available_tickers:
    # Check if this ticker exists in our ESG data
    esg_match = esg[esg["ticker"] == ticker]
    if esg_match.empty:
        skipped.append(ticker)
        continue

    # Load price data
    price_path = f"data/raw/prices/{ticker}.csv"
    prices = pd.read_csv(price_path, index_col=0, parse_dates=True)

    # Drop any price rows with missing values
    prices = prices.dropna(subset=["Close", "Volume"])

    # Need at least 252 rows (1 trading year) for features to be meaningful
    if len(prices) < 252:
        skipped.append(ticker)
        continue

    # Add technical features
    prices = add_technical_features(prices)

    # Add prediction labels
    prices = create_labels(prices)

    # Attach the stock's ESG scores to every row
    # Note: ESG is a static snapshot — same value for all dates for this stock
    esg_row = esg_match.iloc[0]
    for col in ESG_FEATURE_COLS:
        prices[col] = esg_row[col]

    prices["industry"] = esg_row["industry"]
    prices["ticker"]   = ticker

    all_stocks.append(prices)

# Combine all stocks into one large dataframe
master_df = pd.concat(all_stocks, axis=0)

# Drop rows with NaN values (these arise from rolling window warmup)
master_df = master_df.dropna()

master_df.to_csv("data/processed/features.csv")

print(f"\nMaster dataset shape : {master_df.shape}")
print(f"Total stocks included: {len(all_stocks)}")
print(f"Stocks skipped       : {len(skipped)}")
print(f"\nLabel distribution:")
print(master_df["label"].value_counts(normalize=True).round(3))
```

**Understanding the output:** The label distribution printout tells you your class balance. `1: 0.53, 0: 0.47` means 53% of your examples are "up" days — reasonably balanced. If it is more extreme than 60/40, note it down. We handle it in Phase 6.

### Step 7 — Compute and Save Class Weight

```python
import json

label_counts = master_df["label"].value_counts()
num_down = label_counts[0]
num_up   = label_counts[1]

print(f"Up   labels: {num_up:,}")
print(f"Down labels: {num_down:,}")

# pos_weight > 1.0 means there are more "down" days — we penalize the model
# more for missing "up" predictions to compensate
pos_weight = num_down / num_up
print(f"\npos_weight: {pos_weight:.3f}")
print("(Values near 1.0 = balanced. Values > 1.3 = imbalanced, use weighted loss.)")

with open("data/processed/class_weight.json", "w") as f:
    json.dump({"pos_weight": pos_weight}, f)
```

### Step 8 — Build Sequences for the LSTM

The LSTM expects input in the shape `(batch_size, sequence_length, num_features)`. We convert our flat table into overlapping 30-day windows.

```python
LOOKBACK = 30  # Use 30 past trading days to predict the next direction

# Define which columns are model features — order matters and must be consistent
FEATURE_COLS = [
    # Price-based features (9 total)
    "return_1d", "return_5d", "return_20d",
    "ma_ratio", "volatility_20d", "rsi",
    "volume_change", "price_vs_52w_high", "price_vs_52w_low",
    # ESG features (8 total)
    "environment_score", "social_score", "governance_score", "total_score",
    "environment_grade_encoded", "social_grade_encoded",
    "governance_grade_encoded", "total_grade_encoded"
]

# Save the feature list — we will need this in later notebooks
with open("data/processed/feature_cols.json", "w") as f:
    json.dump(FEATURE_COLS, f)

def build_sequences(df, feature_cols, lookback=LOOKBACK):
    """
    Convert a flat time-series dataframe into (X, y) sequence pairs.

    For each day i (starting from day 30):
      X[i] = feature values on days i-30 through i-1  → shape: (30, num_features)
      y[i] = label on day i (0 or 1)

    This is like sliding a 30-day window across the data, one day at a time.
    """
    X_list = []
    y_list = []

    feature_values = df[feature_cols].values   # Shape: (total_days, num_features)
    label_values   = df["label"].values         # Shape: (total_days,)

    for i in range(lookback, len(feature_values)):
        X_list.append(feature_values[i - lookback : i])  # 30-day window
        y_list.append(label_values[i])                    # Label for day i

    return np.array(X_list), np.array(y_list)

# Build sequences per ticker (so windows never cross from one stock to another)
X_list_all = []
y_list_all = []

for ticker, group in master_df.groupby("ticker"):
    group_sorted = group.sort_index()  # Ensure chronological order!
    X_t, y_t = build_sequences(group_sorted, FEATURE_COLS)
    X_list_all.append(X_t)
    y_list_all.append(y_t)

X_all = np.concatenate(X_list_all, axis=0)
y_all = np.concatenate(y_list_all, axis=0)

print(f"Total sequences: {len(X_all):,}")
print(f"Sequence shape : {X_all.shape}  → (num_sequences, lookback_days, num_features)")
```

### Step 9 — Split Into Training and Test Sets (Time-Based)

```python
# Use 80% for training, 20% for testing
# CRITICAL: We split by position (time order), NEVER randomly
split_idx = int(len(X_all) * 0.80)

X_train = X_all[:split_idx]
y_train = y_all[:split_idx]
X_test  = X_all[split_idx:]
y_test  = y_all[split_idx:]

print(f"Training sequences: {X_train.shape[0]:,}")
print(f"Test sequences    : {X_test.shape[0]:,}")
print(f"\nTraining label balance: {y_train.mean():.3f} (proportion of 'up' days)")
print(f"Test label balance    : {y_test.mean():.3f}")

# Save the raw (unnormalized) sequences first
np.savez(
    "data/processed/sequences.npz",
    X_train=X_train, y_train=y_train,
    X_test=X_test,   y_test=y_test
)
```

### Step 10 — Normalize the Price-Based Features

Price features (returns, RSI, etc.) vary in scale. We normalize them using training data statistics only — then apply those same statistics to the test data.

**Critical rule:** Fit the scaler on training data only. If you fit on all data, test statistics contaminate training normalization — another form of data leakage.

```python
from sklearn.preprocessing import StandardScaler

# Reshape: (sequences, timesteps, features) → (sequences × timesteps, features)
n_train, T, F = X_train.shape
n_test        = X_test.shape[0]

X_train_2d = X_train.reshape(-1, F)
X_test_2d  = X_test.reshape(-1, F)

# Only normalize the price-based features (first 9 columns)
# ESG features (columns 9–16) were already normalized in Step 3
PRICE_FEATURE_INDICES = list(range(9))

scaler_price = StandardScaler()

# Fit on TRAINING data only — then apply to both train and test
X_train_2d[:, PRICE_FEATURE_INDICES] = scaler_price.fit_transform(
    X_train_2d[:, PRICE_FEATURE_INDICES]
)
X_test_2d[:, PRICE_FEATURE_INDICES]  = scaler_price.transform(
    X_test_2d[:, PRICE_FEATURE_INDICES]
)

# Reshape back to 3D
X_train = X_train_2d.reshape(n_train, T, F)
X_test  = X_test_2d.reshape(n_test,  T, F)

# Save the normalized version — this is what the model will use
np.savez(
    "data/processed/sequences_normalized.npz",
    X_train=X_train, y_train=y_train,
    X_test=X_test,   y_test=y_test
)

print("Normalized sequences saved to data/processed/sequences_normalized.npz")
print("\nPre-processing complete. You are ready for model training.")
```

---

## 6. Model Training

Create a new notebook called `03_model_training.ipynb`. Start with the seed-setting code from Phase 3 Step 6.

### Step 1 — Load the Processed Data

```python
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
import matplotlib.pyplot as plt
import os

os.makedirs("results/figures", exist_ok=True)

# Load sequences
data = np.load("data/processed/sequences_normalized.npz")
X_train = data["X_train"]
y_train = data["y_train"]
X_test  = data["X_test"]
y_test  = data["y_test"]

# Load feature list and class weight
with open("data/processed/feature_cols.json") as f:
    FEATURE_COLS = json.load(f)

with open("data/processed/class_weight.json") as f:
    pos_weight = json.load(f)["pos_weight"]

INPUT_SIZE = len(FEATURE_COLS)
LOOKBACK   = X_train.shape[1]

print(f"Training sequences : {X_train.shape[0]:,}")
print(f"Test sequences     : {X_test.shape[0]:,}")
print(f"Sequence length    : {LOOKBACK} days")
print(f"Number of features : {INPUT_SIZE}")
print(f"Class weight (pos) : {pos_weight:.3f}")
```

### Step 2 — Create a PyTorch Dataset

PyTorch expects data to be wrapped in a `Dataset` object. Think of it as a container PyTorch knows how to read from in batches.

```python
class StockSequenceDataset(Dataset):
    """
    A PyTorch Dataset that holds our sequence data.
    PyTorch calls __getitem__ to fetch one sample at a time during training.
    """
    def __init__(self, X, y):
        # Convert numpy arrays to PyTorch tensors (the format PyTorch works with)
        self.X = torch.FloatTensor(X)  # Shape: (num_sequences, lookback, num_features)
        self.y = torch.FloatTensor(y)  # Shape: (num_sequences,)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Return one sample (a sequence + its label) by index
        return self.X[idx], self.y[idx]


# Create Dataset objects for training and testing
train_dataset = StockSequenceDataset(X_train, y_train)
test_dataset  = StockSequenceDataset(X_test,  y_test)

# DataLoaders feed data to the model in batches of 64
# shuffle=False is critical for time-series — we must preserve order
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=64, shuffle=False)

print(f"Training batches: {len(train_loader)}")
print(f"Test batches    : {len(test_loader)}")
```

### Step 3 — Build the Attention Layer

The attention layer learns to assign importance weights to each of the 30 time steps.

```python
class AttentionLayer(nn.Module):
    """
    Self-attention over LSTM outputs.

    Given the LSTM's hidden state at each of the 30 timesteps,
    this layer learns a single weight (importance score) per timestep.
    Higher weight = the model thinks that day was more important.
    The final output is a weighted sum of all timestep hidden states.
    """
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        # A linear layer that maps each hidden state to a single score
        self.attention_weights = nn.Linear(hidden_size, 1)

    def forward(self, lstm_output):
        # lstm_output shape: (batch_size, seq_len=30, hidden_size=64)

        # Compute a raw score for each of the 30 timesteps
        scores = self.attention_weights(lstm_output)   # (batch, 30, 1)

        # Softmax converts raw scores to probabilities that sum to 1.0
        weights = torch.softmax(scores, dim=1)         # (batch, 30, 1)

        # Weighted sum: multiply each timestep's hidden state by its weight, then sum
        # This collapses the 30 timesteps into one summary vector
        context = (weights * lstm_output).sum(dim=1)   # (batch, 64)

        return context, weights
```

### Step 4 — Build the Full LSTM + Attention Model

```python
class ESGStockPredictor(nn.Module):
    """
    Full model architecture:
      1. LSTM layers     → process the 30-day sequence, extract temporal patterns
      2. Attention layer → weight the most informative timesteps
      3. Dropout         → randomly zero some values to prevent overfitting
      4. Linear layer    → map from 64 hidden units to 1 output value
      5. Sigmoid         → squash output to a probability between 0 and 1
    """
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3):
        super(ESGStockPredictor, self).__init__()

        self.lstm = nn.LSTM(
            input_size  = input_size,   # Number of features per day (17)
            hidden_size = hidden_size,  # Size of the LSTM's memory (64 units)
            num_layers  = num_layers,   # Stack 2 LSTM layers on top of each other
            batch_first = True,         # Input shape: (batch, seq, features)
            dropout     = dropout       # Dropout applied between stacked LSTM layers
        )

        self.attention = AttentionLayer(hidden_size)
        self.dropout   = nn.Dropout(dropout)
        self.fc        = nn.Linear(hidden_size, 1)  # 64 hidden → 1 output
        self.sigmoid   = nn.Sigmoid()               # Map to [0, 1] probability

    def forward(self, x):
        # x shape: (batch_size, 30, num_features)

        lstm_out, _ = self.lstm(x)
        # lstm_out shape: (batch_size, 30, 64)

        context, attn_weights = self.attention(lstm_out)
        # context shape: (batch_size, 64)

        out = self.dropout(context)
        out = self.fc(out)             # (batch_size, 1)
        out = self.sigmoid(out)        # (batch_size, 1) — probability of "up"

        return out.squeeze(-1), attn_weights   # Return prediction and attention weights


# Count total parameters — useful to report in your paper
model_test   = ESGStockPredictor(INPUT_SIZE)
total_params = sum(p.numel() for p in model_test.parameters())
print(f"Total model parameters: {total_params:,}")
```

### Step 5 — Set Hyperparameters

Hyperparameters are settings you choose before training. They control how the model is structured and how fast it learns.

```python
# --- Model architecture ---
HIDDEN_SIZE = 64    # Number of memory units in each LSTM layer
NUM_LAYERS  = 2     # How many LSTM layers to stack
DROPOUT     = 0.3   # Fraction of neurons randomly dropped during training

# --- Training settings ---
EPOCHS     = 60     # Maximum number of full passes through the training data
BATCH_SIZE = 64     # How many sequences to process at once
LR         = 0.001  # Learning rate: step size for the optimizer each update
PATIENCE   = 8      # Early stopping: stop after this many epochs without improvement

# --- Hardware ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")
# If you have an NVIDIA GPU: 'cuda' — training will be faster
# If not: 'cpu' — fine for this project, just a bit slower
```

**Understanding each hyperparameter:**

`HIDDEN_SIZE = 64` — Each LSTM layer has 64 memory units. More units = more capacity to learn complex patterns, but also more risk of overfitting and slower training. 64 is a safe starting point.

`NUM_LAYERS = 2` — Two stacked LSTM layers. The second layer receives the output of the first and learns more abstract patterns. Using more than 3 layers rarely helps for this type of task.

`DROPOUT = 0.3` — During each training step, 30% of neurons are randomly switched off. This prevents the model from memorizing specific training examples, forcing it to learn more general patterns.

`LR = 0.001` — The learning rate controls how large a step the optimizer takes when updating weights. Too high = training is noisy and unstable. Too low = training is very slow. 0.001 is the standard starting value for Adam.

`PATIENCE = 8` — If the test loss has not improved for 8 consecutive epochs, we stop training and keep the best model seen so far.

### Step 6 — Initialize the Model, Loss, and Optimizer

```python
os.makedirs("results", exist_ok=True)

# Create the model and move it to the correct device (CPU or GPU)
model = ESGStockPredictor(
    input_size  = INPUT_SIZE,
    hidden_size = HIDDEN_SIZE,
    num_layers  = NUM_LAYERS,
    dropout     = DROPOUT
).to(device)

# Loss function: measures how wrong the model's predictions are
# BCELoss = Binary Cross-Entropy Loss, standard for binary classification
criterion = nn.BCELoss()

# If your class imbalance ratio (pos_weight) was > 1.3, use this instead:
# It penalizes the model more for errors on the minority class
# criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))
# (If using BCEWithLogitsLoss, remove the sigmoid from the model's forward() method)

# Optimizer: Adam updates model weights after each batch
# It adapts the learning rate per parameter — better than basic gradient descent
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# Learning rate scheduler: automatically reduces LR when validation loss plateaus
# factor=0.5 means LR is halved; patience=4 means wait 4 epochs before reducing
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=4, verbose=True
)

print("Model, loss function, and optimizer are ready.")
print(f"Total trainable parameters: {sum(p.numel() for p in model.parameters()):,}")
```

### Step 7 — The Training Loop

This is the core of model training. Each epoch is one full pass through all training data.

```python
best_val_loss    = float("inf")   # Track the lowest validation loss seen
patience_counter = 0              # Count epochs without improvement
train_losses     = []             # Store training loss history
val_losses       = []             # Store validation loss history

print("Starting training...\n")
print(f"{'Epoch':>6}  {'Train Loss':>12}  {'Val Loss':>10}")
print("-" * 35)

for epoch in range(1, EPOCHS + 1):

    # ============================
    # TRAINING PHASE
    # ============================
    model.train()  # Enable dropout (training mode)
    epoch_train_loss = 0.0

    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        # Step 1: Clear gradients from previous batch
        # (Gradients accumulate by default — we must clear them each step)
        optimizer.zero_grad()

        # Step 2: Forward pass — feed data through the model
        predictions, _ = model(X_batch)

        # Step 3: Compute the loss (how wrong the predictions are)
        loss = criterion(predictions, y_batch)

        # Step 4: Backward pass — compute gradients (directions to improve weights)
        loss.backward()

        # Step 5: Gradient clipping — prevents exploding gradients (common in RNNs)
        # This caps gradient magnitudes so training stays stable
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Step 6: Update model weights using the computed gradients
        optimizer.step()

        epoch_train_loss += loss.item()

    avg_train_loss = epoch_train_loss / len(train_loader)

    # ============================
    # VALIDATION PHASE
    # ============================
    model.eval()   # Disable dropout (evaluation mode)
    epoch_val_loss = 0.0

    with torch.no_grad():  # No gradient calculation needed — saves memory and time
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            predictions, _ = model(X_batch)
            loss = criterion(predictions, y_batch)
            epoch_val_loss += loss.item()

    avg_val_loss = epoch_val_loss / len(test_loader)

    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)

    # Adjust learning rate if validation loss has plateaued
    scheduler.step(avg_val_loss)

    # Print progress every 5 epochs
    if epoch % 5 == 0 or epoch == 1:
        print(f"{epoch:>6}  {avg_train_loss:>12.4f}  {avg_val_loss:>10.4f}")

    # ============================
    # EARLY STOPPING CHECK
    # ============================
    if avg_val_loss < best_val_loss:
        best_val_loss    = avg_val_loss
        patience_counter = 0
        # Save the best model weights seen so far
        torch.save(model.state_dict(), "results/best_model_esg.pt")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping triggered at epoch {epoch}.")
            print(f"Best validation loss: {best_val_loss:.4f}")
            break

print("\nTraining complete. Best model saved to: results/best_model_esg.pt")
```

### Step 8 — Plot Training Curves

Always plot training vs validation loss. This is a key diagnostic figure for your paper.

```python
plt.figure(figsize=(10, 4))
plt.plot(train_losses, label="Training loss",   color="steelblue", linewidth=2)
plt.plot(val_losses,   label="Validation loss", color="coral",     linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Binary Cross-Entropy Loss")
plt.title("Training vs Validation Loss — ESG + Price Model")
plt.legend()
plt.tight_layout()
plt.savefig("results/figures/training_curves_esg.png", dpi=150)
plt.show()
```

**What to look for in this plot:**

Both curves trending downward together = the model is learning correctly.

Validation loss rising while training loss keeps falling = overfitting. Try increasing dropout from 0.3 to 0.4, or reducing HIDDEN_SIZE from 64 to 32.

Both curves flat from the start = learning rate might be too low, or data has a normalization issue. Check that your sequences do not contain NaN values: `print(np.isnan(X_train).sum())`.

### Step 9 — Train the Baseline Model (Price Features Only)

This step is essential for your academic paper. The baseline uses the same architecture but receives no ESG features. The comparison isolates ESG's contribution.

```python
# Select only price-based feature indices (first 9 columns)
PRICE_ONLY_COLS    = FEATURE_COLS[:9]  # The 9 price features
PRICE_ONLY_INDICES = list(range(9))
INPUT_SIZE_BASELINE = len(PRICE_ONLY_COLS)

print(f"Baseline model features ({INPUT_SIZE_BASELINE}): {PRICE_ONLY_COLS}")

# Extract only price columns from normalized sequences
X_train_price = X_train[:, :, PRICE_ONLY_INDICES]
X_test_price  = X_test[:,  :, PRICE_ONLY_INDICES]

# Create datasets and loaders for the baseline
train_loader_bl = DataLoader(
    StockSequenceDataset(X_train_price, y_train), batch_size=64, shuffle=False
)
test_loader_bl  = DataLoader(
    StockSequenceDataset(X_test_price,  y_test),  batch_size=64, shuffle=False
)

# Identical architecture — only input_size is smaller
model_baseline = ESGStockPredictor(
    input_size  = INPUT_SIZE_BASELINE,
    hidden_size = HIDDEN_SIZE,
    num_layers  = NUM_LAYERS,
    dropout     = DROPOUT
).to(device)

optimizer_baseline = torch.optim.Adam(model_baseline.parameters(), lr=LR)
scheduler_baseline = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_baseline, mode="min", factor=0.5, patience=4
)

# Run the same training loop — copy Step 7 exactly,
# replacing: model → model_baseline
#            optimizer → optimizer_baseline
#            scheduler → scheduler_baseline
#            train_loader → train_loader_bl
#            test_loader  → test_loader_bl
#            save path → "results/best_model_baseline.pt"

print("\nRun the same training loop as Step 7 for the baseline model.")
print("Save to: results/best_model_baseline.pt")
```

---

## 7. Evaluation

Create a new notebook called `04_evaluation.ipynb`.

### Step 1 — Load Both Models and Run Predictions

```python
import torch
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve
)
from torch.utils.data import DataLoader

# Load data
data   = np.load("data/processed/sequences_normalized.npz")
X_test = data["X_test"]
y_test = data["y_test"]

with open("data/processed/feature_cols.json") as f:
    FEATURE_COLS = json.load(f)

device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_SIZE = len(FEATURE_COLS)
LOOKBACK   = X_test.shape[1]

PRICE_ONLY_INDICES  = list(range(9))
INPUT_SIZE_BASELINE = 9

# Load ESG model
model_esg = ESGStockPredictor(INPUT_SIZE, hidden_size=64, num_layers=2, dropout=0.3).to(device)
model_esg.load_state_dict(torch.load("results/best_model_esg.pt", map_location=device))
model_esg.eval()

# Load baseline model
model_baseline = ESGStockPredictor(INPUT_SIZE_BASELINE, hidden_size=64, num_layers=2, dropout=0.3).to(device)
model_baseline.load_state_dict(torch.load("results/best_model_baseline.pt", map_location=device))
model_baseline.eval()

def get_predictions(model, X_np, device):
    """Run data through a model and return probabilities, binary predictions, and attention weights."""
    model.eval()
    all_probs = []
    all_attn  = []
    dataset   = StockSequenceDataset(X_np, np.zeros(len(X_np)))
    loader    = DataLoader(dataset, batch_size=256, shuffle=False)

    with torch.no_grad():
        for X_batch, _ in loader:
            probs, attn = model(X_batch.to(device))
            all_probs.append(probs.cpu().numpy())
            all_attn.append(attn.cpu().numpy())

    probs = np.concatenate(all_probs)
    preds = (probs > 0.5).astype(int)  # Convert probability to binary prediction
    attn  = np.concatenate(all_attn)
    return probs, preds, attn

# Get predictions from both models
probs_esg,  preds_esg,  attn_esg = get_predictions(model_esg, X_test, device)

X_test_price                      = X_test[:, :, PRICE_ONLY_INDICES]
probs_base, preds_base, _         = get_predictions(model_baseline, X_test_price, device)
```

### Step 2 — Compute and Compare All Metrics

```python
def print_metrics(name, y_true, y_pred, y_prob):
    """Compute and print accuracy, F1, and AUC-ROC."""
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)

    print(f"\n{'='*45}")
    print(f"  {name}")
    print(f"{'='*45}")
    print(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  AUC-ROC   : {auc:.4f}")
    print(f"\n{classification_report(y_true, y_pred, target_names=['Down (0)', 'Up (1)'])}")

    return {"accuracy": acc, "f1": f1, "auc": auc}

metrics_esg  = print_metrics("ESG + Price Model",    y_test, preds_esg,  probs_esg)
metrics_base = print_metrics("Price-Only Baseline",  y_test, preds_base, probs_base)

# Calculate the improvement
print(f"\n--- ESG Model Improvement Over Baseline ---")
print(f"  Accuracy : {(metrics_esg['accuracy'] - metrics_base['accuracy'])*100:+.2f} pp")
print(f"  F1 Score : {(metrics_esg['f1']       - metrics_base['f1'])*100:+.2f} pp")
print(f"  AUC-ROC  : {(metrics_esg['auc']      - metrics_base['auc'])*100:+.2f} pp")
print("  (pp = percentage points)")
```

**Interpreting these metrics:**

**Accuracy** — what percentage of predictions were correct. Can be misleading if classes are imbalanced. A model that always predicts "up" will get 53% accuracy on a 53/47 dataset without learning anything.

**F1 Score** — the harmonic mean of precision (when we predict "up", how often are we right?) and recall (of all the actual "up" days, how many did we catch?). This is a better metric than accuracy for imbalanced datasets.

**AUC-ROC** — the probability that the model ranks a random "up" day higher than a random "down" day. 0.5 = random guessing, 1.0 = perfect. In financial prediction, anything above 0.55 is considered meaningful. Even 0.52 is worth discussing in an academic paper, because the stock market is very hard to predict.

### Step 3 — Confusion Matrices

A confusion matrix shows the breakdown of correct and incorrect predictions in detail.

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, preds, name in zip(
    axes,
    [preds_base, preds_esg],
    ["Price-Only Baseline", "ESG + Price Model"]
):
    cm = confusion_matrix(y_test, preds)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=["Predicted Down", "Predicted Up"],
        yticklabels=["Actual Down",    "Actual Up"]
    )
    ax.set_title(name)
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")

plt.suptitle("Confusion Matrices: Baseline vs ESG Model", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig("results/figures/confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.show()
```

**Reading the confusion matrix:**
- Top-left: correctly predicted "down" days (True Negatives)
- Bottom-right: correctly predicted "up" days (True Positives)
- Top-right: predicted "up" but it went down (False Positives)
- Bottom-left: predicted "down" but it went up (False Negatives)

### Step 4 — ROC Curves

```python
fpr_esg,  tpr_esg,  _ = roc_curve(y_test, probs_esg)
fpr_base, tpr_base, _ = roc_curve(y_test, probs_base)

plt.figure(figsize=(7, 6))
plt.plot(fpr_esg,  tpr_esg,
         label=f"ESG + Price Model  (AUC = {metrics_esg['auc']:.3f})",
         color="steelblue", linewidth=2)
plt.plot(fpr_base, tpr_base,
         label=f"Price-Only Baseline (AUC = {metrics_base['auc']:.3f})",
         color="coral", linewidth=2, linestyle="--")
plt.plot([0, 1], [0, 1],
         color="gray", linestyle=":", label="Random guessing (AUC = 0.500)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison: ESG Model vs Baseline")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("results/figures/roc_curves.png", dpi=150)
plt.show()
```

### Step 5 — SHAP Feature Importance

SHAP values explain which features the model relied on most. This is one of the most important outputs for your paper — it shows whether ESG features genuinely influenced predictions.

```python
import shap

# We use a subset because SHAP is computationally slow on large datasets
SHAP_SAMPLE_SIZE = 200

def predict_fn(X_2d):
    """
    Wrapper function for SHAP.
    SHAP feeds flattened 2D input — we reshape it back to 3D for our model.
    """
    X_3d   = X_2d.reshape(-1, LOOKBACK, INPUT_SIZE)
    tensor = torch.FloatTensor(X_3d).to(device)
    with torch.no_grad():
        probs, _ = model_esg(tensor)
    return probs.cpu().numpy()

# Background sample: a small set of training examples SHAP uses as a reference
background = X_test[:100].reshape(100, -1)
explainer  = shap.KernelExplainer(predict_fn, background)

# Compute SHAP values for a sample of test examples
sample    = X_test[:SHAP_SAMPLE_SIZE].reshape(SHAP_SAMPLE_SIZE, -1)
shap_vals = explainer.shap_values(sample, nsamples=100)

# Feature names for the flattened input
flat_feature_names = [
    f"{col}_t-{LOOKBACK - i}"
    for i in range(LOOKBACK)
    for col in FEATURE_COLS
]

# Summary plot — shows which features pushed predictions up or down
shap.summary_plot(
    shap_vals, sample,
    feature_names=flat_feature_names,
    max_display=20,
    show=False
)
plt.tight_layout()
plt.savefig("results/figures/shap_summary.png", dpi=150, bbox_inches="tight")
plt.show()
```

**Reading the SHAP plot:**
- Features at the top had the most influence on predictions
- Red dots = high feature value; blue dots = low feature value
- Dots on the right = pushed the prediction toward "up"
- Dots on the left = pushed the prediction toward "down"

If ESG features (environment_score, governance_grade_encoded, etc.) appear in the top 10, you have strong evidence for your paper that ESG information was used by the model.

### Step 6 — Attention Weight Visualization

```python
# Average attention weights across many test samples
avg_attention = attn_esg.mean(axis=0).flatten()  # Shape: (30,)

days = [f"t-{LOOKBACK - i}" for i in range(LOOKBACK)]

plt.figure(figsize=(13, 4))
plt.bar(days, avg_attention, color="steelblue", edgecolor="white")
plt.xlabel("Days before prediction date")
plt.ylabel("Average attention weight")
plt.title("Average Attention Weights Across Test Set")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("results/figures/attention_weights.png", dpi=150)
plt.show()

top_day = days[avg_attention.argmax()]
print(f"Day with highest average attention: {top_day}")
print(f"(This means the model found {top_day} most informative on average)")
```

### Step 7 — Save All Metrics

```python
import json

results = {
    "esg_model"      : metrics_esg,
    "baseline_model" : metrics_base,
    "improvement"    : {
        "accuracy" : round(metrics_esg["accuracy"] - metrics_base["accuracy"], 4),
        "f1"       : round(metrics_esg["f1"]       - metrics_base["f1"],       4),
        "auc"      : round(metrics_esg["auc"]      - metrics_base["auc"],       4)
    }
}

with open("results/metrics.json", "w") as f:
    json.dump(results, f, indent=2)

print("All metrics saved to results/metrics.json")
print("\nFinal summary:")
print(f"  ESG Model AUC   : {metrics_esg['auc']:.4f}")
print(f"  Baseline AUC    : {metrics_base['auc']:.4f}")
print(f"  AUC improvement : {results['improvement']['auc']:+.4f}")
```

---

## 8. Writing the Academic Paper

### Suggested Structure and Word Counts

```
Section                       Target length
──────────────────────────────────────────────
1.  Abstract                  200–250 words
2.  Introduction              500–800 words
3.  Literature Review         900–1,200 words
4.  Data and Features         500–700 words
5.  Methodology               800–1,100 words
6.  Results                   600–900 words
7.  Discussion                500–700 words
8.  Conclusion                250–350 words
9.  References                20–40 citations
──────────────────────────────────────────────
Total (approx.)               4,500–6,500 words
```

---

### Section 1 — Abstract (Write This Last)

The abstract is 4–5 sentences covering: (1) the problem, (2) your data and method, (3) your main quantitative result, (4) the implication. Write it after all other sections are complete.

**Template to adapt:**
> "This study investigates whether ESG (Environmental, Social, and Governance) scores carry predictive information for short-term stock price direction. Using a dataset of [N] US-listed stocks rated by Finnhub in 2022, we train an LSTM model with an attention mechanism to classify 5-day-ahead price direction as up or down. To isolate ESG's contribution, we compare performance against a price-only baseline model trained on technical indicators alone. The ESG-augmented model achieves an AUC-ROC of [X.XX], versus [X.XX] for the baseline, suggesting that ESG information provides marginal but measurable predictive value. These findings contribute to the growing literature on alternative data in quantitative equity prediction."

---

### Section 2 — Introduction

Write five paragraphs:

**Paragraph 1 — Context:** Open with the growth of ESG investing. Cite a data point (e.g., global ESG AUM has grown to tens of trillions of dollars). Explain that ESG has become a mainstream investment consideration used by institutional investors worldwide.

**Paragraph 2 — The research gap:** Note that while deep learning has been applied extensively to stock price prediction, most studies use only price and volume data. ESG scores are widely used by human analysts but are rarely incorporated into machine learning models. This is the gap your study fills.

**Paragraph 3 — Research question:** State it clearly and explicitly:
> *"This paper asks: does incorporating ESG scores into a deep learning model improve short-term stock price direction prediction compared to a price-only baseline?"*

**Paragraph 4 — Your approach:** Briefly describe what you did — LSTM with attention, 5-day forward prediction, binary classification, ESG + price features compared against a price-only baseline, evaluated on US equity data from 2020–2023.

**Paragraph 5 — Your contributions:** List 3–4 specific contributions. For example:
- A feature engineering pipeline that combines ESG scores, letter grade encodings, and technical price indicators
- An LSTM-attention architecture evaluated specifically on ESG-augmented input
- A controlled ablation study isolating ESG's incremental contribution over technical features alone
- A SHAP-based analysis of which ESG components (E, S, G) were most predictive

---

### Section 3 — Literature Review

Organize into three subsections:

**3.1 ESG and Financial Performance**

Review empirical finance papers on whether high-ESG stocks outperform lower-ESG stocks. The most important reference to cite here is Friede, Busch & Bassen (2015), which aggregated evidence from over 2,000 empirical studies and found that roughly 90% showed a non-negative relationship between ESG and financial performance. Mention that results in the literature are mixed — some studies find outperformance, others find no effect or that it depends on the ESG component or the time period. This mixed evidence is precisely the motivation for a data-driven, machine-learning approach.

**3.2 Deep Learning for Stock Price Prediction**

Review papers that apply LSTM and related architectures to stock price prediction. Key examples include Fischer & Krauss (2018), who applied LSTM to S&P 500 stocks. Note that the vast majority of these papers use only technical price indicators, confirming the gap your study addresses.

**3.3 Alternative Data in Quantitative Finance**

Review papers that combine non-price data with price data for prediction. ESG scores are a type of "alternative data" — the same category as satellite imagery, credit card transaction data, or news sentiment. Citing papers that successfully used text-based sentiment alongside price data strengthens the argument that ESG scores (another structured alternative dataset) may also be informative.

**Papers to search for on Google Scholar:**
- Friede, Busch & Bassen (2015) — "ESG and financial performance: aggregated evidence from more than 2000 empirical studies"
- Fischer & Krauss (2018) — "Deep learning with long short-term memory networks for financial market predictions"
- Hochreiter & Schmidhuber (1997) — the original LSTM paper (always cite this as a foundational reference)
- Search "ESG machine learning stock prediction 2021 2022 2023" for recent related work

---

### Section 4 — Data and Features

**4.1 ESG Data**

Describe: your source (Finnhub), the approximate date of the data snapshot (April–November 2022 based on `last_processing_date` in the CSV), which exchanges are included (NYSE and NASDAQ), and the number of stocks before and after cleaning. Include a summary statistics table:

| Metric | Environment Score | Social Score | Governance Score | Total Score |
|---|---|---|---|---|
| Mean | — | — | — | — |
| Std | — | — | — | — |
| Min | — | — | — | — |
| Max | — | — | — | — |

Fill this in from `esg[score_cols].describe()` before normalization.

**4.2 Price Data**

Describe: source (Yahoo Finance via yfinance), date range (January 2020 – December 2023), number of tickers that successfully downloaded, minimum history requirement (200 trading days), and columns used (Close and Volume).

**4.3 Label Construction**

Be explicit about how labels were created. State the formula: label equals 1 if the closing price 5 days ahead is greater than the current closing price, and 0 otherwise. Report the overall class balance across your full dataset.

**4.4 Feature Summary Table**

| Feature Name | Category | Description |
|---|---|---|
| return_1d | Price | 1-day percentage return |
| return_5d | Price | 5-day cumulative return |
| return_20d | Price | 20-day cumulative return |
| ma_ratio | Price | 10-day MA divided by 30-day MA |
| volatility_20d | Price | 20-day rolling standard deviation of returns |
| rsi | Price | 14-day Relative Strength Index |
| volume_change | Price | 1-day percentage change in trading volume |
| price_vs_52w_high | Price | Current price as fraction of 52-week high |
| price_vs_52w_low | Price | Current price as fraction of 52-week low |
| environment_score | ESG | Normalized environment pillar score |
| social_score | ESG | Normalized social pillar score |
| governance_score | ESG | Normalized governance pillar score |
| total_score | ESG | Normalized total ESG score |
| environment_grade_encoded | ESG | Ordinal-encoded environment grade (1–6) |
| social_grade_encoded | ESG | Ordinal-encoded social grade (1–6) |
| governance_grade_encoded | ESG | Ordinal-encoded governance grade (1–6) |
| total_grade_encoded | ESG | Ordinal-encoded total grade (1–6) |

**4.5 Known Data Limitations**

State these clearly in your paper — academic reviewers expect to see them. Acknowledging limitations shows intellectual honesty and strengthens your paper, not weakens it.

1. ESG scores are a static snapshot from 2022. In reality, ESG ratings change over time. The model assigns the same ESG value to every day in the price history, which is an approximation.
2. Coverage is limited to US-listed stocks on NYSE and NASDAQ. Results may not generalize to other geographies.
3. The 2020–2023 window includes the COVID-19 pandemic and the 2022 bear market. These are unusual market conditions that may affect model generalizability.
4. The industry distribution in the dataset may not be representative of the full US equity universe.

---

### Section 5 — Methodology

**5.1 Sequence Construction**

Explain the 30-day lookback window. State that for each trading day, a 30-day sliding window of historical features is constructed as input, and the model predicts the direction of price movement 5 days into the future. Justify the 30-day window: it is a standard choice in financial ML literature, long enough to capture short-term momentum and mean-reversion patterns.

**5.2 Model Architecture**

Describe all layers in order. Include a figure of the architecture if possible (draw it in PowerPoint or draw.io). Report:
- Input shape: (batch size, 30, 17) for the ESG model; (batch size, 30, 9) for the baseline
- LSTM: 2 layers, 64 hidden units, dropout = 0.3
- Attention: single linear layer with softmax activation over time dimension
- Fully connected: 64 → 1
- Activation: sigmoid
- Total parameters: (your number from Step 4 of Phase 6)

**5.3 Training Protocol**

State all training decisions explicitly. Reviewers will check these:
- Loss function: binary cross-entropy
- Optimizer: Adam with learning rate 0.001
- Learning rate schedule: ReduceLROnPlateau with factor 0.5 and patience 4
- Early stopping: patience of 8 epochs, monitoring validation loss
- Gradient clipping: max norm 1.0
- Random seed: 42

**5.4 Data Split Strategy**

Explicitly state that the dataset was split chronologically — the first 80% of sequences (by time) were used for training and the remaining 20% for testing. State that random shuffling was deliberately avoided to prevent data leakage.

**5.5 Baseline Model**

Describe the price-only baseline: identical LSTM-attention architecture trained on the same data, but with all 8 ESG features removed. The only difference between the two models is the input.

---

### Section 6 — Results

Present your results in this order. Every number must come from your saved `results/metrics.json` file.

**Table 1 — Performance Comparison**

| Metric | Price-Only Baseline | ESG + Price Model | Difference |
|---|---|---|---|
| Accuracy | — | — | — |
| F1 Score | — | — | — |
| AUC-ROC | — | — | — |

**Figure 1** — Training and validation loss curves for both models (`training_curves_esg.png`). Describe what the curves show: whether the model converged, when early stopping triggered, and whether overfitting was observed.

**Figure 2** — Side-by-side confusion matrices (`confusion_matrices.png`). Calculate and report false positive and false negative rates explicitly.

**Figure 3** — ROC curves (`roc_curves.png`). Compare the area under each curve. State clearly which model performed better.

**Figure 4** — SHAP feature importance (`shap_summary.png`). List the top 5 most important features. State which ESG features appeared in the top 10 and what their SHAP values indicate.

**Figure 5** — Average attention weights (`attention_weights.png`). Describe which past days received the highest attention weights. Discuss what this might mean from a financial perspective — for example, if very recent days (t-1, t-2) have the highest weights, it suggests recent price momentum is the dominant signal.

---

### Section 7 — Discussion

**Did ESG help?**

State directly whether the ESG model outperformed the baseline. Report the AUC-ROC difference. Even a small improvement (0.01–0.02 AUC) is academically meaningful to discuss, because equity markets are notoriously hard to predict. If the ESG model did not outperform, discuss why — this is still a valid and publishable finding.

**Which ESG component mattered most?**

Reference the SHAP summary plot. If governance_score had the highest SHAP value, discuss possible financial explanations — governance scandals (accounting fraud, executive misconduct) tend to produce sharp negative price reactions, so governance information may be particularly relevant for short-term prediction.

**Why might ESG carry predictive signal?**

Discuss 2–3 possible explanations. Candidates include: high-ESG firms may be more stable with lower tail risk; ESG ratings are increasingly incorporated into institutional mandates, creating measurable demand effects; ESG scores may proxy for management quality and operational discipline, which are otherwise unobservable.

**Limitations to acknowledge explicitly:**

1. Static ESG scores — a future study with quarterly or annual ESG updates could test whether time-varying ESG ratings improve results further.
2. Single geographic market — the model was trained on US data only.
3. Single time period — the 2020–2023 window includes unusual market conditions.
4. Industry concentration — if the dataset is dominated by specific sectors, results may reflect sector patterns rather than ESG signal.
5. No transaction costs — a model that is marginally better in classification terms may not be profitable after accounting for real-world trading costs.

---

### Section 8 — Conclusion

**Paragraph 1:** Restate the research question: does ESG data improve short-term stock price direction prediction in a deep learning framework?

**Paragraph 2:** Summarize the key findings in 3–4 sentences. State your AUC-ROC numbers explicitly. State which ESG features were most important per SHAP.

**Paragraph 3:** State the practical implication for quantitative investors or ESG data providers.

**Paragraph 4:** Propose future work. Good suggestions include: using time-varying ESG data updated quarterly, extending to international equity markets, trying longer prediction horizons (1 month, 1 quarter), exploring Transformer-based architectures, or combining ESG with news sentiment data.

---

### Reproducibility Checklist

Before submitting your paper, verify all of the following:

- [ ] `torch.manual_seed(42)` and `np.random.seed(42)` are set at the top of every notebook
- [ ] Data split is chronological (80% train, 20% test by time position)
- [ ] All hyperparameters are listed in a table in Section 5
- [ ] The exact date range of price data is stated in Section 4
- [ ] The number of stocks before and after filtering is stated in Section 4
- [ ] Code is uploaded to GitHub or provided as supplementary material
- [ ] All figures are saved at 150 DPI or higher
- [ ] A `requirements.txt` file is included: run `pip freeze > requirements.txt`

---

## 9. Common Mistakes to Avoid

| Mistake | Why It Is a Problem | How to Avoid It |
|---|---|---|
| Shuffling before splitting | Future data leaks into training — results are artificially inflated and not reproducible in real markets | Split by date position only: `X_train = X_all[:split_idx]` |
| Fitting the price scaler on all data | Test set statistics contaminate training normalization | Fit `scaler.fit_transform()` on `X_train` only, then `scaler.transform()` on `X_test` |
| Reporting only accuracy | A model that always predicts "up" on a 55/45 dataset gets 55% accuracy for free | Always report F1 and AUC-ROC alongside accuracy |
| Skipping the baseline model | Without it, you cannot prove ESG added anything beyond price alone | Train both models with identical architecture |
| Not fixing the random seed | Results change every run, making the paper unreproducible | Set seeds at the top of every notebook |
| Building sequences across stock boundaries | Day 29 of Stock A + Day 1 of Stock B is a meaningless hybrid sequence | Build sequences per ticker using `groupby("ticker")`, then concatenate |
| Only training one model run | A single run may have gotten lucky or unlucky with initialization | Train 3 runs with seeds 42, 123, 456 and report mean ± standard deviation |
| Not plotting the training curve | You cannot diagnose overfitting or underfitting without it | Always plot training and validation loss at the end of every training run |
| Ignoring NaN values in sequences | NaN values silently corrupt model weights during training | Check `np.isnan(X_train).sum()` before training — it must be 0 |
| Reporting results without context | "The model got 58% accuracy" means nothing without a reference | Always compare to (a) the baseline model and (b) the naive strategy of always predicting "up" |

---

*This guide is your complete roadmap from raw data to submitted thesis.
Follow each phase in order, run every diagnostic before moving forward,
and write down every number you find — each one becomes a sentence in your paper.*

*Good luck.*