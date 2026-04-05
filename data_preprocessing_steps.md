# Data Pre-Processing Pipeline — Technical Reference

> **What this document covers:** A complete explanation of every step performed by `preprocess.py`.
> Read this alongside the script. Each section maps directly to a numbered function in the code.

---

## Table of Contents

1. [How to Run the Script](#1-how-to-run-the-script)
2. [What the Script Produces](#2-what-the-script-produces)
3. [Step-by-Step Explanation](#3-step-by-step-explanation)
   - [Step 1 — Create Project Folders](#step-1--create-project-folders)
   - [Step 2 — Load and Clean the ESG CSV](#step-2--load-and-clean-the-esg-csv)
   - [Step 3 — Encode Letter Grades as Integers](#step-3--encode-letter-grades-as-integers)
   - [Step 4 — Normalise ESG Scores to 0–1](#step-4--normalise-esg-scores-to-01)
   - [Step 5 — Download Historical Stock Prices](#step-5--download-historical-stock-prices)
   - [Step 6 — Data Diagnostics](#step-6--data-diagnostics)
   - [Step 7 — Create Prediction Labels](#step-7--create-prediction-labels)
   - [Step 8 — Compute Technical Price Features](#step-8--compute-technical-price-features)
   - [Step 9 — Merge ESG and Price Data](#step-9--merge-esg-and-price-data)
   - [Step 10 — Compute Class Imbalance Weight](#step-10--compute-class-imbalance-weight)
   - [Step 11 — Build Sliding-Window Sequences](#step-11--build-sliding-window-sequences)
   - [Step 12 — Chronological Train/Test Split](#step-12--chronological-traintest-split)
   - [Step 13 — Normalise Price-Based Features](#step-13--normalise-price-based-features)
   - [Step 14 — Save Metadata Files](#step-14--save-metadata-files)
   - [Step 15 — Post-Normalisation Diagnostic Chart](#step-15--post-normalisation-diagnostic-chart)
4. [Feature Reference Table](#4-feature-reference-table)
5. [Output Files Reference](#5-output-files-reference)
6. [Configuration Flags](#6-configuration-flags)
7. [Troubleshooting](#7-troubleshooting)
8. [Design Decisions and Academic Justifications](#8-design-decisions-and-academic-justifications)

---

## 1. How to Run the Script

### Prerequisites

Make sure all required libraries are installed before running:

```bash
pip install pandas numpy scikit-learn yfinance matplotlib seaborn tqdm
```

### Basic usage (all defaults)

Place your ESG CSV at `data/raw/esg_scores.csv`, then run:

```bash
python preprocess.py
```

The script will create all folders, download prices, process everything, and save outputs automatically. The full run (including downloads) takes approximately **20–40 minutes** on a standard internet connection.

### Custom options

```bash
# Use a different ESG file and predict 3 days ahead instead of 5
python preprocess.py --esg_path my_data.csv --horizon 3

# Skip re-downloading prices (use CSVs already in data/raw/prices/)
python preprocess.py --skip_download

# Change the lookback window to 20 days and the train split to 75%
python preprocess.py --lookback 20 --split 0.75

# Full example with all options specified
python preprocess.py \
  --esg_path   data/raw/esg_scores.csv \
  --price_dir  data/raw/prices \
  --out_dir    data/processed \
  --fig_dir    results/figures \
  --start      2020-01-01 \
  --end        2023-12-31 \
  --horizon    5 \
  --lookback   30 \
  --min_days   252 \
  --split      0.80 \
```

### Progress tracking

The script prints timestamped log messages to the console as each step completes:

```
10:02:15  INFO      STEP 1 — Creating project folders
10:02:15  INFO        Folders ready: [...]
10:02:16  INFO      STEP 2 — Loading and cleaning ESG data
10:02:16  INFO        Raw ESG shape: (712, 21)
10:02:16  INFO        Stocks after cleaning: 700
...
```

---

## 2. What the Script Produces

After a successful run, the following files will exist:

```
data/
├── raw/
│   ├── esg_scores.csv                ← your original input (unchanged)
│   └── prices/
│       ├── AAPL.csv                  ← one price file per ticker
│       ├── MSFT.csv
│       └── ...
└── processed/
    ├── features.csv                  ← merged ESG + price master dataframe
    ├── sequences.npz                 ← raw (unnormalised) sequences
    ├── sequences_normalized.npz      ← normalised sequences (model input)
    ├── feature_cols.json             ← ordered list of feature names
    ├── class_weight.json             ← imbalance correction weight
    ├── pipeline_config.json          ← full configuration snapshot
    ├── download_summary.json         ← which tickers succeeded/failed
    └── diagnostics_summary.txt       ← text report of data statistics

results/
└── figures/
    ├── esg_score_distributions.png
    ├── esg_grade_distributions.png
    ├── industry_distribution.png
    └── feature_distributions_post_norm.png
```

The two most important outputs for model training are:
- `sequences_normalized.npz` — the model reads this directly
- `feature_cols.json` — the training notebook reads this to know the feature order

---

## 3. Step-by-Step Explanation

---

### Step 1 — Create Project Folders

**Function:** `create_folders(args)`

**What it does:**
Creates all directories required by the pipeline if they do not already exist. Uses `os.makedirs(..., exist_ok=True)` which means it is safe to run multiple times — it will not raise an error if the folder already exists.

**Folders created:**

| Folder | Purpose |
|---|---|
| `data/raw/prices/` | Stores one CSV per ticker downloaded from Yahoo Finance |
| `data/processed/` | Stores all pipeline outputs (sequences, configs, summaries) |
| `results/figures/` | Stores all diagnostic charts saved as PNG files |
| `results/` | Parent folder for evaluation outputs |

**Why this step exists:**
Python raises a `FileNotFoundError` when you try to save a file to a folder that does not exist. Creating folders upfront prevents all downstream steps from failing silently.

---

### Step 2 — Load and Clean the ESG CSV

**Function:** `load_esg(args)`

**What it does:**

1. Reads the ESG CSV file into a pandas DataFrame.
2. Verifies that all required columns are present. If any column is missing, the script stops with a clear error message rather than crashing cryptically later.
3. Keeps only the columns needed for the pipeline (drops metadata like logo URLs and web URLs).
4. Standardises the `ticker` column to uppercase with no leading/trailing spaces. This prevents mismatches like `"aapl"` not matching `"AAPL"` when joining with price files.
5. Drops any row where one or more of the four score columns (`environment_score`, `social_score`, `governance_score`, `total_score`) is missing.

**Required columns in your ESG CSV:**

| Column | Type | Description |
|---|---|---|
| `ticker` | string | Stock ticker symbol |
| `name` | string | Company name |
| `industry` | string | Industry sector |
| `exchange` | string | Stock exchange (NYSE, NASDAQ, etc.) |
| `environment_score` | float | Numerical environment pillar score |
| `social_score` | float | Numerical social pillar score |
| `governance_score` | float | Numerical governance pillar score |
| `total_score` | float | Combined total ESG score |
| `environment_grade` | string | Letter grade (B, BB, BBB, A, AA, AAA) |
| `social_grade` | string | Letter grade |
| `governance_grade` | string | Letter grade |
| `total_grade` | string | Letter grade |

**What the log will show:**
```
STEP 2 — Loading and cleaning ESG data
  Raw ESG shape: (712, 21)
  Stocks after cleaning: 700  (dropped 12 rows with missing scores)
```

---

### Step 3 — Encode Letter Grades as Integers

**Function:** `encode_grades(esg)`

**What it does:**
Creates four new columns — one for each grade column — by mapping the letter grade to an integer using this scale:

| Letter Grade | Encoded Integer | Meaning |
|---|---|---|
| B | 1 | Lowest / basic ESG compliance |
| BB | 2 | Below average |
| BBB | 3 | Average |
| A | 4 | Above average |
| AA | 5 | High |
| AAA | 6 | Excellent (rarely seen) |

**Why this is necessary:**
Machine learning models work with numbers, not text. The grades carry *ordinal* information — AA is meaningfully better than BB — so we convert them to integers that preserve that ordering. This is different from one-hot encoding, which would treat each grade as a separate unrelated category.

**New columns created:**
- `environment_grade_encoded`
- `social_grade_encoded`
- `governance_grade_encoded`
- `total_grade_encoded`

**What happens if an unexpected grade value is found:**
The script logs a warning and sets that cell to `NaN`. It does not crash. Check the log for any warnings after this step.

**Example:**

| ticker | environment_grade | environment_grade_encoded |
|---|---|---|
| AAPL | BB | 2 |
| HASI | AA | 5 |
| ABNB | A | 4 |

---

### Step 4 — Normalise ESG Scores to 0–1

**Function:** `normalise_esg_scores(esg)`

**What it does:**
Applies `MinMaxScaler` from scikit-learn to the four raw ESG score columns. For each column, the formula is:

```
normalised_value = (value - column_min) / (column_max - column_min)
```

After scaling, every score sits in the range [0.0, 1.0].

**Why this is necessary:**
Raw ESG scores have values roughly between 200 and 650. The price-based features we compute later (such as daily returns) will be small decimals — typically between -0.10 and +0.10. If we feed both into the LSTM without normalising, the model's weight updates will be dominated by the magnitude of the ESG scores, and the price features will be effectively ignored.

Normalisation puts all features on the same footing so the model can weigh them fairly.

**Why MinMaxScaler and not StandardScaler here:**
MinMaxScaler is applied to ESG scores specifically because they are bounded (there is a known minimum and maximum possible score). StandardScaler (which centres on mean and scales by standard deviation) is better suited for price features that follow a roughly Gaussian distribution — that is why we use it in Step 13.

**Important note for your paper:**
Because ESG scores are static (the same for every date), we apply MinMaxScaler to the entire ESG dataset before any train/test split. This is acceptable because we are not fitting on future information — the ESG snapshot is from 2022, which predates the end of the price data. If your ESG scores were time-varying, you would need to be more careful here.

---

### Step 5 — Download Historical Stock Prices

**Function:** `download_prices(esg, args)`

**What it does:**
For every ticker in the ESG dataset, downloads daily OHLCV (Open, High, Low, Close, Volume) data from Yahoo Finance using the `yfinance` library.

Each ticker's data is saved as a CSV file at `data/raw/prices/{TICKER}.csv`.

**Skip logic:**
If a CSV for a ticker already exists, the script skips re-downloading it. This means you can run the script again after a partial failure and it will only download missing tickers.

**Filtering:**
Tickers are discarded (not saved) if they have fewer than `--min_days` (default: 252) trading days in the date range. 252 is approximately one calendar year of US trading days. Stocks with very short histories do not have enough data to generate meaningful sequences.

**Failure reasons:**
A ticker can fail to download for several reasons:
- The company was **delisted** (e.g., acquired, went bankrupt) after the ESG snapshot date
- The **ticker symbol was changed** (e.g., after a merger or rebrand)
- A temporary **network error** or Yahoo Finance rate limit

**Expected results:**
From ~700 tickers, expect approximately 400–550 successful downloads. This is normal. The summary is saved to `data/processed/download_summary.json` for your records.

**What to write in your paper:**
State the number of stocks after download filtering. For example: *"Of 700 stocks in the ESG dataset, 487 had sufficient price history in the 2020–2023 period and were retained for analysis."*

---

### Step 6 — Data Diagnostics

**Function:** `run_diagnostics(esg, args)`

**What it does:**
Generates four diagnostic outputs:

**Chart 1 — ESG Score Distributions** (`esg_score_distributions.png`)
A 2×2 grid of histograms showing the distribution of each normalised ESG score across all stocks. Helps you identify whether scores are evenly spread or heavily skewed toward one end.

**Chart 2 — ESG Grade Distributions** (`esg_grade_distributions.png`)
A 2×2 grid of bar charts showing how many stocks received each letter grade. Most datasets are dominated by "B" and "BB" grades at the low end. This is expected — note it as a limitation in your paper.

**Chart 3 — Industry Concentration** (`industry_distribution.png`)
A bar chart of the top 15 industries by stock count. If one industry dominates (> 30% of stocks), the script logs a warning. This is important for your paper's limitations section — the model may learn sector patterns rather than pure ESG signal.

**Text summary** (`diagnostics_summary.txt`)
A plain text file containing descriptive statistics (mean, std, min, max, quartiles) for all ESG score columns, plus the grade distributions for all four grade columns. Use these numbers when writing Section 4 (Data) of your paper.

---

### Step 7 — Create Prediction Labels

**Function:** `create_labels(price_df, horizon)`

**What it does:**
For every trading day in a stock's price history, computes a binary label:

```
label = 1  if  Close[t + horizon] > Close[t]   → price went UP
label = 0  if  Close[t + horizon] ≤ Close[t]   → price went DOWN or flat
```

The default `horizon` is 5 (trading days), which corresponds to approximately one calendar week.

**How it works technically:**
`pandas.shift(-5)` moves the `Close` column 5 rows upward in the DataFrame. So on the row representing day `t`, the column `future_close` contains the closing price from day `t+5`. The label is then just a comparison of those two values.

**Why the last `horizon` rows are dropped:**
On the final 5 rows of the dataset, the future closing price does not yet exist. Keeping these rows would leave `future_close` as `NaN`. We drop them with `dropna()`.

**Visual illustration:**

```
Date       Close    future_close (shift -5)    label
─────────────────────────────────────────────────────
2022-01-03  100.0       103.0                    1  ← price went up in 5 days
2022-01-04   98.0        99.0                    1
2022-01-05  101.0        96.0                    0  ← price fell in 5 days
...
2022-12-23  115.0         NaN                  NaN  ← dropped (no future data)
2022-12-27  116.0         NaN                  NaN  ← dropped
```

---

### Step 8 — Compute Technical Price Features

**Function:** `add_technical_features(df)`

**What it does:**
Computes 11 technical indicators from the daily OHLCV data. Every indicator uses only past data — there is no look-ahead bias here.

**Feature explanations:**

#### Returns

| Feature | Formula | What it measures |
|---|---|---|
| `return_1d` | `(Close[t] - Close[t-1]) / Close[t-1]` | Yesterday's price change |
| `return_5d` | `(Close[t] - Close[t-5]) / Close[t-5]` | Weekly price change |
| `return_20d` | `(Close[t] - Close[t-20]) / Close[t-20]` | Monthly price change |

Returns capture **momentum** — stocks that have been rising tend to continue rising over short horizons (this is called the momentum anomaly in finance).

#### Moving Average Ratio

| Feature | Formula | What it measures |
|---|---|---|
| `ma_10` | 10-day rolling mean of Close | Short-term trend |
| `ma_30` | 30-day rolling mean of Close | Medium-term trend |
| `ma_ratio` | `ma_10 / ma_30` | Short vs medium trend comparison |

`ma_ratio > 1.0` → short-term average is above medium-term average → **bullish signal** (trend is strengthening).
`ma_ratio < 1.0` → short-term average is below medium-term average → **bearish signal** (trend is weakening).

#### Volatility

| Feature | Formula | What it measures |
|---|---|---|
| `volatility_20d` | Rolling 20-day std of `return_1d` | How uncertain / risky the stock is right now |

High volatility periods are harder to predict. Including this feature lets the model learn to be less confident during turbulent periods.

#### RSI — Relative Strength Index

| Feature | Formula | What it measures |
|---|---|---|
| `rsi` | `100 - 100 / (1 + avg_gain / avg_loss)` over 14 days | Whether the stock is overbought or oversold |

RSI is one of the most widely used technical indicators in practice:
- `RSI > 70` → stock may be **overbought** — a pullback may be due
- `RSI < 30` → stock may be **oversold** — a bounce may be due
- `RSI ≈ 50` → neutral momentum

The `1e-8` added to the denominator prevents a division-by-zero error on days when there are no losing periods.

#### Volume Change

| Feature | Formula | What it measures |
|---|---|---|
| `volume_change` | `(Volume[t] - Volume[t-1]) / Volume[t-1]` | Unusual trading activity |

Spikes in trading volume often accompany or precede significant price moves. High volume on an up day is considered a stronger bullish signal than low volume.

#### 52-Week Position

| Feature | Formula | What it measures |
|---|---|---|
| `price_vs_52w_high` | `Close[t] / max(Close, 252 days)` | Where the stock sits relative to its annual peak |
| `price_vs_52w_low` | `Close[t] / min(Close, 252 days)` | Where the stock sits relative to its annual trough |

A stock trading near its 52-week high may have strong momentum; one trading near its 52-week low may be under pressure. These contextualise the current price level without using the raw price number (which is not meaningful across different stocks).

**Note on rolling window warmup:**
All rolling calculations (moving averages, RSI, volatility, 52-week position) produce `NaN` for the first N rows of each stock's history because there is not enough past data yet. These rows are dropped in Step 9 when we call `master_df.dropna()`.

---

### Step 9 — Merge ESG and Price Data

**Function:** `build_master_dataframe(esg, args)`

**What it does:**
Iterates over every ticker that has a downloaded price CSV and:

1. Loads the price CSV
2. Handles any multi-level column headers that `yfinance` sometimes produces
3. Drops rows with missing `Close` or `Volume` values
4. Skips tickers with fewer than `min_days` rows
5. Calls `add_technical_features()` and `create_labels()`
6. Attaches the stock's static ESG scores to every row of its price history
7. Adds the stock's industry and ticker as columns

After all stocks are processed, concatenates everything into one large DataFrame and drops any remaining `NaN` rows (from rolling window warmup).

**Why ESG is attached as static values:**
Your ESG scores are from a single 2022 snapshot. Every row of AAPL's price history — from January 2020 to December 2023 — gets the same `environment_score`, `social_score`, etc. This is an approximation of reality (ESG scores do change over time), but it is a necessary simplification given the data available. Acknowledge this in your paper's limitations section.

**The resulting master DataFrame has this structure:**

| Date | Close | return_1d | ... | rsi | environment_score | ... | label | ticker |
|---|---|---|---|---|---|---|---|---|
| 2020-02-15 | 78.23 | 0.012 | ... | 54.3 | 0.412 | ... | 1 | AAPL |
| 2020-02-18 | 79.01 | 0.010 | ... | 56.1 | 0.412 | ... | 0 | AAPL |
| ... | | | | | | | | |
| 2020-02-15 | 45.10 | -0.003 | ... | 48.2 | 0.731 | ... | 1 | MSFT |

Each row is one trading day for one stock, fully described by all 17 features plus the label.

**Output file:** `data/processed/features.csv`

---

### Step 10 — Compute Class Imbalance Weight

**Function:** `save_class_weight(master_df, args)`

**What it does:**
Counts the number of "up" labels (1) and "down" labels (0) across the entire dataset, then computes:

```
pos_weight = num_down / num_up
```

**Why this matters:**
If the dataset has more "down" days than "up" days (which is common in periods including market downturns like 2022), the model will naturally learn to predict "down" more often — not because it has learned anything meaningful, but because doing so reduces its average error.

The `pos_weight` is passed to PyTorch's `BCEWithLogitsLoss` during training. It tells the loss function: *"penalise missed 'up' predictions by this factor more than missed 'down' predictions."*

**Interpreting the value:**

| pos_weight | Meaning | Action |
|---|---|---|
| 0.90 – 1.10 | Essentially balanced | No correction needed (use standard BCELoss) |
| 1.10 – 1.30 | Mild imbalance | Worth using weighted loss |
| > 1.30 | Significant imbalance | Use weighted loss; mention in paper |
| < 0.77 | More "up" than "down" | Use weighted loss with inverse interpretation |

**Output file:** `data/processed/class_weight.json`

```json
{
  "pos_weight": 1.142
}
```

---

### Step 11 — Build Sliding-Window Sequences

**Function:** `build_sequences(master_df, args)` and `_build_sequences_for_ticker(...)`

**What it does:**
Converts the flat two-dimensional master DataFrame into three-dimensional arrays of overlapping time windows.

**The sliding window concept:**

Imagine a stock with 750 trading days. The LSTM needs to see 30 consecutive days at a time to make a prediction for the 31st day. We slide a 30-day window across the entire history, one day at a time:

```
Window 1: days  1–30  → predicts day 31
Window 2: days  2–31  → predicts day 32
Window 3: days  3–32  → predicts day 33
...
Window 720: days 720–749 → predicts day 750
```

This gives ~720 sequences from one stock with 750 days.

**Why sequences are built per ticker:**
If we built sequences from the concatenated master DataFrame without grouping by ticker, a window could span the last few days of one stock and the first few days of another. That would be meaningless — AAPL's returns from December 2022 have nothing to do with MSFT's returns from January 2020. By building per ticker and then concatenating, we guarantee every sequence belongs to exactly one stock.

**Output shape:**

```
X_all shape: (total_sequences, 30, 17)
               └──────────────┘  └─┘  └┘
               e.g. 360,000      days  features

y_all shape: (total_sequences,)
```

**Output files:** `X_all` and `y_all` are not saved directly — they are immediately passed to Step 12. The final saved version (after normalisation) is `sequences_normalized.npz`.

---

### Step 12 — Chronological Train/Test Split

**Function:** `split_sequences(X_all, y_all, args)`

**What it does:**
Divides the sequence arrays into training and test sets by position:

```python
split_idx = int(len(X_all) * 0.80)

X_train = X_all[:split_idx]   # First 80% of sequences
X_test  = X_all[split_idx:]   # Last  20% of sequences
```

**Why chronological splitting is mandatory:**
This is the most important design decision in the entire pipeline. Here is what goes wrong if you shuffle:

Suppose the dataset spans January 2020 to December 2023. If you shuffle randomly and then split 80/20, sequences from December 2023 can end up in the training set. The model then effectively "trains on the future" — it sees patterns from late 2023 and uses them to predict events from 2020. This is called **data leakage**, and it produces results that look excellent in your notebook but would completely fail in real-world use.

By splitting chronologically, the model only ever sees the past when making predictions about the test period.

**Output files:**

- `data/processed/sequences.npz` — raw (unnormalised) X_train, y_train, X_test, y_test
- The normalised version is saved in Step 13

**What the log will show:**
```
STEP 12 — Splitting sequences (chronological 80/20 split)
  Training sequences : 289,442  (51.8% 'up' labels)
  Test sequences     : 72,361   (48.3% 'up' labels)
```

Note: if the label balance differs noticeably between train and test (e.g., train is 52% up but test is 45% up), this reflects genuine differences between the two time periods. This is normal and expected — market conditions in 2020–2022 differ from late 2022–2023.

---

### Step 13 — Normalise Price-Based Features

**Function:** `normalise_sequences(X_train, y_train, X_test, y_test, args)`

**What it does:**
Applies `StandardScaler` to the 9 price-based feature columns (indices 0–8 in every sequence). The 8 ESG features (indices 9–16) were already normalised to [0,1] in Step 4 and are left unchanged.

`StandardScaler` transforms each feature so that it has mean ≈ 0 and standard deviation ≈ 1:

```
standardised_value = (value - mean) / std
```

**The critical rule — fit on training data only:**

```python
# CORRECT
scaler.fit_transform(X_train_price_columns)   # learns mean and std from training data
scaler.transform(X_test_price_columns)         # applies SAME mean and std to test data

# WRONG — this is data leakage
scaler.fit_transform(X_all_price_columns)      # learns from future test data too
```

If you fit the scaler on all data (train + test combined), the scaler's mean and standard deviation are contaminated by statistics from the test period. The model then benefits from knowing, for example, that the average return during the test period was negative — information it should not have during training.

**Workflow in detail:**

1. The 3D sequences `(num_sequences, 30, 17)` are reshaped to 2D `(num_sequences × 30, 17)` because scikit-learn scalers expect 2D arrays.
2. `fit_transform` is called on the training set's price columns only.
3. `transform` (not `fit_transform`) is called on the test set's price columns using the fitted scaler.
4. The 2D arrays are reshaped back to 3D.

**Sanity check:**
After normalisation, the script checks for any remaining `NaN` or `Inf` values. If any are found, they are replaced with 0 and a warning is logged. The presence of such values usually indicates a stock with unusual price data (e.g., a zero volume day causing a division-by-zero in one of the features).

**Output file:** `data/processed/sequences_normalized.npz`

This is the primary input file for model training.

---

### Step 14 — Save Metadata Files

**Function:** `save_metadata(args)`

**What it does:**
Saves two JSON files that the training and evaluation notebooks read to ensure consistency.

**`feature_cols.json`**
An ordered list of all 17 feature names. The training notebook loads this to confirm it is reading features in the same order the preprocessing pipeline used. Feature order matters because `X_train[:, :, 0]` must always refer to `return_1d`, `X_train[:, :, 9]` must always refer to `environment_score`, and so on.

**`pipeline_config.json`**
A snapshot of every configuration parameter used in this pipeline run. This is your reproducibility record — include the key values from this file in your paper's methodology section.

```json
{
  "esg_path": "data/raw/esg_scores.csv",
  "start_date": "2020-01-01",
  "end_date": "2023-12-31",
  "forecast_horizon": 5,
  "lookback": 30,
  "min_days": 252,
  "train_split": 0.8,
  "num_features": 17,
  "price_feature_indices": [0, 1, 2, 3, 4, 5, 6, 7, 8],
  "esg_feature_indices": [9, 10, 11, 12, 13, 14, 15, 16],
  "seed": 42
}
```

---

### Step 15 — Post-Normalisation Diagnostic Chart

**Function:** `plot_sequence_stats(X_train, X_test, args)`

**What it does:**
Creates a box plot showing the distribution of each feature's values across all training sequences (using the last timestep of each sequence as a representative sample).

After normalisation, price-based features (indices 0–8) should be centred near 0 with most values between -3 and +3. ESG features (indices 9–16) will be between 0 and 1.

This chart is a final sanity check before you start training. If any feature has a wildly different scale compared to the others, something went wrong in the normalisation step.

**Output file:** `results/figures/feature_distributions_post_norm.png`

**What a healthy chart looks like:**
- Price features: median near 0, interquartile range roughly -1 to +1
- ESG features: median varies by dataset, range 0 to 1
- No boxes extending to ±10 or beyond (that would indicate outliers that survived normalisation)

---

## 4. Feature Reference Table

Complete list of all 17 features fed into the model, in the exact order they appear in the sequence arrays.

| Index | Feature Name | Category | Formula / Source | Typical Range (after scaling) |
|---|---|---|---|---|
| 0 | `return_1d` | Price | `(Close[t] - Close[t-1]) / Close[t-1]` | -3 to +3 |
| 1 | `return_5d` | Price | `(Close[t] - Close[t-5]) / Close[t-5]` | -3 to +3 |
| 2 | `return_20d` | Price | `(Close[t] - Close[t-20]) / Close[t-20]` | -3 to +3 |
| 3 | `ma_ratio` | Price | `MA_10 / MA_30` | -3 to +3 |
| 4 | `volatility_20d` | Price | Rolling 20-day std of `return_1d` | -3 to +3 |
| 5 | `rsi` | Price | 14-day RSI (0–100 before scaling) | -3 to +3 |
| 6 | `volume_change` | Price | `(Volume[t] - Volume[t-1]) / Volume[t-1]` | -3 to +3 |
| 7 | `price_vs_52w_high` | Price | `Close[t] / rolling_max(Close, 252)` | -3 to +3 |
| 8 | `price_vs_52w_low` | Price | `Close[t] / rolling_min(Close, 252)` | -3 to +3 |
| 9 | `environment_score` | ESG | MinMax-scaled environment score | 0.0 to 1.0 |
| 10 | `social_score` | ESG | MinMax-scaled social score | 0.0 to 1.0 |
| 11 | `governance_score` | ESG | MinMax-scaled governance score | 0.0 to 1.0 |
| 12 | `total_score` | ESG | MinMax-scaled total ESG score | 0.0 to 1.0 |
| 13 | `environment_grade_encoded` | ESG | Grade → integer (1–6) | 1 to 6 |
| 14 | `social_grade_encoded` | ESG | Grade → integer (1–6) | 1 to 6 |
| 15 | `governance_grade_encoded` | ESG | Grade → integer (1–6) | 1 to 6 |
| 16 | `total_grade_encoded` | ESG | Grade → integer (1–6) | 1 to 6 |

---

## 5. Output Files Reference

| File | Location | Format | Used by |
|---|---|---|---|
| `features.csv` | `data/processed/` | CSV | Inspection, debugging |
| `sequences.npz` | `data/processed/` | NumPy compressed | Backup; not used directly by model |
| `sequences_normalized.npz` | `data/processed/` | NumPy compressed | **Model training** |
| `feature_cols.json` | `data/processed/` | JSON list | Training + evaluation notebooks |
| `class_weight.json` | `data/processed/` | JSON object | Training notebook |
| `pipeline_config.json` | `data/processed/` | JSON object | Reproducibility reference |
| `download_summary.json` | `data/processed/` | JSON object | Paper writing (data section) |
| `diagnostics_summary.txt` | `data/processed/` | Plain text | Paper writing (data section) |
| `esg_score_distributions.png` | `results/figures/` | PNG | Paper figure |
| `esg_grade_distributions.png` | `results/figures/` | PNG | Paper figure |
| `industry_distribution.png` | `results/figures/` | PNG | Paper figure |
| `feature_distributions_post_norm.png` | `results/figures/` | PNG | Sanity check |

### How to load the normalised sequences in your training notebook

```python
import numpy as np
import json

data    = np.load("data/processed/sequences_normalized.npz")
X_train = data["X_train"]   # shape: (n_train, 30, 17)
y_train = data["y_train"]   # shape: (n_train,)
X_test  = data["X_test"]    # shape: (n_test,  30, 17)
y_test  = data["y_test"]    # shape: (n_test,)

with open("data/processed/feature_cols.json") as f:
    FEATURE_COLS = json.load(f)   # list of 17 feature names

with open("data/processed/class_weight.json") as f:
    pos_weight = json.load(f)["pos_weight"]   # float

print(f"Training: {X_train.shape}, Test: {X_test.shape}")
print(f"Features: {FEATURE_COLS}")
print(f"Class weight: {pos_weight:.3f}")
```

---

## 6. Configuration Flags

All flags have sensible defaults. Only change them if you have a specific reason.

| Flag | Default | Type | Description |
|---|---|---|---|
| `--esg_path` | `data/raw/esg_scores.csv` | string | Path to the ESG CSV input file |
| `--price_dir` | `data/raw/prices` | string | Directory to save/load per-ticker price CSVs |
| `--out_dir` | `data/processed` | string | Directory for all pipeline outputs |
| `--fig_dir` | `results/figures` | string | Directory for diagnostic charts |
| `--start` | `2020-01-01` | date string | Start of price history download |
| `--end` | `2023-12-31` | date string | End of price history download |
| `--horizon` | `5` | int | Forward prediction horizon in trading days |
| `--lookback` | `30` | int | LSTM sequence length in trading days |
| `--min_days` | `252` | int | Minimum trading days required to keep a ticker |
| `--split` | `0.80` | float | Fraction of sequences allocated to training |
| `--skip_download` | `False` | flag | If set, skips Yahoo Finance download |

**Choosing the forecast horizon (`--horizon`):**
- `5` days (default) → approximately one calendar week ahead
- `1` day → very short-term; harder to predict; suitable for a separate experiment
- `20` days → approximately one calendar month ahead; may suit long-term ESG signal better

For your thesis, keep `--horizon 5` as the primary result and optionally run with `--horizon 1` and `--horizon 20` as ablation experiments to show how ESG's contribution changes across horizons.

**Choosing the lookback window (`--lookback`):**
30 days is the standard choice in the literature. You can experiment with 20 and 60 as sensitivity analyses. Longer windows capture more historical context but increase memory usage and training time.

---

## 7. Troubleshooting

**"ESG file not found" error:**
Make sure your CSV is at `data/raw/esg_scores.csv` (or pass the correct path with `--esg_path`). The file must be named correctly and in the right folder.

**"Missing columns in ESG CSV" error:**
Your CSV column names must exactly match the required list. Check for typos, extra spaces, or different capitalisation. Run `pd.read_csv("data/raw/esg_scores.csv").columns.tolist()` to see what your file actually contains.

**Download step is very slow:**
Yahoo Finance rate-limits requests. The `tqdm` progress bar shows you which ticker is being downloaded. If the script stops mid-download, run it again with `--skip_download` — it will re-use already-downloaded files and only attempt the missing ones.

**"No stocks were merged" error:**
This usually means the ticker format in your ESG CSV does not match the filenames in your price directory. Ensure tickers are uppercase in both. Check: `esg["ticker"].head()` and `os.listdir("data/raw/prices")[:5]`.

**NaN or Inf warning after normalisation:**
Some stocks have days with zero volume or unusual price data that produce infinity in feature calculations (e.g., division by zero in RSI or 52-week ratio). The script replaces these with 0 and continues. If many such warnings appear, inspect the raw price files for the affected tickers.

**Out of memory on the sequence-building step:**
If you have very many stocks and a large date range, `X_all` can become large. Reduce `--min_days` to filter out more short-history stocks, or reduce the date range with `--start` and `--end`.

---

## 8. Design Decisions and Academic Justifications

This section explains the reasoning behind key decisions, which you can reference when writing your methodology chapter.

### Why a 5-day prediction horizon?

Five trading days corresponds to one calendar week, which is the shortest horizon at which ESG-related information is likely to manifest in prices. Daily prediction (1-day) is dominated by noise and market microstructure. Monthly prediction (20-day) is too long for ESG's relatively slow-moving signal to be the primary driver. Five days is the standard choice in short-term prediction literature and is easily justified in your paper.

### Why a 30-day lookback?

Thirty days captures enough price history to compute all technical indicators meaningfully (RSI requires 14 days, the MA ratio requires 30 days, volatility requires 20 days). It is long enough to represent short-term trends while being short enough to remain computationally efficient. It is also the standard lookback in LSTM-based stock prediction papers, which aids in comparability with existing literature.

### Why MinMaxScaler for ESG and StandardScaler for price features?

ESG scores are bounded by a known minimum and maximum, so MinMaxScaler (which scales to [0,1]) is the natural choice. Price-based features like returns and RSI follow an approximately Gaussian distribution without strict bounds, making StandardScaler (zero mean, unit variance) more appropriate. Mixing the two scalers matches each feature type to its appropriate normalisation method.

### Why build sequences per ticker rather than globally?

If sequences were built from the concatenated master DataFrame without grouping, a window could span the last rows of Ticker A and the first rows of Ticker B. This creates sequences that mix two unrelated stocks, which is statistically meaningless and would introduce noise. Building per ticker and then concatenating guarantees that every sequence belongs to exactly one stock throughout its entire 30-day window.

### Why the random seed is set but does not fully determine the model?

The seed controls NumPy and Python random operations in the preprocessing pipeline. Model training introduces additional stochasticity from PyTorch's GPU operations and batch ordering. To fully reproduce model results, you must also set `torch.manual_seed(42)` in the training notebook. For academic papers, report results as the mean of 3 runs with seeds 42, 123, and 456 to demonstrate stability.

### Static ESG scores — acknowledged limitation

Attaching a 2022 ESG snapshot to all historical rows is a simplification. In reality, a company's ESG score in 2020 may differ from its 2022 score. However, ESG scores are slow-moving — they change gradually rather than day-by-day — so the 2022 value is a reasonable approximation for the 2020–2023 period. This limitation is clearly stated in the pipeline documentation and should be explicitly acknowledged in your paper's limitations section. A future extension of this work could use time-varying ESG data published quarterly by providers such as Sustainalytics or MSCI.

---

*This document is the complete technical reference for `preprocess.py`.
For model training, open `03_model_training.ipynb` and load the outputs described in Section 5.*