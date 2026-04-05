"""
preprocess.py
=============
Automated data pre-processing pipeline for the ESG Stock Direction Prediction project.

What this script does (in order):
  1.  Creates all required project folders
  2.  Loads and cleans the ESG CSV
  3.  Encodes letter grades as ordinal integers
  4.  Normalises ESG scores to [0, 1]
  5.  Downloads historical price data from Yahoo Finance
  6.  Runs data diagnostics and saves charts
  7.  Creates binary up/down labels (5-day forward return)
  8.  Engineers technical price features (returns, RSI, MA, volatility, etc.)
  9.  Merges ESG features into the price dataframe
  10. Computes and saves class imbalance weight
  11. Builds 30-day sliding-window sequences for the LSTM
  12. Splits sequences chronologically (80% train / 20% test)
  13. Normalises price-based features using training statistics only
  14. Saves all outputs to disk

Usage:
  python preprocess.py

  Optional flags:
    --esg_path   PATH   Path to the ESG CSV  (default: data/raw/data.csv)
    --price_dir  PATH   Where to save/load price CSVs  (default: data/raw/prices)
    --out_dir    PATH   Where to save processed files  (default: data/processed)
    --fig_dir    PATH   Where to save diagnostic charts (default: results/figures)
    --start      DATE   Price history start date  (default: 2020-01-01)
    --end        DATE   Price history end date    (default: 2023-12-31)
    --horizon    INT    Forward prediction horizon in days  (default: 5)
    --lookback   INT    LSTM sequence length in days  (default: 30)
    --min_days   INT    Minimum trading days required per ticker  (default: 252)
    --split      FLOAT  Train/test split ratio  (default: 0.80)
    --skip_download     Skip the Yahoo Finance download step (use cached CSVs)

Example:
  python preprocess.py --esg_path my_data.csv --horizon 3 --lookback 20
"""

# ---------------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------------
import argparse
import json
import logging
import os
import random
import sys
import warnings

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Third-party libraries
# ---------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # Non-interactive backend — works without a display
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tqdm import tqdm

try:
    import yfinance as yf
except ImportError:
    print("[ERROR] yfinance is not installed. Run:  pip install yfinance")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# Logging setup  — prints timestamped messages to the console
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ===========================================================================
# STEP 0  — Parse command-line arguments
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="ESG stock prediction — automated pre-processing pipeline"
    )
    p.add_argument("--esg_path",      default="data/raw/data.csv")
    p.add_argument("--price_dir",     default="data/raw/prices")
    p.add_argument("--out_dir",       default="data/processed")
    p.add_argument("--fig_dir",       default="results/figures")
    p.add_argument("--start",         default="2020-01-01")
    p.add_argument("--end",           default="2023-12-31")
    p.add_argument("--horizon",       type=int,   default=5)
    p.add_argument("--lookback",      type=int,   default=30)
    p.add_argument("--min_days",      type=int,   default=252)
    p.add_argument("--split",         type=float, default=0.80)
    p.add_argument("--skip_download", action="store_true",
                   help="Skip Yahoo Finance download — use already-saved CSVs")
    return p.parse_args()

# ===========================================================================
# STEP 1  — Create project folders
# ===========================================================================

def create_folders(args):
    """Create all directories needed by the pipeline."""
    log.info("STEP 1 — Creating project folders")
    folders = [
        args.price_dir,
        args.out_dir,
        args.fig_dir,
        "results",
    ]
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
    log.info(f"  Folders ready: {folders}")

# ===========================================================================
# STEP 2  — Load and clean the ESG CSV
# ===========================================================================

def load_esg(args):
    """
    Load the ESG CSV and keep only the columns the pipeline needs.
    Drops rows with missing scores and standardises the ticker column.
    """
    log.info("STEP 2 — Loading and cleaning ESG data")

    if not os.path.exists(args.esg_path):
        log.error(f"ESG file not found: {args.esg_path}")
        sys.exit(1)

    esg = pd.read_csv(args.esg_path)
    log.info(f"  Raw ESG shape: {esg.shape}")

    required_cols = [
        "ticker", "name", "industry", "exchange",
        "environment_score", "social_score", "governance_score", "total_score",
        "environment_grade", "social_grade", "governance_grade", "total_grade",
    ]
    missing = [c for c in required_cols if c not in esg.columns]
    if missing:
        log.error(f"  Missing columns in ESG CSV: {missing}")
        sys.exit(1)

    esg = esg[required_cols].copy()

    # Standardise ticker format
    esg["ticker"] = esg["ticker"].str.upper().str.strip()

    # Drop rows with any missing score
    score_cols = ["environment_score", "social_score", "governance_score", "total_score"]
    before = len(esg)
    esg = esg.dropna(subset=score_cols)
    dropped = before - len(esg)

    log.info(f"  Stocks after cleaning: {len(esg)}  (dropped {dropped} rows with missing scores)")
    return esg

# ===========================================================================
# STEP 3  — Encode letter grades as ordinal integers
# ===========================================================================

GRADE_MAP = {"B": 1, "BB": 2, "BBB": 3, "A": 4, "AA": 5, "AAA": 6}
GRADE_COLS = ["environment_grade", "social_grade", "governance_grade", "total_grade"]

def encode_grades(esg):
    """
    Map letter grades to integers so the model can process them numerically.
    B=1 (lowest) → AAA=6 (highest).
    Rows with unrecognised grade values are set to NaN and reported.
    """
    log.info("STEP 3 — Encoding letter grades as ordinal integers")

    for col in GRADE_COLS:
        esg[col + "_encoded"] = esg[col].map(GRADE_MAP)
        unmapped = esg[col + "_encoded"].isnull().sum()
        if unmapped > 0:
            bad_vals = esg.loc[esg[col + "_encoded"].isnull(), col].unique()
            log.warning(f"  {unmapped} unrecognised values in '{col}': {bad_vals}")
        else:
            log.info(f"  '{col}' encoded cleanly")

    return esg

# ===========================================================================
# STEP 4  — Normalise ESG scores to [0, 1]
# ===========================================================================

SCORE_COLS = ["environment_score", "social_score", "governance_score", "total_score"]

def normalise_esg_scores(esg):
    """
    Apply MinMax scaling to raw ESG scores so they sit in [0, 1].
    This prevents large score values from dominating the price features
    (which will be small decimals such as 0.02 for a 2% return).
    """
    log.info("STEP 4 — Normalising ESG scores to [0, 1]")

    scaler = MinMaxScaler()
    esg[SCORE_COLS] = scaler.fit_transform(esg[SCORE_COLS])

    log.info(f"  Score range after scaling — min: {esg[SCORE_COLS].min().min():.3f},"
             f"  max: {esg[SCORE_COLS].max().max():.3f}")
    return esg

# ===========================================================================
# STEP 5  — Download historical stock prices from Yahoo Finance
# ===========================================================================

def download_prices(esg, args):
    """
    Download daily OHLCV price data for every ticker in the ESG dataset.

    Tickers are skipped if:
      - The download fails (delisted, ticker changed, network error)
      - Fewer than args.min_days trading days are available

    A summary of successes, short history, and failures is logged and
    saved to data/processed/download_summary.json for reference.
    """
    log.info("STEP 5 — Downloading historical price data from Yahoo Finance")
    log.info(f"  Date range : {args.start}  →  {args.end}")
    log.info(f"  Min days   : {args.min_days}")
    log.info("  This may take 20–40 minutes depending on internet speed...")

    tickers    = esg["ticker"].tolist()
    successful = []
    too_short  = []
    failed     = []

    for ticker in tqdm(tickers, desc="Downloading", unit="ticker"):
        out_path = os.path.join(args.price_dir, f"{ticker}.csv")

        # If the file already exists, skip re-downloading
        if os.path.exists(out_path):
            successful.append(ticker)
            continue

        try:
            df = yf.download(
                ticker,
                start=args.start,
                end=args.end,
                progress=False,
                auto_adjust=True,
            )
            if df.empty or len(df) < args.min_days:
                too_short.append(ticker)
                continue
            df.to_csv(out_path)
            successful.append(ticker)
        except Exception:
            failed.append(ticker)

    log.info(f"  Successfully downloaded : {len(successful)}")
    log.info(f"  Too short (< {args.min_days} days): {len(too_short)}")
    log.info(f"  Failed                 : {len(failed)}")

    summary = {
        "successful": successful,
        "too_short" : too_short,
        "failed"    : failed,
    }
    summary_path = os.path.join(args.out_dir, "download_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"  Download summary saved to {summary_path}")

    return successful

# ===========================================================================
# STEP 6  — Data diagnostics: charts and summary statistics
# ===========================================================================

def run_diagnostics(esg, args):
    """
    Generate and save diagnostic charts:
      - ESG score distributions (histogram grid)
      - ESG grade distributions (bar chart)
      - Industry concentration (bar chart)
      - Exchange breakdown (printed to log)

    Charts are saved to args.fig_dir as PNG files.
    A text summary is saved to data/processed/diagnostics_summary.txt.
    """
    log.info("STEP 6 — Running data diagnostics")

    # --- ESG score distributions ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    for i, col in enumerate(SCORE_COLS):
        axes[i].hist(esg[col].dropna(), bins=30, color="steelblue", edgecolor="white")
        axes[i].set_title(col.replace("_", " ").title())
        axes[i].set_xlabel("Normalised score (0–1)")
        axes[i].set_ylabel("Number of stocks")
    plt.suptitle("ESG Score Distributions (after normalisation)", fontsize=13)
    plt.tight_layout()
    path = os.path.join(args.fig_dir, "esg_score_distributions.png")
    plt.savefig(path, dpi=150)
    plt.close()
    log.info(f"  Saved: {path}")

    # --- Grade distributions ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    for i, col in enumerate(GRADE_COLS):
        grade_counts = esg[col].value_counts().sort_index()
        axes[i].bar(grade_counts.index, grade_counts.values, color="steelblue", edgecolor="white")
        axes[i].set_title(col.replace("_", " ").title())
        axes[i].set_xlabel("Grade")
        axes[i].set_ylabel("Count")
    plt.suptitle("ESG Grade Distributions", fontsize=13)
    plt.tight_layout()
    path = os.path.join(args.fig_dir, "esg_grade_distributions.png")
    plt.savefig(path, dpi=150)
    plt.close()
    log.info(f"  Saved: {path}")

    # --- Industry concentration ---
    industry_counts = esg["industry"].value_counts()
    plt.figure(figsize=(13, 5))
    industry_counts.head(15).plot(kind="bar", color="steelblue", edgecolor="white")
    plt.title("Top 15 Industries in Dataset")
    plt.xlabel("Industry")
    plt.ylabel("Number of stocks")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    path = os.path.join(args.fig_dir, "industry_distribution.png")
    plt.savefig(path, dpi=150)
    plt.close()
    log.info(f"  Saved: {path}")

    # --- Exchange breakdown (logged only) ---
    log.info(f"  Exchange breakdown:\n{esg['exchange'].value_counts().to_string()}")

    # --- Top industry warning ---
    top_industry       = industry_counts.index[0]
    top_industry_pct   = industry_counts.iloc[0] / len(esg) * 100
    if top_industry_pct > 30:
        log.warning(
            f"  Industry concentration: '{top_industry}' makes up "
            f"{top_industry_pct:.1f}% of stocks. "
            f"Consider noting this in your paper's limitations."
        )

    # --- Save text summary ---
    summary_lines = [
        "DIAGNOSTICS SUMMARY",
        "=" * 50,
        f"Total stocks in ESG file : {len(esg)}",
        f"Top industry             : {top_industry} ({top_industry_pct:.1f}%)",
        "",
        "ESG Score Statistics (after normalisation):",
        esg[SCORE_COLS].describe().round(3).to_string(),
        "",
        "Grade Distributions:",
    ]
    for col in GRADE_COLS:
        summary_lines.append(f"\n  {col}:")
        summary_lines.append(esg[col].value_counts().sort_index().to_string())

    txt_path = os.path.join(args.out_dir, "diagnostics_summary.txt")
    with open(txt_path, "w") as f:
        f.write("\n".join(summary_lines))
    log.info(f"  Diagnostics summary saved to {txt_path}")

# ===========================================================================
# STEP 7  — Create binary prediction labels
# ===========================================================================

def create_labels(price_df, horizon):
    """
    For each trading day t, look 'horizon' days into the future.

      label = 1  if  Close[t + horizon] > Close[t]   (price went UP)
      label = 0  if  Close[t + horizon] <= Close[t]  (price went DOWN or stayed flat)

    The last 'horizon' rows are dropped because their future price is unknown.

    Why shift(-horizon)?
      pandas shift(n) moves every value DOWN by n rows.
      shift(-n) moves values UP by n rows, so on row t the column shows
      the value that will appear n rows later — i.e., the future price.
    """
    price_df = price_df.copy()
    price_df["future_close"] = price_df["Close"].shift(-horizon)
    price_df["label"] = (price_df["future_close"] > price_df["Close"]).astype(int)
    price_df = price_df.dropna(subset=["future_close"])
    return price_df

# ===========================================================================
# STEP 8  — Compute technical price features
# ===========================================================================

def add_technical_features(df):
    """
    Compute standard technical indicators from daily OHLCV data.
    All features use ONLY past price information — no future data leakage.

    Features computed:
      return_1d        — 1-day percentage return
      return_5d        — 5-day percentage return
      return_20d       — 20-day percentage return
      ma_10            — 10-day simple moving average of Close
      ma_30            — 30-day simple moving average of Close
      ma_ratio         — ma_10 / ma_30  (> 1 = short-term bullish, < 1 = bearish)
      volatility_20d   — 20-day rolling standard deviation of daily returns
      rsi              — 14-day Relative Strength Index  (>70 overbought, <30 oversold)
      volume_change    — 1-day percentage change in trading volume
      price_vs_52w_high— Close / 52-week rolling maximum  (1.0 = at 52-week high)
      price_vs_52w_low — Close / 52-week rolling minimum  (1.0 = at 52-week low)
    """
    df = df.copy()

    # Returns — how much did the price change over N days?
    df["return_1d"]  = df["Close"].pct_change(1)
    df["return_5d"]  = df["Close"].pct_change(5)
    df["return_20d"] = df["Close"].pct_change(20)

    # Moving averages — trend direction signal
    df["ma_10"]    = df["Close"].rolling(window=10).mean()
    df["ma_30"]    = df["Close"].rolling(window=30).mean()
    df["ma_ratio"] = df["ma_10"] / df["ma_30"]

    # Volatility — rolling standard deviation of 1-day returns
    df["volatility_20d"] = df["return_1d"].rolling(window=20).std()

    # RSI — Relative Strength Index (14-day)
    delta = df["Close"].diff()
    gain  = delta.clip(lower=0).rolling(window=14).mean()
    loss  = (-delta.clip(upper=0)).rolling(window=14).mean()
    rs    = gain / (loss + 1e-8)   # 1e-8 prevents division-by-zero
    df["rsi"] = 100 - (100 / (1 + rs))

    # Volume change — unusual trading activity can precede price moves
    df["volume_change"] = df["Volume"].pct_change(1)

    # Price relative to 52-week extremes — contextualises current price level
    df["price_vs_52w_high"] = df["Close"] / df["Close"].rolling(252).max()
    df["price_vs_52w_low"]  = df["Close"] / df["Close"].rolling(252).min()

    return df

# ===========================================================================
# STEP 9  — Merge ESG features with price data
# ===========================================================================

ESG_FEATURE_COLS = [
    "environment_score", "social_score", "governance_score", "total_score",
    "environment_grade_encoded", "social_grade_encoded",
    "governance_grade_encoded", "total_grade_encoded",
]


OHLCV_NAMES = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}

def _normalise_columns(df, ticker):
    """
    Robustly flatten column headers produced by different yfinance versions.

    yfinance has changed its CSV output format across versions. Three formats
    are seen in the wild:

      Format A - clean single-level (old yfinance):
        columns = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]

      Format B - true MultiIndex (some yfinance versions with auto_adjust=True):
        columns = MultiIndex([("Close","AAPL"), ("High","AAPL"), ...])

      Format C - tuple-as-string (yfinance >= 0.2.x saving to CSV then re-reading):
        The CSV has two header rows. pandas reads the first as the column name
        and the second as the first data row, OR it concatenates them into
        strings like "(\'Close\', \'AAPL\')" or "Close  AAPL".

    This function detects all three and collapses them to bare OHLCV names.
    It also handles the case where the second header row was absorbed as the
    first data row (visible as a row where Close == "Close" or "Price").
    """
    import ast

    # Format B: true MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [str(col[0]).strip() for col in df.columns]
        return df

    # Format C: tuple-as-string or "Name  Ticker" string
    new_cols = []
    for col in df.columns:
        col_str = str(col).strip()
        # Try parsing as a Python tuple literal: "(\'Close\', \'AAPL\')"
        if col_str.startswith("(") and col_str.endswith(")"):
            try:
                parsed = ast.literal_eval(col_str)
                if isinstance(parsed, tuple) and len(parsed) >= 1:
                    new_cols.append(str(parsed[0]).strip())
                    continue
            except (ValueError, SyntaxError):
                pass
        # Try splitting on whitespace: "Close  AAPL" -> "Close"
        parts = col_str.split()
        if len(parts) >= 2 and parts[0] in OHLCV_NAMES:
            new_cols.append(parts[0])
            continue
        new_cols.append(col_str)

    df.columns = new_cols

    # Drop any spurious header row absorbed as a data row.
    # Symptom: the "Close" column first value is the string "Close" or "Price".
    if "Close" in df.columns and len(df) > 0:
        first_val = df["Close"].iloc[0]
        if isinstance(first_val, str) and first_val.strip() in ("Close", "Price", "Adj Close", "Ticker"):
            df = df.iloc[1:].copy()
            
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[df.index.notna()]
    for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df

def build_master_dataframe(esg, args):
    """
    For every ticker that has a downloaded price CSV:
      1. Load the price CSV
      2. Drop rows with missing Close or Volume
      3. Skip tickers with fewer than args.min_days rows
      4. Compute technical features
      5. Create up/down labels
      6. Attach the stock's ESG scores as additional columns
      7. Append to the master list

    Then concatenate all stocks, drop remaining NaN rows (from rolling
    window warmup), and save to data/processed/features.csv.
    """
    log.info("STEP 9 — Merging ESG features with price data")

    available_tickers = [
        f.replace(".csv", "")
        for f in os.listdir(args.price_dir)
        if f.endswith(".csv")
    ]
    log.info(f"  Price files found : {len(available_tickers)}")

    all_stocks = []
    skipped    = []

    for ticker in tqdm(available_tickers, desc="Merging", unit="ticker"):
        # Check if this ticker has ESG data
        esg_match = esg[esg["ticker"] == ticker]
        if esg_match.empty:
            skipped.append((ticker, "no ESG match"))
            continue

        # Load price data
        price_path = os.path.join(args.price_dir, f"{ticker}.csv")
        try:
            prices = pd.read_csv(price_path, index_col=0, parse_dates=True)
        except Exception as e:
            skipped.append((ticker, f"read error: {e}"))
            continue

        # --- Normalise column headers ---
        # Newer yfinance versions write CSVs with a two-row header.
        # When pandas reads that back it produces either:
        #   (a) a true MultiIndex: ("Close", "AAPL")
        #   (b) plain strings that look like tuples: "('Close', 'AAPL')"
        # Both cases must be collapsed to the bare OHLCV name.
        prices = _normalise_columns(prices, ticker)

        # After column normalisation, verify the essential columns exist
        if "Close" not in prices.columns or "Volume" not in prices.columns:
            skipped.append((ticker, "missing Close or Volume after column normalisation"))
            continue

        # Drop rows with missing Close or Volume
        prices = prices.dropna(subset=["Close", "Volume"])

        if len(prices) < args.min_days:
            skipped.append((ticker, f"only {len(prices)} days"))
            continue

        # Compute features and labels
        prices = add_technical_features(prices)
        prices = create_labels(prices, args.horizon)

        # Attach static ESG values to every row
        esg_row = esg_match.iloc[0]
        for col in ESG_FEATURE_COLS:
            prices[col] = esg_row[col]

        prices["industry"] = esg_row["industry"]
        prices["ticker"]   = ticker

        all_stocks.append(prices)

    if not all_stocks:
        log.error("  No stocks were merged. Check your price directory and ESG file.")
        sys.exit(1)

    master_df = pd.concat(all_stocks, axis=0)

    # Drop NaN rows that arise from rolling window warmup periods
    before = len(master_df)
    master_df = master_df.replace([np.inf, -np.inf], np.nan)
    master_df = master_df.dropna()
    log.info(f"  Rows dropped (NaN from rolling warmup): {before - len(master_df):,}")

    # Save master feature file
    out_path = os.path.join(args.out_dir, "features.csv")
    master_df.to_csv(out_path)

    log.info(f"  Stocks successfully merged : {len(all_stocks)}")
    log.info(f"  Stocks skipped             : {len(skipped)}")
    log.info(f"  Master dataset shape       : {master_df.shape}")
    log.info(f"  Saved to                   : {out_path}")

    # Print label distribution
    label_dist = master_df["label"].value_counts(normalize=True).round(3)
    log.info(f"  Label distribution — Up: {label_dist.get(1, 0):.3f}  Down: {label_dist.get(0, 0):.3f}")

    if abs(label_dist.get(1, 0) - 0.5) > 0.1:
        log.warning(
            "  Class imbalance detected (> 60/40 split). "
            "The pos_weight correction will be applied in training."
        )

    return master_df

# ===========================================================================
# STEP 10 — Compute and save class imbalance weight
# ===========================================================================

def save_class_weight(master_df, args):
    """
    Compute pos_weight = num_down / num_up.

    This ratio is passed to BCEWithLogitsLoss during training so the model
    is penalised more for missing the minority class.

    Values close to 1.0  → balanced dataset, no correction needed.
    Values > 1.3         → more "down" days; model under-predicts "up".
    Values < 0.77        → more "up"   days; model under-predicts "down".
    """
    log.info("STEP 10 — Computing class imbalance weight")

    counts   = master_df["label"].value_counts()
    num_down = counts.get(0, 1)
    num_up   = counts.get(1, 1)
    pos_weight = num_down / num_up

    log.info(f"  Up   labels : {num_up:,}")
    log.info(f"  Down labels : {num_down:,}")
    log.info(f"  pos_weight  : {pos_weight:.4f}  "
             f"({'balanced' if 0.77 <= pos_weight <= 1.3 else 'IMBALANCED — use weighted loss'})")

    out_path = os.path.join(args.out_dir, "class_weight.json")
    with open(out_path, "w") as f:
        json.dump({"pos_weight": pos_weight}, f, indent=2)
    log.info(f"  Saved to {out_path}")

# ===========================================================================
# STEP 11 — Build 30-day sliding-window sequences
# ===========================================================================

# Exact column order used for every sequence — must match training notebook
FEATURE_COLS = [
    # Price-based features  (indices 0–8)
    "return_1d", "return_5d", "return_20d",
    "ma_ratio", "volatility_20d", "rsi",
    "volume_change", "price_vs_52w_high", "price_vs_52w_low",
    # ESG features  (indices 9–16)
    "environment_score", "social_score", "governance_score", "total_score",
    "environment_grade_encoded", "social_grade_encoded",
    "governance_grade_encoded", "total_grade_encoded",
]

PRICE_FEATURE_INDICES = list(range(9))   # First 9 columns are price-based

def _build_sequences_for_ticker(group, feature_cols, lookback):
    """
    Convert a single stock's time-series into overlapping (X, y) pairs.

    For every day i starting at index 'lookback':
      X[i] = feature matrix for days [i-lookback, i)  →  shape (lookback, num_features)
      y[i] = binary label for day i

    Building per-ticker prevents any window from spanning two different stocks.
    """
    feat = group[feature_cols].values
    lbl  = group["label"].values

    X, y = [], []
    for i in range(lookback, len(feat)):
        X.append(feat[i - lookback : i])
        y.append(lbl[i])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def build_sequences(master_df, args):
    """
    Iterate over each stock, build sequences, concatenate, and return
    (X_all, y_all) as numpy arrays of shapes:
      X_all : (total_sequences, lookback, num_features)
      y_all : (total_sequences,)
    """
    log.info("STEP 11 — Building sliding-window sequences")
    log.info(f"  Lookback window : {args.lookback} trading days")
    log.info(f"  Feature columns : {len(FEATURE_COLS)}")

    # Verify all feature columns are present
    missing_cols = [c for c in FEATURE_COLS if c not in master_df.columns]
    if missing_cols:
        log.error(f"  Missing feature columns in master_df: {missing_cols}")
        sys.exit(1)

    X_list, y_list = [], []

    for ticker, group in master_df.groupby("ticker"):
        group_sorted = group.sort_index()  # Chronological order is mandatory
        X_t, y_t = _build_sequences_for_ticker(group_sorted, FEATURE_COLS, args.lookback)
        if len(X_t) > 0:
            X_list.append(X_t)
            y_list.append(y_t)

    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)

    log.info(f"  Total sequences : {len(X_all):,}")
    log.info(f"  Sequence shape  : {X_all.shape}  "
             f"→ (num_sequences, lookback_days={args.lookback}, num_features={len(FEATURE_COLS)})")

    return X_all, y_all

# ===========================================================================
# STEP 12 — Chronological train / test split
# ===========================================================================

def split_sequences(X_all, y_all, args):
    """
    Split the sequence array chronologically.

    CRITICAL: We split by position (time order), NEVER randomly.
    Shuffling would allow future data to appear in the training set —
    a form of data leakage that inflates results artificially.

    The first  args.split fraction  of sequences → training set
    The remaining fraction           → test set
    """
    log.info("STEP 12 — Splitting sequences (chronological 80/20 split)")

    split_idx = int(len(X_all) * args.split)

    X_train = X_all[:split_idx]
    y_train = y_all[:split_idx]
    X_test  = X_all[split_idx:]
    y_test  = y_all[split_idx:]

    log.info(f"  Training sequences : {X_train.shape[0]:,}  "
             f"({y_train.mean()*100:.1f}% 'up' labels)")
    log.info(f"  Test sequences     : {X_test.shape[0]:,}  "
             f"({y_test.mean()*100:.1f}% 'up' labels)")

    # Save raw (unnormalised) sequences
    raw_path = os.path.join(args.out_dir, "sequences.npz")
    np.savez(raw_path, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    log.info(f"  Raw sequences saved to {raw_path}")

    return X_train, y_train, X_test, y_test

# ===========================================================================
# STEP 13 — Normalise price-based features
# ===========================================================================

def normalise_sequences(X_train, y_train, X_test, y_test, args):
    """
    Apply StandardScaler to the 9 price-based feature columns (indices 0–8).
    The ESG columns (indices 9–16) were already normalised to [0,1] in Step 4.

    KEY RULE: Fit the scaler on TRAINING data only.
    Then apply the same fitted scaler to TEST data.

    If we fitted on all data first, the scaler would absorb statistics from
    the test set (future period), which constitutes data leakage.

    Workflow:
      1. Reshape 3D arrays to 2D  (sequences × timesteps, features)
      2. Fit scaler on training price columns
      3. Transform training and test price columns separately
      4. Reshape back to 3D
      5. Save normalised arrays
    """
    log.info("STEP 13 — Normalising price-based features (training stats only)")

    n_train, T, F = X_train.shape
    n_test        = X_test.shape[0]

    # Flatten to 2D so scikit-learn's scaler can process the feature columns
    X_train_2d = X_train.reshape(-1, F)
    X_test_2d  = X_test.reshape(-1, F)

    scaler = StandardScaler()

    # fit_transform: compute mean & std from training data, then apply
    X_train_2d[:, PRICE_FEATURE_INDICES] = scaler.fit_transform(
        X_train_2d[:, PRICE_FEATURE_INDICES]
    )
    # transform only: apply training mean & std to test data (no fitting)
    X_test_2d[:, PRICE_FEATURE_INDICES] = scaler.transform(
        X_test_2d[:, PRICE_FEATURE_INDICES]
    )

    # Reshape back to 3D
    X_train_norm = X_train_2d.reshape(n_train, T, F)
    X_test_norm  = X_test_2d.reshape(n_test,  T, F)

    # Sanity check — no NaN or Inf values should remain
    nan_count = np.isnan(X_train_norm).sum() + np.isnan(X_test_norm).sum()
    inf_count = np.isinf(X_train_norm).sum() + np.isinf(X_test_norm).sum()
    if nan_count > 0 or inf_count > 0:
        log.warning(f"  Detected {nan_count} NaN and {inf_count} Inf values after normalisation.")
        log.warning("  These will be replaced with 0. Check your raw data for anomalies.")
        X_train_norm = np.nan_to_num(X_train_norm, nan=0.0, posinf=0.0, neginf=0.0)
        X_test_norm  = np.nan_to_num(X_test_norm,  nan=0.0, posinf=0.0, neginf=0.0)
    else:
        log.info("  Sanity check passed — no NaN or Inf values in normalised sequences")

    norm_path = os.path.join(args.out_dir, "sequences_normalized.npz")
    np.savez(
        norm_path,
        X_train=X_train_norm, y_train=y_train,
        X_test=X_test_norm,   y_test=y_test,
    )
    log.info(f"  Normalised sequences saved to {norm_path}")

    return X_train_norm, X_test_norm

# ===========================================================================
# STEP 14 — Save metadata files used by training and evaluation notebooks
# ===========================================================================

def save_metadata(args):
    """
    Save the feature column list and pipeline configuration to JSON files.
    These are read by the training and evaluation notebooks to ensure
    consistent feature ordering and hyperparameter settings.
    """
    log.info("STEP 14 — Saving metadata")

    # Feature column list (order matters — must match training notebook)
    feat_path = os.path.join(args.out_dir, "feature_cols.json")
    with open(feat_path, "w") as f:
        json.dump(FEATURE_COLS, f, indent=2)
    log.info(f"  Feature columns saved to {feat_path}")

    # Pipeline configuration (for reproducibility)
    config = {
        "esg_path"         : args.esg_path,
        "price_dir"        : args.price_dir,
        "out_dir"          : args.out_dir,
        "start_date"       : args.start,
        "end_date"         : args.end,
        "forecast_horizon" : args.horizon,
        "lookback"         : args.lookback,
        "min_days"         : args.min_days,
        "train_split"      : args.split,
        "num_features"     : len(FEATURE_COLS),
        "price_feature_indices" : PRICE_FEATURE_INDICES,
        "esg_feature_indices"   : list(range(9, len(FEATURE_COLS))),
        "seed"             : SEED,
    }
    cfg_path = os.path.join(args.out_dir, "pipeline_config.json")
    with open(cfg_path, "w") as f:
        json.dump(config, f, indent=2)
    log.info(f"  Pipeline config saved to {cfg_path}")

# ===========================================================================
# STEP 15 — Generate a post-processing diagnostic chart
# ===========================================================================

def plot_sequence_stats(X_train, X_test, args):
    """
    Save a chart showing the feature value distributions after normalisation.
    This confirms that all features are on a comparable scale before training.
    """
    log.info("STEP 15 — Plotting post-normalisation feature distributions")

    # Use the last timestep of each training sequence for inspection
    sample = X_train[:, -1, :]   # Shape: (n_train, num_features)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.boxplot(
        [sample[:, i] for i in range(sample.shape[1])],
        labels=FEATURE_COLS,
        patch_artist=True,
        medianprops={"color": "navy", "linewidth": 2},
    )
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_title("Feature Distributions After Normalisation (training set, last timestep)")
    ax.set_ylabel("Standardised value")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.tight_layout()

    path = os.path.join(args.fig_dir, "feature_distributions_post_norm.png")
    plt.savefig(path, dpi=150)
    plt.close()
    log.info(f"  Saved: {path}")

# ===========================================================================
# MAIN — orchestrate all steps
# ===========================================================================

def main():
    args = parse_args()

    log.info("=" * 60)
    log.info("  ESG STOCK PREDICTION — DATA PRE-PROCESSING PIPELINE")
    log.info("=" * 60)

    # Step 1 — Folders
    create_folders(args)

    # Step 2 — Load ESG
    esg = load_esg(args)

    # Step 3 — Encode grades
    esg = encode_grades(esg)

    # Step 4 — Normalise ESG scores
    esg = normalise_esg_scores(esg)

    # Step 5 — Download prices (can be skipped if CSVs already exist)
    if args.skip_download:
        log.info("STEP 5 — Skipping download (--skip_download flag set)")
    else:
        download_prices(esg, args)

    # Step 6 — Diagnostics
    run_diagnostics(esg, args)

    # Steps 7–9 — Build master dataframe (labels + features + ESG merge)
    master_df = build_master_dataframe(esg, args)

    # Step 10 — Class weight
    save_class_weight(master_df, args)

    # Step 11 — Build sequences
    X_all, y_all = build_sequences(master_df, args)

    # Step 12 — Train/test split
    X_train, y_train, X_test, y_test = split_sequences(X_all, y_all, args)

    # Step 13 — Normalise
    X_train_norm, X_test_norm = normalise_sequences(
        X_train, y_train, X_test, y_test, args
    )

    # Step 14 — Save metadata
    save_metadata(args)

    # Step 15 — Post-normalisation chart
    plot_sequence_stats(X_train_norm, X_test_norm, args)

    log.info("=" * 60)
    log.info("  PRE-PROCESSING COMPLETE")
    log.info("=" * 60)
    log.info(f"  Outputs saved to : {args.out_dir}/")
    log.info(f"  Charts saved to  : {args.fig_dir}/")
    log.info("")
    log.info("  Files ready for model training:")
    log.info(f"    {args.out_dir}/sequences_normalized.npz  ← model input")
    log.info(f"    {args.out_dir}/feature_cols.json          ← feature names")
    log.info(f"    {args.out_dir}/class_weight.json          ← imbalance correction")
    log.info(f"    {args.out_dir}/pipeline_config.json       ← full config")
    log.info("")
    log.info("  Next step: open 03_model_training.ipynb and run all cells.")


if __name__ == "__main__":
    main()