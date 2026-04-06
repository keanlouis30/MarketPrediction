"""
train.py
========
Improved LSTM + Attention training script for the ESG Stock Direction Prediction project.

Changes from the original approach that address overfitting and low accuracy:

  PROBLEM 1 — Overfitting by epoch 5
  Causes:
    - Model too large for the actual signal strength
    - Sequences heavily correlated (29/30 days overlap between adjacent windows)
    - Not enough regularisation
  Fixes applied:
    - Smaller architecture (hidden=32, layers=1)
    - Higher dropout (0.5)
    - Sequence subsampling: keep every 3rd sequence per stock to reduce correlation
    - Weight decay (L2 regularisation) in the optimiser
    - Gradient clipping tightened to 0.5
    - Label smoothing: prevents over-confident predictions on noisy data

  PROBLEM 2 — Accuracy stuck at ~55%
  Causes:
    - Static ESG scores repeated identically across all dates for a stock
      taught the model to use ESG as a stock identifier, not a market signal
    - Too few diverse features relative to market noise
    - No separate validation set for threshold tuning
  Fixes applied:
    - Industry-relative ESG: each stock's ESG z-score within its industry
      sector — captures whether a stock is above or below its peers
    - 7 derived window features: MACD histogram, Bollinger Band position,
      OBV momentum, return skewness, kurtosis, trend strength, volatility trend
    - Walk-forward validation: train 70% / val 15% / test 15% — clean time split
    - Threshold tuning: find the decision threshold that maximises F1 on val set
    - Ensemble of 3 models with different random seeds — averages out noise

Usage:
  python train.py

  Optional flags:
    --out_dir      PATH   Where processed data lives  (default: data/processed)
    --results_dir  PATH   Where to save results       (default: results)
    --hidden       INT    LSTM hidden size             (default: 32)
    --layers       INT    Number of LSTM layers        (default: 1)
    --dropout      FLOAT  Dropout probability          (default: 0.5)
    --epochs       INT    Max training epochs          (default: 80)
    --batch        INT    Batch size                   (default: 128)
    --lr           FLOAT  Initial learning rate        (default: 0.0005)
    --patience     INT    Early stopping patience      (default: 10)
    --subsample    INT    Keep every Nth sequence      (default: 3)
    --label_smooth FLOAT  Label smoothing epsilon      (default: 0.1)
    --n_ensemble   INT    Number of ensemble models    (default: 3)
    --no_ensemble         Train a single model only
"""

import argparse
import json
import logging
import os
import random
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="ESG stock prediction — improved training")
    p.add_argument("--out_dir",      default="data/processed")
    p.add_argument("--results_dir",  default="results")
    p.add_argument("--hidden",       type=int,   default=32)
    p.add_argument("--layers",       type=int,   default=1)
    p.add_argument("--dropout",      type=float, default=0.5)
    p.add_argument("--epochs",       type=int,   default=80)
    p.add_argument("--batch",        type=int,   default=128)
    p.add_argument("--lr",           type=float, default=0.0005)
    p.add_argument("--patience",     type=int,   default=10)
    p.add_argument("--subsample",    type=int,   default=3)
    p.add_argument("--label_smooth", type=float, default=0.1)
    p.add_argument("--n_ensemble",   type=int,   default=3)
    p.add_argument("--no_ensemble",  action="store_true")
    return p.parse_args()

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ===========================================================================
# IMPROVEMENT 1 — Industry-relative ESG features
# ===========================================================================

def add_industry_relative_esg(master_csv_path, X_train, y_train, X_test, y_test, feature_cols):
    """
    Problem: static ESG scores repeated for every date of a stock teach the
    model to use ESG as a stock identifier ('0.41 = AAPL') rather than a
    predictive signal.

    Fix: compute each stock's ESG z-score within its industry.
    A z-score of +1.5 means the stock scores 1.5 standard deviations above
    its sector peers. This is meaningful even with a static snapshot because
    it captures relative positioning, not absolute values.

    Requires features.csv (produced by preprocess.py) to be present.
    If it is missing the function returns the input arrays unchanged.

    New features added (4 columns):
      env_vs_industry   — environment z-score within industry
      soc_vs_industry   — social z-score within industry
      gov_vs_industry   — governance z-score within industry
      total_vs_industry — total ESG z-score within industry
    """
    import pandas as pd

    log.info("IMPROVEMENT 1 — Computing industry-relative ESG z-scores")

    master = pd.read_csv(master_csv_path, index_col=0, parse_dates=True, low_memory=False)

    score_cols   = ["environment_score", "social_score", "governance_score", "total_score"]
    new_col_names = ["env_vs_industry", "soc_vs_industry", "gov_vs_industry", "total_vs_industry"]

    # One row per ticker with its ESG scores and industry
    ticker_esg = master.groupby("ticker")[score_cols + ["industry"]].first().reset_index()

    # Compute z-score per ticker within its industry
    for score_col, new_col in zip(score_cols, new_col_names):
        industry_stats = (
            ticker_esg.groupby("industry")[score_col]
            .agg(mean="mean", std="std")
            .reset_index()
        )
        ticker_esg = ticker_esg.merge(industry_stats, on="industry", how="left")
        ticker_esg[new_col] = (
            (ticker_esg[score_col] - ticker_esg["mean"])
            / (ticker_esg["std"].fillna(1.0) + 1e-8)
        )
        ticker_esg = ticker_esg.drop(columns=["mean", "std"])

    # Build lookup: ticker -> [z_env, z_soc, z_gov, z_total]
    ticker_rel = ticker_esg.set_index("ticker")[new_col_names].to_dict(orient="index")

    # Derive per-sequence ticker mapping from master dataframe
    LOOKBACK = X_train.shape[1]

    def build_ticker_seq_labels(df, lookback):
        labels = []
        for ticker, group in df.groupby("ticker"):
            n_seq = len(group.sort_index()) - lookback
            if n_seq > 0:
                labels.extend([ticker] * n_seq)
        return np.array(labels)

    all_ticker_labels = build_ticker_seq_labels(master, LOOKBACK)
    n_total = len(X_train) + len(X_test)

    if len(all_ticker_labels) != n_total:
        log.warning(
            f"  Ticker label count ({len(all_ticker_labels)}) != "
            f"total sequences ({n_total}). Skipping industry-relative ESG."
        )
        return X_train, y_train, X_test, y_test, feature_cols

    ticker_train = all_ticker_labels[:len(X_train)]
    ticker_test  = all_ticker_labels[len(X_train):]

    def append_rel_esg(X, tickers):
        n, T, F = X.shape
        rel = np.zeros((n, T, 4), dtype=np.float32)
        for i, ticker in enumerate(tickers):
            if ticker in ticker_rel:
                z = list(ticker_rel[ticker].values())
                rel[i, :, :] = z  # same z-score broadcast across all 30 timesteps
        return np.concatenate([X, rel], axis=2)

    X_train_new = append_rel_esg(X_train, ticker_train)
    X_test_new  = append_rel_esg(X_test,  ticker_test)

    new_feature_cols = feature_cols + new_col_names
    log.info(f"  Features: {len(feature_cols)} -> {len(new_feature_cols)}")

    return X_train_new, y_train, X_test_new, y_test, new_feature_cols


# ===========================================================================
# IMPROVEMENT 2 — Derived window features
# ===========================================================================

def add_derived_features(X, feature_cols):
    """
    Compute 7 additional features from within each 30-day window.
    These capture patterns not visible from individual timesteps alone.

    All computations use only data within the window — no look-ahead.

    New features (appended as 7 extra columns):
      macd_signal    MACD histogram at each timestep (fast EMA - slow EMA of returns)
                     Captures momentum divergence: rising MACD = strengthening uptrend
      bb_position    Bollinger Band position: (return - 20d mean) / (2 * 20d std)
                     Values near +1 or -1 indicate price is at the band edge
      obv_momentum   On-Balance Volume momentum: cumulative signed volume changes
                     Positive = volume flowing in (accumulation) = bullish signal
      return_skew    Skewness of returns in the window (scalar, same at all steps)
                     Negative skew = more large down-days; positive = more up-days
      return_kurt    Excess kurtosis of returns (scalar)
                     High kurtosis = fat tails = crash/spike risk in this window
      trend_strength Normalised linear regression slope across the 30-day window
                     Positive = upward trend; negative = downward trend
      vol_trend      Ratio of last-10-day volatility to first-10-day volatility
                     > 1 = volatility increasing (uncertainty rising)
                     < 1 = volatility decreasing (calming down)
    """
    ret1d_idx = feature_cols.index("return_1d")
    vol_idx   = feature_cols.index("volume_change")

    n, T, F = X.shape
    derived = np.zeros((n, T, 7), dtype=np.float32)

    for i in range(n):
        r = X[i, :, ret1d_idx].astype(np.float64)  # (30,) daily returns
        v = X[i, :, vol_idx].astype(np.float64)    # (30,) volume changes

        # --- MACD signal ---
        def ema(x, span):
            alpha = 2.0 / (span + 1)
            out = np.zeros_like(x)
            out[0] = x[0]
            for t in range(1, len(x)):
                out[t] = alpha * x[t] + (1 - alpha) * out[t-1]
            return out
        fast = ema(r, 12)
        slow = ema(r, min(26, T))
        macd = fast - slow

        # --- Bollinger Band position at each timestep ---
        bb = np.zeros(T)
        for t in range(T):
            window = r[max(0, t - 19):t + 1]
            if len(window) > 1:
                mu  = window.mean()
                sig = window.std() + 1e-8
                bb[t] = (r[t] - mu) / (2 * sig)

        # --- OBV momentum ---
        signs   = np.sign(r)
        obv_raw = np.cumsum(signs * np.abs(v))
        obv_max = np.abs(obv_raw).max() + 1e-8
        obv     = obv_raw / obv_max

        # --- Return statistics (scalars broadcast across T) ---
        std_r = r.std() + 1e-8
        skew  = float(np.mean(((r - r.mean()) / std_r) ** 3))
        kurt  = float(np.mean(((r - r.mean()) / std_r) ** 4) - 3.0)

        # --- Trend strength: normalised OLS slope ---
        t_idx = np.arange(T, dtype=np.float64)
        if std_r > 1e-8:
            slope = np.polyfit(t_idx, r, 1)[0]
            trend = float(slope / std_r)
        else:
            trend = 0.0

        # --- Volatility trend ---
        vol_first = r[:10].std() + 1e-8
        vol_last  = r[-10:].std() + 1e-8
        vol_ratio = float(vol_last / vol_first)

        derived[i, :, 0] = macd.astype(np.float32)
        derived[i, :, 1] = bb.astype(np.float32)
        derived[i, :, 2] = obv.astype(np.float32)
        derived[i, :, 3] = np.float32(skew)
        derived[i, :, 4] = np.float32(kurt)
        derived[i, :, 5] = np.float32(trend)
        derived[i, :, 6] = np.float32(vol_ratio)

    X_new      = np.concatenate([X, derived], axis=2).astype(np.float32)
    new_cols   = feature_cols + [
        "macd_signal", "bb_position", "obv_momentum",
        "return_skew", "return_kurt", "trend_strength", "vol_trend"
    ]
    return X_new, new_cols


# ===========================================================================
# IMPROVEMENT 3 — Sequence subsampling
# ===========================================================================

def subsample_sequences(X, y, every_n=3):
    """
    Adjacent sequences overlap by (lookback - 1) / lookback = 97% of their data.
    Training on all of them is equivalent to showing the model the same window
    30 times with a 1-day shift. The model memorises it.

    Keeping every Nth sequence reduces this overlap:
      every_n=3 -> adjacent kept sequences overlap by 27/30 = 90%
      every_n=5 -> adjacent kept sequences overlap by 25/30 = 83%

    This reduces training set size but dramatically improves generalisation.
    Validation and test sets are NOT subsampled (we want full coverage there).
    """
    idx = np.arange(0, len(X), every_n)
    return X[idx], y[idx]


# ===========================================================================
# IMPROVEMENT 4 — Walk-forward split
# ===========================================================================

def walk_forward_split(X_all, y_all, train_frac=0.70, val_frac=0.15):
    """
    Adds a dedicated validation set to the 80/20 split.

    Structure:
      First 70% of all sequences  -> training set
      Next  15% of all sequences  -> validation set  (for threshold tuning)
      Final 15% of all sequences  -> test set        (final reported results)

    All splits are chronological — no shuffling.
    """
    n         = len(X_all)
    train_end = int(n * train_frac)
    val_end   = int(n * (train_frac + val_frac))

    return (
        X_all[:train_end],       y_all[:train_end],
        X_all[train_end:val_end], y_all[train_end:val_end],
        X_all[val_end:],          y_all[val_end:],
    )


# ===========================================================================
# MODEL
# ===========================================================================

class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, lstm_out):
        weights = torch.softmax(self.attn(lstm_out), dim=1)  # (batch, T, 1)
        context = (weights * lstm_out).sum(dim=1)             # (batch, hidden)
        return context, weights


class ESGPredictor(nn.Module):
    """
    Smaller, more regularised LSTM with attention.

    Key design choices vs the original:
      hidden_size=32  (was 64) — fewer parameters means less memorisation capacity
      num_layers=1    (was 2)  — one LSTM layer is sufficient for 30-step sequences
      dropout=0.5     (was 0.3) — more aggressive, appropriate for noisy financial data
      BatchNorm1d before FC — stabilises activations, reduces internal covariate shift
      No sigmoid — using BCEWithLogitsLoss which combines sigmoid + BCE more stably
    """
    def __init__(self, input_size, hidden_size=32, num_layers=1, dropout=0.5):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0
        )
        self.attention  = AttentionLayer(hidden_size)
        self.dropout    = nn.Dropout(dropout)
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.fc         = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _           = self.lstm(x)
        context, attn_weights = self.attention(lstm_out)
        out = self.dropout(context)
        out = self.batch_norm(out)
        out = self.fc(out).squeeze(-1)  # raw logit
        return out, attn_weights


# ===========================================================================
# LABEL SMOOTHING LOSS
# ===========================================================================

class LabelSmoothingBCELoss(nn.Module):
    """
    Standard BCE drives the model toward exactly 0.0 or 1.0 confidence.
    For noisy financial labels (the same 5-day window can be 'up' in one
    year and 'down' in another for similar conditions), this causes the
    model to overfit to memorised patterns.

    Label smoothing softens the targets:
      hard 0  →  epsilon / 2       (e.g. 0.05 with epsilon=0.1)
      hard 1  →  1 - epsilon / 2   (e.g. 0.95)

    The model can never achieve zero loss, which prevents over-optimisation
    on any single training example. This is standard practice in NLP and
    increasingly used in financial ML.
    """
    def __init__(self, epsilon=0.1, pos_weight=None):
        super().__init__()
        self.epsilon    = epsilon
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        smooth = targets * (1 - self.epsilon) + (1 - targets) * (self.epsilon / 2)
        loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=(
                self.pos_weight.to(logits.device)
                if self.pos_weight is not None else None
            )
        )
        return loss_fn(logits, smooth)


# ===========================================================================
# TRAINING LOOP
# ===========================================================================

def train_one_model(X_train, y_train, X_val, y_val,
                    input_size, pos_weight, args, seed, device):
    """
    Train a single ESGPredictor.
    Returns: best model state_dict, train loss history, val loss history, best threshold.
    """
    set_seed(seed)

    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_ds   = TensorDataset(torch.FloatTensor(X_val),   torch.FloatTensor(y_val))

    # shuffle=True within the training set is fine here — chronological integrity
    # is already enforced at the split level; within-split shuffling adds robustness
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False)

    model = ESGPredictor(input_size, args.hidden, args.layers, args.dropout).to(device)

    pw        = torch.tensor([pos_weight]) if abs(pos_weight - 1.0) > 0.05 else None
    criterion = LabelSmoothingBCELoss(epsilon=args.label_smooth, pos_weight=pw)

    # Adam with weight_decay applies L2 penalty to all weights every step
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Cosine annealing: LR decays smoothly from lr down to lr/100 over all epochs
    # More stable than ReduceLROnPlateau for small volatile loss curves
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr / 100
    )

    best_val_loss    = float("inf")
    patience_counter = 0
    best_state       = None
    train_losses, val_losses = [], []

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        epoch_loss = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits, _ = model(Xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()
        avg_train = epoch_loss / len(train_loader)

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                logits, _ = model(Xb)
                val_loss += criterion(logits, yb).item()
        avg_val = val_loss / len(val_loader)

        train_losses.append(avg_train)
        val_losses.append(avg_val)

        if epoch % 10 == 0 or epoch == 1:
            log.info(f"    Epoch {epoch:3d}/{args.epochs} | "
                     f"Train: {avg_train:.4f} | Val: {avg_val:.4f} | "
                     f"LR: {scheduler.get_last_lr()[0]:.6f}")

        if avg_val < best_val_loss:
            best_val_loss    = avg_val
            patience_counter = 0
            best_state       = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                log.info(f"    Early stop at epoch {epoch}  (best val: {best_val_loss:.4f})")
                break

    # --- Tune decision threshold on validation set ---
    # Default threshold (0.5) is rarely optimal.
    # We search for the threshold that maximises F1 on the validation set.
    model.load_state_dict(best_state)
    model.eval()
    val_probs = []
    with torch.no_grad():
        for Xb, _ in val_loader:
            probs = torch.sigmoid(model(Xb.to(device))[0]).cpu().numpy()
            val_probs.extend(probs)
    val_probs = np.array(val_probs)

    best_thresh, best_f1 = 0.5, 0.0
    for thresh in np.arange(0.35, 0.66, 0.01):
        preds = (val_probs > thresh).astype(int)
        f1    = f1_score(y_val[:len(val_probs)], preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, thresh

    log.info(f"    Best threshold: {best_thresh:.2f}  (val F1={best_f1:.4f})")
    return best_state, train_losses, val_losses, best_thresh


# ===========================================================================
# EVALUATION
# ===========================================================================

def ensemble_predict(states, X_np, y_true, input_size, threshold, args, device):
    """Average predicted probabilities across all models in the ensemble."""
    all_probs = []
    all_attn  = []

    ds     = TensorDataset(torch.FloatTensor(X_np))
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False)

    for state in states:
        model = ESGPredictor(input_size, args.hidden, args.layers, args.dropout).to(device)
        model.load_state_dict(state)
        model.eval()

        probs_batch, attn_batch = [], []
        with torch.no_grad():
            for (Xb,) in loader:
                logits, attn = model(Xb.to(device))
                probs_batch.extend(torch.sigmoid(logits).cpu().numpy())
                attn_batch.append(attn.cpu().numpy())

        all_probs.append(np.array(probs_batch))
        all_attn.append(np.concatenate(attn_batch, axis=0))

    avg_probs = np.mean(all_probs, axis=0)
    avg_attn  = np.mean(all_attn,  axis=0)
    avg_preds = (avg_probs > threshold).astype(int)
    return avg_probs, avg_preds, avg_attn


def print_metrics(name, y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob)
    log.info(f"\n  {'='*50}")
    log.info(f"  {name}")
    log.info(f"  {'='*50}")
    log.info(f"  Accuracy : {acc:.4f}  ({acc*100:.2f}%)")
    log.info(f"  F1 Score : {f1:.4f}")
    log.info(f"  AUC-ROC  : {auc:.4f}")
    log.info(f"\n{classification_report(y_true, y_pred, target_names=['Down','Up'])}")
    return {"accuracy": float(acc), "f1": float(f1), "auc": float(auc)}


# ===========================================================================
# PLOTTING
# ===========================================================================

def plot_training_curves(all_train, all_val, name, fig_dir):
    plt.figure(figsize=(10, 4))
    max_len = min(len(t) for t in all_train)
    for i, (tr, vl) in enumerate(zip(all_train, all_val)):
        plt.plot(tr, alpha=0.3, color="steelblue")
        plt.plot(vl, alpha=0.3, color="coral")
    plt.plot(np.mean([t[:max_len] for t in all_train], axis=0),
             color="steelblue", linewidth=2.5, label="Mean train")
    plt.plot(np.mean([v[:max_len] for v in all_val], axis=0),
             color="coral",     linewidth=2.5, label="Mean val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training vs Validation Loss — {name}")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(fig_dir, f"training_curves_{name.lower().replace(' ','_')}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    log.info(f"  Saved: {path}")


def plot_roc(y_test, probs_esg, probs_base, m_esg, m_base, fig_dir):
    fpr_e, tpr_e, _ = roc_curve(y_test, probs_esg)
    fpr_b, tpr_b, _ = roc_curve(y_test, probs_base)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr_e, tpr_e, color="steelblue", lw=2,
             label=f"ESG + Price  (AUC={m_esg['auc']:.3f})")
    plt.plot(fpr_b, tpr_b, color="coral", lw=2, linestyle="--",
             label=f"Price-only  (AUC={m_base['auc']:.3f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle=":", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves — ESG Model vs Baseline")
    plt.legend(loc="lower right")
    plt.tight_layout()
    path = os.path.join(fig_dir, "roc_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    log.info(f"  Saved: {path}")


def plot_confusion(y_test, preds_base, preds_esg, fig_dir):
    try:
        import seaborn as sns
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, preds, name in zip(
            axes,
            [preds_base, preds_esg],
            ["Price-Only Baseline", "ESG + Price Model"]
        ):
            cm = confusion_matrix(y_test, preds)
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                        xticklabels=["Pred Down", "Pred Up"],
                        yticklabels=["Actual Down", "Actual Up"])
            ax.set_title(name)
        plt.suptitle("Confusion Matrices", y=1.02)
        plt.tight_layout()
        path = os.path.join(fig_dir, "confusion_matrices.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        log.info(f"  Saved: {path}")
    except ImportError:
        log.warning("  seaborn not installed — skipping confusion matrix plot")


def plot_attention(attn, lookback, fig_dir):
    avg  = attn.mean(axis=0).flatten()[:lookback]
    days = [f"t-{lookback - i}" for i in range(lookback)]
    plt.figure(figsize=(13, 4))
    plt.bar(days, avg, color="steelblue", edgecolor="white")
    plt.xlabel("Days before prediction date")
    plt.ylabel("Average attention weight")
    plt.title("Average Attention Weights Across Test Set")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    path = os.path.join(fig_dir, "attention_weights.png")
    plt.savefig(path, dpi=150)
    plt.close()
    log.info(f"  Saved: {path}")


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.results_dir, exist_ok=True)
    fig_dir = os.path.join(args.results_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    log.info("=" * 60)
    log.info("  ESG STOCK PREDICTION — IMPROVED TRAINING PIPELINE")
    log.info("=" * 60)
    log.info(f"  Device          : {device}")
    log.info(f"  Hidden size     : {args.hidden}")
    log.info(f"  Dropout         : {args.dropout}")
    log.info(f"  Label smoothing : {args.label_smooth}")
    log.info(f"  Subsample every : {args.subsample}")
    log.info(f"  Ensemble size   : {1 if args.no_ensemble else args.n_ensemble}")

    # --- Load base sequences ---
    seq_path = os.path.join(args.out_dir, "sequences_normalized.npz")
    if not os.path.exists(seq_path):
        log.error(f"sequences_normalized.npz not found at {seq_path}. Run preprocess.py first.")
        sys.exit(1)

    raw      = np.load(seq_path)
    X_all    = np.concatenate([raw["X_train"], raw["X_test"]], axis=0)
    y_all    = np.concatenate([raw["y_train"], raw["y_test"]], axis=0)

    with open(os.path.join(args.out_dir, "feature_cols.json")) as f:
        feature_cols = json.load(f)
    with open(os.path.join(args.out_dir, "class_weight.json")) as f:
        pos_weight = json.load(f)["pos_weight"]

    log.info(f"\n  Total sequences : {len(X_all):,}")
    log.info(f"  Base features   : {len(feature_cols)}")

    # --- Improvement 1: industry-relative ESG ---
    master_csv = os.path.join(args.out_dir, "features.csv")
    if os.path.exists(master_csv):
        try:
            X_tr_tmp = X_all[:int(len(X_all)*0.70)]
            y_tr_tmp = y_all[:int(len(y_all)*0.70)]
            X_te_tmp = X_all[int(len(X_all)*0.70):]
            y_te_tmp = y_all[int(len(y_all)*0.70):]

            X_tr_tmp, y_tr_tmp, X_te_tmp, y_te_tmp, feature_cols = \
                add_industry_relative_esg(master_csv, X_tr_tmp, y_tr_tmp,
                                          X_te_tmp, y_te_tmp, feature_cols)
            X_all = np.concatenate([X_tr_tmp, X_te_tmp], axis=0)
            y_all = np.concatenate([y_tr_tmp, y_te_tmp], axis=0)
            log.info(f"  Industry-relative ESG added.")
        except Exception as e:
            log.warning(f"  Industry-relative ESG failed ({e}). Continuing without it.")
    else:
        log.warning(f"  {master_csv} not found. Skipping industry-relative ESG.")

    # --- Walk-forward split ---
    X_train, y_train, X_val, y_val, X_test, y_test = walk_forward_split(X_all, y_all)
    log.info(f"\n  Walk-forward split:")
    log.info(f"    Train : {len(X_train):,}  ({y_train.mean()*100:.1f}% up)")
    log.info(f"    Val   : {len(X_val):,}    ({y_val.mean()*100:.1f}% up)")
    log.info(f"    Test  : {len(X_test):,}   ({y_test.mean()*100:.1f}% up)")

    # --- Improvement 2: derived window features ---
    log.info("\n  Computing derived window features (train)...")
    X_train, feature_cols = add_derived_features(X_train, feature_cols)
    log.info("  Computing derived window features (val)...")
    X_val,   _            = add_derived_features(X_val,   feature_cols[:len(feature_cols)-7])
    log.info("  Computing derived window features (test)...")
    X_test,  _            = add_derived_features(X_test,  feature_cols[:len(feature_cols)-7])

    log.info(f"  Final feature count: {len(feature_cols)}")

    # --- Improvement 3: subsample training sequences ---
    X_train_sub, y_train_sub = subsample_sequences(X_train, y_train, every_n=args.subsample)
    log.info(f"  Training sequences after subsampling: {len(X_train_sub):,}")

    INPUT_SIZE = X_train_sub.shape[2]

    # Price-only indices (first 9 columns are always price-based)
    price_idx = list(range(9))
    X_tr_price  = X_train_sub[:, :, price_idx]
    X_val_price = X_val[:,       :, price_idx]
    X_te_price  = X_test[:,      :, price_idx]

    seeds   = [42, 123, 456] if not args.no_ensemble else [42]
    n_seeds = len(seeds)

    # --- Train ESG models ---
    log.info(f"\n{'='*52}")
    log.info(f"  Training ESG + Price model  ({n_seeds} seed(s))")
    log.info(f"{'='*52}")
    esg_states, esg_tr_hist, esg_vl_hist, esg_thresholds = [], [], [], []
    for i, seed in enumerate(seeds):
        log.info(f"\n  Seed {seed}  ({i+1}/{n_seeds})")
        st, tr, vl, th = train_one_model(
            X_train_sub, y_train_sub, X_val, y_val,
            INPUT_SIZE, pos_weight, args, seed, device
        )
        esg_states.append(st)
        esg_tr_hist.append(tr)
        esg_vl_hist.append(vl)
        esg_thresholds.append(th)

    avg_esg_thresh = float(np.mean(esg_thresholds))

    # --- Train baseline models ---
    log.info(f"\n{'='*52}")
    log.info(f"  Training price-only baseline  ({n_seeds} seed(s))")
    log.info(f"{'='*52}")
    base_states, base_tr_hist, base_vl_hist, base_thresholds = [], [], [], []
    for i, seed in enumerate(seeds):
        log.info(f"\n  Seed {seed}  ({i+1}/{n_seeds})")
        st, tr, vl, th = train_one_model(
            X_tr_price, y_train_sub, X_val_price, y_val,
            len(price_idx), pos_weight, args, seed, device
        )
        base_states.append(st)
        base_tr_hist.append(tr)
        base_vl_hist.append(vl)
        base_thresholds.append(th)

    avg_base_thresh = float(np.mean(base_thresholds))

    # --- Ensemble predictions ---
    log.info("\n  Generating ensemble predictions on test set...")
    probs_esg,  preds_esg,  attn_esg = ensemble_predict(
        esg_states,  X_test,    y_test, INPUT_SIZE,    avg_esg_thresh,  args, device
    )
    probs_base, preds_base, _        = ensemble_predict(
        base_states, X_te_price, y_test, len(price_idx), avg_base_thresh, args, device
    )

    # --- Print results ---
    log.info("\n")
    m_esg  = print_metrics("ESG + Price Model (ensemble)",  y_test, preds_esg,  probs_esg)
    m_base = print_metrics("Price-Only Baseline (ensemble)", y_test, preds_base, probs_base)

    improvement = {k: round(m_esg[k] - m_base[k], 4) for k in m_esg}
    log.info(f"\n  ESG improvement over baseline:")
    for k, v in improvement.items():
        log.info(f"    {k:12s}: {v:+.4f}")

    # --- Save models ---
    for state, seed in zip(esg_states, seeds):
        torch.save(state, os.path.join(args.results_dir, f"model_esg_seed{seed}.pt"))
    for state, seed in zip(base_states, seeds):
        torch.save(state, os.path.join(args.results_dir, f"model_baseline_seed{seed}.pt"))

    # --- Save updated feature list ---
    with open(os.path.join(args.out_dir, "feature_cols_extended.json"), "w") as f:
        json.dump(feature_cols, f, indent=2)

    # --- Plots ---
    plot_training_curves(esg_tr_hist,  esg_vl_hist,  "ESG Model", fig_dir)
    plot_training_curves(base_tr_hist, base_vl_hist, "Baseline",  fig_dir)
    plot_roc(y_test, probs_esg, probs_base, m_esg, m_base, fig_dir)
    plot_confusion(y_test, preds_base, preds_esg, fig_dir)
    plot_attention(attn_esg, X_test.shape[1], fig_dir)

    # --- Save results JSON ---
    results = {
        "config": {
            "hidden_size"    : args.hidden,
            "num_layers"     : args.layers,
            "dropout"        : args.dropout,
            "label_smoothing": args.label_smooth,
            "subsample_n"    : args.subsample,
            "n_ensemble"     : n_seeds,
            "seeds"          : seeds,
            "esg_threshold"  : avg_esg_thresh,
            "base_threshold" : avg_base_thresh,
            "num_features"   : len(feature_cols),
        },
        "esg_model"      : m_esg,
        "baseline_model" : m_base,
        "improvement"    : improvement,
    }
    with open(os.path.join(args.results_dir, "training_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    log.info("\n" + "=" * 60)
    log.info("  TRAINING COMPLETE")
    log.info("=" * 60)
    log.info(f"  ESG Model — Acc: {m_esg['accuracy']*100:.2f}%  "
             f"F1: {m_esg['f1']:.4f}  AUC: {m_esg['auc']:.4f}")
    log.info(f"  Baseline  — Acc: {m_base['accuracy']*100:.2f}%  "
             f"F1: {m_base['f1']:.4f}  AUC: {m_base['auc']:.4f}")
    log.info(f"  ESG lift  — Acc: {improvement['accuracy']:+.4f}  "
             f"F1: {improvement['f1']:+.4f}  AUC: {improvement['auc']:+.4f}")
    log.info(f"\n  Results saved to {args.results_dir}/training_results.json")


if __name__ == "__main__":
    main()
