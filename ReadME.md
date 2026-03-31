# Project Title: ESG-Driven Stock Performance Predictor

## Objective
To determine if **ESG (Environmental, Social, Governance)** scores can serve as leading indicators for stock price returns or volatility.

---

## Phase 1: Data Integration & Augmentation
The provided `data.csv` contains high-quality ESG snapshots, but stock prediction requires time-series pricing data.

* **ESG Feature Extraction:** Use `environment_score`, `social_score`, `governance_score`, and `total_score` as primary features.
* **External Data Fetching:** Use the `ticker` column to fetch 2–5 years of historical stock prices (Open, High, Low, Close, Volume) using APIs like `yfinance` or `Alpha Vantage`.
* **Target Variable Definition:**
    * **Regression:** Predict the "Forward 30-day Return" or "Next Month's Closing Price."
    * **Classification:** Predict if the stock will outperform the S&P 500 index (Binary: 0 or 1).

---

## Phase 2: Feature Engineering
Since Deep Learning models are sensitive to scale:

* **Normalization:** Use `StandardScaler` or `MinMaxScaler` on ESG scores (0–1000 range) and prices.
* **Technical Indicators:** Add RSI, MACD, and Moving Averages to complement the "fundamental" ESG data.
* **Sentiment Analysis (Optional):** Scrape news for the `name` column to add a "Social Sentiment" score.

---

## Phase 3: Model Architectures
Since you've done XGBoost, you should explore these two Deep Learning paths:

### Option A: TabNet (Attentive Interpretable Tabular Learning)
* **Why it fits:** It is a state-of-the-art DL architecture specifically for tabular data. It uses sequential attention to choose which features to reason from at each decision step.
* **Benefit:** It provides **feature importance "masks,"** allowing you to see if "Social" was more important than "Governance" for a specific prediction.

### Option B: Temporal Fusion Transformer (TFT) / Hybrid LSTM
* **Architecture:** Use an LSTM layer for historical price trends and a Dense/Embedding layer for static ESG scores.
* **Structure:**
    1.  **Input 1:** Time-series data (Prices).
    2.  **Input 2:** Categorical/Static data (Industry, ESG Grades).
    3.  **Fusion:** Concatenate outputs into a final **MLP (Multi-Layer Perceptron)** for the prediction.

---

## Phase 4: Implementation Workflow
1.  **Baseline:** Train an XGBoost or LightGBM regressor using the CSV features to set a "score to beat."
2.  **DL Setup:** Build the model using `PyTorch` or `TensorFlow/Keras`.
3.  **Training:** Implement **Early Stopping** to prevent overfitting, as stock data is notoriously noisy.
4.  **Backtesting:** Simulate a portfolio that buys the "Top 10" stocks with the highest predicted ESG-driven returns.

---

## Phase 5: Evaluation Metrics
* **Quantitative:** Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).
* **Financial:** **Sharpe Ratio** and **Maximum Drawdown** of the ESG-predicted portfolio vs. a Benchmark (e.g., SPY).

---

## Phase 6: Visualization & Reporting
* **ESG Heatmaps:** Correlate `total_score` with actual price growth.
* **Prediction vs. Actual:** Plot a time-series graph for sample tickers (e.g., DIS, GM, GWW) showing the model’s predicted price vs. the real price.