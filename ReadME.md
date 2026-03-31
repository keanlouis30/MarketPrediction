# ESG-Driven Stock Price Predictor

A Deep Learning-based financial forecasting project that analyzes the correlation between Environmental, Social, and Governance (ESG) metrics and stock market performance. This project moves beyond traditional boosted tree models (XGBoost) by utilizing a Neural Network architecture to capture non-linear relationships between corporate sustainability and financial valuation.

## 📊 Project Overview
The core objective is to determine if a company’s ESG scores—specifically Environmental, Social, and Governance pillars—can serve as leading indicators for stock price returns. 

The project utilizes a dataset of over 700 companies (including a specialized focus on the **Energy and Gas sectors**) to model how sustainability performance impacts market sentiment and long-term stock growth.

## 🛠️ Tech Stack
* **Language:** Python 3.10+
* **Deep Learning:** PyTorch / TensorFlow (Keras)
* **Data Manipulation:** Pandas, NumPy
* **Data Acquisition:** yfinance (for historical market prices)
* **Visualization:** Matplotlib, Seaborn
* **Baseline Model:** XGBoost Regressor

## 📉 Dataset Description
The project leverages a primary dataset (`data.csv`) containing the following features:
* **Tickers & Identifiers:** `ticker`, `name`, `industry`, `exchange`.
* **ESG Pillar Scores:** `environment_score`, `social_score`, `governance_score`.
* **Overall Metrics:** `total_score`, `total_grade`, `total_level`.
* **Metadata:** `last_processing_date`, `cik`.

*Note: For the predictive component, this data is merged with daily adjusted closing prices fetched via the Yahoo Finance API.*

## 🧠 Deep Learning Architecture
The project implements a **Residual Multi-Layer Perceptron (Res-MLP)**. Unlike standard regression models, this architecture allows for:
1.  **Entity Embeddings:** Categorical data like `industry` is mapped into high-dimensional space to learn sector-specific ESG impacts.
2.  **Dense Residual Blocks:** Multiple hidden layers (256 -> 128 -> 64) with skip connections to prevent vanishing gradients.
3.  **Regularization:** Integration of **Batch Normalization** and **Dropout (0.3)** to handle the high noise-to-signal ratio typical of stock market data.
4.  **Optimization:** AdamW optimizer with Mean Squared Error (MSE) loss function.

## 🚀 Getting Started

### 1. Installation
```bash
pip install torch pandas yfinance scikit-learn matplotlib
```

### 2. Data Preparation
The script first filters the ESG dataset and joins it with historical price data based on the ticker and last_processing_date.

```python
# Load ESG Data
df = pd.read_csv('data.csv')

# Feature Engineering: Calculate Forward 30-day Returns
# Target = (Price in 30 days - Current Price) / Current Price
df['target'] = df['Adj Close'].pct_change(periods=30).shift(-30)
```

### 3. Training the Model
```python
# Initialize the Deep Learning Regressor
model = ESGPricePredictor(input_dim=X_train.shape[1])

# Train using Backpropagation
train_model(model, train_loader, epochs=50, lr=0.001)
```

### 4. Evaluation Metrics
The model is evaluated against a baseline XGBoost model using:

Mean Absolute Error (MAE): Average magnitude of prediction errors.

Sharpe Ratio: Measuring the risk-adjusted return of the ESG-based portfolio.

Directional Accuracy: How often the model correctly predicts the "Sign" (Up/Down) of the stock movement.

### 5.Future Enhancements
Temporal Fusion Transformers (TFT): Moving from static ESG snapshots to time-series ESG data.

Sentiment Analysis: Integrating NLP to analyze ESG-related news alongside numerical scores.

Gas Sector Deep Dive: Specialized sub-models for energy volatility prediction.