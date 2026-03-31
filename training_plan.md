# Project Roadmap: ESG-Driven Stock Price Predictor

This project plan is structured to take you from the static CSV data you have to a fully functioning Deep Learning prediction system. It is designed to bridge the gap between your previous XGBoost experience and a more advanced Neural Network architecture.

---

### **Phase 1: Environment & Data Engineering**
*The goal is to transform the static ESG snapshots into a training-ready dataset with market targets.*

1.  **Environment Setup:**
    * Initialize a Python virtual environment.
    * Install core dependencies: `torch`, `pandas`, `yfinance`, `scikit-learn`, and `matplotlib`.
2.  **Market Data Acquisition:**
    * Write a script to iterate through the `ticker` column in `data.csv`.
    * Use `yfinance` to download historical Adjusted Closing prices for the last 5 years.
3.  **Target Label Generation:**
    * **Calculate Forward Returns:** Define your prediction window (e.g., 30-day or 90-day returns).
    * **Temporal Alignment:** Align the ESG `last_processing_date` with the corresponding stock price at that time to ensure the model isn't "looking into the future" (Data Leakage prevention).
4.  **Feature Selection:**
    * **Numerical:** Keep `environment_score`, `social_score`, `governance_score`, and `total_score`.
    * **Categorical:** Convert the `industry` column using One-Hot Encoding or Label Encoding.

### **Phase 2: Baseline Modeling (The Benchmark)**
*Before going deep, establish a baseline using your previous toolset to prove the Deep Learning model's value.*

1.  **XGBoost Implementation:**
    * Split data into Training (80%) and Testing (20%) based on time.
    * Train an XGBoost Regressor on the ESG scores to predict the Forward Return.
2.  **Evaluation:**
    * Calculate Mean Absolute Error (MAE) and R-Squared ($R^2$).
    * This establishes the "score to beat."

### **Phase 3: Deep Learning Development**
*Transitioning to the Neural Network architecture.*

1.  **Data Preprocessing for DL:**
    * **Scaling:** Apply `StandardScaler` to all numerical inputs. Neural networks are highly sensitive to unscaled data.
    * **Tensors:** Convert Pandas DataFrames into PyTorch Tensors or TensorFlow Tensors.
2.  **Architecture Design (Res-MLP):**
    * Build a Feed-Forward Network with at least 3 hidden layers.
    * **Dropout Layers:** Add `Dropout(0.3)` between layers to handle the volatility of financial data.
    * **Batch Normalization:** Include `BatchNorm` layers to stabilize training and speed up convergence.
3.  **The Training Loop:**
    * **Loss Function:** Use Mean Squared Error (MSE).
    * **Optimizer:** Use Adam or AdamW.
    * **Early Stopping:** Monitor the validation loss and stop training when it stops improving to prevent overfitting.

### **Phase 4: Specialized Sector Analysis (Gas/Energy Focus)**
*Refining the model for high-impact sectors.*

1.  **Sub-group Testing:**
    * Isolate the 51 Energy/Gas companies identified in your CSV.
    * Run the trained model specifically on these tickers to see if ESG scores have a higher predictive power in carbon-intensive industries compared to "Media" or "Tech."
2.  **Feature Importance:**
    * Use **SHAP** (SHapley Additive exPlanations) or Integrated Gradients to see which ESG pillar (E, S, or G) the neural network values most for gas company stock predictions.

### **Phase 5: Evaluation & Financial Backtesting**
*Translating model accuracy into financial logic.*

1.  **Prediction Visualization:**
    * Create "Predicted vs. Actual" scatter plots.
    * Generate a "Residual Plot" to see where the model fails (e.g., during market crashes).
2.  **Simulated Portfolio:**
    * Select the top 10 stocks with the highest "Predicted Returns" according to the Deep Learning model.
    * Compare the performance of this "ESG-Optimized Portfolio" against a simple S&P 500 index.

### **Phase 6: Documentation & Finalization**
*Wrapping up the project for professional presentation.*

1.  **Code Refactoring:** Clean up the notebooks into modular `.py` scripts (e.g., `data_loader.py`, `model.py`, `train.py`).
2.  **Final README.md:**
    * Populate the README with the project results and metrics (MAE of DL vs. XGBoost).
    * Include the "Getting Started" instructions and requirements.
3.  **Project Summary:**
    * Document findings on whether Environmental scores actually impact Gas company valuations more than Governance scores in the 2025-2026 market context.