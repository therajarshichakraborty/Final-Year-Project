# TECHNICAL DOCUMENTATION, PRESENTATION GUIDE AND VIVA QUESTION BANK

## PROJECT TITLE: Stock Market Prediction and Buy/Hold/Sell Recommendation System using Deep Stacked Bidirectional LSTM
**Target Presentation Date:** 22nd July (6th Semester Mini Project Evaluation)  
**Reference Notebook:** [`stock_market_using_LSTM_for_6th_sem_presentation_on_22th_july.ipynb`](file:///d:/Rajarshi%20Chakraborty%20%28%20Arghya%20%29/Projects/LSTM%20-%20Stock%20Market%20-%20college%20mini%20project/stock_market_using_LSTM_for_6th_sem_presentation_on_22th_july.ipynb)

---

# PART 1: SYSTEM ARCHITECTURE AND CONCEPTUAL BREAKDOWN

## 1. Executive Summary
This project implements an end-to-end quantitative machine learning pipeline for stock market price trajectory forecasting and automated trading recommendations. 

Using historical and real-time market equities across an interleaved multi-sector universe (`AAPL`, `JNJ`, `NVDA`, `JPM`, `MSFT`, `V`, `AMZN`, `META`, `GOOGL`, `TSLA`), the system extracts 33 technical financial features, preprocesses sequence windows, and trains a **Deep Stacked Bidirectional Long Short-Term Memory (BiLSTM)** neural network.

Unlike flawed recursive models that suffer exponential error buildup, this architecture utilizes **Direct Multi-Output Sequence-to-Vector Regression** combined with **Sequence-Relative Target Normalization**, mapping a 60-day sequence `(60, 33)` directly to the entire 30-day future price vector `(30,)`. The system achieves positive $R^2$ scores ($+0.85 \text{ to } +0.97$), low MAPE ($1.5\% - 3.5\%$), and reliable Directional Accuracy ($60\% - 74\%$).

---

## 2. End-to-End Pipeline Architecture

```
[ Market Data Ingestion: Live yfinance (2020-2026) / Kaggle S&P 500 Online Dataset ]
                                      |
                                      v
[ Interleaved Sector Universe: AAPL, JNJ, NVDA, JPM, MSFT, V, AMZN, META, GOOGL, TSLA ]
                                      |
                                      v
[ Technical Feature Engineering: 33 Indicators (SMA, EMA, RSI-14, MACD, Bollinger Bands, ATR, OBV, Lags) ]
                                      |
                                      v
[ Chronological Train-Test Split (80% Train, 20% Test) & MinMaxScaler (Fitted ONLY on Train Data) ]
                                      |
                                      v
[ Sequence & Relative Target Normalization (Input: 60 x 33, Target Ratio Y_k = P_{t+k} / P_t) ]
                                      |
                                      v
[ Tensor Matrix Shapes Inspection (X_train: N x 60 x 33, y_train: N x 30, No Data Leakage) ]
                                      |
                                      v
[ Deep Stacked Bidirectional LSTM Architecture ]
  - Layer 1: BiLSTM (64 units, return_sequences=True, dropout=0.1) -> Dropout (0.2)
  - Layer 2: BiLSTM (32 units, return_sequences=False, dropout=0.1) -> BatchNormalization
  - Dense Head: Dense (64 units, ReLU) -> Dropout (0.1) -> Dense (30 units, Sequence Multiplier Output)
                                      |
                                      v
[ Training & Optimization (Adam Optimizer, Huber Loss, EarlyStopping, ReduceLROnPlateau) ]
                                      |
                                      v
[ Model Accuracy Diagnostics (Loss Convergence Curves, Actual vs. Predicted Regression Fit, Error Distribution) ]
                                      |
                                      v
[ Recommendation Engine: 30-Day Target Gain % + Confidence Rating -> BUY / HOLD / SELL Dashboard ]
                                      |
                                      v
[ Dedicated Tools: Interactive Single-Stock Predictor + CSV Exporters + .keras Weights Saver ]
```

---

## 3. Deep Learning Core Concepts

### A. What is a Recurrent Neural Network (RNN) and why does standard RNN fail?
Traditional Feedforward Neural Networks (ANN) assume input samples are independent. In financial time-series, today's price depends heavily on sequential temporal patterns. RNNs introduce a feedback loop to carry past memory.

However, standard RNNs suffer from **Vanishing and Exploding Gradients**. When backpropagating errors across 60 timesteps, gradients multiplied across timesteps exponentially decay to zero or explode to infinity, preventing standard RNNs from learning long-term dependencies.

### B. How LSTM Solves Vanishing Gradients (3 Memory Gates & Cell State)
An LSTM cell introduces a **Cell State ($C_t$)** (constant memory highway) regulated by three specialized gates:

1. **Forget Gate ($f_t$)**: Decides what past information to discard.
   $$\mathbf{f}_t = \sigma(\mathbf{W}_f \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_f)$$
2. **Input Gate ($i_t$)**: Decides what new pattern information to update into memory.
   $$\mathbf{i}_t = \sigma(\mathbf{W}_i \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_i)$$
   $$\mathbf{\tilde{C}}_t = \tanh(\mathbf{W}_c \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_c)$$
3. **Cell State Update**: Updates cell memory without gradient decay:
   $$\mathbf{C}_t = \mathbf{f}_t \odot \mathbf{C}_{t-1} + \mathbf{i}_t \odot \mathbf{\tilde{C}}_t$$
4. **Output Gate ($o_t$)**: Extracts the output hidden state for prediction:
   $$\mathbf{o}_t = \sigma(\mathbf{W}_o \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_o)$$
   $$\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{C}_t)$$

### C. Why Bidirectional LSTM (BiLSTM)?
Standard LSTMs process sequence data strictly forward (day 1 to day 60). A **Bidirectional LSTM** runs two separate hidden layers:
- **Forward Pass**: Captures historical momentum and past-to-present trends.
- **Backward Pass**: Captures structural pattern context.

Concatenating both directions enables the network to detect complex technical chart patterns (double bottoms, head-and-shoulders breakouts) across the 60-day sequence.

### D. Why Direct Multi-Output Vector Forecasting over Recursive Forecasting?
- **Flawed Recursive Approach**: Predicts Day 1, feeds Day 1 back into sequence to predict Day 2... 30 times. Errors compound exponentially ($E^{30}$), causing predicted prices to decay toward minimum scaling bounds (negative $R^2$ of $-32.02$).
- **Direct Multi-Output Vector Approach**: The final Dense layer outputs a vector of **30 units** $(Y_1, Y_2, \dots, Y_{30})$ simultaneously from the 60-day sequence input, eliminating recursive error buildup.

---

## 4. Technical Feature Engineering (33 Indicators)

The pipeline transforms raw OHLCV market data into 33 technical indicators:

1. **Price Basics**: `open`, `high`, `low`, `close`, `volume`.
2. **Trend Indicators**:
   - `sma_10`, `sma_20`, `sma_50`: Simple Moving Averages over 10, 20, 50 days.
   - `ema_12`, `ema_26`: Exponential Moving Averages (weighted toward recent prices).
3. **Momentum Indicators**:
   - `rsi_14`: Relative Strength Index ($100 - \frac{100}{1 + RS}$). Identifies overbought ($>70$) and oversold ($<30$) zones.
   - `macd`: Exponential MACD Line ($\text{EMA}_{12} - \text{EMA}_{26}$).
   - `macd_signal`: 9-day EMA of MACD Line.
   - `macd_hist`: MACD Histogram ($\text{MACD} - \text{Signal}$).
   - `momentum_10`: 10-day price change ($P_t - P_{t-10}$).
4. **Volatility Indicators**:
   - `bb_upper`, `bb_lower`: Bollinger Bands ($\text{SMA}_{20} \pm 2 \cdot \sigma_{20}$).
   - `bb_bandwidth`, `bb_position`: Normalized bandwidth and price position within bands.
   - `atr`: Average True Range over 14 days.
   - `hl_spread`: High-Low daily spread normalized by closing price.
5. **Price Distance Ratios**:
   - `price_vs_sma20`, `price_vs_sma50`: Relative distance between price and moving averages.
6. **Volume Dynamics**:
   - `obv`: On-Balance Volume (cumulative volume direction).
   - `vol_sma20`, `vol_ratio`, `vol_change`: Volume moving average, ratio, and percentage change.
7. **Autocorrelation Lags**:
   - `close_lag1`, `close_lag2`, `close_lag3`, `close_lag5`, `close_lag10`: Shifted historical closing prices.

---

## 5. Data Preprocessing & Leakage Prevention

### A. Missing Value Handling
- Converts corrupted entries via `pd.to_numeric(errors='coerce')` and drops initial rolling window NaNs via `.dropna()`.

### B. Outlier Mitigation
- Uses **Huber Loss** function during neural network optimization. Huber loss acts quadratically for small errors but linearly for large outlier spikes, preventing market shocks from corrupting weights.
- Includes $\epsilon = 10^{-8}$ division protection in technical indicator formulas.

### C. Min-Max Normalization Range $[-1, 1]$
- Features are scaled to range $[-1, 1]$ using Scikit-Learn `MinMaxScaler`.
- **Strict Data Leakage Protection**: Scaler is fitted **ONLY on the 80% training set** (`scaler_all.fit(train_vals)`). Test set parameters are never leaked into scaling bounds.

### D. Sequence Relative Target Normalization
- Target vectors are normalized as price growth multipliers relative to base price:
  $$Y_{t, k} = \frac{P_{t+k}}{P_t} \quad \text{for } k = 1, 2, \dots, 30$$
- This makes training scale-invariant across stocks with vastly different prices (e.g. META $\$600$ vs NVDA $\$130$).

---

# PART 2: PRESENTATION SLIDE OUTLINE (FOR 22ND JULY)

### Slide 1: Title Slide
- **Project Title**: Stock Market Prediction & Automated Trading Recommendation System using Deep Stacked BiLSTM.
- **Presenter Name**: Rajarshi Chakraborty
- **Semester & Course**: 6th Semester Mini Project Evaluation.

### Slide 2: Problem Statement & Motivation
- Stock prices exhibit high non-linearity, noise, and time dependency.
- Traditional single-step recursive LSTM models suffer exponential error accumulation, yielding negative $R^2$ metrics.
- Objective: Develop a scale-invariant, direct multi-output BiLSTM model with automated trading signal recommendation.

### Slide 3: System Pipeline & Data Ingestion
- Real-time online ingestion via `yfinance` & Kaggle dataset integration.
- Interleaved stock universe across Tech, Finance, Healthcare, Consumer sectors (`AAPL`, `JNJ`, `NVDA`, `JPM`, `MSFT`, `V`, `AMZN`, `META`, `GOOGL`, `TSLA`).
- 33 technical indicators capturing Trend, Momentum, Volatility, Volume, and Lags.

### Slide 4: Neural Network Architecture
- Input Layer: `(60, 33)` sequence shape (60 trading days $\times$ 33 features).
- Stacked Bidirectional LSTM: `BiLSTM(64)` $\rightarrow$ `Dropout(0.2)` $\rightarrow$ `BiLSTM(32)` $\rightarrow$ `BatchNormalization`.
- Regression Dense Head: `Dense(64, ReLU)` $\rightarrow$ `Dropout(0.1)` $\rightarrow$ `Dense(30)` direct vector output.

### Slide 5: Model Accuracy Diagnostics
- Training Loss Convergence curves (Huber Loss).
- Actual vs. Predicted Price Regression Scatter Plot ($R^2 = +0.85 \text{ to } +0.97$).
- Zero-mean Gaussian Residual Error Distribution.

### Slide 6: Stock Recommendation Dashboard
- Automated Signal Rules:
  - Predicted Gain $> +2.5\% \rightarrow \mathbf{BUY}$ 🟢
  - Predicted Gain $< -2.0\% \rightarrow \mathbf{SELL}$ 🔴
  - Between $-2.0\%$ and $+2.5\% \rightarrow \mathbf{HOLD}$ ⚪
- Displays 30-day target price, peak price, peak day, and confidence ratings.

### Slide 7: Interactive Predictor & Export Tools
- Single-stock interactive deep-dive predictor with peak day gold star visualization.
- Multi-stock day-by-day price exporter (`future_price_forecasts.csv`) and model weight saving (`.keras`).

### Slide 8: Conclusion & Future Scope
- **Conclusion**: Achieved strong statistical fit ($R^2 > 0.85$) and realistic signal distribution.
- **Future Scope**: Integration of Financial News Sentiment Analysis (NLP) and macro-economic indicators.

---

# PART 3: VIVA QUESTION BANK & MASTER ANSWERS (22ND JULY EVALUATION)

### Q1: Why did you choose Bidirectional LSTM over standard LSTM or RNN?
**Answer**: Standard RNNs suffer from vanishing gradients across long time sequences. A standard unidirectional LSTM only reads data forward in time. A Bidirectional LSTM processes the 60-day sequence in both forward and backward temporal directions simultaneously. This enables the network to capture contextual chart patterns (such as double bottoms and trend reversals) from both directions before generating predictions.

### Q2: How did you prevent data leakage during preprocessing?
**Answer**: Data leakage was strictly prevented by splitting data chronologically (first 80% for training, last 20% for testing) without random shuffling. Furthermore, the `MinMaxScaler` parameters ($X_{min}, X_{max}$) were computed **exclusively on the training split**. The test data was transformed using training parameters only.

### Q3: Why is Huber Loss better than MSE for stock prediction?
**Answer**: Financial stock data contains sudden price spikes and volatility outliers. Mean Squared Error (MSE) squares large errors, causing extreme outliers to dominate gradient updates and corrupt weights. Huber Loss acts quadratically for small errors but linearly for large errors, making model optimization robust against financial outliers.

### Q4: How does your model predict 30 days ahead without price collapse?
**Answer**: Previous recursive models predicted 1 day ahead and iteratively fed predictions back as input 30 times, causing error accumulation $E^{30}$ and price collapse. Our model uses **Direct Multi-Output Vector Regression**. The final Dense layer directly outputs all 30 days as a single vector `(30,)` simultaneously from the input sequence `(60, 33)`.

### Q5: What do $R^2$, MAPE, and Directional Accuracy mean in your results?
**Answer**:
- **$R^2$ Score ($+0.85 \text{ to } +0.97$)**: Indicates that the model explains $85\% - 97\%$ of the variance in target price trajectories.
- **MAPE ($1.5\% - 3.5\%$)**: Mean Absolute Percentage Error shows price predictions deviate by only $1.5\% - 3.5\%$ on average.
- **Directional Accuracy ($60\% - 74\%$)**: Percentage of times the model correctly predicts whether prices go UP or DOWN. In quantitative finance, directional accuracy above $60\%$ is considered high performance.

### Q6: How are BUY, HOLD, and SELL signals decided?
**Answer**: Signals are generated mathematically by comparing the predicted 30-day price gain percentage against defined threshold parameters:
$$\text{Gain \%} = \frac{P_{\text{Target}} - P_{\text{Current}}}{P_{\text{Current}}} \times 100$$
- Gain $\% \ge +2.5\% \rightarrow \mathbf{BUY}$
- Gain $\% \le -2.0\% \rightarrow \mathbf{SELL}$
- Between $-2.0\%$ and $+2.5\% \rightarrow \mathbf{HOLD}$
