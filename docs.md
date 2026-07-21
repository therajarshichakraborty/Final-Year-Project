# TECHNICAL DOCUMENTATION, PRESENTATION GUIDE AND VIVA QUESTION BANK

## PROJECT TITLE: Stock Market Prediction and Buy/Hold/Sell Recommendation System using Deep Stacked Bidirectional LSTM

---

# PART 1: SYSTEM ARCHITECTURE AND CONCEPTUAL BREAKDOWN

## 1. Executive Summary
This project implements an end to end machine learning pipeline for financial stock market price prediction and automated trading recommendation. Using historical stock data from S&P 500 market equities, the system extracts 33 technical indicator features, preprocesses sequential data using sliding windows, and feeds them into a Deep Stacked Bidirectional Long Short-Term Memory (LSTM) neural network.

The trained model forecasts next day closing prices and iteratively projects stock price trajectories 30 days into the future. Based on predicted percentage gains and model confidence, an automated signal generator categorizes stocks into BUY, HOLD, or SELL recommendations.

---

## 2. End to End Pipeline Architecture

```
[ Kaggle S&P 500 Historical Data (505 Tickers, 5 Years) ]
                        |
                        v
[ Data Filter & Cleaning (AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA, JPM, V, JNJ) ]
                        |
                        v
[ Technical Feature Engineering (33 Indicators: RSI, MACD, Bollinger Bands, ATR, OBV, Lags) ]
                        |
                        v
[ Chronological Train-Test Split (80% Train, 20% Test) & MinMaxScaler (Fitted on Train Only) ]
                        |
                        v
[ 60-Day Sliding Window Sequence Generator (Input: 60 x 33, Target: Next-Day Scaled Close) ]
                        |
                        v
[ Deep Stacked Bidirectional LSTM Network Architecture ]
  - Layer 1: BiLSTM (128 units, dropout=0.1) + Dropout (0.3)
  - Layer 2: BiLSTM (64 units, dropout=0.1) + Dropout (0.2)
  - Layer 3: Unidirectional LSTM (32 units) + BatchNormalization
  - Dense Head: Dense(64, ReLU) -> Dropout(0.1) -> Dense(32, ReLU) -> Output Dense(1)
                        |
                        v
[ Training & Optimization (Adam Optimizer, Huber Loss, EarlyStopping, ReduceLROnPlateau) ]
                        |
                        v
[ Performance Evaluation (RMSE, MAE, MAPE, Directional Accuracy %) ]
                        |
                        v
[ 30-Day Autoregressive Iterative Forecasting ]
                        |
                        v
[ Recommendation Generator: Gain % Calculation + Confidence Rating -> BUY / HOLD / SELL ]
```

---

## 3. Deep Learning Core Concepts Explained Simply

### A. What is an Recurrent Neural Network (RNN) and why does standard RNN fail?
Traditional Artificial Neural Networks (ANN) assume that inputs are independent of each other. In time-series data like stock prices, today's price depends heavily on yesterday's price. RNNs introduce a feedback loop to pass past memory into current decisions.

However, standard RNNs suffer from the Vanishing and Exploding Gradient problem. When backpropagating errors across long time sequences (e.g., 60 days), gradients multiplied across many timesteps exponentially shrink to zero or blow up to infinity. This prevents standard RNNs from learning long term dependencies.

### B. How LSTM Solves Vanishing Gradients (The Memory Cell & 3 Gates)
An LSTM (Long Short-Term Memory) cell replaces the standard neuron with a specialized cell state (constant memory highway) and three regulating gates.

1. Forget Gate (Sigmoid layer): Decides what information to drop from the cell state.
   Formula: f_t = sigmoid(W_f * [h_{t-1}, x_t] + b_f)
   Analogy: Filtering out old noise (e.g., discarding stock price behavior from 3 months ago that is no longer relevant).

2. Input Gate (Sigmoid + Tanh layer): Decides what new information to store in the cell state.
   Formulas:
   i_t = sigmoid(W_i * [h_{t-1}, x_t] + b_i)
   C_tilde_t = tanh(W_c * [h_{t-1}, x_t] + b_c)
   Analogy: Identifying a sudden surge in volume or RSI breakout today and writing it to long term memory.

3. Cell State Update (The Constant Memory Highway):
   Formula: C_t = f_t * C_{t-1} + i_t * C_tilde_t
   Analogy: The master notebook holding the ongoing stock trend context.

4. Output Gate (Sigmoid + Tanh layer): Decides what portion of the cell state forms the output hidden state for the current step.
   Formulas:
   o_t = sigmoid(W_o * [h_{t-1}, x_t] + b_o)
   h_t = o_t * tanh(C_t)
   Analogy: Producing today's specific price prediction signal based on overall context.

### C. Why Bidirectional LSTM (BiLSTM)?
A standard LSTM reads time series strictly forward (day 1 to day 60). A Bidirectional LSTM processes the 60-day window in two directions simultaneously:
- Forward LSTM: Learns past to present momentum.
- Backward LSTM: Learns reverse temporal patterns to understand structural cycles.
The outputs are concatenated, allowing the network to capture complex chart patterns such as head-and-shoulders, cup-and-handle, and double bottoms across the 60-day sequence.

---

## 4. Feature Definition and Extraction (FDE) Details
The project transforms raw Open, High, Low, Close, Volume data into 33 technical features:

1. Price Basics: Open, High, Low, Close, Volume.
2. Trend Indicators:
   - Simple Moving Averages: SMA-10, SMA-20, SMA-50 (Identify short, medium, long trends).
   - Exponential Moving Averages: EMA-12, EMA-26 (Weigh recent prices higher).
3. Momentum Indicators:
   - RSI (Relative Strength Index, 14-day): Measures velocity and magnitude of price changes (Overbought > 70, Oversold < 30).
   - MACD (Moving Average Convergence Divergence): Difference between EMA-12 and EMA-26.
   - MACD Signal & MACD Histogram: Identify bullish and bearish crossovers.
   - Momentum (10-day price change): Measures overall speed of movement.
4. Volatility Indicators:
   - Bollinger Bands (20-day mean, 2 std dev): Upper band, Lower band, Bandwidth, and Band Position (Measures price extension).
   - ATR (Average True Range, 14-day): Measures market volatility.
   - High-Low Spread: Daily trading range normalized by close price.
5. Price Distance Features:
   - Price vs SMA-20: Percentage distance from 20-day average.
   - Price vs SMA-50: Percentage distance from 50-day average.
6. Volume Indicators:
   - OBV (On-Balance Volume): Cumulative volume added on up-days and subtracted on down-days.
   - Volume SMA-20 & Volume Ratio: Current volume compared to 20-day moving average.
   - Volume Change: Daily percentage change in trading volume.
7. Price Lag Features:
   - Close price shifts at lag 1, 2, 3, 5, and 10 days to capture immediate autocorrelation.

---

## 5. Iterative 30-Day Forecasting & Recommendation Algorithm

### A. Autoregressive Iterative Forecasting Mechanism
Since stock market data into the future does not contain future feature values, the system uses an iterative step-by-step prediction mechanism:
1. Feed the latest 60 historical feature timesteps into the model.
2. Predict next day scaled close price.
3. Inverse-scale the prediction using `scaler_close` to record real USD target price.
4. Append the predicted scaled close price into the feature window while shifting the window forward by 1 step.
5. Repeat steps 1 to 4 for 30 consecutive trading days.

### B. Buy / Hold / Sell Recommendation Thresholds
- Target Price: Predicted price on Day 30.
- Peak Price: Highest predicted price within the 30-day window.
- Predicted Gain %: ((Target Price - Current Price) / Current Price) * 100.
- Recommendation Rules:
  - BUY Signal: Predicted Gain >= +3.0%
  - SELL Signal: Predicted Gain <= -2.0%
  - HOLD Signal: Predicted Gain between -2.0% and +3.0%
- Confidence Rating:
  - High Confidence: Model Directional Accuracy > 60%
  - Medium Confidence: Model Directional Accuracy between 52% and 60%
  - Low Confidence: Model Directional Accuracy < 52%

---
---

# PART 2: 20-MINUTE PRESENTATION SLIDE-BY-SLIDE GUIDE

Below is the complete presentation script for a 20-minute presentation. Each slide includes title, key slide content, and exact spoken explanation script.

---

## SLIDE 1: Title, Objective and Scope

### Slide Content:
- Project Title: Stock Market Prediction and Automated Recommendation System using Deep Stacked Bidirectional LSTM.
- Presenter: Rajarshi Chakraborty.
- Objective:
  - Develop an intelligent deep learning system to forecast stock prices using high dimensional historical trading indicators.
  - Generate actionable Buy, Hold, and Sell recommendations with 30-day target gains and model confidence metrics.
- Scope:
  - Dataset: S&P 500 stock market equities (5-year daily records, 505 tickers).
  - Target Assets: Major tech and enterprise stocks (AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA, JPM, V, JNJ).
  - Horizon: 60-day historical lookback for 30-day future trajectory prediction.

### Spoken Script (2 Minutes):
"Respected panel members and professors, good morning. Today I am presenting my mini project titled Stock Market Prediction and Automated Recommendation System using Deep Stacked Bidirectional LSTM.

The stock market is inherently volatile and non-linear. The main objective of this project is to eliminate emotional retail trading decisions by building an automated, data-driven system. Our model does not just look at past closing prices, it processes 33 technical indicators across 60 trading days to forecast future price trajectories over a 30-day horizon.

The scope of this project focuses on leading S&P 500 stocks such as Apple, Microsoft, Google, NVIDIA, and Amazon using 5 years of historical data. The ultimate deliverable is not just price forecasting, but clear Buy, Hold, or Sell advice backed by statistical confidence scores."

---

## SLIDE 2: Problem Statement

### Slide Content:
- Financial Market Challenges:
  - High Volatility and Noise: Stock prices are influenced by non-linear factors.
  - Failure of Traditional Statistical Models: Linear models like ARIMA struggle with complex non-linear price patterns.
  - Over-Reliance on Single Indicators: Retail traders rely on single technical indicators which frequently yield false signals.
  - Lack of Actionable Advice: Most academic papers only output RMSE values instead of clear decision framework for investors.
- Proposed Solution:
  - Multi-feature deep temporal neural network combining trend, momentum, volatility, and volume indicators with stacked Bidirectional LSTM for multi-step forecasting and advisory logic.

### Spoken Script (2 Minutes):
"Why is stock prediction so difficult? First, financial time-series data is non-stationary and noisy. Traditional linear time-series models like ARIMA assume stationarity and struggle to capture complex non-linear swings.

Second, retail investors often lose money because they look at isolated indicators like RSI alone, which leads to false breakout traps.

Third, most research models stop at showing a low Root Mean Squared Error number. But as an investor, an RMSE of 2.5 does not tell you whether you should put money into a stock today. Our project solves this by transforming deep learning forecasts into clear trading signals."

---

## SLIDE 3: Proposed Methodology

### Slide Content:
- Pipeline Architecture:
  1. Data Ingestion: Automated retrieval of S&P 500 historical data.
  2. Feature Engineering: Extraction of 33 financial indicators.
  3. Preprocessing: Chronological 80-20 train-test split and leakage-free MinMaxScaler.
  4. Sequence Creation: 60-day sliding window matrix generation.
  5. Deep Learning Core: 3-layer stacked Bidirectional LSTM with Batch Normalization and Dropout.
  6. Multi-Step Forecasting: 30-day autoregressive projection.
  7. Recommendation Engine: Rule-based signal generation (+3% Buy, -2% Sell).

### Spoken Script (2 Minutes):
"This slide shows our end to end proposed methodology. We begin by downloading 5 years of historical data. Instead of feeding raw prices alone, we engineer a rich 33-dimensional feature matrix per stock.

To strictly avoid data leakage, we perform an 80-20 chronological split before fitting our MinMaxScaler on the training set only. We then create 60-day sliding sequence windows.

Our neural architecture uses stacked Bidirectional LSTMs followed by dense regression layers, trained with Huber Loss to handle market spikes. Finally, the output is fed into a 30-day iterative forecast engine to yield final trading recommendations."

---

## SLIDE 4: Data Acquisition

### Slide Content:
- Primary Source: Kaggle S&P 500 Dataset (`camnugent/sandp500`).
- Dataset Specifications:
  - Total Stocks: 505 companies.
  - Duration: 5 Years (2013-02-08 to 2018-02-07).
  - Records: 1,259 daily trading entries per stock (619,040 total rows).
  - Raw Attributes: `date`, `open`, `high`, `low`, `close`, `volume`, `Name`.
- Data Selection:
  - Filtered top market cap assets: AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA, JPM, V, JNJ.

### Spoken Script (2 Minutes):
"For data acquisition, we used the official Kaggle S&P 500 dataset compiled from financial exchanges over 5 years. It contains over 619,000 rows across 505 tickers.

Each record includes standard daily OHLCV values: Open, High, Low, Close, and Volume. We extracted 1,259 trading days per ticker for our target portfolio, ensuring sufficient historical length to cover different market regimes including bull trends, consolidations, and minor pullbacks."

---

## SLIDE 5: Data Preprocessing

### Slide Content:
- Handling Missing Values: Drop incomplete rows resulting from technical indicator windowing.
- Chronological Train-Test Split:
  - Train Set: First 80% of data (approx. 908 trading days).
  - Test Set: Final 20% of data (approx. 182 trading days).
  - Strict Rule: No random shuffling to preserve temporal sequence order.
- Feature Normalization:
  - `MinMaxScaler(feature_range=(0, 1))` fitted ONLY on training data.
  - Separate `scaler_close` maintained for target close price to enable clean inverse transformation back to USD.
- Sliding Window Generation:
  - Input shape per sample: (60 timesteps, 33 features).
  - Target label: Scaled close price at step t+1.

### Spoken Script (2 Minutes):
"Data preprocessing is critical for time-series modeling. First, we strictly avoid random shuffling. In financial data, shuffling causes future data to leak into the past. We split the data chronologically with 80% for training and 20% for testing.

Second, to prevent Data Leakage, we fit our MinMaxScaler strictly on the training set and transform the test set using those exact parameters.

Third, we transform the tabular data into 3D sequential tensors of shape (60, 33), meaning the network looks at 60 past trading days across 33 indicators to predict tomorrow's price."

---

## SLIDE 6: Feature Definition & Extraction (FDE)

### Slide Content:
- 33 Engineered Features Categorized:
  - Trend: SMA-10, SMA-20, SMA-50, EMA-12, EMA-26.
  - Momentum: RSI-14, MACD, MACD Signal, MACD Hist, 10-day Momentum.
  - Volatility: Bollinger Bands (Upper, Lower, Bandwidth, Position), ATR-14, High-Low Spread %.
  - Price Stretch: Price vs SMA-20 %, Price vs SMA-50 %.
  - Volume Signals: On-Balance Volume (OBV), Volume SMA-20, Volume Ratio, Volume Change %.
  - Autocorrelation Lags: Close price shifted at t-1, t-2, t-3, t-5, t-10.

### Spoken Script (2 Minutes):
"In Slide 6, we highlight Feature Definition and Extraction, or FDE. Feeding only raw close price into an LSTM causes lag. To give the model comprehensive market awareness, we engineered 33 domain-specific features.

We capture trend through multiple moving averages, momentum through RSI and MACD crossovers, and volatility using Bollinger Bands and Average True Range. We also include volume indicators like On-Balance Volume, which shows smart money accumulation before price moves occur. Lastly, price lag features provide explicit short term memory connections."

---

## SLIDE 7: Hybrid Model Building (Planned Future Extension)

### Slide Content:
- Current Architecture vs Next Semester Hybrid Extension:
  - Current System: Deep Stacked BiLSTM (End to End Deep Learning).
  - Next Semester Planned Hybrid Architecture:
    1. Statistical + Deep Learning Hybrid (ARIMA + LSTM):
       - ARIMA captures linear trend and seasonal autocorrelation components.
       - LSTM models non-linear residuals and volatility errors.
    2. Attention-Based LSTM (Temporal Attention Mechanism):
       - Assigns dynamic attention weights to crucial historical days within the 60-day window (e.g. high impact earnings breakout day).
- Conceptual Diagram of Planned Hybrid Pipeline:
  - Input Data -> ARIMA (Linear Forecast) + Attention-BiLSTM (Non-linear Residuals) -> Combined Final Prediction.

### Spoken Script (2 Minutes):
"In this slide, we outline our Roadmap for Hybrid Model Building, which we plan to implement for our next semester project phase.

Currently, our system relies on a pure Deep Stacked BiLSTM network. While BiLSTM excels at non-linear patterns, traditional statistical models like ARIMA are mathematically optimal for linear trend decomposition.

For the next phase, we are designing a two stage hybrid architecture. ARIMA will filter out linear autocorrelation, and the residual error will be fed into an Attention-based BiLSTM. The Attention mechanism will dynamically highlight key shock days in the 60-day window, further enhancing multi-step forecast stability."

---

## SLIDE 8: Training, Output and Expected Outcome

### Slide Content:
- Model Hyperparameters & Training Setup:
  - Epochs: 2 (Colab demo run) up to 50 with EarlyStopping (patience=10).
  - Batch Size: 32 | Optimizer: Adam (lr=0.001) | Loss: Huber Loss.
  - Regularization: Dropout (0.3, 0.2, 0.1), Batch Normalization, ReduceLROnPlateau.
- Evaluation Metrics:
  - RMSE, MAE, MAPE, Directional Accuracy %.
- Output & Recommendation Generator:
  - 30-Day Trajectory Curve.
  - Actionable Advisory Output: Signal (BUY/HOLD/SELL), Target Price, Peak Price, Peak Day, Predicted Gain %, Confidence Level.

### Spoken Script (2 Minutes):
"Slide 8 covers model training and performance evaluation. We optimize using Adam with Huber Loss. Huber loss is superior to standard MSE because it behaves linearly for large errors, preventing catastrophic weight adjustments during random market spikes.

To prevent overfitting, we apply Dropout rates up to 0.3 and Batch Normalization.

Our outputs provide complete clarity: for every stock, the model plots the 30-day forecast trajectory and outputs a final decision card showing target price, percentage gain, recommendation signal, and confidence rating based on directional accuracy."

---

## SLIDE 9: Future Work

### Slide Content:
- Key Enhancements for Next Semester:
  1. Multimodal Sentiment Integration:
     - Incorporating financial news sentiment (FinBERT) and Twitter/Reddit market sentiment.
  2. Exogenous Macroeconomic Features:
     - Adding interest rates, VIX volatility index, and crude oil prices.
  3. Real-Time Trading API Integration:
     - Live data streaming via Yahoo Finance / Alpaca API for paper trading.
  4. Transformer & Temporal Fusion Transformer (TFT) Comparison:
     - Benchmarking BiLSTM against Attention Transformers.

### Spoken Script (2 Minutes):
"Looking ahead to future work, stock prices are not driven by numerical price history alone. External events such as Federal Reserve rate hikes or quarterly earnings news cause sudden jumps.

In the next iteration, we will integrate FinBERT sentiment analysis from financial news. We will also incorporate macroeconomic indicators like the VIX index and crude oil prices, and deploy the system to live paper trading using Alpaca API."

---

## SLIDE 10: Conclusion

### Slide Content:
- Summary of Achievements:
  - Built a robust, leakage-free data pipeline with 33 financial indicators.
  - Implemented a Deep Stacked Bidirectional LSTM capable of processing multi-feature temporal sequences.
  - Achieved strong directional accuracy and low error metrics across major S&P 500 equities.
  - Created an end-to-end framework transforming deep learning output into actionable Buy, Hold, or Sell trading signals.
- Final Takeaway:
  - Demonstrates that combining financial domain feature engineering with deep sequence models provides a scalable foundation for quantitative trading strategies.

### Spoken Script (2 Minutes):
"To conclude, this project successfully bridges the gap between complex deep learning models and practical financial decision making.

By combining 33 technical indicators with a Deep Stacked Bidirectional LSTM network, we established a resilient framework for stock prediction. We ensured strict scientific validity by preventing data leakage and focusing on directional accuracy alongside price error.

Thank you panel members for your time and guidance. I am now open to your questions."

---
---

# PART 3: VIVA AND COUNTER-QUESTION BANK (PROFESSOR DEFENSE)

This section contains 30 counter-questions specifically tailored to questions asked by Indian academic project panels. Every answer is written in simple, clear, and direct language without em-dashes.

---

### Q1: What is an LSTM? Define it in simple terms.
Answer:
An LSTM stands for Long Short-Term Memory. It is a special type of Recurrent Neural Network designed to learn long term sequential patterns. Unlike standard neural networks that process inputs independently, an LSTM has a memory cell state that maintains context over long time steps. It uses three internal gating mechanisms: the Forget Gate, Input Gate, and Output Gate, to selectively remember or forget information over time.

---

### Q2: Why did you choose LSTM for stock prediction instead of traditional Machine Learning algorithms like Random Forest or XGBoost?
Answer:
Stock prices are time-series data where temporal order matters. Traditional Machine Learning algorithms like Random Forest or XGBoost treat each input sample as independent and identically distributed. They cannot naturally capture temporal dependencies or sequential trends across time steps. LSTM is explicitly built to process sequential data while preserving historical context across a 60-day window.

---

### Q3: Why did you use LSTM instead of traditional statistical time-series models like ARIMA or SARIMA?
Answer:
ARIMA and SARIMA are linear models that require data to be strictly stationary. Stock market price series are non-stationary, noisy, and contain non-linear relationships influenced by market sentiment and volatility. ARIMA struggles to capture non-linear feature interactions across multiple indicators. LSTM is a non-linear deep learning model that handles raw non-stationary multi-feature sequences without requiring explicit mathematical stationarity transformations.

---

### Q4: Explain the difference between Cell State and Hidden State in an LSTM.
Answer:
The Cell State is the long term memory highway of the LSTM. It runs down the entire chain with minimal linear interactions, carrying old context forward without shrinking gradients.
The Hidden State is the short term working memory. It represents the output of the LSTM cell at the current specific time step after applying the Output Gate filter on the Cell State.

---

### Q5: Explain the function of all 3 gates in an LSTM cell with their mathematical activation functions.
Answer:
1. Forget Gate: Uses a Sigmoid activation function to output values between 0 and 1, deciding what percentage of past cell memory to drop.
2. Input Gate: Uses a Sigmoid layer to decide which features to update, combined with a Tanh layer to generate new candidate values ranging from -1 to +1.
3. Output Gate: Uses a Sigmoid layer to filter the updated cell state, which is then passed through a Tanh function to produce the final hidden state output.

---

### Q6: Why did you use a Bidirectional LSTM instead of a standard Unidirectional LSTM?
Answer:
A standard Unidirectional LSTM only reads the sequence forward from day 1 to day 60. A Bidirectional LSTM runs two separate LSTMs: one forward from past to present, and one backward from present to past inside the 60-day window. Concatenating their hidden states allows the model to learn structural chart patterns such as head-and-shoulders or double bottoms that require looking at surrounding context across the entire window.

---

### Q7: Does a Bidirectional LSTM cause Data Leakage from the future during real-world stock prediction?
Answer:
No, it does not cause data leakage. During inference at day T, the model only receives the past 60 days of historical data from day T-60 to day T. The backward LSTM operates strictly inside this historical 60-day window. It never sees data beyond day T. Therefore, no future market information is leaked.

---

### Q8: What is Data Leakage in time-series forecasting, and how did you prevent it in your preprocessing?
Answer:
Data leakage occurs when information from the future or test dataset inadvertently leaks into the training process, causing artificially high test performance that fails in real trading. We prevented data leakage by:
1. Performing an 80-20 chronological train-test split without random shuffling.
2. Fitting our MinMaxScaler exclusively on the 80% training data portion, and using those exact parameters to scale the test data.

---

### Q9: Why did you choose a lookback window of 60 days? Why not 10 days or 300 days?
Answer:
A 60-day lookback corresponds to approximately 3 months of trading, which equals one financial quarter. This duration provides enough sequence length for the LSTM to capture quarterly trends and technical patterns. A 10-day lookback is too short to learn medium term patterns, while a 300-day lookback increases tensor memory size, causes gradient decay, and introduces outdated market noise.

---

### Q10: Why did you use MinMaxScaler instead of StandardScaler (Z-score normalization)?
Answer:
LSTM networks use Sigmoid and Tanh activation functions, which are sensitive to input scale and can suffer from saturating gradients if inputs are large. MinMaxScaler bounds all feature values strictly between 0 and 1, matching the input distribution expectations of neural network activation gates. StandardScaler produces unbounded values that can slow down gradient convergence.

---

### Q11: What is Huber Loss, and why did you use it instead of standard Mean Squared Error (MSE)?
Answer:
Huber Loss combines the best properties of MSE and Mean Absolute Error (MAE). For small prediction errors, it behaves quadratic like MSE, providing smooth differentiable gradients. For large errors above a threshold delta, it behaves linear like MAE. Stock market data contains random price spikes and outliers. Standard MSE squares large errors, causing extreme gradient updates that ruin model weights. Huber Loss is robust against outliers.

---

### Q12: Why did you train separate LSTM models for each stock instead of one single global model for all stocks?
Answer:
Each stock stock ticker has unique price volatility, trading volume dynamics, and market beta. For example, Tesla (TSLA) exhibits high volatility compared to Johnson & Johnson (JNJ). A single global model tends to average out these distinct behaviors. Training individual models allows the LSTM weights to specialize in the specific price patterns of each company.

---

### Q13: What is Directional Accuracy, and why is it more important than RMSE for stock trading?
Answer:
Directional Accuracy measures the percentage of times the model correctly predicts whether tomorrow's price will move UP or DOWN compared to today.
Formula: Percentage of times Sign(Predicted Price Change) equals Sign(Actual Price Change).
RMSE only measures USD price difference. A model could have a very low RMSE of 1 USD, but if it consistently predicts an UP move when the market goes DOWN, a trader will lose money. Directional accuracy measures actual trading profitability potential.

---

### Q14: Explain how your 30-day iterative forecasting works when future feature values are unknown.
Answer:
We use an autoregressive sliding window approach:
1. Pass the true historical past 60 days into the model to predict Day 1 ahead scaled close price.
2. Inverse-scale the output to record the USD price.
3. Update the 60-day window by dropping the oldest day (Day 1) and appending a new timestep containing the predicted close price.
4. Pass this updated window back into the model to predict Day 2.
5. Repeat this process iteratively for 30 steps.

---

### Q15: What is the main limitation of multi-step iterative forecasting with LSTM?
Answer:
Error accumulation. Because each step prediction is fed back into the model as input for the next step, any minor error in Day 1 prediction compound exponentially over 30 days. This causes the forecast curve to smooth out towards a long term trend mean as the horizon extends.

---

### Q16: Why is stock price data non-stationary, and how does your model handle it?
Answer:
Stock data is non-stationary because its mean, variance, and autocorrelation change over time due to economic inflation, market growth, and volatility shocks.
Traditional models require differencing data to make it stationary. LSTM handles non-stationary inputs because its internal cell gates dynamically update weights to adapt to changing trends over historical sequence windows.

---

### Q17: What features did you engineer, and why did you include On-Balance Volume (OBV)?
Answer:
We engineered 33 technical features including SMA, EMA, RSI, MACD, Bollinger Bands, ATR, Price Lags, and OBV.
OBV (On-Balance Volume) tracks cumulative volume momentum by adding volume on green up days and subtracting volume on red down days. Volume often precedes price breakouts. Including OBV allows the LSTM to detect institutional accumulation before price movements show up on the chart.

---

### Q18: What are Bollinger Bands, and how do they help the LSTM model?
Answer:
Bollinger Bands consist of a 20-day Simple Moving Average surrounded by upper and lower envelope lines set 2 standard deviations away.
They measure market volatility and price stretch. When price touches the upper band, the stock is statistically overextended. Including Bandwidth and Band Position features helps the LSTM recognize volatility squeezes and mean reversion points.

---

### Q19: Why didn't you use ARIMA in this semester's codebase, and what is your plan for Next Semester?
Answer:
In this semester, our primary focus was building a deep learning pipeline using multi-feature sequential inputs. ARIMA only accepts single univariate series and cannot incorporate 33 technical features simultaneously.
For next semester, we will implement a Hybrid ARIMA + Attention-LSTM system. ARIMA will model the primary linear time-series trend, and the Attention-LSTM will learn the non-linear residual errors.

---

### Q20: How does an Attention Mechanism improve an LSTM model?
Answer:
Standard LSTMs compress an entire 60-day sequence into a single final vector, treating recent and older days with fixed structural decay.
An Attention Mechanism computes dynamic scalar weights for every timestep in the 60-day window. It allows the model to pay extra attention to high impact shock days (such as an earnings announcement 40 days ago) when predicting tomorrow's price.

---

### Q21: How did you prevent Overfitting in your deep network?
Answer:
We implemented multiple regularization techniques:
1. Dropout Layers (0.3, 0.2, 0.1): Randomly deactivates neurons during training to prevent co-adaptation.
2. Recurrent Dropout (0.1): Drops connections on recurrent LSTM state updates.
3. Batch Normalization: Normalizes hidden layer outputs to stabilize training.
4. EarlyStopping Callback: Monitors validation loss and stops training if performance fails to improve for 10 consecutive epochs.

---

### Q22: Explain your choice of Adam Optimizer and your Learning Rate strategy.
Answer:
Adam (Adaptive Moment Estimation) computes adaptive learning rates for individual parameters using both first moments (mean) and second moments (uncentered variance) of gradients. It converges fast on noisy financial data.
We set an initial learning rate of 0.001 and used `ReduceLROnPlateau` to automatically drop the learning rate by 50% whenever validation loss plateaued for 5 epochs.

---

### Q23: What are your Buy, Hold, and Sell signal thresholds based on?
Answer:
The signals are calculated from the 30-day predicted percentage gain:
- BUY: Predicted Gain >= +3.0%
- SELL: Predicted Gain <= -2.0%
- HOLD: Predicted Gain between -2.0% and +3.0%
These thresholds reflect realistic transaction cost buffers in equity markets, ensuring recommendations account for brokerage fees and minor market slippage.

---

### Q24: What happens if a major news event or Black Swan event (like COVID crash or earnings surprise) occurs? Can your model predict it?
Answer:
No. Pure technical price-history models cannot predict exogenous shock events because news information is not present in historical price arrays prior to release.
This is why our output includes a Confidence Rating based on Directional Accuracy, and why our Future Work includes integrating real-time FinBERT news sentiment analysis.

---

### Q25: What is the difference between Return Sequences = True and Return Sequences = False in Keras LSTM layers?
Answer:
- `return_sequences=True`: The LSTM outputs hidden states for all 60 timesteps, returning a 3D tensor of shape (batch, 60, units). This is required when stacking another LSTM layer on top.
- `return_sequences=False`: The LSTM only outputs the final hidden state vector for step 60, returning a 2D tensor of shape (batch, units). This is used before passing data into dense regression layers.

---

### Q26: What is the total parameter count of your network, and how is it calculated for an LSTM layer?
Answer:
An LSTM layer parameter count is calculated as:
Formula: 4 * [(input_dim + output_dim + 1) * output_dim]
The factor of 4 comes from the 4 internal neural network layers inside every cell (Forget, Input, Candidate, Output). In our model, Layer 1 BiLSTM contains 165,888 parameters, leading to a total model parameter count of 355,201 trainable parameters.

---

### Q27: How did you evaluate your model performance? State all metrics used.
Answer:
We evaluated models using four metrics:
1. RMSE (Root Mean Squared Error): Measures absolute prediction error magnitude in USD.
2. MAE (Mean Absolute Error): Measures average linear dollar error.
3. MAPE (Mean Absolute Percentage Error): Measures error relative to stock price.
4. Directional Accuracy (%): Percentage of correct directional trend predictions.

---

### Q28: What is RSI (Relative Strength Index) and how does the model use it?
Answer:
RSI is a momentum oscillator measuring the speed and magnitude of recent price changes on a scale from 0 to 100.
RSI above 70 indicates overbought conditions (potential price drop), while RSI below 30 indicates oversold conditions (potential price rebound). Including RSI gives the LSTM explicit signal awareness of trend exhaustion points.

---

### Q29: Can your system be deployed for live algorithmic trading? What components are missing?
Answer:
Currently, the system is an offline predictive pipeline. To make it a live trading bot, we need:
1. Real-time API connection (e.g. Alpaca or Interactive Brokers API) for execution.
2. Automated order execution and stop-loss risk management module.
3. Live streaming data ingest engine to replace historical Kaggle downloads.

---

### Q30: What is the final conclusion and business value of your project?
Answer:
The project proves that combining domain technical indicator engineering with deep stacked Bidirectional LSTMs creates an effective quantitative model for stock price trend forecasting. It transitions standard machine learning from academic RMSE metrics into practical decision advisory cards, providing a reproducible baseline for automated trading strategy research.
