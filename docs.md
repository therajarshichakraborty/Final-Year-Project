# TECHNICAL DOCUMENTATION, PRESENTATION GUIDE AND VIVA QUESTION BANK

## PROJECT TITLE: Hybrid Weather Forecasting with ARIMA, SARIMA and LSTM
**Notebook:** [`app/final_stock_market.ipynb`](app/final_stock_market.ipynb). The filename is kept for the repository; the content is weather forecasting.
**Author:** Rajarshi Chakraborty · 6th-Semester Project
**Dataset:** Jena Climate, 2009–2016 (Max Planck Institute for Biogeochemistry, distributed by TensorFlow/Keras)
**Runtime:** about 5–6 minutes end-to-end (measured on a 12-core CPU); runs on Google Colab with no uploads and no API keys

> All numbers in this document come from the executed notebook committed in the repository. If the notebook is re-run (for example on a Colab GPU), the neural-network numbers can differ in the last decimals. The notebook's own closing summary (§ 14.2) always regenerates from the run.

---

# PART 1: WHAT THE PROJECT DOES

## 1. Executive Summary
The notebook forecasts the **daily mean air temperature** of Jena, Germany, from 1 to 90 days ahead. It compares seven forecasters under one rigorous protocol:

| # | Model | Family | Idea in one line |
|---|---|---|---|
| 1 | Persistence | Baseline | Tomorrow = today |
| 2 | Climatology | Baseline | The long-run average for that calendar day |
| 3 | ARIMA(3,0,1) | Statistical | Past values + past shocks, no seasonality |
| 4 | SARIMA (Fourier) | Statistical | ARIMA errors + a smooth annual cycle (sine/cosine terms) |
| 5 | LSTM | Deep learning | Multi-task recurrent network over 11 weather variables, 3-seed ensemble |
| 6 | **Hybrid-R** (proposed) | Hybrid | SARIMA + an LSTM that learns SARIMA's errors |
| 7 | Hybrid-E | Hybrid | Per-horizon optimal weighted average of SARIMA and LSTM |

**Headline results on the untouched 2016 test year:**
- **Next day:** both hybrids and the LSTM beat every statistical model and baseline. Hybrid-E is the most accurate (RMSE **2.008 °C**, R² **0.931**). Hybrid-R reaches 2.033 °C, **6.0 % better than SARIMA** (statistically significant, p = 0.008).
- **Across all 90 lead times:** Hybrid-R has the lowest mean error (**3.638 °C**). It is **at least as accurate as SARIMA at every lead**, whereas the LSTM alone drifts and ends up to 36 % worse than SARIMA.
- **Uncertainty:** 95 % prediction intervals are well calibrated one day ahead (95–97 % coverage) and conservative at long leads.

---

## 2. End-to-End Pipeline Architecture

```
[ Jena Climate archive: 420,551 ten-minute records × 14 variables, auto-downloaded (Google Cloud / AWS mirror) ]
                                      |
                                      v
[ Data-quality audit: 327 duplicate timestamps · 544 missing stamps · 38 sentinel (-9999) values · physical range checks ]
                                      |
                                      v
[ Wind speed + direction -> (u, v) vectors · resample to daily mean/min/max · 4 low-coverage days time-interpolated ]
                                      |
                                      v
[ 2,922 clean days × 11 variables · STL decomposition · ADF/KPSS stationarity · ACF/PACF ]
                                      |
                                      v
[ Chronological split: 2009–2014 train (2,191 d) | 2015 validation (365 d) | 2016 test (366 d) ]
                                      |
          +---------------------------+---------------------------+---------------------------+
          v                           v                           v                           v
   Baselines                  Statistical                  Deep learning                  Hybrids
 persistence, climatology   ARIMA (AIC grid)          multi-task LSTM ensemble     Hybrid-R: SARIMA + LSTM residuals
                            SARIMA (Fourier + AIC)    (recursive multi-step)       Hybrid-E: optimal weights
          +---------------------------+---------------------------+---------------------------+
                                      |
                                      v
[ Fixed-target rolling-origin evaluation: every test day forecast at every lead 1–90 days ]
                                      |
                                      v
[ RMSE · MAE · MAPE · R² · MASE · skill · Diebold–Mariano · Ljung–Box · interval coverage ]
                                      |
                                      v
[ 90-day out-of-sample forecast (1 Jan – 31 Mar 2017) with 95 % prediction intervals + auto-generated summary ]
```

---

## 3. How to Run
1. Open the notebook in Google Colab (*File → Upload notebook*, or from GitHub).
2. *Runtime → Run all.* The dataset (13.6 MB) downloads automatically. The Google Cloud mirror is tried first, then AWS.
3. The notebook requests a T4 GPU; a CPU runtime also works but is slower.
4. Outputs are written to `outputs/`: 10 figures at 300 dpi, CSV tables, the trained `.keras` models and `outputs.zip`.

**Configuration** (cell 1.2): `SEED = 42`, `TRAIN_END = 2014-12-31`, `VAL_END = 2015-12-31`, `FORECAST_HORIZON = 90` (any value 30–90), `LOOKBACK = 30`, `N_ENSEMBLE = 3` (set to 1 for a quick run), `ALPHA = 0.05` (95 % intervals).

> **Known issue:** the "Open in Colab" badge at the top still points to the old location (`main/final_stock_market.ipynb`). After the move to `app/`, the link should be `https://colab.research.google.com/github/therajarshichakraborty/Final-Year-Project/blob/main/app/final_stock_market.ipynb`.

---

# PART 2: METHODOLOGY IN DETAIL

## 4. Data and Preprocessing

### A. Dataset
Weather station of the Max Planck Institute for Biogeochemistry, Jena. It recorded readings every 10 minutes from 2009 to 2016 and is a standard benchmark (Chollet, *Deep Learning with Python*, 2021).

Variables used: temperature `T` (target), dew point `Tdew`, relative humidity `rh`, pressure `p`, vapour-pressure deficit `VPdef`, wind speed `wv`, maximum gust `max. wv` and wind direction `wd`. Six columns are near-exact functions of these (potential temperature, saturation and actual vapour pressure, specific humidity, water-vapour concentration, air density). They are dropped to avoid multicollinearity.

### B. Data-Quality Audit (Table 1)

| Check | Finding | Action |
|---|---|---|
| Duplicate timestamps | 327 | removed (first kept) |
| Missing 10-minute timestamps | 544 (0.13 %) | absorbed by daily aggregation |
| Sentinel values (−9999, wind sensors) | 38 | set to missing |
| Physically implausible readings | 0 | (rule in place) |
| Days below 50 % coverage | 4 (25–28 Oct 2016) | time-interpolated |
| Final series | 2,922 days, 01 Jan 2009 → 31 Dec 2016 | 0 missing values |

### C. Why Wind Vectors?
Direction is circular: 350° and 10° are neighbours, but their arithmetic mean is 180°. Wind is therefore converted to east–west and north–south components before averaging:
$$u = -w\,\sin\theta, \qquad v = -w\,\cos\theta$$

### D. Daily Aggregation
10-minute readings are resampled to daily **mean** values. Temperature also keeps the daily **minimum and maximum**, and gusts keep the daily **maximum**. A daily mean built from only a few hours is biased, so days with fewer than 72 of their 144 readings are treated as missing.

### E. Normalisation (no leakage)
Z-score standardisation, $z = (x - \mu_{train})/\sigma_{train}$, uses `StandardScaler` **fitted on the 2009–2014 training days only**.

---

## 5. Time-Series Diagnostics

### A. STL Decomposition (Fig. 3)
STL is applied to **weekly** means with a 52-week period. Weekly averaging removes day-to-day weather noise first.
- Seasonal strength **0.87** (strong annual cycle) and trend strength **0.09** (negligible).
- Strength formula: $F = \max\!\left(0,\, 1 - \tfrac{\operatorname{Var}(R)}{\operatorname{Var}(X+R)}\right)$
- STL looks at future values (it is two-sided), so it is used **only for description, never as a model input**.

### B. Stationarity Tests (Table 3)

| Series (training years) | ADF p | KPSS level p | KPSS trend p | Verdict |
|---|---|---|---|---|
| Raw daily temperature | 0.012 | 0.100 | 0.100 | Stationary |
| First difference | 0.000 | 0.100 | 0.100 | Stationary |
| Deseasonalised anomaly | 0.000 | 0.010 | 0.100 | Trend-stationary (no unit root) |

**Consequences:** d = 0 (no differencing), and the annual cycle is deterministic, so it is modelled with Fourier terms rather than seasonal differencing.

### C. ACF / PACF (Fig. 4)
The raw ACF oscillates with a 365-day period, which shows the seasonality. After the annual cycle is removed, the PACF is significant only at the first few lags (1–5), a short-memory AR structure that matches the AIC choice of p = 3.

---

## 6. Experimental Design

### A. Fixed-Target Rolling-Origin Protocol
- Every test day $j$ is forecast at every lead $h = 1 \dots 90$, from origin $j - h$, using only data available up to that day.
- Parameters are estimated **once** on 2009–2014 and then frozen. Models only update their state as new days arrive: the Kalman filter for ARIMA/SARIMA, the input window for the LSTM.
- **Every lead is scored on the same 366 test days.** Built-in sanity check: climatology's error is exactly flat across leads (3.76 °C).

### B. Leakage Safeguards
Chronological split with no shuffling · scaler, climatology and all parameters fitted on training data only · the 2015 validation year is used only for early stopping, hybrid weights and interval calibration · the test year is used **once**.

### C. Metrics

| Metric | Formula | Meaning |
|---|---|---|
| RMSE | $\sqrt{\frac1n\sum e_t^2}$ | penalises large misses (°C) |
| MAE | $\frac1n\sum \lvert e_t\rvert$ | typical error size (°C) |
| MAPE | $\frac{100}{n}\sum \frac{\lvert e_t\rvert}{y_t + 273.15}$ | relative error, computed in **kelvin** |
| R² | $1 - \frac{\sum e_t^2}{\sum (y_t-\bar y)^2}$ | share of variance explained |
| MASE | $\text{MAE} / \text{MAE}_{\text{naive, train}}$ | below 1 beats the naïve forecast |
| Skill | $1 - \text{MSE}/\text{MSE}_{\text{persistence}}$ | above 0 beats persistence |

**Why MAPE is computed in kelvin:** Jena's winter temperatures cross 0 °C, where MAPE in °C explodes (dividing by about 0). Percentages need a scale with a true zero, which for temperature is kelvin. MAPE values therefore look small (≈ 0.55 %); RMSE and MAE are the primary comparison.

---

## 7. The Models

### A. ARIMA(3,0,1) (Table 6)
$$\phi(B)(y_t - \mu) = \theta(B)\varepsilon_t$$
- Orders are chosen from a 16-model AIC grid ($p, q \in 0..3$) with $d = 0$ from the stationarity tests. AIC = 9882.6.
- Estimates: $\mu = 8.20$, $\phi = (1.882, -1.117, 0.232)$, $\theta_1 = -0.843$, $\sigma^2 = 5.29$.
- There is no seasonal term, so long-range forecasts drift away from reality. This is the deliberate "no seasonality" reference.

### B. SARIMA with Fourier Seasonality (Tables 7–9)
$$y_t = \mu + \sum_{k=1}^{K}\left[a_k \sin\tfrac{2\pi k t}{365.25} + b_k\cos\tfrac{2\pi k t}{365.25}\right] + u_t, \qquad \phi(B)u_t = \theta(B)\varepsilon_t$$
- **Why not textbook SARIMA$(P,D,Q)_{365}$?** It needs a 732-dimensional state vector and more than 9 GB of RAM. Differencing at lag 365 also doubles the noise, and 365.25 is not an integer. Fourier terms are the standard remedy (Hyndman & Athanasopoulos, 2021, §10.5).
- Selection: $K = 1$ by AIC (Table 7), then ARMA(3,1) by AIC.
- Estimates: mean 9.16 °C, $\sin_1 = -2.54$, $\cos_1 = -9.34$, $\phi = (1.533, -0.798, 0.184)$, $\theta_1 = -0.532$.
- Fitted annual cycle: amplitude **±9.7 °C**, warmest around **19 July**, coldest around **17 January**.
- **ψ-weights** (impulse response, the share of a weather "surprise" still felt $j$ days later): 1.00, 1.00, 0.74, 0.51, …, 0.23 at day 7, 0.08 at day 14.

### C. Exact Vectorised Forecasting (an engineering contribution)
Multi-step forecasts from **every** origin come from a single Kalman-filter pass, by propagating the predicted state with the transition matrix:
$$a_{i+h|i} = T^{h-1}a_{i+1|i}, \qquad \hat y_{i+h|i} = x_{i+h}'\beta + Z a_{i+h|i}$$
The notebook verifies the result against statsmodels' own `get_forecast()`; the maximum difference is 0.0 °C. The method is about 1000× faster than re-filtering once per origin.

### D. LSTM: Multi-Task Recurrent Network
Gate equations (Hochreiter & Schmidhuber, 1997):
$$f_t=\sigma(W_f[h_{t-1},x_t]+b_f),\; i_t=\sigma(W_i[\cdot]+b_i),\; o_t=\sigma(W_o[\cdot]+b_o),\; c_t=f_t\odot c_{t-1}+i_t\odot\tanh(W_c[\cdot]+b_c),\; h_t=o_t\odot\tanh(c_t)$$

```
input (30 days × 13 features: 11 weather variables + sin/cos of the annual phase)
  -> LSTM 64 (return sequences) -> Dropout 0.2
  -> LSTM 32 -> Dropout 0.2
  -> Dense 32 (ReLU)
  -> Dense 11: tomorrow's CHANGE in every weather variable   [33,803 trainable parameters]
```

| Design choice | Why |
|---|---|
| **Multi-task output** (all 11 variables) | richer training signal; it also enables recursive forecasting, because every input is predicted |
| **Δ (change) targets**, unit-variance per task | "no change" (persistence) becomes the default; stationary targets |
| **Recursive multi-step** | each predicted day is fed back; calendar features stay exact |
| **3-seed ensemble** | averaging reduces variance (Lakshminarayanan et al., 2017) |
| Adam 1e-3, batch 32, early stopping (patience 15), LR decay | avoids over-fitting on about 2,161 training windows |

Training: the three members stopped at their best epochs 15, 21 and 25 (Fig. 5).

### E. Hybrid-R: SARIMA + LSTM Residual Learner (proposed model)
This follows Zhang (2003): series = linear part + non-linear part.
1. SARIMA's one-step error: $e_{t+1} = y_{t+1} - \hat y^{\,S}_{t+1|t}$.
2. An LSTM with the same architecture learns to predict $e_{t+1}$ from the weather window plus SARIMA's recent errors. **Auxiliary tasks** (changes in the other weather variables) give it a dense training signal, because SARIMA's errors are nearly white noise.
3. Day ahead: $\hat y^{HR}_{t+1} = \hat y^{S}_{t+1|t} + \hat e_{t+1}$.
4. Multi-step, using **ψ-weight propagation**: $\hat y^{HR}_{t+h} = \hat y^{S}_{t+h|t} + \psi_{h-1}\,\hat e_{t+1}$. The correction fades exactly like a real weather anomaly, so the hybrid never inherits the LSTM's drift.

### F. Hybrid-E: Optimal Per-Horizon Ensemble
$$\hat y^{HE}_{t+h} = w_h\hat y^{S}_{t+h} + (1-w_h)\hat y^{L}_{t+h}, \qquad w_h = \operatorname{clip}_{[0,1]}\frac{\sum(\hat y^L-y)(\hat y^L-\hat y^S)}{\sum(\hat y^L-\hat y^S)^2}$$
The weights are fitted on 2015 (Bates & Granger, 1969). The weight on SARIMA is 0.22 at day 1, 0.27 at day 2, 0.42 at day 3, 0.36 at day 7 and **1.00 from day 14 onward**. The ensemble hands over automatically from the LSTM to SARIMA.

---

# PART 3: RESULTS (2016 TEST YEAR)

## 8. Next-Day Accuracy (Table 10, Fig. 7)

| Model | RMSE (°C) | MAE (°C) | MAPE (%) | R² | MASE | Skill vs persistence |
|---|---|---|---|---|---|---|
| Persistence | 2.340 | 1.769 | 0.626 | 0.9062 | 0.952 | +0.0 % |
| Climatology | 3.764 | 2.934 | 1.036 | 0.7573 | 1.579 | −158.7 % |
| ARIMA | 2.226 | 1.718 | 0.608 | 0.9151 | 0.924 | +9.5 % |
| SARIMA | 2.164 | 1.672 | 0.591 | 0.9198 | 0.900 | +14.5 % |
| LSTM | 2.017 | 1.566 | 0.555 | 0.9303 | 0.843 | +25.7 % |
| **Hybrid-R** | 2.033 | 1.596 | 0.565 | 0.9292 | 0.859 | +24.5 % |
| **Hybrid-E** | **2.008** | **1.549** | **0.549** | **0.9309** | **0.834** | **+26.4 %** |

## 9. Statistical Significance (Table 11, Diebold–Mariano with the HLN correction)

| Hybrid-R vs | DM (h=1) | p (h=1) | DM (h=7) | p (h=7) | Verdict (day ahead) |
|---|---|---|---|---|---|
| Persistence | −3.97 | 0.0001 | −5.27 | 0.0000 | Hybrid-R significantly better |
| Climatology | −9.77 | 0.0000 | −0.34 | 0.735 | Hybrid-R significantly better |
| ARIMA | −3.52 | 0.0005 | −2.70 | 0.007 | Hybrid-R significantly better |
| SARIMA | −2.68 | 0.0077 | −0.94 | 0.350 | Hybrid-R significantly better |
| LSTM | +0.55 | 0.585 | −2.02 | 0.044 | no significant difference |
| Hybrid-E | +1.02 | 0.310 | −1.30 | 0.194 | no significant difference |

**Reading:** a negative DM statistic means Hybrid-R is more accurate. At day 1 it is level with the LSTM. At day 7 it is **significantly better than the LSTM** (p = 0.044).

## 10. Skill by Lead Time (Table 12, Fig. 8)

| Model | h=1 | h=2 | h=3 | h=7 | h=14 | h=30 | h=60 | h=90 | Mean 1–90 |
|---|---|---|---|---|---|---|---|---|---|
| Persistence | 2.34 | 3.49 | 4.11 | 5.16 | 5.44 | 6.52 | 8.75 | 10.92 | 7.56 |
| Climatology | 3.76 | 3.76 | 3.76 | 3.76 | 3.76 | 3.76 | 3.76 | 3.76 | 3.76 |
| ARIMA | 2.23 | 3.26 | 3.73 | 4.43 | 4.73 | 5.75 | 7.48 | 8.43 | 6.46 |
| SARIMA | 2.16 | 3.08 | 3.43 | 3.73 | 3.65 | 3.67 | 3.67 | 3.67 | 3.64 |
| LSTM | 2.02 | 2.85 | 3.24 | 3.92 | 4.16 | 4.56 | 4.72 | 4.99 | 4.52 |
| **Hybrid-R** | 2.03 | 2.92 | 3.29 | 3.72 | 3.65 | 3.67 | 3.67 | 3.67 | **3.64** |
| Hybrid-E | 2.01 | 2.85 | 3.25 | 3.80 | 3.65 | 3.67 | 3.67 | 3.67 | 3.64 |

**Story of the figure:**
- The LSTM is sharpest for 1–4 days, falls behind SARIMA from day 5, and drifts upward through iterated (recursive) error accumulation.
- ARIMA, with no seasonality, drifts to 8.43 °C.
- SARIMA and both hybrids level off at about 3.67 °C, close to climatology. This is the **predictability limit** for daily temperature without a numerical weather-prediction model.
- **Hybrid-R is at least as accurate as SARIMA at every lead and up to 6.0 % better at short leads.** It keeps the LSTM's gains and never inherits its drift.
- Over all 90 leads, Hybrid-R (3.638), Hybrid-E (3.641) and SARIMA (3.645) are a practical tie (less than 1 % apart). They separate only at short leads.

## 11. Residual Diagnostics (Table 13, Fig. 9)

| Model | Bias (°C) | Error sd (°C) | Ljung–Box Q(10) | p |
|---|---|---|---|---|
| Persistence | −0.009 | 2.340 | 35.7 | 0.0001 |
| Climatology | +0.821 | 3.673 | 432.0 | 0.0000 |
| ARIMA | +0.016 | 2.226 | 17.2 | 0.070 |
| SARIMA | +0.138 | 2.160 | 19.5 | 0.034 |
| LSTM | +0.038 | 2.017 | 25.5 | 0.004 |
| Hybrid-R | +0.130 | 2.029 | 26.0 | 0.004 |
| Hybrid-E | +0.060 | 2.007 | 24.1 | 0.007 |

- Errors are centred near zero and close to normal, with slightly heavy tails.
- With 366 days the Ljung–Box test detects even small leftover autocorrelation. It is far smaller than the baselines' and points to multi-day weather regimes, which is future work.
- Climatology's +0.82 °C bias shows that 2016 was warmer than the 2009–2014 average.

## 12. Uncertainty Quantification (Table 14)
- **ARIMA and SARIMA** use analytic Gaussian intervals: $\hat y \pm 1.96\,\sigma_h$.
- **LSTM and the hybrids** use **split-conformal** intervals (Vovk et al., 2005). For each lead, the half-width is the $\lceil (n+1)(0.95)\rceil$-th smallest absolute validation error. This needs no distributional assumption.

| Model | Method | Coverage h=1 | Coverage h=7 | Coverage h=30 | Coverage, all leads | Width h=1 (°C) | Mean width (°C) |
|---|---|---|---|---|---|---|---|
| ARIMA | analytic | 95.4 % | 94.5 % | 98.1 % | 96.8 % | 9.0 | 25.0 |
| SARIMA | analytic | 96.4 % | 96.4 % | 97.3 % | 96.9 % | 8.8 | 15.9 |
| LSTM | conformal | 95.1 % | 97.0 % | 97.0 % | 96.8 % | 8.5 | 20.3 |
| Hybrid-R | conformal | 96.7 % | 98.1 % | 98.6 % | 98.5 % | 8.8 | 17.6 |
| Hybrid-E | conformal | 95.1 % | 97.0 % | 98.6 % | 98.4 % | 8.5 | 17.6 |

**Reading:** day-ahead intervals are **well calibrated** (95–97 % against a 95 % target). At long leads the conformal intervals are **conservative**, wider than necessary, because the 2015 calibration year was more variable than 2016.

## 13. 90-Day Out-of-Sample Forecast (Table 15, Fig. 10)
Forecast for 1 January – 31 March 2017, beyond the last observation:

| Lead · Date | SARIMA | Hybrid-R | Hybrid-E | Normal (climatology) |
|---|---|---|---|---|
| Day 1 · 01 Jan 2017 | −1.7 [−6.1, +2.7] | −0.6 [−5.0, +3.8] | −1.3 [−5.5, +2.9] | −0.3 |
| Day 7 · 07 Jan 2017 | −0.6 [−8.4, +7.2] | −0.3 [−8.7, +8.1] | +0.8 [−7.3, +8.9] | −0.8 |
| Day 30 · 30 Jan 2017 | −0.3 [−8.3, +7.8] | −0.2 [−9.3, +8.8] | −0.3 [−9.3, +8.8] | −1.1 |
| Day 60 · 01 Mar 2017 | +2.1 [−6.0, +10.1] | +2.1 [−6.9, +11.1] | +2.1 [−6.9, +11.1] | +2.0 |
| Day 90 · 31 Mar 2017 | +6.3 [−1.8, +14.3] | +6.3 [−2.7, +15.2] | +6.3 [−2.7, +15.2] | +7.3 |

The forecasts follow the turn from winter to spring. Beyond about 2 weeks they converge to the seasonal normal, and the 95 % interval widens to the natural spread of the season. That is the honest behaviour of a skilful model.

---

## 14. Figure and Table Index

| Figure | Content | | Table | Content |
|---|---|---|---|---|
| 1 | Daily weather 2009–2016 + split ribbon | | 1 | Data-quality audit |
| 2 | Correlation heatmap | | 2 | Descriptive statistics |
| 3 | STL decomposition (weekly) | | 3 | ADF / KPSS stationarity |
| 4 | ACF / PACF, raw vs anomaly | | 4 | Chronological split |
| 5 | LSTM learning curves | | 5–6 | ARIMA AIC search, parameters |
| 6 | Test-year forecasts vs observed | | 7–9 | SARIMA selection, parameters |
| 7 | RMSE · MAE · MAPE · R² comparison | | 10 | Day-ahead metrics |
| 8 | Skill vs lead time (1–90 days) | | 11 | Diebold–Mariano tests |
| 9 | Residual diagnostics (Hybrid-R) | | 12 | RMSE by horizon |
| 10 | 90-day forecast with 95 % intervals | | 13–15 | Error diagnostics · coverage · future forecast |

All figures are saved as 300-dpi PNG files in `outputs/figures/` and are ready for slides and the paper.

## 15. Honest Limitations
1. **One station, one test year.** A multi-year rolling test would tighten the comparison.
2. **Small dataset for deep learning.** There are only about 2,161 training windows (6 years).
3. **Frozen parameters.** Models are not re-estimated during the test year.
4. **No numerical weather-prediction inputs.** Beyond about a week, no statistical model can beat climatology.
5. **Conformal validity** assumes exchangeability, and the validation year also serves early stopping and weights. Coverage is therefore checked empirically.
6. **Four imputed days** (25–28 Oct 2016) remain in the test year (about 1 %).

---

# PART 4: PRESENTATION SLIDE OUTLINE

### Slide 1: Title
Hybrid Weather Forecasting with ARIMA, SARIMA and LSTM · Rajarshi Chakraborty · 6th Semester.

### Slide 2: Problem and Motivation
- Weather has a linear part (annual cycle, persistence) and a non-linear part (fronts, humidity, wind).
- ARIMA/SARIMA capture the linear part; an LSTM learns the non-linear part but drifts when iterated.
- Research questions: Is a hybrid more accurate? How does skill decay with the horizon? Are the intervals trustworthy?

### Slide 3: Data and Cleaning
- 420,551 ten-minute records → 2,922 clean days.
- Real defects fixed: 327 duplicates, 38 sentinel values, 4 low-coverage days, wind converted to vectors.

### Slide 4: Diagnostics
- Stationary (ADF p = 0.012, KPSS p ≥ 0.10) → d = 0.
- STL seasonal strength 0.87; PACF shows short memory → AR(3).

### Slide 5: Models
- ARIMA(3,0,1), and SARIMA with Fourier seasonality (with the reason it is not $(P,D,Q)_{365}$).
- Multi-task LSTM (33.8 k parameters, 3-seed ensemble).
- Hybrid-R (residual learner + ψ-propagation) and Hybrid-E (optimal weights).

### Slide 6: Evaluation Protocol
- Chronological split, test used once, every day forecast at every lead 1–90.
- Metrics, DM tests, interval coverage.

### Slide 7: Results — Next Day
- Table 10 and Fig. 7: hybrids/LSTM R² ≈ 0.93; +24–26 % skill over persistence.
- Hybrid-R 6 % better than SARIMA (p = 0.008).

### Slide 8: Results — Horizon (key slide)
- Fig. 8: the LSTM drifts, ARIMA drifts, Hybrid-R is never worse than SARIMA → best of both worlds.

### Slide 9: Uncertainty and 90-Day Forecast
- Fig. 10 and Table 14: calibrated day-ahead intervals; a forecast that follows winter into spring.

### Slide 10: Conclusion and Future Work
- Contributions: a leakage-free benchmark, exact vectorised state-space forecasting, a stabilised residual hybrid, conformal intervals.
- Next: a 36-year Indian dataset (ERA5, New Delhi), NWP inputs, Transformer models.

---

# PART 5: VIVA QUESTION BANK AND MODEL ANSWERS

### Q1: Why is the file called final_stock_market.ipynb if it is about weather?
**Answer:** The project grew out of the earlier stock-market LSTM project, and the repository keeps that filename. The content is a hybrid weather-forecasting study that reuses the earlier project's lessons: chronological splits, scalers fitted on training data only, and awareness of recursive error accumulation.

### Q2: Why did you combine ARIMA/SARIMA with LSTM?
**Answer:** Following Zhang (2003), a series is a linear part plus a non-linear part. SARIMA captures the annual cycle and persistence efficiently. The LSTM learns non-linear multivariate effects, such as pressure, humidity and wind patterns. In our results the LSTM is sharpest for 1–4 days but drifts at long leads, and SARIMA is stable but less sharp. Hybrid-R keeps both strengths: it is 6 % better than SARIMA one day ahead and never worse than SARIMA at any lead.

### Q3: Why is your SARIMA written with sine/cosine terms instead of SARIMA(P,D,Q)₃₆₅?
**Answer:** With daily data the season is 365.25 days. A textbook SARIMA(1,0,1)(1,1,1)₃₆₅ has a 732-dimensional state vector and needs more than 9 GB of memory for the Kalman filter. Seasonal differencing at lag 365 doubles the noise, and 365.25 is not an integer. The standard solution (Hyndman & Athanasopoulos, §10.5) represents seasonality with Fourier terms and keeps ARMA errors. AIC selected one harmonic and ARMA(3,1).

### Q4: How did you prevent data leakage?
**Answer:**
- The split is strictly chronological.
- The scaler, climatology and all parameters are fitted on 2009–2014 only.
- 2015 is used only for early stopping, ensemble weights and interval calibration.
- 2016 is used once.
- STL is two-sided, so it is used only for description, never as an input.
- Every forecast uses only data available at its origin.

### Q5: Why is MAPE so small (about 0.55 %)?
**Answer:** It is computed in kelvin. Jena's temperature crosses 0 °C in winter, and MAPE in °C divides by values near zero and explodes. Percentage errors are only meaningful on a scale with a true zero. RMSE and MAE are the primary metrics.

### Q6: Hybrid-E has the lowest next-day RMSE. Why do you call Hybrid-R the proposed model?
**Answer:**
- The day-ahead differences between Hybrid-E, the LSTM and Hybrid-R are not statistically significant (DM p = 0.31 and 0.59).
- Hybrid-R's design contribution is that its correction fades through SARIMA's ψ-weights. It is therefore guaranteed never to drift, and it was at least as accurate as SARIMA at every lead.
- It also has the lowest mean error over 1–90 days (3.638 °C) and is significantly better than the LSTM at 7 days (p = 0.044).

### Q7: What are ψ-weights?
**Answer:** Any ARMA model can be written as $u_t = \sum \psi_j \varepsilon_{t-j}$. $\psi_j$ is the share of today's surprise still present $j$ days later. For our SARIMA it is 1.00, 1.00, 0.74, 0.51 … and 0.08 after two weeks. Hybrid-R multiplies the LSTM's correction by these weights, so the correction decays exactly like a real weather anomaly.

### Q8: Why does the LSTM predict changes of all 11 variables, not just temperature?
**Answer:**
- **Multi-task learning:** predicting the whole weather state forces the network to learn joint atmospheric dynamics, a richer training signal.
- **Recursion:** forecasting several days ahead recursively needs future values of every input, and the network provides them.
- **Δ targets:** "no change" (persistence) becomes the default behaviour.

### Q9: Why does the LSTM get worse at long horizons?
**Answer:** Recursive forecasting feeds predictions back as inputs, so small errors accumulate. Its RMSE rises from 2.02 °C at day 1 to 4.99 °C at day 90, versus 3.67 °C for SARIMA. This is the same compounding-error problem we met in the stock project. The hybrids solve it: Hybrid-E moves its weight to SARIMA (100 % from day 14), and Hybrid-R's correction fades with the ψ-weights.

### Q10: What does the Diebold–Mariano test tell you?
**Answer:** It tests whether two models' squared errors differ systematically, not by chance, allowing for autocorrelated errors. Hybrid-R is significantly better than SARIMA (DM = −2.68, p = 0.008), ARIMA and persistence one day ahead. It is statistically level with the LSTM at day 1 and better than the LSTM at day 7 (p = 0.044).

### Q11: How did you create the confidence intervals for the LSTM?
**Answer:** Split-conformal prediction. For each lead, I took the 95th-percentile (finite-sample corrected) absolute error on the validation year as the half-width. It needs no normality assumption. On the test year the day-ahead coverage was 95.1–96.7 % against the 95 % target.

### Q12: Why are 90-day forecasts close to the seasonal normal?
**Answer:** Day-to-day weather is predictable for only about 1–2 weeks from past observations. Beyond that, the best honest forecast is the seasonal normal plus uncertainty, which every good model converges to (≈ 3.67 °C RMSE). Fig. 8 shows this explicitly; models that do not converge (ARIMA, the recursive LSTM) are worse.

### Q13: What is the "fixed-target" evaluation and why does it matter?
**Answer:** Every lead time is scored on exactly the same 366 test days, so differences between leads reflect the lead alone, not a different season being scored. The proof is that climatology's RMSE is perfectly flat at 3.76 °C across all 90 leads.

### Q14: What would you improve next?
**Answer:**
- A longer and richer dataset: 36 years of ERA5 reanalysis for New Delhi, which gives about 5× more training data, physical drivers such as clouds, radiation and soil moisture, and real heatwaves.
- Systematic hyperparameter optimisation.
- Anomaly-based LSTM inputs to stop the drift.
- Numerical weather-prediction inputs.
- Transformer and N-BEATS architectures.

---

## References
1. Bates, J. M., & Granger, C. W. J. (1969). The combination of forecasts. *Operational Research Quarterly*, 20(4), 451–468.
2. Box, G. E. P., & Jenkins, G. M. (1970). *Time Series Analysis: Forecasting and Control*. Holden-Day.
3. Chollet, F. (2021). *Deep Learning with Python* (2nd ed.). Manning.
4. Cleveland, R. B., et al. (1990). STL: A seasonal-trend decomposition procedure based on loess. *Journal of Official Statistics*, 6(1), 3–73.
5. Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics*, 13(3), 253–263.
6. Harvey, D., Leybourne, S., & Newbold, P. (1997). Testing the equality of prediction mean squared errors. *International Journal of Forecasting*, 13(2), 281–291.
7. Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8), 1735–1780.
8. Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed.). OTexts.
9. Hyndman, R. J., & Koehler, A. B. (2006). Another look at measures of forecast accuracy. *International Journal of Forecasting*, 22(4), 679–688.
10. Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). Simple and scalable predictive uncertainty estimation using deep ensembles. *NeurIPS 30*.
11. Vovk, V., Gammerman, A., & Shafer, G. (2005). *Algorithmic Learning in a Random World*. Springer.
12. Zhang, G. P. (2003). Time series forecasting using a hybrid ARIMA and neural network model. *Neurocomputing*, 50, 159–175.
