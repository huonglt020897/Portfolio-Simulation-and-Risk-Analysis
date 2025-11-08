*This repository was part of the assignment for Risk Analysis course in Business Intelligence master program.*

## **1. Problem Introduction**

This report combines **Problem 1** and **Problem 2** to analyze and compare two approaches for modeling stock price evolution and portfolio risk:

* **Problem 1:** A baseline *constant-variance Monte Carlo simulation* assuming i.i.d. normal log-returns.
* **Problem 2:** An advanced *time-varying volatility simulation* using **J.P. Morgan’s RiskMetrics** model to incorporate volatility clustering.

The study evaluates a diversified investment portfolio consisting of **Intel Corporation (INTC)** and **SSAB AB (SSAB-A.ST)** from **January 2021 to June 2025**, with an initial investment of **$1,000 (50/50 split)**.
The main objectives are to:

* Estimate expected growth and volatility,
* Quantify downside risk using **Value at Risk (VaR)**,
* Compare model outputs with actual outcomes, and
* Evaluate how time-varying volatility improves upon the constant-variance model.

---

## **2. Common Methodological Foundations**

Both problems follow a similar computational structure:

1. **Extract** and align market data (Intel & SSAB, Jan 2011 – Jun 2025) & **Convert** SSAB prices to USD using daily SEK/USD exchange rates.
2. **Compute** daily log-returns: 
   $$R_t = \log\left(\frac{X_t}{X_{t-1}}\right)$$

3. **Estimate** mean ($\mu$) and variance ($\sigma^2$) of log-returns using **Maximum Likelihood Estimation (MLE)**.
4. **Simulate** 4.5 years (≈1134 trading days) of future prices using stochastic processes (with constant variance for Problem 1 and J.P. Morgan’s RiskMetrics for Problem 2).
5. **Perform Monte Carlo simulations** (10,000 iterations) to obtain distributions of ending prices and portfolio value.
6. **Derive Value at Risk (VaR)** (Problem 2)
7. **Compare** simulated vs actual outcomes.

---
## **3. Step-by-Step Description and Results**

### **3.1 Extract Data**
Daily closing prices were extracted for Intel (USD) and SSAB A (SEK) from Yahoo Finance between **2011-01-01 and 2025-06-30**.

For comparability, SSAB prices were converted into USD using daily SEK/USD exchange rates. The merged dataset ensured aligned trading days and consistent currency values. The data were split into two subsets:
- Historical (for parameter estimation): before 2021-01-01
- Observed (for simulation validation): from 2021-01-01 onwards.

The SSAB conversion introduces an additional stochastic component — exchange-rate volatility, which partially explains why SSAB simulations later show heavier variance than Intel. Dividing the dataset ensures that model parameters are out-of-sample validated, making the forecast process more realistic and reducing look-ahead bias.

**Key code function**:
```
def extract_data(ticker_symbol, start_date,end_date):
    """
    Fetch ticker data from Yahoo Finance
    """
    ticker=yf.Ticker(ticker_symbol)
    ticker_data=ticker.history(start=start_date,end=end_date)
    ticker_data.index = ticker_data.index.tz_localize(None)
    return ticker_data
```
---

### **3.2 Calculate Log-Returns**

Log-returns were computed as:

$$R_t = \log\left(\frac{X_t}{X_{t-1}}\right)$$

This transformation stabilizes variance and normalizes the return distribution.
The sample mean ($\mu$) and variance ($\sigma^2$) of log-returns were calculated to summarize the historical behavior of each asset.

**Key code function**:
```
def compute_log_return(hist_data):
    '''
    Compute log returns from historical price data
    '''
    log_returns = pd.DataFrame()
    log_returns['daily_log_return'] = np.log(hist_data['Close'] / hist_data['Close'].shift(1))
    log_returns.dropna(inplace=True)
    return log_returns
```
<img width="1189" height="490" alt="step2" src="https://github.com/user-attachments/assets/a907e6f0-c892-49a7-ae5d-294d5d12049c" />

**Inference:**
<br>For Intel and SSAB, variance differences already indicated that SSAB is the riskier asset, likely due to smaller market capitalization and currency sensitivity.

---

### **3.3 Use MLE to Estimate Parameters**

Using the assumption that log-returns follow a normal distribution, the MLE estimators for the mean and variance are:

$$
\hat{\mu} = \bar{R} = \frac{1}{n}\sum_{t=1}^nR_t, \quad
\hat{\sigma}^2 = \frac{1}{n}\sum_{t=1}^n (R_t - \bar{R})^2
$$

These estimates serve as inputs for the next step’s stochastic simulation.

**Key code function**:
```
def compute_norm_parameters(data):
    '''Compute mean, variance and standard deviation of log returns'''
    mean = data['daily_log_return'].mean()
    var = data['daily_log_return'].var(ddof=0)
    std = data['daily_log_return'].std(ddof=0)
    return mean, var, std
```

**Output:**
| Stock |      Mean | Variance | Standard Deviation |
|------:|----------:|---------:|-------------------:|
| Intel |  0.000467 | 0.000329 |           0.018148 |
|  SSAB | -0.000436 | 0.000715 |           0.026747 |

<img width="1189" height="490" alt="step3" src="https://github.com/user-attachments/assets/990c13bb-4848-4fc0-b2f9-455feca94077" />

**Inference:**
- Intel’s positive mean reflects a slight upward drift over the decade, whereas SSAB’s negative mean shows long-term stagnation or cyclical performance. The variance of SSAB returns is more than double that of Intel, consistent with the earlier discussion on foreign-exchange risk.
- KDE plots confirmed that both datasets approximately follow normal distributions.

---
### **3.4 Simulate Daily Share Prices**

#### **3.4.1 Problem 1 — Simulate Daily Share Prices with Constant Variance Simulation**

The first problem uses a **constant-variance, normal-return model**:

$$X_t = X_{t-1} \times e^{R_t}, \quad R_t \sim \mathcal{N}(\mu, \sigma^2)$$

Each trading day’s log-return is drawn independently from the estimated normal distribution. This approach assumes that variance remains constant over time.

**Key code function**:
```
#Function to simulate closing prices
def simulate_closing_price(initial_price, mean, var, number_of_trading_days=252*4.5):
    simulated_returns = np.random.normal(mean, np.sqrt(var), int(number_of_trading_days))
    simulated_closing_prices = [initial_price]
    for i in simulated_returns:
        next_price = simulated_closing_prices[-1] * np.exp(i)
        simulated_closing_prices.append(next_price)
    return simulated_closing_prices
```
**Description:**
- `initial_price` represents the stock price on the first trading day of 2021.
- Daily returns are generated randomly from a normal distribution defined by the estimated mean ($\mu$) and standard deviation ($\sigma$).
- 	Each subsequent day’s price is calculated as: $X_t = X_{t-1} \times e^{R_t}$
- The function uses a fixed random seed to ensure reproducibility of the simulation results.

<img width="1189" height="989" alt="1pricepath1" src="https://github.com/user-attachments/assets/d9632242-44c1-4902-a1e9-f6f2c87293d4" />

**Inference:**
- **Intel:** This divergence indicates an upward drift bias in the model—caused by using the long-run historical mean return $(\mu>0)$ estimated from a bullish decade (2011–2020).
- **SSAB:** The model fails to capture the cyclical recovery and higher-volatility bursts in the real market and foreign-exchange environment.
- **Portfolio:** The simulated portfolio still mirrors the general market trend, demonstrating that diversification dampens individual-asset errors.

#### **3.4.2 Problem 2 — Simulate Daily Share Prices with Time-Varying Volatility Simulation (JP Morgan's RiskMetrics)**

The **J.P. Morgan's RiskMetrics** model was used to update daily variance dynamically:

$$\sigma_t^2 = \lambda \sigma_{t-1}^2 + (1 - \lambda)R_{t-1}^2$$
with a decay factor ($\lambda = 0.94$).

A custom simulation function (`morgan_var_log_return`) was implemented to evolve prices through time.
At each step, a random log-return $(R_t \sim \mathcal{N}(\mu, \sigma_{t}^2))$ is drawn, price is updated as ($P_t = P_{t-1} e^{R_t}$), and variance is adjusted recursively. Each call to the function generates a unique random path, ensuring variability across simulations.

```
def simulate_morgan_closing_price(initial_price, mean, initial_var, number_of_trading_days=252*4.5, alpha=0.94):
    """
    Simulate closing prices using Morgan's method
    """
    variances = [initial_var] #list of variances for INTC
    prices = [initial_price]  #list of prices for INTC
    initial_log_return = np.random.normal(mean, np.sqrt(initial_var))
    log_returns = [initial_log_return]  #list of log returns for INTC
    for t in range(1, int(number_of_trading_days)):
        var_t = alpha * variances[-1] + (1-alpha) * (log_returns[-1] ** 2) #update variance
        r_t = np.random.normal(mean, np.sqrt(var_t)) #simulate return_t based on updated variance var_t
        price_t = prices[-1]*np.exp(r_t) #compute price_t based on return_t and price_(t-1)

        variances.append(var_t)
        log_returns.append(r_t)
        prices.append(price_t)
        
    return variances, log_returns, prices
```

<img width="1189" height="989" alt="1pricepath2" src="https://github.com/user-attachments/assets/5255dc5d-6967-4ec2-b759-744fd97d72c2" />

**Inference:**
<br>This structure captures volatility persistence, allowing shocks in returns to have prolonged effects on future variance. However, since the model assumes normal innovations, extreme market events are understated.

When comparing simulated with actual prices, Intel’s simulated path shows a persistent upward bias, whereas SSAB’s path underestimates volatility and recovery. This mismatch arises because:

- Drift ($\mu$) is estimated from a historical regime that may not persist post-2021.
- Volatility decay ($\lambda = 0.94$) smooths short-term shocks, producing overly stable trajectories.
- Real market behavior exhibits asymmetric volatility and mean reversion, not captured by Gaussian simulation.

Despite these limitations, the J.P. Morgan’s RiskMetrics model effectively demonstrates how volatility clustering can propagate through simulated price paths.

---
### **3.5 Monte Carlo Simulation for Ending Prices**
10,000 Monte Carlo simulations were run for both Intel and SSAB to generate a distribution of possible ending prices and portfolio value in **June 2025**.

**Key code function**:
```
from tqdm import tqdm

#Function to simulate ending prices
def simulate_ending_price(initial_price, mean, initial_var, number_of_trading_days=252*4.5, num_of_simulations=10000):
    """
    Simulate ending prices for a given number of simulations
    """
    simulated_ending_prices=[]
    for i in tqdm(range(num_of_simulations)):
        simulated_var, simulated_returns, simulated_closing_prices= simulate_morgan_closing_price(initial_price, mean, initial_var, number_of_trading_days)
        simulated_ending_price=simulated_closing_prices[-1]
        simulated_ending_prices.append(simulated_ending_price)
    return simulated_ending_prices
```

#### **3.5.1 Problem 1 — Simulation Ending Prices with Constant Variance Simulation**

|       | INTC ending price | SSAB ending price | INTC ending value | SSAB ending value | Portfolio value |
|------:|------------------:|------------------:|------------------:|------------------:|----------------:|
| count |          10000.00 |          10000.00 |          10000.00 |          10000.00 |        10000.00 |
|  mean |             91.95 |              2.38 |           1023.93 |            457.40 |         1481.33 |
|   std |             61.27 |              2.60 |            682.20 |            499.10 |         1174.95 |
|   min |              7.54 |              0.05 |             84.01 |             10.09 |           94.10 |
|   25% |             50.34 |              0.86 |            560.55 |            165.45 |          726.00 |
|   50% |             76.82 |              1.61 |            855.43 |            308.47 |         1163.90 |
|   75% |            115.26 |              2.92 |           1283.42 |            560.90 |         1844.31 |
|   max |            669.66 |             39.05 |           7456.78 |           7501.90 |        14958.69 |

<img width="1490" height="490" alt="output1" src="https://github.com/user-attachments/assets/516484dc-d8e4-490f-8116-f04f934434be" />

**Inference:**
- The long right tail shows that extreme positive outcomes are possible, but with very low probability.
- The 95% confidence interval reflects an asymmetrical risk–reward structure: downside losses are limited, but upside gains, though rare, can be large.

#### **3.5.2 Problem 2 — Simulation Ending Prices with Time-Varying Volatility Simulation (J.P. Morgan’s RiskMetrics)**

|       | INTC ending price | SSAB ending price | INTC ending value | SSAB ending value | Portfolio value |
|------:|------------------:|------------------:|------------------:|------------------:|----------------:|
| count |          10000.00 |          10000.00 |          10000.00 |      1.000000e+04 |    1.000000e+04 |
|  mean |            575.72 |            593.86 |           6410.75 |      1.140844e+05 |    1.204951e+05 |
|   std |          41175.96 |          54694.70 |         458499.65 |      1.050727e+07 |    1.096518e+07 |
|   min |              0.00 |              0.00 |              0.00 |      0.000000e+00 |    0.000000e+00 |
|   25% |             59.58 |              1.12 |            663.44 |      2.159600e+02 |    8.803800e+02 |
|   50% |             75.76 |              1.60 |            843.60 |      3.078100e+02 |    1.150830e+03 |
|   75% |             97.20 |              2.28 |           1082.33 |      4.377700e+02 |    1.521610e+03 |
|   max |        4097694.86 |        5463155.06 |       45628363.29 |      1.049514e+09 |    1.095142e+09 |

<img width="1489" height="490" alt="output2" src="https://github.com/user-attachments/assets/1ed535d6-9cb9-4b45-9bab-d9fd6a5cf400" />

#### **Inference:**
- The mean is much larger than the median, confirming the presence of right-skewed distributions with large outliers.

---

### **3.6 Value at Risk (VaR)** (problem 2)

The **90% VaR** was estimated from the simulated distribution of portfolio values:

$$\text{VaR}_{0.9} = V_0 - Q_{0.10}(V_T)$$

where $Q_{0.10}(V_T)$ is the 10th percentile of simulated portfolio values. This represents the potential loss that the portfolio will not exceed with 90% confidence.

$$V_0=1000, \qquad Q_{0.10}(V_T)=639.40$$

$$\text{VaR}_{0.9} = 1000 - 639.40 = 360.60 \space \text{(USD)}$$

The portfolio has a **360.60 USD** potential loss (~36.06% on initial value) with 10% probability, indicating moderate downside risk. This measure captures both individual asset volatilities and diversification effects.

Intel contributes more to upward bias (overperformance in simulations), while SSAB contributes to tail thickness.

---

### **3.7 Compare the Simulated with the Actual Results**
#### **3.7.1 Problem 1 — Compare the Constant Variance Simulation with the Actual Results**
|                        Stock |               INTC |               SSAB |         Portfolio |
|-----------------------------:|-------------------:|-------------------:|------------------:|
|                Initial Value |                500 |                500 |              1000 |
|                       Shares |              11.14 |             192.11 |               NaN |
|                Initial Price |               44.9 |                2.6 |               NaN |
|        Expected Ending Price |              91.95 |               2.38 |               NaN |
| Expected Ending Price 95% CI |     (23.0, 251.67) |       (0.27, 9.23) |               NaN |
|          Actual Ending Price |              22.69 |               5.96 |               NaN |
|        Expected Ending Value |            1023.93 |              457.4 |           1481.33 |
| Expected Ending Value 95% CI |  [256.11, 2802.38] |   [51.87, 1773.15] |  [308.3, 4575.64] |
|          Actual Ending Value |             252.66 |            1145.61 |           1398.27 |
|         Expected Profit/Loss |             523.93 |              -42.6 |            481.33 |
|  Expected Profit/Loss 95% CI | (-243.89, 2302.38) | (-448.13, 1273.15) | (-691.7, 3575.64) |
|           Actual Profit/Loss |            -247.34 |             645.61 |            398.27 |

**Inference:**
- The simulation model does not successfully provide an expected growth trajectory (fail to predict the trajectory of both Intel and SSAB).
- However, it quantifies the uncertainty of outcomes.

#### **3.7.2 Problem 2 — Compare the Time-Varying Volatility Simulation (J.P. Morgan’s RiskMetrics) with the Actual Results**
|                        Stock |    INTC |      SSAB | Portfolio |
|-----------------------------:|--------:|----------:|----------:|
|                Initial Value |  500.00 |    500.00 |   1000.00 |
|                       Shares |   11.14 |    192.11 |       NaN |
|                Initial Price |   44.90 |      2.60 |       NaN |
|        Expected Ending Price |  575.72 |    593.86 |       NaN |
|          Median Ending Price |   75.76 |      1.60 |       NaN |
| 10th Percentile Ending Price |   44.82 |      0.73 |       NaN |
|          Actual Ending Price |   22.69 |      5.96 |       NaN |
|        Expected Ending Value | 6410.75 | 114084.39 | 120495.14 |
|          Median Ending Value |  843.60 |    307.81 |   1150.83 |
| 10th Percentile Ending Value |  499.11 |    139.39 |    639.40 |
|          Actual Ending Value |  252.66 |   1145.61 |   1398.27 |
|         Expected Profit/Loss | 5910.75 | 113584.39 | 119495.14 |
|           Actual Profit/Loss | -247.34 |    645.61 |    398.27 |
|            Value at Risk 90% |    0.89 |    360.61 |    360.60 |

***Because the simulations produced several extreme outliers, the median value is included alongside the mean to give a clearer picture of the overall distribution.*
#### **Inference:**

- **Intel**: 
  - The model predicted an expected ending price of $371.73, far above the actual ending price of $22.69.
  - The median simulated price ($75.46) was closer to reality but still much higher — showing that the model’s average drift from historical data created an upward bias.
- **SSAB**:
  - SSAB’s simulation shows an expected ending price of $50.76, while the actual price was $5.96.
  - The median simulated price ($1.61) and 10th percentile ($0.69) suggest that the simulation captured the possibility of very low outcomes, even if the mean was inflated by rare extreme values.
- **Portfolio**: Despite offsetting biases, the portfolio simulation produced a realistic risk–return balance, showing the value of diversification. Intel’s optimistic bias andSSAB’s volatility partly offset each other, producing a more balanced total portfolio outcome.

  
---
## **4. Comparative Analysis: Problem 1 vs Problem 2**

| Feature                 | Problem 1: Constant Variance | Problem 2: J.P. Morgan’s RiskMetrics    |
| ----------------------- | ---------------------------- | ------------------------------ |
| Volatility behavior     | Constant (homoscedastic)     | Time-varying (heteroscedastic) |
| Innovation distribution | Normal, i.i.d.               | Normal, conditional variance   |
| Model realism           | Simple and stable            | Captures volatility clustering |
| Sensitivity to shocks   | None                         | Moderate (depends on λ)        |
| Tail risk capture       | Poor                         | Improved, but still thin tails |
| Intel prediction        | Overestimated                | Overestimated                  |
| SSAB prediction         | Underestimated               | Slightly improved but smoother |
| Portfolio VaR           | Not computed                 | 229.6 USD (22.9%)              |
| Interpretability        | High                         | Moderate                       |
| Practical application   | Teaching baseline            | Risk analysis, VaR estimation  |

**Inference:**

* The **J.P. Morgan’s RiskMetrics model (Problem 2)** adds realism through adaptive volatility, aligning better with financial theory.
* The **constant-variance model (Problem 1)** is computationally simpler but fails to represent real-world heteroskedasticity.
* Both models highlight the sensitivity of Monte Carlo forecasts to small parameter changes, reinforcing that such models serve best as **risk quantification tools**, not as deterministic predictors.

---

## **5. Discussion and Conclusion**

Both models successfully demonstrated core principles of **quantitative risk analysis**:

1. **Monte Carlo simulation** reveals a range of possible outcomes, not single-point forecasts.
2. **Normality assumption limitations:** Fat tails and correlation clustering make real risks heavier than simulated ones.
3. **Volatility dynamics matter:** The J.P. Morgan’s RiskMetrics framework better mirrors observed market behavior, though still underestimates extreme risk events.
4. **Diversification effect:** The portfolio shows smoother and more robust performance than individual assets, validating the foundational principle of risk pooling.
