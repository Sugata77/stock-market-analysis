#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
warnings.filterwarnings("ignore")


# In[2]:


# Objective: To analyze market trends and predict future market behavior using machine learning techniques.


# In[3]:


# Import the data


# In[4]:


data = pd.read_csv(r'D:\Work\Unified Mentor\Data\stocks.csv')


# In[5]:


# Initial exploration:


# In[6]:


data.head()


# In[7]:


print(f"Dataset shape: {data.shape}")
print("\nData types:\n", data.dtypes)
print("\nMissing values:\n", data.isnull().sum())
print("\nUnique tickers:", data['Ticker'].unique())


# In[8]:


# There are no missing values to deal with.


# In[9]:


# Data Cleaning


# In[10]:


# Remove any duplicate rows
data = data.drop_duplicates()
data.head()


# In[11]:


# Convert Date column to datetime
data['Date'] = pd.to_datetime(data['Date'])
data.head()


# In[12]:


# Summary Statistics
display(data.describe())


# In[13]:


# Step 1: Calculate Mean of 'Close' column
mean_close = data['Close'].mean()

# Step 2: Calculate Standard Deviation of 'Close' column
std_close = data['Close'].std()

# Step 3: Apply CV Formula
cv_close = (std_close / mean_close) * 100

# Step 4: Print the result
print(f"Coefficient of Variation of Close Price = {round(cv_close,2)}%")

## Summary of Descriptive Analysis:

# Observations:

248 trading days, confirming a complete dataset.

# Price & Volume Averages:

Open: $215.25 | Close: $215.38 | High: $217.92 | Low: $212.70

Avg. Volume: 32M shares, which indicates moderate liquidity.

Minimal difference between open & close prices suggests stable daily sentiment.

# Market Volatility:

Closing Price Std Dev: $91.46, which implies high volatility, significant price swings.

Volume Std Dev: 22.33M. Some days show extreme trading activity.

# Extreme Values:

Lowest Close: $89.35 | Highest Close: $366.83

Lowest Volume: 2.66M | Highest Volume: 113M

Implication: Sharp price spikes, possibly linked to market news/events.

# Price Distribution (Quartiles):

Q1: $136.35, Q2 (Median): $209.92, Q3: $303.94

50% of prices were below $209.92, indicating the stock spent more time in lower price ranges.

Wide range ($136 - $303) confirms strong volatility.

# Key Takeaways:
High volatility – Large price swings.
News-driven movements – Possible external triggers.
Fluctuating trading volume – Institutional activity likely.
Median < Mean – High positive skewness.
# In[14]:


# Plot Closing Price Over Time
plt.figure(figsize=(14, 7))

sns.lineplot(data=data, x='Date', y='Close', hue='Ticker', marker='o')

plt.title('Closing Price Trend Over Time (AAPL, MSFT, NFLX, GOOG)', fontsize=18)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Closing Price (USD)', fontsize=14)
plt.xticks(rotation=45)
plt.legend(title='Ticker', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

## Interpretation of the Closing Price Plot for This Stock

# Overall Trend:

If the stock price gradually increases, it suggests a bullish trend (investors are confident, and the stock is gaining value).

If it declines over time, it indicates a bearish trend (selling pressure, potential negative sentiment).

If the price moves sideways, it means the stock is range-bound, lacking a clear direction.

# Volatility Analysis:

If the line fluctuates drastically, the stock is experiencing high volatility—possibly due to market news or economic conditions.

If the movements are smooth and gradual, the stock is more stable with fewer external shocks.

# Key Observations from Peaks & Dips:

Sharp Spikes : Could be linked to positive news, earnings reports, or institutional buying.

Sudden Drops : Possible negative news, market crashes, or profit-taking by investors.

# Support & Resistance:

If the stock keeps bouncing off a certain price level and rising, it means strong support (buyers enter at that price).

If it struggles to break a certain high and reverses down, that’s a resistance level (sellers taking profit).

# Scenario-Specific Implication:

If the stock shows an upward movement overall, it indicates growth potential, and investors may have confidence in it.
If there are wild swings, the stock is likely being influenced by news, speculation, or external factors.
If it is steadily declining, it may indicate weak fundamentals or investor fear.
# In[15]:


# Plot Trading Volume Over Time
plt.figure(figsize=(14, 7))

sns.lineplot(data=data, x='Date', y='Volume', hue='Ticker', marker='o')

plt.title('Trading Volume Over Time (AAPL, MSFT, NFLX, GOOG)', fontsize=18)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Trading Volume', fontsize=14)
plt.xticks(rotation=45)
plt.legend(title='Ticker', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

## Key Observations from the Plot

Key Observations from the Plot
Trading Volume = The number of shares traded during a specific period (daily in this case). It indicates the liquidity and investor interest.
1. AAPL (Apple) - Highest & Most Active Stock
•	Apple consistently shows the highest trading volume among all four companies.
•	Peaks:
o	Around mid-March 2023 — A noticeable spike in volume.
o	Early May 2023 — Highest trading volume peak (~110 million shares).
Interpretation:

These spikes might be due to:
•	Earnings announcements.
•	New product launches.
•	Market news or stock split rumours.

2. MSFT (Microsoft) - Moderate & Stable Trading Volume
•	MSFT maintains a moderate trading volume throughout the period.
•	Peaks:
o	Mid-March 2023 spike — aligning with AAPL's volume increase.
o	Mid-April 2023 — Possibly due to quarterly results or strategic announcements.
Interpretation:
Microsoft stock trading behaviour is stable but reacts sharply to market news/events.
3. NFLX (Netflix) - Lowest Trading Volume
•	Netflix consistently shows the lowest trading volume among the four.
•	Small spike in mid-March and mid-April 2023.
Interpretation:

Lower trading volume indicates:
•	Fewer active traders.
•	Possibly less liquidity.
•	Investors may be holding positions rather than trading actively.

4. GOOG (Google) - Mid-range Volume with Occasional Spikes
•	Google's trading volume fluctuates between MSFT and NFLX levels.
•	Significant spike in early February 2023 (close to 100 million shares).
•	Another spike aligns with AAPL & MSFT in mid-March.
Interpretation:
Google's spikes could be related to:
•	Regulatory news.
•	AI product announcements.
•	Earnings results.

General Trend Patterns:
Stock	Trading Volume Pattern	Remarks
AAPL	High Volume, Frequent Spikes	Highly traded; responsive to market events.
MSFT	Moderate Volume, Stable Trend	Regular trading; sensitive to news.
NFLX	Low Volume, Occasional Spikes	Less liquid; investor holding behaviour.
GOOG	Medium Volume, Irregular Spikes	Reactive to specific events.# Practical Use of MA in Stock Trading

🔹 Moving averages help filter out noise and identify trends.
🔹 Traders use crossovers as buy/sell signals.
🔹 Long-term investors use MAs to confirm trend strength before making decisions.
# In[16]:


# Compute Moving Averages (7-day and 14-day)
data["MA7"] = data["Close"].rolling(window=7).mean()
data["MA14"] = data["Close"].rolling(window=14).mean()
data.head(20)

Moving Average (MA) smooths out daily price fluctuations to show overall trend direction.

7-day MA (MA7): A short-term trend indicator that reacts quickly to price changes.
14-day MA (MA14): A longer-term trend indicator that reacts more slowly, filtering out minor price fluctuations.

MA Type	Meaning	Use Case
MA7	Average of closing prices over the last 7 days	Captures short-term trend
MA14	Average of closing prices over the last 14 days	Captures medium-term trend

Interpretation:

When the 7-day MA is above the 14-day MA → Short-term bullish trend (upward momentum).

Possible buying opportunity if price confirms upward movement.

When the 7-day MA is below the 14-day MA → Short-term bearish trend (downward momentum).

Could signal a potential sell-off.

# Crossovers Between the Two MAs

If MA7 crosses above MA14 → Bullish signal (uptrend beginning).

If MA7 crosses below MA14 → Bearish signal (downtrend beginning).

The two moving averages may stay close together, indicating sideways movement.

No clear bullish or bearish trend.

# Application to this stock data.

For an uptrend:

MA7 will be above MA14, confirming strong momentum.

Suggests investors are buying and pushing prices higher.

For a Downtrend:

MA7 will fall below MA14, showing weak momentum.

Could indicate investor uncertainty or selling pressure.

# Final Takeaways:

The 7-day MA reacts faster, capturing short-term price swings.
The 14-day MA provides a broader view, filtering out daily volatility.
Crossovers between MA7 and MA14 can signal trend reversals or continuations.
Helps in making data-driven trading decisions rather than reacting to short-term price movements.
# In[17]:


# Plot Closing Price with Moving Averages for all Tickers
plt.figure(figsize=(14, 8))

# Loop through each ticker
for ticker in data['Ticker'].unique():
    temp_data = data[data['Ticker'] == ticker]
    
    # Plot Closing Price
    plt.plot(temp_data['Date'], temp_data['Close'], label=f'{ticker} Closing', alpha=0.7)
    
    # Plot 7-Day Moving Average
    plt.plot(temp_data['Date'], temp_data['MA7'], label=f'{ticker} 7-Day MA', linestyle='--')
    
    # Plot 14-Day Moving Average
    plt.plot(temp_data['Date'], temp_data['MA14'], label=f'{ticker} 14-Day MA', linestyle='--')

plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.title("Stock Price with Moving Averages (AAPL, MSFT, NFLX, GOOG)")
plt.legend(loc='best')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

Components of the Plot:
1. X-axis → Date
•	Time component (daily stock prices)
•	Shows the timeline from start to end of the dataset (2023-02 to 2023-05)
•	Helps us observe how stock prices change over time.
2. Y-axis → Price (USD)
•	Represents the stock price in US dollars.
•	Ranges from 90 USD to 370 USD.
3. Lines plotted → Each line represents:
Line Type	Description	Why is it used?
Solid Line	Closing Price	Real stock price on that date
Dashed Line (--)	7-Day Moving Average (MA7)	To smooth out short-term fluctuations
Dashed Line (--)	14-Day Moving Average (MA14)	To show medium-term trend

Interpretation:
AAPL (Apple) — Light Blue Lines
•	Closing Price fluctuates between 90 to 160 USD.
•	MA7 and MA14 lines are smoother — show trend direction.
•	If Closing Price crosses above MA — Bullish Signal.
•	If Closing Price crosses below MA — Bearish Signal.

MSFT (Microsoft) — Red Lines
•	Price Range: 240 to 290 USD.
•	MA lines closely follow the closing price, showing a steady upward trend.
•	MA7 reacts faster to price changes than MA14.

NFLX (Netflix) — Pink Lines
•	Price Range: 300 to 370 USD.
•	Fluctuations are higher.
•	Trend shows moderate volatility.
•	MA7 and MA14 lines reduce noise from price spikes.

GOOG (Google) — Cyan Lines
•	Price Range: 90 to 150 USD.
•	Stable upward trend.
•	MA7 and MA14 indicate a strong bullish trend from March onwards.

Final Insights from this Plot:
Observation	Implication
MA lines smoothen the data	Help in identifying trend direction
If MA7 > MA14	Indicates short-term bullish trend
If MA7 < MA14	Indicates short-term bearish trend
The crossing of lines	Signal for buy/sell decision (used in technical analysis)

Key Observations:

Data Issues – The blue line has extreme price fluctuations that are inconsistent with a natural stock price movement. This suggests possible data errors or outliers.
Moving Averages Provide Stability – Despite erratic closing prices, the 7-day and 14-day MAs smooth out trends, making it easier to understand stock movements.
Short-Term vs Long-Term Trends – If the 7-day MA crosses above the 14-day MA, it could signal bullish momentum (buy signal). If it crosses below, it could indicate a bearish trend (sell signal).
# In[18]:


# Calculate Correlation between Closing Price and Volume
correlation = data[["Close", "Volume"]].corr()
display(correlation)

This table shows the correlation between Closing Price and Trading Volume in your stock data.

Key Observations:

Correlation between Closing Price (Close) vs. Volume: -0.544

This indicates a moderate negative correlation between stock price and trading volume.

As volume increases, the stock price tends to decrease (and vice versa).

A negative correlation of -0.544 suggests that when trading activity is high, stock prices may experience a decline.

What This Means in the Market Context:

# High Volume, Lower Prices:

Large selling pressure (panic selling) can push prices down.

Institutional investors may offload large positions, increasing volume but driving prices lower.

# Low Volume, Higher Prices:

When volume is low, there may be fewer sellers, allowing prices to rise due to supply-demand dynamics.

Trading Insights:
🔹 If high volume consistently results in lower prices, this could indicate bearish trends or market uncertainty.
🔹 If price rises with low volume, it may indicate weak buying interest, making the trend unsustainable.
🔹 Traders can use volume-price analysis to spot breakouts, trend reversals, or market sentiment shifts.
# In[19]:


# Correlation heatmap:

import seaborn as sns
import matplotlib.pyplot as plt

# Compute the correlation matrix
data_numeric = data.select_dtypes(include=["number"])  # Keep only numerical columns
correlation_matrix = data_numeric.corr()

# Plot the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, vmin=-1, vmax=1)
plt.title("Correlation Heatmap of Stock Data")
plt.show()

This correlation heatmap visually represents the relationship between different stock-related variables.
Correlation values range from -1 to 1 (inclusively):

1 (or close to 1): Strong positive correlation (variables move together).

-1 (or close to -1): Strong negative correlation (variables move in opposite directions).

0: No correlation (variables are independent).

Key Observations from the Heatmap:


1. Strong Positive Correlations (Close to 1, Dark Red Areas)
Open, High, Low, Close, and Adjusted Close (Adj Close) are nearly 1.00, meaning they move almost identically.

Moving Averages (MA7 and MA14) are highly correlated (~0.98), meaning short-term and long-term trends are similar.

Open, High, Low, and Close vs. MA7 and MA14 (~0.93 - 0.97): Stock prices and moving averages follow similar trends.

Implication: Stock prices are strongly related, so past prices can predict short-term trends.


2. Negative Correlation Between Price and Volume (-0.54 to -0.55, Blue Areas)
Closing Price vs. Volume (-0.54): Higher stock prices tend to be associated with lower trading volumes.

Other price-related metrics (Open, High, Low, Adj Close) also have a similar negative correlation with volume (-0.54 to -0.55).

Implication: When the stock price rises, trading volume tends to decrease. This may indicate that investors hold onto stocks when prices are high and trade more when prices are low.


3. Volume vs. Moving Averages (~-0.53 to -0.51)
Slight negative correlation (-0.51 to -0.53): As stock prices stabilize over time, volume tends to drop.

Implication: Less volatility over time may reduce trading activity.

Summary of Insights:

Stock price metrics (Open, High, Low, Close) are nearly identical, so focusing on Close Price is enough for trend analysis.
Stock price and volume have a moderate negative correlation (-0.54), meaning higher prices generally lead to lower trading activity.
Moving Averages (MA7 & MA14) are strongly correlated (~0.98), confirming that price trends are stable in short- and long-term movements.
Volume has a weaker correlation with moving averages, suggesting trading activity does not always align with price trends.
# In[20]:


# Calculate Daily Returns for each Ticker
data["Daily_Return"] = data.groupby("Ticker")["Close"].pct_change()

# Plot Daily Return Distribution for each Ticker
tickers = ['AAPL', 'MSFT', 'NFLX', 'GOOG']

plt.figure(figsize=(14, 10))

for ticker in tickers:
    plt.subplot(2, 2, tickers.index(ticker)+1)  # 2x2 grid
    sns.histplot(
        data[data["Ticker"] == ticker]["Daily_Return"].dropna(), 
        bins=50, kde=True
    )
    plt.xlabel("Daily Return")
    plt.ylabel("Frequency")
    plt.title(f"Daily Return Distribution - {ticker}")

plt.tight_layout()
plt.show()

Interpretation:
Term	Meaning	Impact
Daily Return	% gain or loss per day	Higher return → Higher reward potential
Return Distribution	Shape & spread of returns	Tells us about Risk & Volatility
Volatility	Spread of returns	Higher volatility → Higher Risk
A well-behaved stock has a "Normal Distribution" of returns — narrow and cantered near zero.
Extreme "tails" or "flat curves" indicate riskier behaviour.
1. AAPL — Daily Return Distribution
•	Range of Daily Returns: –3% to 4.5% (approximately)
•	Shape: Symmetrical, close to bell-shaped (normal distribution)
•	Interpretation:
o	Most daily returns are between –1% and 2%.
o	Few extreme returns (left tail & right tail).
•	Conclusion:
o	AAPL stock shows moderate risk.
o	Stable stock with balanced volatility.
o	Suitable for risk-averse investors.
2. MSFT — Daily Return Distribution
•	Range of Daily Returns: –2.5% to +6.5% (approximately)
•	Shape: Right-skewed (slight positive bias)
•	Interpretation:
o	Most returns are between –1% and +2%.
o	Few days with higher positive returns (6%).
•	Conclusion:
o	MSFT is slightly more aggressive than AAPL.
o	Stable stock with occasional higher upside.
o	Suitable for medium-risk investors.
3. NFLX — Daily Return Distribution
•	Range of Daily Returns: –4% to +8.5% (approximately)
•	Shape: Wider spread — indicates higher volatility.
•	Interpretation:
o	Most returns are between –2% and +2%.
o	Occasional large positive returns (8%).
•	Conclusion:
o	NFLX is highly volatile.
o	Suitable for aggressive investors or short-term traders.
o	Potential for high rewards but carries higher risk.

4. GOOG — Daily Return Distribution
•	Range of Daily Returns: -7% to +4.5% (approximately)
•	Shape: Left-skewed (more extreme negative returns)
•	Interpretation:
o	Most returns between –2% and +2%.
o	Large left tail indicates higher loss events.
•	Conclusion:
o	GOOG has higher downside risk compared to others.
o	Suitable for well-diversified portfolios.
o	Risk-averse investors may avoid pure GOOG exposure.
Final Insights Summary:
Stock	Return Distribution	Risk Level	Investor Suitability
AAPL	Normal, Stable	Low	Conservative, Long-Term
MSFT	Slightly Aggressive	Low-Moderate	Balanced Portfolio
NFLX	Wide Spread, Volatile	High	Aggressive, Traders
GOOG	Downside Risk Visible	Moderate-High	Diversified Portfolio Required
# In[21]:


plt.figure(figsize=(14, 7))

for ticker in tickers:
    df_ticker = data[data['Ticker'] == ticker].copy()
    df_ticker['Daily_Return'] = df_ticker['Close'].pct_change()
    df_ticker['Volatility'] = df_ticker['Daily_Return'].rolling(window=14).std()
    
    plt.plot(df_ticker['Date'], df_ticker['Volatility'], label=f"{ticker} 14-Day Volatility")

plt.xlabel('Date')
plt.ylabel('Volatility')
plt.title('Stock Volatility Comparison (14-Day Rolling)')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.show()

Explanation of the Stock Volatility Over Time Chart:

Volatility measures how much the stock price fluctuates over time.

Higher volatility = Higher Risk/Uncertainty
Lower volatility = Stability/Safe Investment
In this case:
•	Volatility is calculated using the standard deviation of daily returns over a 14-day window.

Interpretation of the Graph:
X-axis:
•	Time Period: From March 2023 to May 2023
Y-axis:
•	Volatility Values (Standard Deviation of Daily Returns)
Stock-wise Insights:
1. AAPL (Blue Line)
•	Observation: Lowest Volatility among all 4 stocks consistently.
•	Range: Between 0.010 and 0.017
•	Inference:
o	AAPL shows stable price behaviour.
o	Suitable for risk-averse investors.
o	Strong brand, market dominance, less sensitive to market shocks.
2. MSFT (Orange Line)
•	Observation: Moderate volatility.
•	Sudden Spike: Around end of April 2023 — possibly due to earnings report or product announcement.
•	Range: 0.013 to 0.022
•	Inference:
o	Stable growth with occasional market reactions.
o	Balanced risk-reward stock.
3. NFLX (Green Line)
•	Observation: Highest Volatility consistently.
•	Range: Spikes up to 0.032 (Very High)
•	Inference:
o	Streaming industry is highly competitive.
o	NFLX stock reacts aggressively to market news — earnings, subscriber numbers, competition.
o	Suitable for high-risk traders, not for conservative investors.
4. GOOG (Red Line)
•	Observation: Moderate volatility, relatively stable after April.
•	Range: 0.014 to 0.024
•	Inference:
o	Strong fundamentals but reacts moderately to market events.
o	Safe for medium-risk portfolios.
Key Takeaways for Portfolio Strategy:
Stock	Volatility	Investment Suitability
AAPL	Lowest	Safe Haven / Defensive Stock
MSFT	Moderate	Balanced Portfolio Holding
NFLX	Highest	High-Risk / Short-Term Trading
GOOG	Moderate	Long-Term Growth / Medium-Risk Holding

Final Recommendations:
For Risk-Averse Investors:
•	Allocate higher weight to AAPL & MSFT.
For Aggressive Traders:
•	NFLX offers potential for higher returns but comes with risk.
For Balanced Portfolio:
•	Diversify between AAPL, MSFT, and GOOG.
•	Minimize exposure to NFLX unless willing to tolerate risk.
# ### Machine learinig Implimentation:

# In[22]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV file
df = pd.read_csv(r'D:\Work\Unified Mentor\Data\stocks.csv')

# Convert Date column to datetime
df["Date"] = pd.to_datetime(df["Date"])

# Display first few rows
df.head()


# In[23]:


# Outlier detection: (Box plot approach)

# Interpretation: Dots outside the box represent potential outliers. If many extreme values exist, outlier removal might be needed.


# In[24]:


plt.figure(figsize=(8,5))
sns.boxplot(x=df["Close"])
plt.title("Box Plot of Closing Price")
plt.show()


# In[25]:


# Using Interquartile Range to find outliers mathematically.


# In[26]:


# Compute Q1, Q3, and IQR
Q1 = df["Close"].quantile(0.25)  # 25th percentile
Q3 = df["Close"].quantile(0.75)  # 75th percentile
IQR = Q3 - Q1  # Interquartile range

# Define the outlier boundaries
lower_bound = Q1 - 1.5 * IQR  
upper_bound = Q3 + 1.5 * IQR  

# Detect outliers
outliers = df[(df["Close"] < lower_bound) | (df["Close"] > upper_bound)]
print(f"Number of outliers detected: {len(outliers)}")

# Display outliers
outliers


# In[27]:


# Cap Outliers Instead of Removing (This method preserves all data points but limits extreme values.)


# In[28]:


df["Close"] = df["Close"].clip(lower_bound, upper_bound)


# In[29]:


# Log Transformation (If the data is skewed, this reduces the impact of extreme values.)


# In[30]:


import numpy as np

df["Close_Log"] = np.log(df["Close"])

# Replot Boxplot to Check Transformation Effect
plt.figure(figsize=(8,5))
sns.boxplot(x=df["Close_Log"])
plt.title("Box Plot of Log-Transformed Closing Price")
plt.show()


# In[31]:


# Using z score method. Z > 3 or Z < -3 means a value is an outlier.


# In[32]:


from scipy import stats

data["z_score"] = stats.zscore(data["Close"])
data_no_outliers = data[data["z_score"].abs() < 3]  # Keeping values within 3 standard deviations

print(f"Data points after Z-score filtering: {len(data_no_outliers)}")


# In[33]:


# All of the methods indicate that there is no outliers.

High Volatility Does Not Always Mean Outliers.

Stock prices naturally fluctuate, especially in highly volatile stocks.

If the price jumps and drops are frequent but still within a reasonable range, they won’t be classified as outliers.

Outlier detection methods (e.g., IQR, Z-score) only flag values that are extremely far from the norm—if all fluctuations fall within the expected range, they won’t be labeled as outliers.

# Data Errors vs. Statistical Outliers:

Extreme price movements could be due to data inconsistencies, missing values, or incorrect entries, which are not always outliers.

If incorrect values were smoothed during preprocessing (e.g., missing values filled with the mean), this could remove extreme points before they could be detected as outliers.

# Moving Averages Can Amplify Perceived Volatility:

A moving average lags behind actual price movements, so if the stock price is fluctuating rapidly, the difference between the moving average and the real price might look extreme.

# In[34]:


## Machine Learning Model (After Handling Outliers):


# In[35]:


# Select Features (X) and Target (y)
features = ["Open", "High", "Low", "Volume"]
target = "Close"

X = data_no_outliers[features]
y = data_no_outliers[target]

# Check for missing values again
print(X.isnull().sum())
print(y.isnull().sum())


# In[36]:


#  Split Data into Training and Testing Sets:


# In[37]:


from sklearn.model_selection import train_test_split

# Split Data (80% Training, 20% Testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training Set: {X_train.shape}, Testing Set: {X_test.shape}")


# In[38]:


# Scale the Data: (Normalizes data for better ML performance)


# In[39]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[40]:


# Train a Machine Learning Model (Random Forest)


# In[41]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Train Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Predictions
y_pred = model.predict(X_test_scaled)

# Evaluate Model Performance
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"R² Score: {r2}")

Mean Absolute Error (MAE) measures the average absolute difference between the actual stock prices and the predicted stock prices.

Lower is better. A value of 1.586 means that, on average, your model’s predictions are off by $1.59 from the actual stock price.

Interpretation:
    
Since stock prices usually range in hundreds of dollars, an error of $1.59 is very low.
This indicates that the model is making very precise predictions with minimal deviation.Mean Squared Error calculates the average of the squared differences between actual and predicted values. Squaring emphasizes larger errors more than smaller ones.

Lower is better. A value of 4.504 suggests that the model’s larger errors are still very low.

Interpretation:
    
Since squaring exaggerates large errors, a low MSE means that the model rarely makes large prediction mistakes.
A small MSE confirms that the model's predictions are tightly clustered around actual prices.R² Score (Coefficient of Determination) tells us how well the model explains variance in stock prices.

Higher is better. Your model’s R² = 0.9994 is extremely close to 1, meaning it explains 99.94% of the variation in stock prices.

Interpretation:
    
The model has an exceptionally high predictive power.
99.94% of stock price fluctuations are captured by the model.
There is only 0.06% unexplained variance, meaning almost all factors influencing stock price movements are accounted for.
# In[42]:


# Compute Adjusted R²
n = X.shape[0]  # Number of observations
p = X.shape[1]  # Number of predictors (features)

adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))

print(f"R² Score: {r2}")
print(f"Adjusted R² Score: {adjusted_r2}")

Adjusted R² is very close to R², meaning that all the features used are contributing meaningfully to the prediction.

Since Adjusted R² penalizes unnecessary variables, the minimal difference suggests that none of the features are redundant or negatively impacting the model.

Excellent Model Performance – Your model explains almost all variability in stock prices.
No Overfitting Issues Detected from R² – The Adjusted R² is still very high, meaning the model generalizes well.
Strong Feature Selection – The chosen variables (Open, High, Low, Volume, etc.) are all useful for predicting the Closing Price.
# In[43]:


# Cross-Validation (K-Fold) for Model Robustness:


# In[44]:


from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
import numpy as np

# Initialize Linear Regression model
model = LinearRegression()

# Perform 10-Fold Cross-Validation
cv_scores = cross_val_score(model, X, y, cv=10, scoring="r2")

# Print results
print(f"Cross-Validation R² Scores: {cv_scores}")
print(f"Mean R² Score: {np.mean(cv_scores)}")
print(f"Standard Deviation of R²: {np.std(cv_scores)}")

Cross-Validation R² Scores Analysis:
    
A 10-Fold Cross-Validation is performed, which tests model’s ability to generalize.

The R² scores range from 0.99927 to 0.99999, meaning the model explains almost all the variance across different folds.

Most scores are above 0.9999, showing a consistent, strong predictive ability across different subsets of the data.

Mean R² Score = 0.99987 (~99.99%)
This is very close to the original R² score (~0.999998), meaning your model does not overfit significantly and generalizes well.

A mean R² above 0.9998 indicates that your model is highly reliable.

Standard Deviation = 0.00022
Very low standard deviation means the model's performance is consistent across different test sets.

If the standard deviation was high (e.g., >0.01), it would suggest instability, but in this case, the model is stable.

Conclusion:
    
Model performs consistently well across different data splits.
No signs of significant overfitting.
R² and Adjusted R² are both high, confirming that all features contribute meaningfully. 🚀Final Conclusion:

The model is highly accurate:
    
Very low errors (MAE & MSE).
Extremely high predictive power (R² = 0.9994).
Reliable for stock price forecasting.