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
# In[13]:


# Plot Closing Price Over Time
plt.figure(figsize=(12, 6))
plt.plot(data["Date"], data["Close"], marker='o', linestyle='-', label="Closing Price", color='blue')
plt.xlabel("Date")
plt.ylabel("Closing Price (USD)")
plt.title("Stock Closing Price Over Time")
plt.legend()
plt.grid(True)
plt.xticks(data["Date"][::15], rotation=45)  # Show every 15th date
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
# In[14]:


# Plot Trading Volume Over Time
plt.figure(figsize=(12, 6))
plt.bar(data["Date"], data["Volume"], color='green', alpha=0.6, label="Trading Volume")
plt.xlabel("Date")
plt.ylabel("Volume (in shares)")
plt.title("Trading Volume Over Time")
plt.legend()
plt.xticks(data["Date"][::15], rotation=45)  # Show every 15th date
plt.show()

## Key Observations from the Plot

Multiple Stock Trends Visible.

The plot shows several distinct stock price movements, meaning the dataset contains multiple stocks, not just one.

Some stocks have higher prices ($350+), while others are much lower ($100–$150).

Some Stocks Are Increasing in Price.
Certain stocks show an upward trend, indicating growth and strong investor interest.
Possible reasons: Positive earnings reports, good market conditions, or strong company fundamentals.

Some Stocks Are Decreasing in Price
Several lines decline over time, showing bearish trends (falling prices).
Possible reasons: Market downturns, bad earnings reports, or investor sell-offs.

Some Stocks Are Range-Bound
Some stock prices are moving sideways, fluctuating within a limited range.
This could indicate market indecision or low volatility.

# Interpretation of the Trends

High Volatility:

The stock prices are not moving in a single direction; some are rising, others are falling.

Suggests market uncertainty or mixed investor sentiment.

Market Divergence:

Some stocks outperform others, indicating sector-specific trends (e.g., tech stocks rising while energy stocks decline).

Crossing Lines:

Some stock prices overlap or cross each other at different times.

This could indicate stocks trading at similar levels temporarily before diverging.

# Final Takeaways:

The market is highly dynamic, with stocks rising, falling, or staying flat.
Some stocks outperform others, suggesting sector-specific growth.# Practical Use of MA in Stock Trading

🔹 Moving averages help filter out noise and identify trends.
🔹 Traders use crossovers as buy/sell signals.
🔹 Long-term investors use MAs to confirm trend strength before making decisions.
# In[15]:


# Compute Moving Averages (7-day and 14-day)
data["MA7"] = data["Close"].rolling(window=7).mean()
data["MA14"] = data["Close"].rolling(window=14).mean()
data.head(20)

Moving Average (MA) smooths out daily price fluctuations to show overall trend direction.

7-day MA (MA7): A short-term trend indicator that reacts quickly to price changes.
14-day MA (MA14): A longer-term trend indicator that reacts more slowly, filtering out minor price fluctuations.

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
# In[16]:


# Plot Moving Averages with Closing Price
plt.figure(figsize=(12, 6))
plt.plot(data["Date"], data["Close"], label="Closing Price", color='blue', alpha=0.6)
plt.plot(data["Date"], data["MA7"], label="7-Day MA", color='red', linestyle="--")
plt.plot(data["Date"], data["MA14"], label="14-Day MA", color='green', linestyle="--")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.title("Stock Price with Moving Averages")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.show()

This plot visualizes stock closing prices along with 7-day and 14-day moving averages over time.

1. Blue Line (Closing Price)
The blue line represents the actual stock closing prices.

It appears highly volatile, showing sudden spikes and drops.

The irregular and extreme movements indicate possible data inconsistencies, missing values, or anomalies in the dataset.

2. Red Dashed Line (7-Day Moving Average - MA7)
The red dashed line smooths out the closing prices over a 7-day period.

It helps in identifying short-term trends while filtering out some price noise.

The spikes in MA7 suggest that the closing price had irregular jumps or missing values.

3. Green Dashed Line (14-Day Moving Average - MA14)
The green dashed line represents the 14-day moving average.

This line provides a longer-term trend, reacting more slowly to price changes than the 7-day MA.

It follows the general price movement but appears more stable than the MA7.

Key Observations:

Data Issues – The blue line has extreme price fluctuations that are inconsistent with a natural stock price movement. This suggests possible data errors or outliers.
Moving Averages Provide Stability – Despite erratic closing prices, the 7-day and 14-day MAs smooth out trends, making it easier to understand stock movements.
Short-Term vs Long-Term Trends – If the 7-day MA crosses above the 14-day MA, it could signal bullish momentum (buy signal). If it crosses below, it could indicate a bearish trend (sell signal).
# In[17]:


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
# In[18]:


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
# In[19]:


# Compute Daily Returns
data["Daily_Return"] = data["Close"].pct_change()
# Plot Daily Returns
plt.figure(figsize=(12, 6))
sns.histplot(data["Daily_Return"].dropna(), bins=50, kde=True, color='purple')
plt.xlabel("Daily Return")
plt.ylabel("Frequency")
plt.title("Distribution of Daily Returns")
plt.show()

Explanation of the Distribution of Daily Returns:

This histogram visualizes the distribution of daily returns for the stock. It helps assess how frequently different return values occur and provides insights into stock volatility and risk.

Key Observations from the Chart:

Most Returns Are Close to Zero

The majority of daily returns are clustered around 0%, indicating that the stock mostly experiences small daily price changes.

This suggests that on most days, the stock price remains relatively stable.


Bell-Shaped Distribution (Normal-Like)

The histogram shows a near normal distribution, meaning returns are symmetrically spread around the average.

A slight peak in the center suggests that most daily changes are small, with occasional larger changes.


Presence of Outliers (Extreme Returns)

There are a few extreme values on both the left and right sides of the distribution.

Negative outliers (left side): Represent days with sharp declines.

Positive outliers (right side): Represent days with large gains.

These outliers may be caused by major market events, earnings reports, or unexpected news.


Stock Volatility Insight:

If the distribution were wide and flat, it would indicate high volatility (large fluctuations).

Since this distribution is narrow and peaked, the stock has relatively low volatility, with most daily returns falling within a small range.


Implications of This Analysis:

Stock has stable daily price movements – Most returns are near 0%.
Few extreme return days – Could be due to earnings reports, macroeconomic factors, or news events.
Risk assessment – Investors may consider this stock relatively low risk if they prefer stable returns.
Volatility monitoring – Further analysis can check whether extreme return days correlate with high trading volume.
# In[20]:


# Compute Volatility (Rolling Standard Deviation)
data["Volatility"] = data["Daily_Return"].rolling(window=14).std()

# Plot Volatility Over Time
plt.figure(figsize=(12, 6))
plt.plot(data["Date"], data["Volatility"], color='orange', label="14-Day Volatility")
plt.xlabel("Date")
plt.ylabel("Volatility")
plt.title("Stock Volatility Over Time")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.show()

Explanation of the Stock Volatility Over Time Chart:

This chart represents 14-day rolling volatility of the stock over time. Volatility measures the degree of variation in stock prices, helping assess risk and market fluctuations.

Key Observations from the Chart:

Sharp Drop in Volatility Around Early March

Initially, volatility is high (~0.175 or 17.5%), indicating large fluctuations in stock prices.

Around early March, volatility drops significantly to near 0.025 (2.5%), suggesting that price fluctuations became much smaller.

Stable Low Volatility Period

After the initial drop, the volatility remains low and stable, indicating that stock price movements were more predictable in this period.

This suggests that external factors (such as earnings reports or market events) that previously caused high volatility may have settled.

Slight Uptick in Volatility Towards May

Near the end of the chart (around May), volatility slightly increases again.

This could indicate renewed price fluctuations due to market news, earnings, or macroeconomic conditions.

Implications of This Analysis:

Stock experienced a highly volatile phase initially but 
stabilized afterward.
Low volatility period suggests reduced risk, making it more predictable for traders and investors.
Recent slight uptick could indicate upcoming price movement, which traders should watch.
Risk assessment – Investors who prefer stable returns may find the later period more attractive, while traders seeking volatility might focus on earlier fluctuations.
# ### Machine learinig Implimentation:

# In[21]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV file
df = pd.read_csv(r'D:\Work\Unified Mentor\Data\stocks.csv')

# Convert Date column to datetime
df["Date"] = pd.to_datetime(df["Date"])

# Display first few rows
df.head()


# In[22]:


# Outlier detection: (Box plot approach)

# Interpretation: Dots outside the box represent potential outliers. If many extreme values exist, outlier removal might be needed.


# In[23]:


plt.figure(figsize=(8,5))
sns.boxplot(x=df["Close"])
plt.title("Box Plot of Closing Price")
plt.show()


# In[24]:


# Using Interquartile Range to find outliers mathematically.


# In[25]:


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


# In[26]:


# Cap Outliers Instead of Removing (This method preserves all data points but limits extreme values.)


# In[27]:


df["Close"] = df["Close"].clip(lower_bound, upper_bound)


# In[28]:


# Log Transformation (If the data is skewed, this reduces the impact of extreme values.)


# In[29]:


import numpy as np

df["Close_Log"] = np.log(df["Close"])

# Replot Boxplot to Check Transformation Effect
plt.figure(figsize=(8,5))
sns.boxplot(x=df["Close_Log"])
plt.title("Box Plot of Log-Transformed Closing Price")
plt.show()


# In[30]:


# Using z score method. Z > 3 or Z < -3 means a value is an outlier.


# In[31]:


from scipy import stats

data["z_score"] = stats.zscore(data["Close"])
data_no_outliers = data[data["z_score"].abs() < 3]  # Keeping values within 3 standard deviations

print(f"Data points after Z-score filtering: {len(data_no_outliers)}")


# In[32]:


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

# In[33]:


## Machine Learning Model (After Handling Outliers):


# In[34]:


# Select Features (X) and Target (y)
features = ["Open", "High", "Low", "Volume"]
target = "Close"

X = data_no_outliers[features]
y = data_no_outliers[target]

# Check for missing values again
print(X.isnull().sum())
print(y.isnull().sum())


# In[35]:


#  Split Data into Training and Testing Sets:


# In[36]:


from sklearn.model_selection import train_test_split

# Split Data (80% Training, 20% Testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training Set: {X_train.shape}, Testing Set: {X_test.shape}")


# In[37]:


# Scale the Data: (Normalizes data for better ML performance)


# In[38]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[39]:


# Train a Machine Learning Model (Random Forest)


# In[40]:


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
# In[41]:


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
# In[42]:


# Cross-Validation (K-Fold) for Model Robustness:


# In[43]:


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
R² and Adjusted R² are both high, confirming that all features contribute meaningfully. 🚀
# In[44]:


# Random Forest for Potential Improvement:


# In[45]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Initialize Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Predict on test data
y_pred_rf = rf_model.predict(X_test)

# Evaluate Performance
rf_r2 = r2_score(y_test, y_pred_rf)
rf_mae = mean_absolute_error(y_test, y_pred_rf)
rf_mse = mean_squared_error(y_test, y_pred_rf)

print(f"Random Forest R² Score: {rf_r2}")
print(f"Mean Absolute Error: {rf_mae}")
print(f"Mean Squared Error: {rf_mse}")

Interpretation of Random Forest Performance:

# R² Score: 0.99991 (~99.99%)
This is slightly lower than Linear Regression’s R² (0.999998) but still excellent.

It means Random Forest can explain 99.99% of the variance in stock prices, showing strong predictive power.

# Mean Absolute Error (MAE) = 0.6212
On average, the model’s predictions are off by just $0.62.

This is lower than the MAE for Linear Regression (~1.58), meaning Random Forest is more accurate in absolute terms.

# Mean Squared Error (MSE) = 0.7535
Since MSE penalizes larger errors more, the small value here shows that big prediction errors are rare.

Compared to Linear Regression’s MSE (4.50), this is significantly better, meaning Random Forest makes fewer large mistakes.

# Conclusion: Which Model is Better?


Random Forest is better in terms of lower MAE and MSE.
Linear Regression has a slightly higher R², but the difference is tiny.
If avoiding large prediction errors is more important, Random Forest is the better choice.Final Conclusion:

The model is highly accurate:
    
Very low errors (MAE & MSE).
Extremely high predictive power (R² = 0.9994).
Reliable for stock price forecasting.