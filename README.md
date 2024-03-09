# Stock Predictor Interactive Dashboard

This application provides an interactive dashboard to predict US stock prices primarily based on price changes from other markets that open before the US market. It leverages historical data to guide users on taking on long/short positions.

US stocks chosen are Apple, Alibaba, Nvdia, Taiwan Semiconductor, Advanced Micro Devices, Intel and Tesla.

Markets used for prediction are S&P500, NASDAQ, DJI, CAC40, Dax Performance Index, All Ordinaries, Hang Seng Index and Nikkei 225.

## Tech Stack

Python and Streamlit

## Features

- **Predictive Modeling:** Utilize international market shifts to predict change in US stock prices.
- **Interactive Dashboard:** Analyze and visualize data in real-time using an interactive interface.
- **Decision-Making Criteria:** Offers long/short guidance based on predicted gains and moving average indicators (SMA10 vs. SMA50 and EMA10 vs. EMA50).
- **Model Evaluation:** Leverage Sharpe Ratio and Maximum Drawdown to evaluate model.

## Challenges and Future Directions

### Model Accuracy

- The current model exhibits a high RMSE, indicating a need for predictive accuracy improvements.

### Consistency Issues

- Adjusted R^2 values differ between the Python file execution and Streamlit display, requiring further investigation.

### Overfitting

- Evidence suggests potential overfitting, as indicated by comparative analysis of Adjusted R^2 and RMSE between train and test set.

### Data Sources

- Data is sourced from Yahoo Finance; exploring additional or alternative data sources like Alpha Vantage could be beneficial as data provided by Yahoo Finance is very limited.

### Algorithm Exploration

- Investigating various machine learning algorithms, including XGBoost, Logistic Regression, and LSTM, may enhance model predictive capabilities.
