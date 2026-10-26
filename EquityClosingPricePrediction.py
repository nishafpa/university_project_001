# Import the Libraries
import streamlit as st
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore") # Suppress Python warnings
warnings.filterwarnings("ignore", category=FutureWarning) # Suppress the specific FutureWarning from pandas/yfinance multi-index behavior
warnings.filterwarnings("ignore", category=UserWarning) # Suppress the specific FutureWarning from pandas/yfinance multi-index behavior

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # 0 = all, 1 = no info, 2 = no warning, 3 = no error/warning
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import yfinance as yf
from fredapi import Fred

from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import pmdarima as pm
from statsmodels.tsa.statespace.sarimax import SARIMAX

import tensorflow as tf
import keras_tuner as kt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

import random
os.environ['PYTHONHASHSEED'] = '0' # Set environment variables for determinism
random.seed(42) # Set seed for Python
np.random.seed(42) # Set seed for NumPy
tf.random.set_seed(42) # Set seed for TensorFlow

# Constants
# Please note that you need to sign up for a free FRED account and obtain an API key from the FRED website (https://fred.stlouisfed.org/) to access the data
# You can replace your FRED API key in the code below OR use my provided API key for demonstration purpose
FRED_API_KEY = "c0f88df6d55d87792b278b64733d9b24"

# Header Display
def show_header():
    # Page Config & Title
    st.set_page_config(page_title="US Equity Closing Price Prediction", layout="wide")
    st.title("UC Berkely Professional Certificate for ML & AI - Capstone Project: US Equity Closing Price Prediction")

    st.subheader("Description")
    st.write(f"This project is to predict the Equity closing price of the US stock market listed equities based on the historical data such as Endogenous and Exogenous variables (Features).")
    st.write(f"Endogenous variables of the equity closing price prediction are Open, Low, High, Close/Adjusted Close and Volume from Yahoo Finance, and calculated values for Simple Moving Average (SMA), Exponential Moving Average (EMA), Moving Average Convergence Divergence (MACD) and Relative Strength Index (RSI).")
    st.write(f"Exogenous variables of the equity closing price prediction are Benchmark Indices (S&P500 Index, NASDAQ Composite Index) from Yahoo Finance and Macroeconomic indicators from Federal Reserve Economic Data (FRED) database API (Interest Rates & Monetary Policy, Yield Curve & Bond Market Signals, Credit Spreads & Financial Stress, Liquidity & Money Supply and Inflation & Macro Growth).")
    st.write(f"The classical Time Series Analysis & Forecasting and LSTM (Long Short-Term Memory) in Recurrent Neural Networks (RNN) are used to predict the US Equity price. These models are compared against the Baseline Model Linear Regression. These are Supervised and Regression problem solving models.")

    st.divider() # The separator
    st.info("💡 Please Enter a US Equity Ticker Symbol in the sidebar to proceed!")

# Footer Display
def show_footer():
    st.divider() # The separator
    st.subheader("Brief Report to the Trader or Equity Fund Manager")
    st.write(f"When a Trader or Equity Fund Manager wants to invest in a particular Equity or the Equity Fund on behalf of their firm or their clients, the Prediction of the future closing price would help to make their decision. Without a Quantitative analysis, the decision would be based on guess work and it can potentially end up in wrong decision and can potentially lose the money. Quantitative analysis would help to decrease or minimize the money loss for the firm or their clients. But, be careful that even the best models cannot predict 100% accuracy, but can minimize the loss.")
    st.write(f"Next day's predicted closing price is provided in the report to get the guideline of the investment decision, for the ticker inputted by the client.")

    st.subheader("Next Steps and Recommendations")
    st.write(f"It is important to actively monitor already trained models if its prediction quality is improved or deteriorated over a period of time, due to the underlying dataset's statistical properties change. If deteriorated, care must be taken to further hypertune the existing models or even find out other better performing models. Also, some of the highly influential input features such as real-time financial market news, company fundamentals (valuation ratios, profitability, growth etc.), investor sentiments etc. are not considered in the current deployed models, which are important while predicting the future closing price of the equity and must be considered.")

    st.divider() # The separator
    st.markdown(f"Author: Nishaf P Abdul Majeed (Senior Data Architect/Modeler) | [nishafpa@gmail.com](mailto:nishafpa@gmail.com) | [LinkedIn](https://www.linkedin.com/in/nishaf-abdul-majeed-04b13ba3/)")
    st.write(f"Acknowledgements: UC Berkely Tutors for this Professional Course in Machine Learning and Artificial Intelligence!")
    st.write(f"Copyright (c) 2026 Nishaf P Abdul Majeed. All rights reserved.")
    st.divider() # The separator

# Validate the ticker symbol using yfinance
def ticker_validation(ticker: str) -> dict:
    try:
        if ticker == "":
            return {"status": False, "message": "ERROR: Empty US Equity Ticker Symbol entered!"}
        elif 'regularMarketPrice' in yf.Ticker(ticker).info:
            return {"status": True, "message": ""}
        else:
            return {"status": False, "message": f"ERROR: Invalid US Equity Ticker Symbol {ticker} entered or Yahoo Finance does not return the data requested!"}
    except Exception as e:
        return {"status": False, "message": f"ERROR: Invalid US Equity Ticker Symbol {ticker} entered or Yahoo Finance does not return the data requested!"}

def download_historical_equity_data(ticker: str) -> dict:
    try:
        # Download historical data for a single Equity from Yahoo Finance
        # The Adjusted Closing Price is generally more important for long term stock prediction, since it accounts for Corporate Actions like Stock Splits, Dividends, and Rights Offerings
        # auto_adjust=True will return only Adjusted Close Price, with the column name 'Close'
        # period="max" will download the maximum available data, interval="1d" will download daily data
        df = yf.download(ticker, period="max", interval="1d", auto_adjust=True)

        # Check if the dataframe is empty (i.e., no data was returned)
        if len(df) > 0:
            # Clean multi-level column index if present in newer yfinance versions
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            # Rename the columns to lowercase and prefix with 'equity_'
            df.columns = ['equity_'+col.lower() for col in df.columns]
            # Create all Calendar Days and forward-fill values
            temp_calendar_days_df = pd.date_range(start=df.index.min(), end=df.index.max(), freq="D")
            df = df.reindex(temp_calendar_days_df).ffill()
            # Drop the intermediate variables to save memory
            del temp_calendar_days_df

            return {"data": df, "status": True, "message": ""}
        else:
            return {"data": pd.DataFrame(), "status": False, "message": f"ERROR: Yahoo Finance does not return the data requested for {ticker}!"}
    except Exception as e:
        return {"data": pd.DataFrame(), "status": False, "message": f"ERROR: Yahoo Finance does not return the data requested for {ticker}!"}

def calculate_technical_indicators(df: pd.DataFrame) -> dict:
    try:
        # Calculate the Technical Indicators for historical data of a single Equity

        # Calculate Simple Moving Average (SMA)
        # It is used to Identify Trend Direction, Support and Resistance Levels, and Generate Buy/Sell Signals
        # An SMA is an arithmetic moving average calculated by adding recent prices and then dividing that figure by the number of time periods in the calculation average
        # Short Term Price Movement - 20 days SMA
        # Medium Term Price Movement - 50 days SMA
        # Long Term Price Movement - 200 days SMA
        df['equity_sma_20'] = df['equity_close'].rolling(window=20).mean()
        df['equity_sma_50'] = df['equity_close'].rolling(window=50).mean()
        df['equity_sma_200'] = df['equity_close'].rolling(window=200).mean()

        # Calculate Exponential Moving Average (EMA)
        # It is used to Identify Fast Trend Direction, Support and Resistance Levels, and Generate Buy/Sell Signals
        # An EMA is a type of moving average that places a greater weight and significance on the most recent price data.
        # The EMA reacts more significantly to recent price changes than a simple moving average (SMA), which applies an equal weight to all observations in the period.
        # adjust=False ensures recursive calculation matching standard financial formula.
        # EMA = ((Current Closing Price - Previous EMA) * multiplier) + Previous EMA where multiplier = 2 / (span + 1)
        # span is the number of periods for the EMA calculation
        # Short Term Price Movement - 9 days EMA
        # Medium Term Price Movement - 50 days EMA
        # Long Term Price Movement - 100 days EMA
        df['equity_ema_9'] = df['equity_close'].ewm(span=9, adjust=False).mean()
        df['equity_ema_50'] = df['equity_close'].ewm(span=50, adjust=False).mean()
        df['equity_ema_100'] = df['equity_close'].ewm(span=100, adjust=False).mean()

        # Calculate Moving Average Convergence Divergence (MACD)
        # It is used to Identify Trend Direction, Momentum, and Generate Buy/Sell Signals
        exp12 = df['equity_close'].ewm(span=12, adjust=False).mean()
        exp26 = df['equity_close'].ewm(span=26, adjust=False).mean()
        df['equity_macd_line'] = exp12 - exp26 #  12-period EMA minus 26-period EMA
        df['equity_macd_signal'] = df['equity_macd_line'].ewm(span=9, adjust=False).mean() # 9-period EMA of the MACD line
        df['equity_macd_histogram'] = df['equity_macd_line'] - df['equity_macd_signal'] # MACD Histogram = MACD Line - Signal Line
        # Drop the intermediate variables to save memory
        del exp12, exp26
        # Drop MACD Line and MACD Signal columns, as they are intermediate calculations
        df.drop(columns=['equity_macd_line', 'equity_macd_signal'], inplace=True)

        # Calculate Relative Strength Index (RSI)
        # It is used to Identify Overbought and Oversold Conditions, Trend Strength, and Generate Buy/Sell Signals
        delta = df['equity_close'].diff() # Calculate the difference between the current and previous closing price
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean() # Calculate the average gain over a 14-day period
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean() # Calculate the average loss over a 14-day period
        rs = gain / loss # Calculate the relative strength (RS) as the ratio of average gain to average loss
        df['equity_rsi_14'] = 100 - (100 / (1 + rs)) # Calculate the RSI using the formula: RSI = 100 - (100 / (1 + RS))
        # Drop the intermediate variables to save memory
        del delta, gain, loss, rs

        return {"data": df, "status": True, "message": ""}
    except Exception as e:
        return {"data": pd.DataFrame(), "status": False, "message": f"ERROR: Error occurred while calculating technical indicators!"}

def download_historical_sp500_data() -> dict:
    try:
        # Download historical data for S&P 500 Index (as a benchmark) from Yahoo Finance
        # S&P 500 Index Ticker Symbol: ^GSPC
        # For Index Tickers, Closing Price and Adjusted Closing Price are the same, since Indexes do not have Corporate Actions like Stock Splits, Dividends, and Rights Offerings
        # auto_adjust=True will return only Close Price, since Adjusted Close Price is the same as Close Price for Indexes
        # period="max" will download the maximum available data, interval="1d" will download daily data
        df = yf.download('^GSPC', period="max", interval="1d", auto_adjust=True)

        # check if the dataframe is empty (i.e., no data was returned)
        if len(df) > 0:
            # Clean multi-level column index if present in newer yfinance versions
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            # Drop Columns that are not needed for analysis (e.g., 'Open', 'High', 'Low', 'Volume')
            df = df.drop(columns=['Open', 'High', 'Low', 'Volume'])
            # Rename the columns to lowercase and prefix with 'sp500_'
            df.columns = ['sp500_'+col.lower() for col in df.columns]
            # Create all Calendar Days and forward-fill values
            temp_calendar_days_df = pd.date_range(start=df.index.min(), end=df.index.max(), freq="D")
            df = df.reindex(temp_calendar_days_df).ffill()
            # Drop the intermediate variables to save memory
            del temp_calendar_days_df

            return {"data": df, "status": True, "message": ""}
        else:
            return {"data": pd.DataFrame(), "status": False, "message": f"ERROR: Yahoo Finance does not return the data requested for S&P 500 Index!"}
    except Exception as e:
        return {"data": pd.DataFrame(), "status": False, "message": f"ERROR: Yahoo Finance does not return the data requested for S&P 500 Index!"}

def download_historical_nasdaq_data() -> dict:
    try:
        # Download historical data for NASDAQ Composite Index (as a benchmark) from Yahoo Finance
        # NASDAQ Composite Index Ticker Symbol: ^IXIC
        # For Index Tickers, Closing Price and Adjusted Closing Price are the same, since Indexes do not have Corporate Actions like Stock Splits, Dividends, and Rights Offerings
        # auto_adjust=True will return only Close Price, since Adjusted Close Price is the same as Close Price for Indexes
        # period="max" will download the maximum available data, interval="1d" will download daily data
        df = yf.download('^IXIC', period="max", interval="1d", auto_adjust=True)

        # check if the dataframe is empty (i.e., no data was returned)
        if len(df) > 0:
            # Clean multi-level column index if present in newer yfinance versions
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            # Drop Columns that are not needed for analysis (e.g., 'Open', 'High', 'Low', 'Volume')
            df = df.drop(columns=['Open', 'High', 'Low', 'Volume'])
            # Rename the columns to lowercase and prefix with 'nasdaq_'
            df.columns = ['nasdaq_'+col.lower() for col in df.columns]
            # Create all Calendar Days and forward-fill values
            temp_calendar_days_df = pd.date_range(start=df.index.min(), end=df.index.max(), freq="D")
            df = df.reindex(temp_calendar_days_df).ffill()
            # Drop the intermediate variables to save memory
            del temp_calendar_days_df

            return {"data": df, "status": True, "message": ""}
        else:
            return {"data": pd.DataFrame(), "status": False, "message": f"ERROR: Yahoo Finance does not return the data requested for NASDAQ Composite Index!"}
    except Exception as e:
        return {"data": pd.DataFrame(), "status": False, "message": f"ERROR: Yahoo Finance does not return the data requested for NASDAQ Composite Index!"}

def download_historical_fred_data(api_key: str) -> dict:
    try:
        # Download historical data from FRED (Federal Reserve Economic Data) database for the below:

        # Initialize the FRED object with API key
        # Please note that you need to sign up for a free FRED account and obtain an API key from the FRED website (https://fred.stlouisfed.org/) to access the data
        # You can replace your FRED API key in the code below OR use my provided API key for demonstration purpose
        fred = Fred(api_key=api_key)

        # Define the dictionary of series IDs
        fred_series_dict = {
            # Interest Rates & Monetary Policy - Central bank decisions drive equity discount rates and corporate borrowing costs
            'effective_federal_reserve_funds_rate': 'FEDFUNDS', # Monthly. The primary benchmark rate set by the Fed
            # Yield Curve & Bond Market Signals - The bond market often prices economic reality faster than equity markets
            'treasury_constant_maturity_10_year_minus_2_year': 'T10Y2Y', # Daily. The classic yield curve inversion indicator. When it dips below 0, it signals a looming recession, traditionally acting as a strong mid-to-long term bearish flag for equities
            'treasury_constant_maturity_10_year_minus_3_month': 'T10Y3M', # Daily. The Federal Reserve Bank of New York uses this specific 10-year minus 3-month spread to calculate the statistical probability of a U.S. recession over the next 12 months
            '10_year_treasury_constant_maturity_rate': 'DGS10', # Daily. Used as the risk-free rate in discounted cash flow (DCF) stock valuation models. A rising 10Y yield compresses equity valuation multiples
            # Credit Spreads & Financial Stress - These indicators serve as proxies for systemic risk and corporate bankruptcy risk
            'st_louis_fed_financial_stress_index': 'STLFSI4', # Weekly. A composite index measuring market stress. Readings above 0 indicate above-average financial market strain
            'cboe_volatility_index': 'VIXCLS', #  Daily. The "fear gauge". While technically a market output, incorporating it helps predict short-term regime changes and momentum shifts
            # Liquidity & Money Supply - Stocks require capital flow to move higher. These indicators track macro liquidity
            'm2_money_supply': 'M2SL', # Monthly. Measures cash, checking deposits, and easily convertible money. Rapid M2 growth historically signals asset price inflation (bullish for stocks)
            # Inflation & Macro Growth - These indicators affect consumer spending power and corporate margins
            'consumer_price_index': 'CPIAUCSL', # Monthly. The headline inflation indicator. Persistent high inflation forces aggressive Fed tightening, squeezing stock multiples
            'unemployment_rate': 'UNRATE', # Monthly. A lagging but vital health indicator of the consumer economy
            'industrial_production_index': 'INDPRO', # Monthly. A reliable, monthly proxy for GDP growth before official quarterly numbers are released
            'real_gross_domestic_product': 'GDPC1' # Quarterly. is generally more important for long-term stock market trends because it reflects true economic health and corporate volume growth without price distortions
        }

        # Download each series into a DataFrame and combine them
        fred_data_frames = []
        for name, series_id in fred_series_dict.items():
            try:
                series_data = fred.get_series(series_id)
                df_temp = pd.DataFrame(series_data, columns=[name])
                fred_data_frames.append(df_temp)
            except Exception as e:
                return {"data": pd.DataFrame(), "status": False, "message": f"ERROR: FRED database does not return the data requested for {name}! Please check the series ID and your FRED API key!"}

        # Concatenate all series horizontally into a single DataFrame
        fred_original_df = pd.concat(fred_data_frames, axis=1)

        # Create all Calendar Days and forward-fill values
        temp_calendar_days_df = pd.date_range(start=fred_original_df.index.min(), end=fred_original_df.index.max(), freq="D")
        fred_original_df = fred_original_df.reindex(temp_calendar_days_df).ffill()

        # Drop the intermediate variables to save memory
        del fred_data_frames, temp_calendar_days_df

        return {"data": fred_original_df, "status": True, "message": ""}
    except Exception as e:
        return {"data": pd.DataFrame(), "status": False, "message": f"ERROR: FRED database does not return the data! Please check the series ID and your FRED API key!"}

def data_preparation(ticker_df: pd.DataFrame, sp500_df: pd.DataFrame, nasdaq_df: pd.DataFrame, fred_df: pd.DataFrame) -> dict:
    try:
        # Get the data from 1/1/2021 from ticker_df dataframe for Post Covid analysis
        ticker_df_copy = ticker_df.copy()
        ticker_df_copy = ticker_df_copy[ticker_df_copy.index >= '2021-01-01']

        # Get the data from 1/1/2021 from sp500_df dataframe for Post Covid analysis
        sp500_df_copy = sp500_df.copy()
        sp500_df_copy = sp500_df_copy[sp500_df_copy.index >= '2021-01-01']

        # Get the data from 1/1/2021 from nasdaq_df dataframe for Post Covid analysis
        nasdaq_df_copy = nasdaq_df.copy()
        nasdaq_df_copy = nasdaq_df_copy[nasdaq_df_copy.index >= '2021-01-01']

        # Get the data from 1/1/2021 from fred_df dataframe for Post Covid analysis
        fred_df_copy = fred_df.copy()
        fred_df_copy = fred_df_copy[fred_df_copy.index >= '2021-01-01']

        # Inner Join the dataframe ticker_df with sp500_df, nasdaq_df and fred_df to create a merged dataframe
        df = ticker_df_copy.join([sp500_df_copy, nasdaq_df_copy, fred_df_copy], how="inner")

        # Drop the intermediate variables to save memory
        del ticker_df_copy, sp500_df_copy, nasdaq_df_copy, fred_df_copy

        # Create target column for next trading day's closing price in the current row, which will be used for supervised learning
        df['equity_predicted_close'] = df['equity_close'].shift(-1)

        # Drops the last row since it will have NaN value in the target column 'equity_predicted_close' after shifting
        df.dropna(inplace=True)

        # Get the number of missing values in each column
        # There should not be any missing values in the merged dataframe, since we have forward filled the FRED data and dropped the last row after shifting the target column
        if (df.isnull().sum().sum() > 0):
            return {"data": pd.DataFrame(), "status": False, "message": f"ERROR: Data Quality Check Failed since the Data contains Missing Values!"}

        return {"data": df, "status": True, "message": ""}
    except Exception as e:
        return {"data": pd.DataFrame(), "status": False, "message": f"ERROR: Data Preparation and/or Data Quality Check Failed!"}

def data_plot(df: pd.DataFrame) -> dict:
    try:
        # Plot the time-series data for all the features in the merged dataframe using Seaborn's lineplot function
        sns.set(style="darkgrid") # Set the Seaborn style to "darkgrid" for better aesthetics in the plots

        # Get the list of features (columns) in the dataframe, except the target column 'equity_predicted_close'
        features_list = [col for col in df.columns if col != 'equity_predicted_close']

        # Create a grid of subplots
        fig, axes = plt.subplots(
            nrows=len(features_list), # Number of rows equal to the number of features
            ncols=1, # Number of columns is 1 since we want a single column of plots
            figsize=(15, 100), # Set the figure size to 15 inches wide and 100 inches tall to accommodate all the subplots
            sharex=True,  # Share x-axis across all time-series panels
        )

        # Flatten axes array in case it's multi-dimensional, loop through each feature
        for i, feature in enumerate(features_list):
            ax = axes[i] # Get the current axis for the subplot
            sns.lineplot(data=df, x=df.index, y=feature, ax=ax, color="tab:blue") # Plot the time-series data for the current feature using Seaborn's lineplot function, with the x-axis as the index (date) and y-axis as the feature values, and set the color to blue
            ax.set_title(f"Feature: {feature}", loc="left", fontsize=12) # Set the title of the subplot to indicate the feature being plotted, aligned to the left, with a font size of 12
            ax.set_ylabel(feature) # Set the y-axis label to the feature name to indicate what is being plotted on the y-axis
            ax.set_xlabel("") # Set the x-axis label to an empty string to avoid cluttering the plots with x-axis labels for each subplot

        axes[-1].set_xlabel("Trade Date") # Label the bottom x-axis
        plt.tight_layout() # Adjust the layout to prevent overlapping of subplots and ensure that titles and labels are clearly visible
        plt.show() # Display the plots

        return {"fig": fig, "status": True, "message": ""}
    except Exception as e:
        return {"fig": fig, "status": False, "message": f"ERROR: Data Plotting Failed!"}

def linear_regression_model(df: pd.DataFrame, ticker: str) -> dict:
    try:
        # Linear Regression Model

        start_time = time.perf_counter() # Record start time

        # Create X and y from the equity_df dataset where 'equity_predicted_close' column is the target and the remaining columns are the features
        X = df.drop(columns=['equity_predicted_close']) # Data Frame for Features
        y = df["equity_predicted_close"] # Series for Target

        # Split the time series data into training and testing sets
        # Since this is time series data, we will not shuffle the data and will use the first 80% of the data for training and the last 20% for testing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

        # Create a Pipeline with Scaling and Linear Regression
        pipeline = Pipeline(steps=[
                ("scaler", MinMaxScaler()),  # Scaling is important to ensure that all features are on the same scale
                ("regressor", LinearRegression())
            ])

        # Define the Parameter Grid
        # The parameter grid is used to specify the hyperparameters that we want to tune for the Linear Regression model.
        # In this case, we are tuning the 'fit_intercept' and 'positive' parameters of the Linear Regression model.
        # fit_intercept: Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations (e.g., data is expected to be already centered).
        # positive: When set to True, forces the coefficients to be positive. This can be useful when we know that the relationship between the features and the target variable is positive (e.g., as the value of a feature increases, the value of the target variable also increases).
        param_grid = {"regressor__fit_intercept": [True, False],
                    "regressor__positive": [True, False]}

        # Time Series Split which is essential for financial data to prevent looking ahead into future data
        tscv = TimeSeriesSplit(n_splits=5)

        # Create a Grid Search with Cross-Validation
        # Cross-Validation is used to avoid data leakage and overfitting. The number of splits is set to 5, which means that the dataset will be split into 5 folds
        # Scoring is set to "neg_mean_squared_error" since we are predicting a continuous variable (equity closing price)
        # n_jobs=-1 means that all available CPU cores will be used for parallel processing, which can speed up the grid search process.
        grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=tscv, scoring="neg_mean_squared_error", n_jobs=-1)

        # Fit the Grid Search to the training data
        grid_search.fit(X_train, y_train)

        # Predict using the Best Model
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)

        # Evaluate Metrics
        mae = round(mean_absolute_error(y_test, y_pred),4) # Best Value is 0. Measures the average absolute distance between actual and predicted values. It treats all errors equally.
        mse = round(mean_squared_error(y_test, y_pred),4) # Best Value is 0. Measures the average squared distance between actual and predicted values. It penalizes larger errors more than smaller errors.
        rmse = round(np.sqrt(mse),4) # Best Value is 0. Measures the square root of the average squared distance between actual and predicted values. It is in the same unit as the target variable and is more interpretable than MSE.
        r2 = round(r2_score(y_test, y_pred),4) # Best Value is 1. Measures the proportion of variance in the target variable that is explained by the model. It ranges from 0 to 1, with higher values indicating better model performance.

        # Plot the Actual vs Predicted Closing Price for the Test Set
        # Create DatetimeIndex object to store the index of the test set
        test_data_DatetimeIndex = X_test.index

        # Create a DataFrame in Wide Format for Seaborn Line Plot
        df_wide = pd.DataFrame({'Actual Closing Price': y_test, 'Predicted Closing Price': y_pred}, index=test_data_DatetimeIndex)

        # Define a custom color palette for the line plot to make it visually appealing and easy to distinguish between actual and predicted values
        custom_palette = {
            'Actual Closing Price': '#0A2540',    # Deep Navy Blue
            'Predicted Closing Price': '#00D4B2'  # Bright Teal
        }

        prediction_fig = plt.figure(figsize=(15, 5)) # Set the figure size to 15 inches wide and 5 inches tall
        sns.lineplot(data=df_wide, palette=custom_palette, linewidth=2.5) # Plot the actual and predicted closing prices using Seaborn's lineplot function, with the specified color palette and line width
        plt.title(f'Linear Regression - Actual vs Predicted Closing Price for {ticker}') # Set the title of the plot to include the ticker symbol for clarity
        plt.xlabel('Trade Date') # Set the x-axis label to 'Trade Date' to indicate that the x-axis represents the dates of the trades
        plt.ylabel('Closing Price in USD') # Set the y-axis label to 'Closing Price' to indicate that the y-axis represents the closing prices of the equity
        plt.xticks(rotation=45) # Rotate the x-axis labels by 45 degrees for better readability, especially when there are many dates on the x-axis
        plt.tight_layout() # Adjust the padding of the plot to ensure that all elements are visible and not cut off
        plt.show() # Display the plot

        # Plot the most influential features in the top using Seaborn
        # Create dataframe for Features and Coefficients
        feature_coefficient_df = pd.DataFrame({
            'Feature': [name.split('__')[-1] for name in list(best_model.named_steps['scaler'].get_feature_names_out())],
            'Coefficient': best_model.named_steps['regressor'].coef_
        })

        feature_fig = plt.figure(figsize=(15, 8)) # Set the figure size to 15 inches wide and 8 inches tall
        sns.barplot(data=feature_coefficient_df.sort_values(by='Coefficient', ascending=False), x='Coefficient', y='Feature', palette='viridis') # Plot the coefficients of the features using Seaborn's barplot function, sorting the features by their coefficient values in descending order, and using the 'viridis' color palette for better visual distinction
        plt.title(f'Linear Regression - Feature Importance (Most Influential at the Top)') # Set the title of the plot to indicate that it shows the features most influencing the equity closing price for the specified ticker
        plt.xlabel('Coefficient Value') # Set the x-axis label to 'Coefficient Value' to indicate that the x-axis represents the values of the coefficients for each feature
        plt.ylabel('Features') # Set the y-axis label to 'Feature Name' to indicate that the y-axis represents the names of the features used in the model
        plt.tight_layout() # Adjust the padding of the plot to ensure that all elements are visible and not cut off
        plt.show() # Display the plot

        end_time = time.perf_counter() # Record end time
        elapsed_seconds = round(end_time - start_time, 0) # Calculate elapsed time in Seconds

        # Create Summary Statistics
        lr_metrics_dict = {"Model Name" : "LINEAR REGRESSION",
                        "Elapsed Time (Seconds)" : int(elapsed_seconds),
                        "Mean Absolute Error (MAE)" : mae,
                        "Mean Squared Error (MSE)" : mse,
                        "Root Mean Squared Error (RMSE)" : rmse.item(),
                        "R-squared (R2)" : r2,
                        "Last Trading Day (MM/DD/YYYY)": test_data_DatetimeIndex[-1].strftime("%m/%d/%Y"),
                        "Actual Closing Price (USD)" : round(y_test.iloc[-1],2).item(),
                        "Predicted Closing Price (USD)" : round(y_pred[-1],2).item()
                        }
        
        return {"metrics": lr_metrics_dict, "prediction_fig": prediction_fig, "feature_fig": feature_fig, "status": True, "message": ""}
    except Exception as e:
        return {"metrics": lr_metrics_dict, "prediction_fig": prediction_fig, "feature_fig": feature_fig, "status": False, "message": f"ERROR: Linear Regression Model Failed!"}

def sarimax_model(df: pd.DataFrame, ticker: str) -> dict:
    try:
        # SARIMAX Model

        start_time = time.perf_counter() # Record start time

        # Create X and y from the equity_df dataset where 'equity_predicted_close' column is the target and the remaining columns are the features
        X = df.drop(columns=['equity_predicted_close']) # Data Frame for Features
        y = df["equity_predicted_close"] # Series for Target

        # Split the time series data into training and testing sets
        # Since this is time series data, we will not shuffle the data and will use the first 80% of the data for training and the last 20% for testing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

        # Scaling the dataset (Separately to avoid leakage)
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)

        y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
        y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1))

        # Tune Hyperparameters
        auto_model = pm.auto_arima (
            y = y_train_scaled,       # Target variable which has been scaled/normalized to help the model algorithm converge faster
            X = X_train_scaled,       # Exogenous variables which has been scaled/normalized
            start_p = 1,              # Limits the search for the core non-seasonal parts of the model. It searches for the best Autoregressive (p) terms between start_p and max_p
            max_p = 3,
            start_q = 1,              # Limits the search for the core non-seasonal parts of the model. It searches for the best Moving Average (q) terms between start_q and max_q
            max_q = 3,
            d = None,                 # Let auto_arima find optimal differencing 'd' for non-seasonal differences needed to make your target data stationary (removing trends)
            seasonal = True,          # Tells the model to look for a seasonal pattern. Setting m=5 specifies a weekly cycle based on 5 days
            m = 5,
            start_P = 0,              # Limits the search for the seasonal Autoregressive component up to 2 lags specified by max_P
            max_P = 2,
            trace = False,            # Cleanup parameter - Run the code completely silently in the background
            error_action = "ignore",  # Cleanup parameter - Ignore optimization errors
            suppress_warnings = True, # Cleanup parameter - Hide long lists of text readouts
        )

        # Train the Model
        final_model = SARIMAX (
            endog = y_train_scaled,       # Target variable which has been scaled/normalized to help the model algorithm converge faster
            exog = X_train_scaled,        # Exogenous variables which has been scaled/normalized
            order = auto_model.order,     # Optimal (p, d, q) non-seasonal hyperparameters. p is Autoregressive order (past values), d is Differencing order (to make the data stationary), q is Moving average order (past forecast errors)
            seasonal_order = auto_model.seasonal_order, # Optimal (P, D, Q, s) seasonal hyperparameters
        ).fit(disp = False)               # Suppresses all convergence output and logs during the fit

        # Forecast requires the scaled test exogenous features
        y_pred_scaled = final_model.forecast (
            steps = len(X_test),   # The number of out-of-sample time steps to forecast
            exog = X_test_scaled   # Future values of the exogenous regressors corresponding only to the forecasted time steps
        )

        # Bring back original values
        # Reshape to 2D array for inverse_transform
        y_pred_scaled_2d = y_pred_scaled.reshape(-1, 1)

        # Inverse transform to bring back original dollar scale
        y_pred_original = scaler_y.inverse_transform(y_pred_scaled_2d)
        y_test_original = scaler_y.inverse_transform(y_test_scaled)

        # Convert to clean 1D arrays or series
        y_pred_original = y_pred_original.flatten()
        y_test_original = y_test_original.flatten()

        # Evaluate Metrics
        mae = round(mean_absolute_error(y_test_original, y_pred_original),4) # Best Value is 0. Measures the average absolute distance between actual and predicted values. It treats all errors equally.
        mse = round(mean_squared_error(y_test_original, y_pred_original),4) # Best Value is 0. Measures the average squared distance between actual and predicted values. It penalizes larger errors more than smaller errors.
        rmse = round(np.sqrt(mse),4) # Best Value is 0. Measures the square root of the average squared distance between actual and predicted values. It is in the same unit as the target variable and is more interpretable than MSE.
        r2 = round(r2_score(y_test_original, y_pred_original),4) # Best Value is 1. Measures the proportion of variance in the target variable that is explained by the model. It ranges from 0 to 1, with higher values indicating better model performance.

        # Plot the Actual vs Predicted Closing Price for the Test Set
        # Create DatetimeIndex object to store the index of the test set
        test_data_DatetimeIndex = X_test.index

        # Create a DataFrame in Wide Format for Seaborn Line Plot
        df_wide = pd.DataFrame({'Actual Closing Price': y_test_original, 'Predicted Closing Price': y_pred_original}, index=test_data_DatetimeIndex)

        # Define a custom color palette for the line plot to make it visually appealing and easy to distinguish between actual and predicted values
        custom_palette = {
            'Actual Closing Price': '#0A2540',    # Deep Navy Blue
            'Predicted Closing Price': '#00D4B2'  # Bright Teal
        }

        prediction_fig = plt.figure(figsize=(15, 5)) # Set the figure size to 15 inches wide and 5 inches tall
        sns.lineplot(data=df_wide, palette=custom_palette, linewidth=2.5) # Plot the actual and predicted closing prices using Seaborn's lineplot function, with the specified color palette and line width
        plt.title(f'SARIMAX - Actual vs Predicted Closing Price for {ticker}') # Set the title of the plot to include the ticker symbol for clarity
        plt.xlabel('Trade Date') # Set the x-axis label to 'Trade Date' to indicate that the x-axis represents the dates of the trades
        plt.ylabel('Closing Price in USD') # Set the y-axis label to 'Closing Price' to indicate that the y-axis represents the closing prices of the equity
        plt.xticks(rotation=45) # Rotate the x-axis labels by 45 degrees for better readability, especially when there are many dates on the x-axis
        plt.tight_layout() # Adjust the padding of the plot to ensure that all elements are visible and not cut off
        plt.show() # Display the plot

        # Plot the most influential features in the top
        features_list = X.columns.to_list()
        # Exogenous feature parameters in statsmodels are prefixed with 'x' (e.g., x1, x2, x3...)
        # Filter for these specific coefficients and map them back to your features_list
        exog_params = [final_model.params[i] for i in range(len(features_list))]

        feature_importance = pd.DataFrame(
            {
                "Feature": features_list,
                "Coefficient": exog_params
            }
        )

        # Sort features by their impact
        feature_importance = feature_importance.sort_values(by="Coefficient", ascending=False)

        feature_fig = plt.figure(figsize=(15, 10))
        colors = ["salmon" if c < 0 else "skyblue" for c in feature_importance["Coefficient"]]

        plt.barh(
            feature_importance["Feature"],
            feature_importance["Coefficient"],
            color=colors,
        )
        plt.axvline(0, color="black", linestyle="--", linewidth=0.8)
        plt.xlabel("Coefficient Value")
        plt.ylabel("Features")
        plt.title("SARIMAX - Feature Importance (Most Influential at the Top)")
        plt.gca().invert_yaxis()
        plt.grid(axis="x", linestyle=":", alpha=0.6)
        plt.show()

        end_time = time.perf_counter() # Record end time
        elapsed_seconds = round(end_time - start_time, 0) # Calculate elapsed time in Seconds

        # Create Summary Statistics
        sarimax_metrics_dict = {"Model Name" : "SARIMAX",
                                "Elapsed Time (Seconds)" : int(elapsed_seconds),
                                "Mean Absolute Error (MAE)" : mae,
                                "Mean Squared Error (MSE)" : mse,
                                "Root Mean Squared Error (RMSE)" : rmse.item(),
                                "R-squared (R2)" : r2,
                                "Last Trading Day (MM/DD/YYYY)": test_data_DatetimeIndex[-1].strftime("%m/%d/%Y"),
                                "Actual Closing Price (USD)" : round(y_test_original[-1],2).item(),
                                "Predicted Closing Price (USD)" : round(y_pred_original[-1],2).item()
                                }

        return {"metrics": sarimax_metrics_dict, "prediction_fig": prediction_fig, "feature_fig": feature_fig, "status": True, "message": ""}
    except Exception as e:
        return {"metrics": sarimax_metrics_dict, "prediction_fig": prediction_fig, "feature_fig": feature_fig, "status": False, "message": f"ERROR: Linear Regression Model Failed!"}

def lstm_model(df: pd.DataFrame, ticker: str) -> dict:
    try:
        # LSTM Model
        # Create Time-Series Sequences (Lookback window) to transform a time series into input (X) and target (y) pairs
        # Input Parameters:
        #  data: Time-series data
        #  lookback: Number of time steps to use as input
        #  target_idx: Index Position of the Target variable in the dataframe
        # Returns:
        #  X: Array of shape (num_samples, lookback)
        #  y: Array of shape (num_samples,) predicting the next value
        def create_sequences(data, lookback, target_idx):
            X, y = [], []
            for i in range(len(data) - lookback):
                X.append(data[i:(i + lookback), :])
                y.append(data[i + lookback, target_idx])
            return np.array(X), np.array(y)

        # Define Hypermodel for Bayesian Optimization
        # Input Parameters:
        #  hp: Special object passed by the tuning algorithm for HyperParameters
        # Returns:
        #  model: A fully configured and compiled Keras Model instance
        def build_model(hp):
            model = Sequential() # Initialize the empty linear container
            
            # Add LSTM Input first layer to process the sequence data
            model.add(LSTM(
                units = hp.Int('units_1', min_value=32, max_value=128, step=32), # Tunes the number of units (neurons) inside the LSTM layer from 32 to 128 increment of 32
                return_sequences = hp.Choice('return_seq', [True, False]), # Try both True/False during training to see which one yields better validation accuracy
                input_shape = (X_train.shape[1], X_train.shape[2])  # Parameters are Training dataset Number of Rows and Number of Columns respectively
            ))
            
            # 'return_seq'=True: Outputs a 3D tensor (batch_size, timesteps, output_features), Required by the next LSTM layer
            # 'return_seq'=False: Outputs a 2D tensor (batch_size, output_features), Required by Dense layers, causes an error if sent to another LSTM
            if hp.get('return_seq'):
                # Add LSTM second layer to process the sequence data
                model.add(LSTM(units = hp.Int('units_2', min_value=16, max_value=64, step=16), # Tunes the number of units (neurons) inside the LSTM layer from 16 to 64 increment of 16
                            return_sequences = False # Outputs a 2D tensor
                            ))
                
            # Add Dropout Layer to randomly "turn off" a percentage of neurons during training to avoid overfitting
            model.add(Dropout(rate = hp.Float('dropout', min_value=0.1, max_value=0.4, step=0.1))) # Lowest dropout rate the tuner will try is 10% of neurons to be dropped, Highest 40%, Increment of 10%

            # Add Output Dense Layer to return the Predicted Closing Price
            model.add(Dense(1)) # 1 Neuron
            
            # It configures the learning process by defining how the model should update its weights and how it should measure its performance
            # Adam is the optimization algorithm used to update the model's weights during training to minimize errors
            # Min/Max Value defines the boundaries for the search. The tuner will experiment with values inside this range
            # 'log' tells the tuner to explore the range on a logarithmic scale
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate = hp.Float('lr', min_value=0.0001, max_value=0.01, sampling='log')),
                loss='mse' # Loss function for Regression Model - Mean Squared Error
            )
            return model

        start_time = time.perf_counter() # Record start time

        # Fix Naive Trap: Transform entire dataset to Percentage Returns (Stationary)
        # This removes both linear trend and scaling disparities across features
        df_returns = df.pct_change()
        df_returns = df_returns.iloc[1:] # Drop the First row which is a NULL row after Percentage Change
        df_returns.fillna(0, inplace=True) # Replace NaN by Zero
        df_returns.replace([np.inf, -np.inf], 0, inplace=True) # Replace +/- Infinity by Zero

        # Train/Test Split (Chronological split to preserve temporal patterns)
        train_size = int(len(df_returns) * 0.8)
        train_df = df_returns.iloc[:train_size]
        test_df = df_returns.iloc[train_size:]

        # Extract the matching datetime index for test_df
        test_index_df = df_returns.index[train_size:]

        # Scale the Features (Fit ONLY on training data to prevent leakage)
        scaler = MinMaxScaler(feature_range=(-1, 1)) # -1 to 1 works better for returns centering around 0
        train_scaled = scaler.fit_transform(train_df)
        test_scaled = scaler.transform(test_df)

        # Separate scaler for the target returns column to make inverse-transform seamless
        target_scaler = MinMaxScaler(feature_range=(-1, 1))
        target_scaler.fit(train_df[['equity_predicted_close']])

        # Create Time-Series Sequences (Lookback window) to transform a time series into input (X) and target (y) pairs
        LOOKBACK = 15 # Number of time steps to use as input
        X_train, y_train = create_sequences(train_scaled, LOOKBACK, df.columns.get_loc('equity_predicted_close'))
        X_test, y_test = create_sequences(test_scaled, LOOKBACK, df.columns.get_loc('equity_predicted_close'))

        # Execute Bayesian Optimization (Faster than Random/Grid Search) with Early Stopping
        tuner = kt.BayesianOptimization(
            hypermodel = build_model, # This is a user-defined function passed into the tuner. It contains the blueprint of your LSTM model and specifies which hyperparameters are allowed to change
            objective = 'val_loss', # This tells the tuner how to judge success. It will try to minimize the validation loss
            max_trials = 5, # The tuner will build, train, and evaluate exactly 5 different model configurations before stopping and picking the winner
            directory = 'temp', # temp directory where tuned model is stored
            project_name = 'lstm_tuned_model', # project name for the tuned model
            overwrite = True # Created the tuned model every time
        )

        # It is a regularization technique used in deep learning to prevent a neural network from overfitting and to save training time
        # It tells to watch the validation loss for 7 epochs to wait for an improvement before stopping
        # restore_best_weights=True means automatically rolls back the model's weights to the exact epoch where the validation loss was at its lowest
        early_stopping = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=0)

        # Execute hyperparameter tuning search
        tuner.search(
            X_train, # Training Data for Input Features
            y_train, # Training Data for Target
            epochs = 50, # Maximum number of times each model trial will see the entire training dataset
            batch_size = 32, # Number of training samples the model processes at once before updating its internal weights
            validation_split = 0.1, # Tells the tuner to automatically set aside 10% of your training data to use as validation data
            callbacks = [early_stopping], # Ensures that if a specific model trial stops improving on the validation data, it cuts the training short instead of wasting time running all 50 epochs
            verbose = 0 # Hide the Output Messages
        )

        # It retrieves the single top-performing model found during a KerasTuner hyperparameter search
        best_model = tuner.get_best_models(num_models=1)[0]

        # Predict in Scaled value
        pred_scaled = best_model.predict(X_test, verbose=0)

        # Inverse transform predictions and actuals back into raw Return values
        pred_returns = target_scaler.inverse_transform(pred_scaled).flatten()
        actual_returns = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

        # Calculate the exact chronological alignment index in the original dataframe
        # Account for offsets: 1 (pct_change) + LOOKBACK (sequencing)
        alignment_offset = 1 + LOOKBACK 
        test_start_idx = train_size + alignment_offset

        # Extract actual target prices for the test window duration
        actual_prices = df.loc[df.index[test_start_idx : test_start_idx + len(actual_returns)], 'equity_predicted_close'].values

        # Extract the true preceding prices (t-1) for every step in the test window
        previous_true_prices = df.loc[df.index[test_start_idx - 1 : test_start_idx + len(actual_returns) - 1], 'equity_predicted_close'].values

        # Reconstruct predictions: True Previous Price * (1 + Predicted Return)
        # This removes cumulative drift or mirror inversion caused by feedback loops
        pred_prices = previous_true_prices * (1 + pred_returns)

        # Evaluate Metrics
        mae = round(mean_absolute_error(actual_prices, pred_prices),4) # Best Value is 0. Measures the average absolute distance between actual and predicted values. It treats all errors equally.
        mse = round(mean_squared_error(actual_prices, pred_prices),4) # Best Value is 0. Measures the average squared distance between actual and predicted values. It penalizes larger errors more than smaller errors.
        rmse = round(np.sqrt(mse),4) # Best Value is 0. Measures the square root of the average squared distance between actual and predicted values. It is in the same unit as the target variable and is more interpretable than MSE.
        r2 = round(r2_score(actual_prices, pred_prices),4) # Best Value is 1. Measures the proportion of variance in the target variable that is explained by the model. It ranges from 0 to 1, with higher values indicating better model performance.

        # Plot the Actual vs Predicted Closing Price for the Test Set
        feature_names = df.columns.tolist() # List of all Feature Names
        # Create DatetimeIndex object to store the index of the test set
        test_data_DatetimeIndex = test_index_df[-1 * len(actual_prices) :]

        # Create a DataFrame in Wide Format for Seaborn Line Plot
        df_wide = pd.DataFrame({'Actual Closing Price': actual_prices, 'Predicted Closing Price': pred_prices}, index=test_data_DatetimeIndex)

        # Define a custom color palette for the line plot to make it visually appealing and easy to distinguish between actual and predicted values
        custom_palette = {
            'Actual Closing Price': '#0A2540',    # Deep Navy Blue
            'Predicted Closing Price': '#00D4B2'  # Bright Teal
        }

        prediction_fig = plt.figure(figsize=(15, 5)) # Set the figure size to 15 inches wide and 5 inches tall
        sns.lineplot(data=df_wide, palette=custom_palette, linewidth=2.5) # Plot the actual and predicted closing prices using Seaborn's lineplot function, with the specified color palette and line width
        plt.title(f'LSTM - Actual vs Predicted Closing Price for {ticker}') # Set the title of the plot to include the ticker symbol for clarity
        plt.xlabel('Trade Date') # Set the x-axis label to 'Trade Date' to indicate that the x-axis represents the dates of the trades
        plt.ylabel('Closing Price in USD') # Set the y-axis label to 'Closing Price' to indicate that the y-axis represents the closing prices of the equity
        plt.xticks(rotation=45) # Rotate the x-axis labels by 45 degrees for better readability, especially when there are many dates on the x-axis
        plt.tight_layout() # Adjust the padding of the plot to ensure that all elements are visible and not cut off
        plt.show() # Display the plot

        # Plot the most influential features in the top
        rng = np.random.default_rng(seed=42) # Initialize the Generator with a specific seed
        baseline_preds = best_model.predict(X_test, verbose=0) # Predict in Scaled value for Baseline Prediction
        baseline_mse = mean_squared_error(y_test, baseline_preds) # Calculate MSE for Baseline Prediction

        feature_importance = {}
        # Loop through all Input feature indices to evaluate performance drops
        for f_idx in range(X_test.shape[2]-1):
            X_test_permuted = X_test.copy()
            
            # Shuffle this specific feature across all samples and lookback timesteps
            for seq in range(X_test.shape[0]):
                rng.shuffle(X_test_permuted[seq, :, f_idx])
                
            permuted_preds = best_model.predict(X_test_permuted, verbose=0) # Predict in Scaled value after the Shuffle for each feature
            permuted_mse = mean_squared_error(y_test, permuted_preds) # Calculate MSE after the Shuffle for each feature
            
            # Importance score = how much worse the error got when this feature was ruined
            feature_importance[feature_names[f_idx]] = permuted_mse - baseline_mse # If the difference is high, that feature is important

        # Sort features strictly from Most Influential to Least Influential
        sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=False) # False for horizontal plot alignment
        feature_important_df = pd.DataFrame(sorted_importance, columns=['Feature', 'Importance Drop (MSE)'])

        # Plot Most Influential Features (Top to Least)
        feature_fig = plt.figure(figsize=(15, 10))
        # Color code by direction of impact (positive = green, negative = red)
        colors = ["salmon" if c < 0 else "skyblue" for c in feature_important_df['Importance Drop (MSE)']]

        plt.barh(
            feature_important_df['Feature'],
            feature_important_df['Importance Drop (MSE)'],
            color=colors,
        )
        plt.axvline(0, color="black", linestyle="--", linewidth=0.8)
        plt.xlabel("Importance Drop (MSE)")
        plt.ylabel("Features")
        plt.title("LSTM - Feature Importance (Most Influential at the Top)")
        plt.grid(axis="x", linestyle=":", alpha=0.6)
        plt.show()

        end_time = time.perf_counter() # Record end time
        elapsed_seconds = round(end_time - start_time, 0) # Calculate elapsed time in Seconds

        # Create Summary Statistics
        lstm_metrics_dict = {"Model Name" : "LSTM",
                            "Elapsed Time (Seconds)" : int(elapsed_seconds),
                            "Mean Absolute Error (MAE)" : mae,
                            "Mean Squared Error (MSE)" : mse,
                            "Root Mean Squared Error (RMSE)" : rmse.item(),
                            "R-squared (R2)" : r2,
                            "Last Trading Day (MM/DD/YYYY)": test_data_DatetimeIndex[-1].strftime("%m/%d/%Y"),
                            "Actual Closing Price (USD)" : round(actual_prices[-1],2).item(),
                            "Predicted Closing Price (USD)" : round(pred_prices[-1],2).item()
                            }

        return {"metrics": lstm_metrics_dict, "prediction_fig": prediction_fig, "feature_fig": feature_fig, "status": True, "message": ""}
    except Exception as e:
        return {"metrics": lstm_metrics_dict, "prediction_fig": prediction_fig, "feature_fig": feature_fig, "status": False, "message": f"ERROR: Linear Regression Model Failed!"}

def main():
    show_header()

    # User inputs a US Equity Ticker Symbol
    with st.sidebar.form(key="ticker_form"):
        ticker = st.text_input("Please Enter a US Equity Ticker Symbol (e.g., AAPL):").strip().upper()
        submitted = st.form_submit_button(label="Submit")

    # This block waits until the submit button is clicked
    if submitted:
        return_dict = ticker_validation(ticker)
        if return_dict["status"]:
            st.write(f"##### US Equity Ticker Symbol: {ticker}")
        else:
            st.sidebar.error(return_dict['message'])
            show_footer()
            st.stop()

        st.write(f"##### Step# 1: Downloading the data from Yahoo Finance, FRED API and Calculating Technical Indicators...")
        download_progress_bar = st.progress(0, text="Progress: 0%")

        return_dict = download_historical_equity_data(ticker=ticker)
        if not return_dict["status"]:
            st.error(return_dict['message'])
            show_footer()
            st.stop()
        yf_original_equity_df = return_dict["data"]
        download_progress_bar.progress(20, text="Progress: 20%")
        st.write(f"Downloaded the data for {ticker} from Yahoo Finance!")

        return_dict = calculate_technical_indicators(df=yf_original_equity_df)
        if not return_dict["status"]:
            st.error(return_dict['message'])
            show_footer()
            st.stop()
        yf_original_equity_df = return_dict["data"]
        download_progress_bar.progress(30, text="Progress: 30%")
        st.write(f"Calculated the Technical Indicators for {ticker}!")

        return_dict = download_historical_sp500_data()
        if not return_dict["status"]:
            st.error(return_dict['message'])
            show_footer()
            st.stop()
        yf_original_sp500_df = return_dict["data"]
        download_progress_bar.progress(40, text="Progress: 40%")
        st.write(f"Downloaded the data for S&P 500 Index from Yahoo Finance!")
        
        return_dict = download_historical_nasdaq_data()
        if not return_dict["status"]:
            st.error(return_dict['message'])
            show_footer()
            st.stop()
        yf_original_nasdaq_df = return_dict["data"]
        download_progress_bar.progress(50, text="Progress: 50%")
        st.write(f"Downloaded the data for NASDAQ Composite Index from Yahoo Finance!")

        return_dict = download_historical_fred_data(api_key=FRED_API_KEY)
        if not return_dict["status"]:
            st.error(return_dict['message'])
            show_footer()
            st.stop()
        fred_original_df = return_dict["data"]
        download_progress_bar.progress(100, text="Progress: 100%")
        st.write(f"Downloaded the Economic Indicators data from FRED database!")

        st.write(f"##### Step# 2: Creating Final Merged dataset from 01/01/2021, Checking Data Quality, Preparting the Plots...")
        return_dict = data_preparation(ticker_df=yf_original_equity_df, sp500_df=yf_original_sp500_df, nasdaq_df=yf_original_nasdaq_df, fred_df=fred_original_df)
        if not return_dict["status"]:
            st.error(return_dict['message'])
            show_footer()
            st.stop()
        equity_df = return_dict["data"]
        st.write(f"Final Dataset Created and Data Quality Check Passed!")
        st.dataframe(equity_df, width='stretch')
        
        return_dict = data_plot(df=equity_df)
        if not return_dict["status"]:
            st.error(return_dict['message'])
            show_footer()
            st.stop()
        st.write(f"##### Visualize all Input Feature Columns")
        st.pyplot(return_dict["fig"])

        st.write(f"##### Step# 3: Executing Linear Regression Model...")
        return_dict = linear_regression_model(df=equity_df, ticker=ticker)
        if not return_dict["status"]:
            st.error(return_dict['message'])
            show_footer()
            st.stop()
        st.pyplot(return_dict["prediction_fig"])
        st.pyplot(return_dict["feature_fig"])
        lr_metrics_dict = return_dict["metrics"]

        st.write(f"##### Step# 4: Executing SARIMAX Model...")
        return_dict = sarimax_model(df=equity_df, ticker=ticker)
        if not return_dict["status"]:
            st.error(return_dict['message'])
            show_footer()
            st.stop()
        st.pyplot(return_dict["prediction_fig"])
        st.pyplot(return_dict["feature_fig"])
        sarimax_metrics_dict = return_dict["metrics"]

        st.write(f"##### Step# 5: Executing LSTM Model...")
        return_dict = lstm_model(df=equity_df, ticker=ticker)
        if not return_dict["status"]:
            st.error(return_dict['message'])
            show_footer()
            st.stop()
        st.pyplot(return_dict["prediction_fig"])
        st.pyplot(return_dict["feature_fig"])
        lstm_metrics_dict = return_dict["metrics"]

        st.write(f"##### Step# 6: Summary Statistics for all the Models")
        # Combine the Summary Statistics of the 3 Models into a list and create the DataFrame
        summary_df = pd.DataFrame([lr_metrics_dict, sarimax_metrics_dict, lstm_metrics_dict])
        st.dataframe(summary_df, width='stretch')
        st.write(f"(Best Value is 0 for MAE, MSE, RMSE)")
        st.write(f"(Best Value is 1 for R2)")

    show_footer()

if __name__ == "__main__":
    main()