# yfinance is the API and ta is a technical analysis tool, other important libraries to import data
import yfinance as yf
import ta

"""
    This file contains three functions to pull and preprocess stock data.
    For function calls, refer to the model_training.ipynb or lstm_predict.py files.
"""

def pull_stock_data(stocks, start_date, end_date, interval):
    """
    Retrieves historical stock data from Yahoo Finance.

    Parameters:
    stocks (list): List of stock ticker symbols.
    start_date (str): Start date for the data in 'YYYY-MM-DD' format.
    end_date (str): End date for the data in 'YYYY-MM-DD' format.
    interval (str): Data interval (e.g., '1d', '1h').

    Returns:
    dict: Dictionary with stock tickers as keys and their corresponding data as values.
    """
    stock_data = dict()

    for stock in stocks:
        stock_data[stock] = yf.download(stock, start=start_date, end=end_date, interval=interval)
        if len(stock_data[stock])==0:
          del stock_data[stock]

    return stock_data

def pull_stock_indicators(stock_data):
    """
    Adds technical indicators to stock data.

    Parameters:
    stock_data (dict): Dictionary with stock tickers as keys and their corresponding data as values.

    Returns:
    dict: Dictionary with stock tickers as keys and their data including technical indicators as values.
    """

    # Using ta to add some indicators. This helps remove noise
    for stock in stock_data.keys():
        # Adding simple moving average 20 and 50
        stock_data[stock]['SMA_20'] = ta.trend.sma_indicator(close=stock_data[stock]['Close'], window=20)
        stock_data[stock]['SMA_50'] = ta.trend.sma_indicator(close=stock_data[stock]['Close'], window=50)

        # Adding stochastic oscillators
        stock_data[stock]['%K'] = ta.momentum.stoch(high=stock_data[stock]['High'], low=stock_data[stock]['Low'], close=stock_data[stock]['Close'], window=14)
        stock_data[stock]['%D'] = ta.momentum.stoch_signal(high=stock_data[stock]['High'], low=stock_data[stock]['Low'], close=stock_data[stock]['Close'], window=14)

        # Adding RSI indicator
        stock_data[stock]['RSI'] = ta.momentum.RSIIndicator(close=stock_data[stock]['Close'], window=14).rsi()

        # Calculate Bollinger Bands
        bollinger_bands = ta.volatility.BollingerBands(close=stock_data[stock]['Close'], window=20, window_dev=2)

        # Add Bollinger Bands
        stock_data[stock]['BB_Middle'] = bollinger_bands.bollinger_mavg()
        stock_data[stock]['BB_Upper'] = bollinger_bands.bollinger_hband()
        stock_data[stock]['BB_Lower'] = bollinger_bands.bollinger_lband()

        # Calculate MACD
        macd = ta.trend.MACD(close=stock_data[stock]['Close'], window_fast=12, window_slow=26, window_sign=9)

        # Add MACD and signal line
        stock_data[stock]['MACD'] = macd.macd()
        stock_data[stock]['MACD_Signal'] = macd.macd_signal()

        # Calculate ATR
        atr = ta.volatility.AverageTrueRange(high=stock_data[stock]['High'], low=stock_data[stock]['Low'], close=stock_data[stock]['Close'], window=14)

        # Add ATR
        stock_data[stock]['ATR'] = atr.average_true_range()

    return stock_data

def cleaning_data(stock_data):
    """
    Cleans stock data by ensuring consistent format and removing NaN values.

    Parameters:
    stock_data (dict): Dictionary with stock tickers as keys and their corresponding data as values.

    Returns:
    dict: Cleaned stock data dictionary.
    """

    temp_dict = dict()

    for stock in stock_data.keys():
      try:
          temp_dict[stock_data[stock].shape][stock] = stock_data[stock]
      except:
          temp_dict[stock_data[stock].shape] = dict()
          temp_dict[stock_data[stock].shape][stock] = stock_data[stock]

    max_l = max([len(temp_dict[shape]) for shape in temp_dict.keys()])

    for shape in temp_dict.keys():
      if len(temp_dict[shape])==max_l:
        stock_data = temp_dict[shape]
        break

    # Remove all rows where there is a nan/non number entry
    for stock in stock_data.keys():
      stock_data[stock] = stock_data[stock].dropna()

    return stock_data