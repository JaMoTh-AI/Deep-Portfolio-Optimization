import pandas as pd
import yfinance as yf

def get_risk_free_rate():
    """
    Get the current 3-month Treasury Bill (risk-free rate) from Yahoo Finance.

    Returns:
    float: Current 3-month T-Bill rate as a percentage.
    """
    # Ticker symbol for 3-month T-Bill on Yahoo Finance
    t_bill_symbol = "^IRX"

    # Download historical data for the 3-month T-Bill
    t_bill_data = yf.download(t_bill_symbol, period="1d")

    # Get the most recent value of the 3-month T-Bill rate
    current_rate = t_bill_data['Close'].iloc[-1]

    # Convert to a percentage
    return current_rate / 100

def sectors(tkrs):
    """
    Given a list of tickers, return a dictionary matching the tickers to the sector that they belong to. 
    
    Inputs:
    tkrs: List of tickers
    df: DataFrame of stock data

    Outputs:
    sectors: Dictionary of sectors to the tickers that belong to that sector
    """
    
    output = {}
    for ticker in tkrs:
        stock = yf.Ticker(ticker)
        info = stock.info
        if 'sector' in info:
            output[ticker] = info['sector']
        else:
            output[ticker] = ""
    
    return output

def dividend_yield(tkrs, date):
    """
    Given a list of tickers, return a dictionary matching the tickers 
    to their dividend yield as a percentage of price. 
    If the dividend yield is not available, then return 0.
    
    Inputs:
    tkrs: List of tickers
    date: input strings

    Outputs:
    dividend_yield: Dictionary of tickers to their dividend yield
    """
    if date.tzinfo is None:
        date = date.tz_localize('America/New_York')
    else:
        date = date.tz_convert('America/New_York')
    dividend_yields = {}

    for ticker in tkrs:

        stock = yf.Ticker(ticker)
        dividends = stock.dividends

        if not dividends.empty:
            dividends_before_date = dividends[dividends.index < date]

            if not dividends_before_date.empty:
                most_recent_yield = dividends_before_date.iloc[-1]
                dividend_yields[ticker] = most_recent_yield

            else:
                dividend_yields[ticker] = 0

        else:
            dividend_yields[ticker] = 0

    return dividend_yields

def risk_free_rate(df, start_date, end_date=None):
    """
    Given a time interval, return the risk free rate for that interval,
    which is typically the US Treasury Bill rate for that time interval.

    Inputs:
    df: DataFrame of stock data
    start_date: Start of the time period
    end_date: End of the time period

    Outputs:
    risk_free_rate: The risk free rate for the time period
    """

    if end_date is None:
        end_date = df.index[-1]

    time_period = df.index[(df.index >= start_date) & (df.index <= end_date)]

    return df['^IRX'][time_period].mean()

def all_stock_calc_funcs(tkrs, df, start_date, end_date=None):
    """
    Return a dictionary of all the returned data from the 
    stock calculation functions.

    Inputs:
    tkrs: List of tickers
    df: DataFrame of stock data
    start_date: Start of the time period
    end_date: End of the time period

    Outputs:
    all_data: Dictionary of all the data returned from the stock calculation functions
    """

    risk = risk_func(tkrs, df, start_date, end_date)
    expected_return = expected_returns(tkrs, df, start_date, end_date)
    sectors = sectors(tkrs, df, start_date, end_date)
    price = prices(tkrs, df, start_date, end_date)
    dividend_yields = dividend_yield(tkrs, df, start_date, end_date)
    risk_free = risk_free_rate(df, start_date, end_date)

    all_data = {
        'Risks': risk,
        'Expected Returns': expected_return,
        'Sectors': sectors,
        'Prices': price,
        'Dividend Yields': dividend_yields,
        'Risk Free Rate': risk_free
    }

    return all_data