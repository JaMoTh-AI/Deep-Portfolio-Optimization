import pandas as pd

def sectors(tkrs):
    """
    Given a list of tickers, return a dictionary matching the tickers to the sector that they belong to. 
    
    Inputs:
    tkrs: List of tickers
    df: DataFrame of stock data

    Outputs:
    sectors: Dictionary of sectors to the tickers that belong to that sector
    """
    
    all_tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]

    output = all_tickers[all_tickers['Symbol'].isin(tkrs)][['Symbol', 'GICS Sector']]
    output = output.set_index('Symbol')['GICS Sector'].to_dict()
    
    return output

def dividend_yield(tkrs, df, start_date, end_date=None):
    """
    Given a list of tickers, return a dictionary matching the tickers 
    to their dividend yield as a percentage of price. 
    If the dividend yield is not available, then return 0.
    
    Inputs:
    tkrs: List of tickers
    df: DataFrame of stock data

    Outputs:
    dividend_yield: Dictionary of tickers to their dividend yield
    """

    if end_date is None:
        end_date = df.index[-1]

    time_period = df.index[(df.index >= start_date) & (df.index <= end_date)]
    
    dividend_yield = {}
    
    for t in tkrs:
        try:
            dividend_yield = df[t]['Dividends'][time_period] / df[t]['Close'][time_period]
            dividend_yield[t] = dividend_yield.mean()
        except:
            dividend_yield[t] = 0

    return dividend_yield

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