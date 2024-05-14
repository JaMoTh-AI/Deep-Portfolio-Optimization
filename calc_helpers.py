def risk_func(tkrs, df, start_date, end_date=None):
    """
    Given a list of tickers, return a dictionary matching 
    the tickers to their risk. Risk can be calculated as the 
    standard deviation of the normalized prices over some period of time.
    Alternative risk calculations can be the range of the high and low 
    points of the stock price in that period of time. 

    Inputs:
    tkrs: List of tickers
    df: DataFrame of stock data
    start_date: Start of the time period
    end_date: End of the time period

    Outputs:
    risk_dict: Dictionary of tickers to their risk, defined as the standard deviation of the normalized closing price from a given day
    """

    if end_date is None:
        end_date = df.index[-1]

    time_period = df.index[(df.index >= start_date) & (df.index <= end_date)]

    risk_dict = {}

    for t in tkrs:
        risk_dict[t] = df[t]['Close'][time_period].std()

    return risk_dict


def expected_returns(tkrs, df, time_period_start, time_period_end=None):
    """
    Given a list of tickers, return a dictionary matching the tickers 
    to their expected return. Expected return can be calculated as the 
    average return over some period of time. For example, if you are 
    trying to calculate the expected annual return, then take the 
    average of the annual returns for a past certain number of years.

    Inputs:
    tkrs: List of tickers
    df: DataFrame of stock data
    time_period_start: Start of the time period
    time_period_end: End of the time period

    Outputs:
    expected_return_dict: Dictionary of tickers to their expected return
    """

    if time_period_end is None:
        time_period_end = df.index[-1]

    time_period = df.index[(df.index >= time_period_start) & (df.index <= time_period_end)]
    expected_return_dict = {}

    for t in tkrs:
        expected_return_dict[t] = df[t][time_period].mean()

    return expected_return_dict


def sectors(tkrs, df, start_date, end_date=None):
    """
    Given a list of tickers, return a dictionary matching the tickers to the sector that they belong to. 
    
    Inputs:
    tkrs: List of tickers
    df: DataFrame of stock data

    Outputs:
    sectors: Dictionary of sectors to the tickers that belong to that sector
    """
    
    if end_date is None:
        end_date = df.index[-1]

    time_period = df.index[(df.index >= start_date) & (df.index <= end_date)]

    sectors = {}
    for t in tkrs:

        t_sector = df[df['Symbol'] == t]['GICS Sector'][time_period].values[0]
        
        if t_sector not in sectors:
            sectors[t_sector] = [t]
        else:
            sectors[t_sector].append(t)

    return sectors


def prices(tkrs, df, start_date, end_date=None):
    """
    Given a list of tickers, return a dictionary matching the tickers to their current price. 
    
    Inputs:
    tkrs: List of tickers
    df: DataFrame of stock data

    Outputs:
    prices: Dictionary of tickers to their current price
    """

    if end_date is None:
        end_date = df.index[-1]
    
    prices = {}
    for t in tkrs:
        prices[t] = df[t]['Close'][end_date]

    return prices

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