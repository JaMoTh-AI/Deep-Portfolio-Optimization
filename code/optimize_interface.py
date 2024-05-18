import streamlit as st
import numpy as np
import cvxpy as cp
import pandas as pd
from portfolio_optimizer import *
from lstm_predict import *
from calc_helpers import *
import yfinance as yf

# Define helper functions
def get_portfolio_details(tickers, stock_data, optimized_shares, total_expected_return, total_risk, total_investment, total_dividend_yield):
    """
    Return a DataFrame containing the details of each asset in the optimized portfolio.

    Arguments:
    tickers (list): List of asset tickers.
    stock_data (dict): Dictionary of stock data with expected returns, risk, and price.
    optimized_shares (list): List of the number of shares for each asset.
    total_expected_return (float): Total expected return of the portfolio.
    total_risk (float): Total risk of the portfolio.
    total_investment (float): Total amount of money spent in the portfolio.
    """
    data = []
    total_shares = 0
    for i, ticker in enumerate(tickers):
        shares = optimized_shares[i]
        price = stock_data[ticker]['price']
        expected_return = stock_data[ticker]['expected_returns'] * 100  # Convert to percentage
        risk = stock_data[ticker]['risk']
        dividend = stock_data[ticker]['dividend_yield'] * 100  # Convert to percentage
        data.append([ticker, shares, price, expected_return, risk, dividend])
        total_shares += shares
    
    data.append(['Total', total_shares, total_investment, total_expected_return * 100, total_risk, total_dividend_yield * 100])
    df = pd.DataFrame(data, columns=['Ticker', 'Shares', 'Price ($)', 'Expected Return (%)', 'Risk', 'Dividend Yield (%)'])
    df = df.round(2)

    return df


def display_share_constraints(min_shares, max_shares):
    """
    Display the current share constraints as a DataFrame in Streamlit.

    Arguments:
    min_shares (dict): Dictionary of minimum shares constraints.
    max_shares (dict): Dictionary of maximum shares constraints.
    """
    constraints_data = []
    for ticker in set(min_shares.keys()).union(max_shares.keys()):
        min_share = min_shares.get(ticker, 'N/A')
        max_share = max_shares.get(ticker, 'N/A')
        constraints_data.append([ticker, min_share, max_share])
    
    constraints_df = pd.DataFrame(constraints_data, columns=['Ticker', 'Min Shares', 'Max Shares'])
    if len(constraints_df) > 0:
        st.subheader("Current Share Constraints")
        st.dataframe(constraints_df)


def display_sector_constraints(min_sector, max_sector):
    """
    Display the current sector constraints as a DataFrame in Streamlit.

    Arguments:
    min_sector (dict): Dictionary of minimum sector constraints.
    max_sector (dict): Dictionary of maximum sector constraints.
    """
    constraints_data = []
    for ticker in set(min_sector.keys()).union(max_sector.keys()):
        min_sec = min_sector.get(sector, 'N/A')
        max_sec = max_sector.get(sector, 'N/A')
        constraints_data.append([sector, min_sec, max_sec])
    
    constraints_df = pd.DataFrame(constraints_data, columns=['Sector', 'Min Percentage', 'Max Percentage'])
    if len(constraints_df) > 0:
        st.subheader("Current Sector Constraints")
        st.dataframe(constraints_df)

def get_fake_stock_data():
    """
    Return dictionary of fake stock data for testing purposes.
    """
    stock_data = {
    'AAPL': {
        'price': 139.24,
        'expected_returns': 0.0574,
        'risk': 0.0384,
        'sector': 'Technology',
        'dividend_yield': 0.0351
    },
    'MSFT': {
        'price': 211.56,
        'expected_returns': 0.0742,
        'risk': 0.0273,
        'sector': 'Technology',
        'dividend_yield': 0.0213
    },
    'GOOGL': {
        'price': 1456.78,
        'expected_returns': 0.0675,
        'risk': 0.0421,
        'sector': 'Communication Services',
        'dividend_yield': 0.0342
    },
    'AMZN': {
        'price': 3156.21,
        'expected_returns': 0.0819,
        'risk': 0.0376,
        'sector': 'Consumer Discretionary',
        'dividend_yield': 0.0284
    },
    'FB': {
        'price': 265.78,
        'expected_returns': 0.0598,
        'risk': 0.0312,
        'sector': 'Communication Services',
        'dividend_yield': 0.0294
    }
    }
    return stock_data


# Initialize session states
if 'min_shares' not in st.session_state:
    st.session_state.min_shares = {}
if 'max_shares' not in st.session_state:
    st.session_state.max_shares = {}
if 'min_sector' not in st.session_state:
    st.session_state.min_sector = {}
if 'max_sector' not in st.session_state:
    st.session_state.max_sector = {}
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = False
if 'portfolio_details' not in st.session_state:
    st.session_state.portfolio_details = None
if 'tickers' not in st.session_state:
    st.session_state.tickers = []


# Page title
st.title("Portfolio Optimization")
st.markdown("""
**Portfolio optimization with a deep learning backbone**  
Jack Jansons (jcj59), Rob Mosher (ram487), Raphael Thesmar (rft38)  
CS 4701 Spring 2024 Final Project

""")

st.markdown("""
**How to get started**
1. Select a start date for model predictions. The start date is the date from which expected returns and risk predictions are generated. This date is converted to the closest valid trading day.
2. Select a time interval for model predictions. The time interval is the frequency at which expected returns and risk predictions are generated.
    - **Daily**: Will generate expected returns predictions for the closing time of the next valid trading day. This would be beneficial for someone trying to buy a portfolio today and selling it tomorrow.
    - **Weekly**: Will generate expected returns predictions for the closing time five valid trading days from the start date (a trading week).
    - **Monthly**: Will generate expected returns predictions for the closing time twenty-one valid trading days from the start date (a trading month).
    - **Yearly**: Will generate expected returns predictions for the closing time two hundred fifty-two valid trading days from the start date (a trading year).
3. Click the "Generate Stock Data" button to generate stock data. This will generate expected returns and risk predictions for the selected time interval. **Note**: Prediction generation may take a few minutes. Please be patient.
4. Once stock data is generated, you can select an optimization objective function. This function defines what aspects of the portfolio you wish to maximize or minimize. See objective function descriptions below.
5. Set optimization parameters for both constraints and the objective function. Optimization parameters include budget, risk-free rate, risk aversion, minimum investment percentage, and minimum expected returns percentage. See parameter descriptions below.
6. Add preferences for individual assets and sectors. Preferences include minimum and maximum shares for individual assets and minimum and maximum percentages for each sector. See descriptions below.
7. Click the "Optimize Portfolio" button to optimize the portfolio. This will generate an optimized portfolio based on the selected objective function and constraints which you can then download as a .csv file.
""")

st.markdown("""
    **Disclamer:** We make no representation or warranties regarding the accuracy of the projections and predicitons contained in this application or its ability to achieve projected results. Any financial projections are estimated. Thus, we recommend consulting with a professional financial advisor before relying on any information provided herein for real investment decisions.
""")


# Select start date and time interval for LSTM predictions
start_date = st.date_input("Select Start Date", value=datetime.today(), max_value=datetime.today())
timestep = st.selectbox(
    "Select Time Interval for Predictions",
    ("d", "w", "m", "y"),
    index=0,
    format_func=lambda x: {'d': 'Daily', 'w': 'Weekly', 'm': 'Monthly', 'y': 'Yearly'}[x]
)


# Generate Stock Data
predict_button = st.button("Generate Stock Data")
if predict_button:
    placeholder = st.empty()
    progress_bar = placeholder.progress(0)
    message = st.empty()
    message.text("Generating stock data... this may take a moment. Please be patient.")
    
    # Generate predictions
    stock_data = predict(timestep, str(start_date), progress_bar)
    #stock_data = get_fake_stock_data() # Use for testing



    # After prediction is complete
    st.session_state.stock_data = stock_data
    st.session_state.tickers = stock_data.keys()
    progress_bar.progress(100)
    st.success("Stock data generation complete.")
    

if 'stock_data' in st.session_state:
    # Construct stock data DataFrames
    stock_data = st.session_state.stock_data
    stock_data_df = pd.DataFrame(st.session_state.stock_data).T
    stock_data_df = stock_data_df.rename(columns={'expected_returns': 'Expected Returns (%)', 'risk': 'Risk ($)', 'price': 'Closing Price ($)', 'dividend_yield': 'Dividend Yield (%)', 'sector': 'Sector'})
    
    # Convert to percentage
    stock_data_df['Expected Returns (%)'] = stock_data_df['Expected Returns (%)'] * 100  
    stock_data_df['Dividend Yield (%)'] = stock_data_df['Dividend Yield (%)'] * 100

    # Convert to ($) Units
    stock_data_df['Risk ($)'] = pd.to_numeric(stock_data_df['Risk ($)'] * stock_data_df['Closing Price ($)']).round(2)

    # Display stock data
    columns=['Ticker', 'Min Shares', 'Max Shares']
    st.subheader("Stock Data")
    st.markdown("""
    Listed below are the possible stocks that can be included in the portfolio. The table includes the expected returns, risk, closing price of valid trading day closest to start day, dividend yield, and sector for each asset. 
    Expected returns and risk 
    """)
    st.dataframe(stock_data_df)


    ### Portfolio Optimization
    st.subheader("Set Optimization Parameters")

    # Select objective function
    st.markdown("""
    **Objective Functions:**
    1. **Minimize Risk**: Minimize the total risk of the portfolio. Total risk is calculated as sum of the number of shares multiplied by the variance in ($) for each asset. 
       - For this objective function, it is critical to set the minimum investment percentage.
       - The risk-free rate and risk aversion coefficient are not used.
    2. **Maximize Sharpe Ratio**: Maximize the Sharpe Ratio of the portfolio.
       - Typically, Sharpe Ratio is calculated as (Expected Return - Risk-Free Rate) / Total Risk. This is the return to risk ratio.
       - Here, Sharpe Ratio is calculated as the (Expected Return - Risk-Free Rate) - Risk Aversion Coefficient * Total Risk to ensure linearity while maintaining the reward to risk tradeoff.
       - The risk-free rate and risk aversion coefficient are required in this objective function.
       - **Note:** If the risk-free rate and risk aversion coefficient are set to 0, the objective function is equivalent to maximizing expected returns.
    3. **Maximize Dividend Yield**: Maximize the total dividend yield of the portfolio.
       - The dividend yield is calculated as the dividend yield in ($) of each asset multiplied by the number of shares - risk aversion coefficient multiplied the total risk.
       - The risk-free rate is not used.
    """)
    objective_function = st.selectbox(
        "Select Objective Function",
        ("Minimize Risk", "Maximize Sharpe Ratio", "Maximize Dividend Yield")
    )

    # Input constraint parameters
    budget = st.number_input("Enter Portfolio Budget", min_value=0.0, value=10000.0)
    st.markdown("""**Note:** Risk Free Rate is taken from the current 3-month US Treasury Bill rate""")
    risk_free_rate = st.number_input("Enter Risk-free Rate", min_value=0.0, value=get_risk_free_rate())
    st.markdown("""**Note:** Risk Aversion Coefficient is a measure of the investor's risk tolerance. A higher value indicates a lower risk tolerance. A value of 0 indicates no risk aversion.""")
    risk_aversion = st.number_input("Enter Risk Aversion Coefficient", min_value=0.0, value=0.0)
    st.markdown("""**Note:** Minimum Investment Percentage is the minimum percentage of the budget that must be invested in the portfolio. This parameter is necessary for the minimum risk objective function to avoid having an empty portfolio.""")
    min_investment = st.number_input("Enter Minimum Investment Percentage", min_value=0.0, max_value=1.0, value=0.8)
    st.markdown("""**Note:** Minimum Expected Returns Percentage is the minimum expected returns percentage that the portfolio must achieve.""")
    min_returns = st.number_input("Enter Minimum Expected Returns Percentage", value=0.0)

    # Define the selected objective function
    if objective_function == "Minimize Risk":
        objective_fun = minimize_risk
    elif objective_function == "Maximize Sharpe Ratio":
        objective_fun = maximize_sharpe_ratio
    elif objective_function == "Maximize Dividend Yield":
        objective_fun = maximize_dividend_yield

    # Input min and max shares for each asset
    st.subheader("Select Which Assets to Include")
    st.markdown("""
    Select which assets you want to require in your portfolio. You can set a minimum and maximum number of shares for each asset. To add an asset:
    1. Select the ticker symbol from the dropdown.
    2. Set the minimum number of shares of the asset required in the portfolio.
    3. Set the maximum number of shares of the asset allowed in the portfolio.
    4. Click the "Add Stock" button.  
    In order to remove an asset, select the stock ticker in the "Ticker Symbol to Remove" dropdown and click the "Remove Stock" button.
    """)
    ticker = st.selectbox("Ticker Symbol", sorted(st.session_state.tickers))
    min_share = st.number_input("Minimum Shares", min_value=0, value=0)
    max_share = st.number_input("Maximum Shares", min_value=0, value=100)

    if st.button("Add Stock"):
        st.session_state.min_shares[ticker] = min_share
        st.session_state.max_shares[ticker] = max_share
        st.success(f"Constraint added for {ticker}: Min {min_share}, Max {max_share}")
        
    # Remove constraints for a ticker
    st.subheader("Remove Asset for Individual Assets")
    remove_ticker = st.selectbox("Ticker Symbol to Remove", sorted(st.session_state.max_shares.keys()))
    
    if st.button("Remove Stock"):
        if remove_ticker in st.session_state.max_shares or remove_ticker in st.session_state.min_shares:
            st.success(f"Constraint removed for {remove_ticker}")
        if remove_ticker in st.session_state.min_shares:
            del st.session_state.min_shares[remove_ticker]
        if remove_ticker in st.session_state.max_shares:
            del st.session_state.max_shares[remove_ticker]
        if remove_ticker not in stock_data:
            st.error(f"Ticker {remove_ticker} not found in the stock data")

    # Display current constraints
    display_share_constraints(st.session_state.min_shares, st.session_state.max_shares)


    # Input min and max percentage for each sector
    st.subheader("Select Which Sectors to Include")
    st.markdown("""
    Select which sectors you want to require in your portfolio. You can set a minimum and maximum percentage of the portfolio budget for each sector. To add a sector:
    1. Select the sector from the dropdown.
    2. Set the minimum percentage of the budget required in the portfolio.
    3. Set the maximum percentage of the budget allowed in the portfolio.
    4. Click the "Add Sector" button.  
    In order to remove a sector, select the sector in the "Select Sector to Remove" dropdown and click the "Remove Sector" button.
    """)
    sector = st.selectbox("Select Sector", sorted(stock_data_df['Sector'].unique()))
    min_sector = st.number_input("Minimum Percentage", min_value=0.0, max_value = 1.0, value=0.0)
    max_sector = st.number_input("Maximum Percentage", min_value=0.0, max_value = 1.0, value=1.0)

    if st.button("Add Sector"):
        st.session_state.min_sector[sector] = min_sector
        st.session_state.max_sector[sector] = max_sector
        st.success(f"Constraint added for {sector}: Min {min_sector}, Max {max_sector}")
        
    # Remove constraints for a ticker
    st.subheader("Remove Sector")
    remove_sector = st.selectbox("Select Sector to Remove", sorted(st.session_state.max_sector.keys()))
    
    if st.button("Remove Sector"):
        if remove_sector in st.session_state.max_sector or remove_sector in st.session_state.min_sector:
            st.success(f"Constraint removed for {remove_sector}")
        if remove_sector in st.session_state.min_sector:
            del st.session_state.min_sector[remove_sector]
        if remove_sector in st.session_state.max_sector:
            del st.session_state.max_sector[remove_sector]
        if remove_sector not in stock_data:
            st.error(f"Sector {remove_sector} not found in constraints")

    # Display current constraints
    display_sector_constraints(st.session_state.min_sector, st.session_state.max_sector)

    # Define the constraints functions
    constraints = sequential_constraints([
        budget_constraint,
        minimum_investment_constraint,
        minimum_returns,
        assets_constraint,
        sector_allocation_constraint
    ])
    
    # Optimize portfolio
    if st.button("Optimize Portfolio"):
        try:
            portfolio, expected_return, risk, total_investment, total_dividend_yield = optimize_portfolio(
                stock_data,
                objective_fun,
                constraint_fun=constraints,
                budget=budget,
                risk_free_rate=risk_free_rate,
                risk_aversion=risk_aversion,
                min_investment=min_investment,
                min_returns=min_returns,
                min_shares=st.session_state.min_shares,
                max_shares=st.session_state.max_shares,
                min_sector=st.session_state.min_sector,
                max_sector=st.session_state.max_sector
            )
            st.success("Portfolio optimization successful!")

            # Display portfolio details and return the dataframe
            st.session_state.portfolio = True
            st.session_state.portfolio_details = get_portfolio_details(list(portfolio.keys()), stock_data, np.array(list(portfolio.values())), expected_return, risk, total_investment, total_dividend_yield)
        except Exception as e:
            st.error(f"Optimization failed: {e}")

    # Display portfolio details if available
    if st.session_state.portfolio:
        st.subheader("Optimized Portfolio Details")
        st.markdown("""
        Listed below are the details of the optimized portfolio. The table includes the ticker, number of shares, closing price, expected returns, risk, and dividend yield for each asset. The total expected return, total risk, total investment, and total dividend yield for the portfolio are also included.
        """)
        st.dataframe(st.session_state.portfolio_details)

        # Add a download button
        csv = st.session_state.portfolio_details.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Portfolio as CSV",
            data=csv,
            file_name='optimized_portfolio.csv',
            mime='text/csv',
        )

    
