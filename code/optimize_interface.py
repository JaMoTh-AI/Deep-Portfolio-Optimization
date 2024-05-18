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
    message.text("Generating stock data... this may take a moment.")
    
    # Generate predictions
    stock_data = predict(timestep, str(start_date), progress_bar)
    
    # After prediction is complete
    st.session_state.stock_data = stock_data
    st.session_state.tickers = stock_data.keys()
    progress_bar.progress(100)
    message.text("Stock data generation complete.")
    
if 'stock_data' in st.session_state:
    stock_data = st.session_state.stock_data
    stock_data_df = pd.DataFrame(st.session_state.stock_data).T
    st.subheader("Stock Data")
    st.dataframe(stock_data_df)

    st.subheader("Set Optimization Parameters")
    # Select objective function
    st.write("Note: Maximizing Sharpe Ratio with risk free rate and risk aversion of 0 is equivalent to maximizing expected returns.")
    objective_function = st.selectbox(
        "Select Objective Function",
        ("Minimize Risk", "Maximize Sharpe Ratio", "Maximize Dividend Yield")
    )

    # Input constraint parameters
    budget = st.number_input("Enter Budget", min_value=0.0, value=10000.0)
    risk_free_rate = st.number_input("Enter Risk-free Rate", min_value=0.0, value=get_risk_free_rate())
    risk_aversion = st.number_input("Enter Risk Aversion Coefficient", min_value=0.0, value=0.0)
    min_investment = st.number_input("Enter Minimum Investment Percentage", min_value=0.0, max_value=1.0, value=0.8)
    min_returns = st.number_input("Enter Minimum Expected Returns Percentage", value=0.0)

    # Define the selected objective function
    if objective_function == "Minimize Risk":
        objective_fun = minimize_risk
    elif objective_function == "Maximize Sharpe Ratio":
        objective_fun = maximize_sharpe_ratio
    elif objective_function == "Maximize Dividend Yield":
        objective_fun = maximize_dividend_yield

    # Input min and max shares for each asset
    st.subheader("Set Min and Max Shares for Individual Assets")
    ticker = st.selectbox("Ticker Symbol", sorted(st.session_state.tickers))
    min_share = st.number_input("Minimum Shares", min_value=0, value=0)
    max_share = st.number_input("Maximum Shares", min_value=0, value=100)

    if st.button("Add Stock Constraint"):
        st.session_state.min_shares[ticker] = min_share
        st.session_state.max_shares[ticker] = max_share
        st.success(f"Constraint added for {ticker}: Min {min_share}, Max {max_share}")
        
    # Remove constraints for a ticker
    st.subheader("Remove Constraints for Individual Assets")
    remove_ticker = st.selectbox("Ticker Symbol to Remove Constraint", sorted(st.session_state.max_shares.keys()))
    
    if st.button("Remove Stock Constraint"):
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
    st.subheader("Set Min and Max Percentages for Each Sector")
    sector = st.selectbox("Select Sector", sorted(stock_data_df['sector'].unique()))
    min_sector = st.number_input("Minimum Percentage", min_value=0.0, max_value = 1.0, value=0.0)
    max_sector = st.number_input("Maximum Percentage", min_value=0.0, max_value = 1.0, value=1.0)

    if st.button("Add Sector Constraint"):
        st.session_state.min_sector[sector] = min_sector
        st.session_state.max_sector[sector] = max_sector
        st.success(f"Constraint added for {sector}: Min {min_sector}, Max {max_sector}")
        
    # Remove constraints for a ticker
    st.subheader("Remove Constraints for Sector")
    remove_sector = st.selectbox("Select Sector Constraint to Remove", sorted(st.session_state.max_sector.keys()))
    
    if st.button("Remove Sector Constraint"):
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
        st.dataframe(st.session_state.portfolio_details)

        # Add a download button
        csv = st.session_state.portfolio_details.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Portfolio as CSV",
            data=csv,
            file_name='optimized_portfolio.csv',
            mime='text/csv',
        )

    
