import streamlit as st
import numpy as np
import cvxpy as cp
import pandas as pd
from portfolio_optimizer import *

# Define helper functions
def get_portfolio_details(tickers, stock_data, optimized_shares, total_expected_return, total_risk, total_investment):
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
        data.append([ticker, shares, price, expected_return, risk])
        total_shares += shares
    
    data.append(['Total', total_shares, total_investment, total_expected_return * 100, total_risk])
    df = pd.DataFrame(data, columns=['Ticker', 'Shares', 'Price ($)', 'Expected Return (%)', 'Risk'])
    df = df.round(2)

    return df

# Function to display current share constraints as a DataFrame
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

# Initialize session states
if 'min_shares' not in st.session_state:
    st.session_state.min_shares = {}
if 'max_shares' not in st.session_state:
    st.session_state.max_shares = {}
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = False
if 'portfolio_details' not in st.session_state:
    st.session_state.portfolio_details = None

# Page title
st.title("Portfolio Optimization")
st.markdown("""
**Portfolio optimization with a deep learning backbone**  
Jack Jansons (jcj59), Rob Mosher (ram487), Raphael Thesmar (rft38)  
CS 4701 Spring 2024 Final Project

""")

# Upload stock data
uploaded_file = st.file_uploader("Choose a file with stock data", type="csv")

if uploaded_file is not None:
    stock_data_df = pd.read_csv(uploaded_file)
    stock_data = stock_data_df.set_index('ticker').T.to_dict('list')
    for ticker in stock_data:
        stock_data[ticker] = {
            'expected_returns': stock_data[ticker][0],
            'risk': stock_data[ticker][1],
            'price': stock_data[ticker][2]
        }

    st.subheader("Stock Data")
    st.dataframe(stock_data_df)

    st.subheader("Set Optimization Parameters")
    # Select objective function
    objective_function = st.selectbox(
        "Select Objective Function",
        ("Minimize Risk", "Maximize Sharpe Ratio", "Maximize Dividend Yield")
    )

    # Input constraint parameters
    budget = st.number_input("Enter Budget", min_value=0.0, value=10000.0)
    risk_free_rate = st.number_input("Enter Risk-free Rate", min_value=0.0, value=0.0)
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
    ticker = st.text_input("Ticker Symbol")
    min_share = st.number_input("Minimum Shares", min_value=0, value=0)
    max_share = st.number_input("Maximum Shares", min_value=0, value=100)

    if st.button("Add Constraint"):
        if ticker in stock_data:
            st.session_state.min_shares[ticker] = min_share
            st.session_state.max_shares[ticker] = max_share
            st.success(f"Constraint added for {ticker}: Min {min_share}, Max {max_share}")
        else:
            st.error(f"Ticker {ticker} not found in the stock data")

    # Remove constraints for a ticker
    st.subheader("Remove Constraints for Individual Assets")
    remove_ticker = st.text_input("Ticker Symbol to Remove Constraint", key="remove_ticker")
    
    if st.button("Remove Constraint"):
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
            portfolio, expected_return, risk, total_investment = optimize_portfolio(
                stock_data,
                objective_fun,
                constraint_fun=constraints,
                budget=budget,
                risk_free_rate=risk_free_rate,
                risk_aversion=risk_aversion,
                min_investment=min_investment,
                min_returns=min_returns,
                min_shares=st.session_state.min_shares,
                max_shares=st.session_state.max_shares
            )
            st.success("Portfolio optimization successful!")

            # Display portfolio details and return the dataframe
            st.session_state.portfolio = True
            st.session_state.portfolio_details = get_portfolio_details(list(portfolio.keys()), stock_data, np.array(list(portfolio.values())), expected_return, risk, total_investment)
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
