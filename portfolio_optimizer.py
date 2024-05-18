import pandas as pd
import numpy as np
import gurobipy
import cvxpy as cp
import matplotlib.pyplot as plt
from datetime import datetime


### Optimization Function (Integer Programming)
# Returns an optimal portfolio given an objective value and constraints.
# Portfolios are in the form of a dictionary with tickers as keys and number of shares to buy as values.
def optimize_portfolio(stock_data, objective_fun, constraint_fun=None, **kwargs):
    """
    Optimize a portfolio of assets based on specified objectives and constraints.

    Arguments:
    stock_data (dict): Dictionary of stock tickers with tuples: (expected returns, risk, current price).
    objective_fun (callable): Function to minimize that takes weights and stock data.
    constraint_fun (list): Function that takes in weights, stock data, and budget and returns a list of additional constraints.
    **kwargs:  Used to pass additional parameters directly to the objective and constraints functions.

    Returns:
    portfolio (dict): Dictionary of stock tickers mapped to the number of shares to buy.
    """
    portfolio = {}
    num_assets = len(stock_data)
    tickers = list(stock_data.keys())

    # Define share variables
    variables = {}
    shares = cp.Variable(num_assets, integer=True)  # Create a vector of variables for the weights of each ticker in the portfolio

    # Define constraints
    constraints = []
    constraints += [shares >= 0]  # Basic problem constraints

    if constraint_fun is not None:  # Add additional constraints if necessary
        constraints += constraint_fun(shares, stock_data, **kwargs)

    # Define objective function
    objective = objective_fun(shares, stock_data, **kwargs)

    # Define and solve optimization problem
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.GUROBI)

    # Construct optimal portfolio
    if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        optimized_shares = shares.value
        portfolio = dict(zip(tickers, optimized_shares))

        # Calculate expected return and risk of the optimized portfolio
        expected_returns = np.array([stock_data[ticker]['expected_returns'] for ticker in tickers])
        risks = np.array([stock_data[ticker]['risk'] for ticker in tickers])
        prices = np.array([stock_data[ticker]['price'] for ticker in tickers])
        dividend_yields = np.array([stock_data[ticker]['price'] * stock_data[ticker]['dividend_yield'] for ticker in stock_data.keys()])

        
        budget = kwargs.get('budget')
        total_investment = np.sum(prices * optimized_shares)
        portfolio_return = (expected_returns * prices) @ optimized_shares / total_investment
        portfolio_risk = np.sqrt(np.dot(risks**2, optimized_shares) / total_investment)
        total_dividend_yield = (dividend_yields @ optimized_shares) / total_investment
    else:
        # If optimization infeasible throw error
        raise Exception(f"Optimization failed with status {problem.status}")

    # Keep non-zero tickers
    portfolio = {k: v for k, v in portfolio.items() if v > 0}

    return portfolio, portfolio_return, portfolio_risk, total_investment, total_dividend_yield



#### Objective Functions
def minimize_risk(shares, stock_data, **kwargs):
    """
    Define objective function to minimize risk.

    Arguments:
    shares (list): Number of shares to buy of each stock ticker in portfolio.
    stock_data (dict): Dictionary with stock tickers as keys and dictionaries as values.

    Returns:
    objective (expression): CVXPY objective function to optimize.
    """
    # Calculate risks for each asset
    risks = np.array([stock_data[ticker]['price'] * stock_data[ticker]['risk'] for ticker in stock_data.keys()])

    # Define objective function
    objective = cp.Minimize(risks @ shares)

    return objective


def maximize_sharpe_ratio(shares, stock_data, **kwargs):
    """
    Define objective function to maximize sharpe ratio. Sharpe ratio with risk aversion 
    of 0 and risk free rate of 0 maximizes expected returns. Sharpe ratio with risk free 
    rate of 0 maximizes return to risk ratio.

    Arguments:
    shares (list): Number of shares to buy of each stock ticker in portfolio.
    stock_data (dict): Dictionary with stock tickers as keys and dictionaries as values.
    risk_free_rate (float): Risk-free rate of return (typically US Treasury Bill rate).
    risk_aversion (float): Risk aversion coefficient.
    budget (float): Maximum amount of money that can be spent.

    Returns:
    objective (expression): CVXPY objective function to optimize.
    """
    # Get keyword arguments
    risk_free_rate = kwargs.get('risk_free_rate', 0)
    risk_aversion = kwargs.get('risk_aversion', 0)
    budget = kwargs.get('budget')

    # Calculate expected returns and variances for each asset
    expected_returns = np.array([stock_data[ticker]['price'] * (stock_data[ticker]['expected_returns'] - risk_free_rate) for ticker in stock_data.keys()])
    variances = np.array([(stock_data[ticker]['price'] * stock_data[ticker]['risk']) ** 2 for ticker in stock_data.keys()])

    # Calculate portfolio variances and returns
    portfolio_returns = expected_returns @ shares
    portfolio_variances = variances @ shares

    # Modified Sharpe Ratio objective function to ensure convexity
    sharpe_ratio = (portfolio_returns) - (risk_aversion * portfolio_variances)
    objective = cp.Maximize(sharpe_ratio)

    return objective


def maximize_dividend_yield(shares, stock_data, **kwargs):
    """
    Define objective function to maximize dividend yields.

    Arguments:
    shares (list): Number of shares to buy of each stock ticker in portfolio.
    stock_data (dict): Dictionary with stock tickers as keys and dictionaries as values.
    risk_free_rate (float): Risk-free rate of return (typically US Treasury Bill rate).
    risk_aversion (float): Risk aversion coefficient.
    budget (float): Maximum amount of money that can be spent.

    Returns:
    objective (expression): CVXPY objective function to optimize.
    """
    # Get keyword arguments
    risk_aversion = kwargs.get('risk_aversion', 0)
    budget = kwargs.get('budget')

    # Calculate dividend yields and variances for each asset
    dividend_yields = np.array([stock_data[ticker]['price'] * stock_data[ticker]['dividend_yield'] for ticker in stock_data.keys()])
    variances = np.array([(stock_data[ticker]['price'] * stock_data[ticker]['risk']) ** 2 for ticker in stock_data.keys()])

    # Calculate portfolio dividend returns and variances
    portfolio_dividends = dividend_yields @ shares
    portfolio_variances = variances @ shares

    # Define objective function
    objective = cp.Maximize(portfolio_dividends - risk_aversion * portfolio_variances)

    return objective


#### Constraint Functions
def sequential_constraints(constraint_funs):
    """
    Combines a list of constraints functions. Similar tp nn.Sequential.

    Argument:
    constraint_funs (list): List of constraint functions.

    Returns:
    constraint_fun (function): Callable constraints function
    """
    return lambda shares, stock_data, **kwargs: sequential_constraints_helper(constraint_funs, shares, stock_data, **kwargs)


def sequential_constraints_helper(constraint_funs, shares, stock_data, **kwargs):
    """
    Helper function which combines a list of constraints functions.

    Arguments:
    constraints_funs (list): List of constraint functions.
    shares (list): Number of shares to buy of each stock ticker in portfolio.
    stock_data (dict): Dictionary with stock tickers as keys and dictionaries as values.
    **kwargs:  Used to pass additional parameters directly to the constraints function.

    Returns:
    constraints (list): A list of constraints.
    """
    constraints = []

    # Loop through all constraint functions
    for constraint_fun in constraint_funs:
        constraints += constraint_fun(shares, stock_data, **kwargs)

    return constraints


def budget_constraint(shares, stock_data, **kwargs):
    """
    Ensure portfolio is feasible with given budget.

    Arguments:
    shares (list): Number of shares to buy of each stock ticker in portfolio.
    stock_data (dict): Dictionary with stock tickers as keys and dictionaries as values.
    budget (float): Total budget to be invested in the portfolio.

    Returns:
    constraint (list): List containing the constraints.
    """
    # Get keyword arguments
    budget = kwargs.get('budget')

    prices = np.array([stock_data[ticker]['price'] for ticker in stock_data.keys()])
    total_cost = shares @ prices
    return [total_cost <= budget]


def minimum_investment_constraint(shares, stock_data, **kwargs):
    """
    Ensure portfolio is feasible with given budget.

    Arguments:
    shares (list): Number of shares to buy of each stock ticker in portfolio.
    stock_data (dict): Dictionary with stock tickers as keys and dictionaries as values.
    budget (float): Total budget to be invested in the portfolio.
    min_investment (float): Percentage indicating minimum percentage of budget to utilize.

    Returns:
    constraint (list): List containing the budget constraint.
    """
    # Get keyword arguments
    budget = kwargs.get('budget')
    min_investment = kwargs.get('min_investment', 0)

    prices = np.array([stock_data[ticker]['price'] for ticker in stock_data.keys()])
    total_cost = shares @ prices
    return [total_cost >= budget * min_investment]


def minimum_returns(shares, stock_data, **kwargs):
    """
    Define objective function to maximize returns.

    Arguments:
    shares (list): Number of shares to buy of each stock ticker in portfolio.
    stock_data (dict): Dictionary with stock tickers as keys and dictionaries as values.
    min_returns (float): Percentage indicating minimum desired expected returns.
    budget (float): Total budget to be invested in the portfolio.

    Returns:
    constraint (list): List containing the constraints.
    """
    # Get keyword arguments
    min_returns = kwargs.get('min_returns', 0)
    budget = kwargs.get('budget')

    expected_returns = np.array([stock_data[ticker]['price'] * stock_data[ticker]['expected_returns'] for ticker in stock_data.keys()])
    portfolio_returns = expected_returns @ shares

    return [portfolio_returns / budget >= min_returns]


def assets_constraint(shares, stock_data, **kwargs):
    """
    Ensure portfolio is feasible with given budget.

    Arguments:
    shares (list): Number of shares to buy of each stock ticker in portfolio.
    stock_data (dict): Dictionary with stock tickers as keys and dictionaries as values.
    min_shares (dict): Dictionary with stock tickers as keys and minimum number of shares to buy as values.
    max_shares (dict): Dictionary with stock tickers as keys and maximum number of shares to buy as values.

    Returns:
    constraint (list): List containing the constraints.
    """
    min_shares = kwargs.get('min_shares', {})
    max_shares = kwargs.get('max_shares', {})
    
    constraints = []
    for idx, ticker in enumerate(stock_data.keys()):
        min_share = min_shares.get(ticker, 0)
        max_share = max_shares.get(ticker, 100)
        
        constraints.append(shares[idx] >= min_share)
        constraints.append(shares[idx] <= max_share)

    return constraints


def sector_allocation_constraint(shares, stock_data, **kwargs):
    """
    Ensure portfolio is feasible with given budget.

    Arguments:
    shares (list): Number of shares to buy of each stock ticker in portfolio.
    stock_data (dict): Dictionary with stock tickers as keys and dictionaries as values.
    min_sector (dict): Dictionary of minimum sector constraints.
    max_sector (dict): Dictionary of maximum sector constraints.
    budget (float): Total budget to be invested in the portfolio.

    Returns:
    constraint (list): List containing the constraints.
    """
    # Get keyword arguments
    min_sector = kwargs.get('min_sector', {})
    max_sector = kwargs.get('max_sector', {})
    budget = kwargs.get('budget')

    constraints = []
    for sector in min_sector.keys():
        sector_shares = np.array([shares[idx] for idx, ticker in enumerate(stock_data.keys()) if stock_data[ticker]['sector'] == sector])
        sector_prices = np.array([stock_data[ticker]['price'] for ticker in stock_data.keys() if stock_data[ticker]['sector'] == sector])
        constraints.append(sector_shares @ sector_prices <= budget*max_sector[sector])
        constraints.append(sector_shares @ sector_prices >= budget*min_sector[sector])
    return constraints
