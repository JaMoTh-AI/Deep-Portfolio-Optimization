from data_utils import pull_stock_data, pull_stock_indicators, cleaning_data
from lstm_datasets import ValTimeSeriesDataset
from lstm import LSTM
import torch
import torch.nn as nn
import os
from datetime import timedelta
from dateutil.relativedelta import relativedelta
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from calc_helpers import sectors, dividend_yield
import math


"""
    This file contains functions for evaluating and predicting stock prices using pre-trained LSTM models.
    The main function is `predict`, which takes a start date and a timestep and makes predictions using the LSTM models.
    It makes predictions on all stocks with available LSTM models.
"""

def val_on_stock(model, val_dataloader, criterion, device):
    """
    Evaluate the model on validation data.

    Parameters:
    model (nn.Module): The LSTM model to evaluate.
    val_dataloader (DataLoader): DataLoader for the validation dataset.
    criterion (nn.Module): Loss function.
    device (torch.device): Device to run the evaluation on.

    Returns:
    float: The average validation loss.
    """
    val_running_loss = 0
    tot = 0

    model.eval()
    with torch.no_grad():
        for i, (X,y) in enumerate(val_dataloader):
            labels = y.view(-1, 1).to(device)
            inputs = X.to(device)

            outputs = model(inputs) # Forward path

            loss = criterion(outputs, labels) # loss function applied

            val_running_loss += loss.item()
            tot += 1

    return val_running_loss/tot

def find_closest_date_index(target_date, df):
    """
    Finds the index in the DataFrame that's closest to the given date string.

    Parameters:
    date_str (str): The date string to compare, e.g., '2023-01-01'.
    df (pd.DataFrame): The DataFrame with DatetimeIndex.

    Returns:
    closest_index: The index in the DataFrame that's closest to the given date.
    """
    # Convert the date string to a datetime object
    if df.index.name=="Datetime":
        if target_date.tzinfo is None:
            target_date = target_date.tz_localize('America/New_York')
        else:
            target_date = target_date.tz_convert('America/New_York')
    else:
        target_date = target_date.tz_localize(None)


    # Calculate the absolute difference between the target date and each index
    time_diffs = (df.index - target_date).map(abs)

    # Find the index with the minimum difference
    closest_index = time_diffs.argmin()

    return closest_index

def evaluate(ticker, dataloader, dataset, model_dir, start_date, device, num_layers, input_size, hidden_size):
    """
    Evaluate a specific stock model and make a prediction.

    Parameters:
    ticker (str): Stock ticker symbol.
    dataloader (DataLoader): DataLoader for the dataset.
    dataset (Dataset): The dataset object.
    model_dir (str): Directory where the model weights are stored.
    start_date (str): Start date for the evaluation.
    device (torch.device): Device to run the evaluation on.
    num_layers (int): Number of LSTM layers.
    input_size (int): Input size for the LSTM model.
    hidden_size (int): Hidden size for the LSTM model.

    Returns:
    dict: Dictionary containing the variance and prediction.
    """
    # Load the model for the specified ticker
    path_to_model = os.path.join(model_dir, ticker +".pt")
    model = LSTM(num_layers, input_size, hidden_size, device).to(device)
    state_dict = torch.load(path_to_model, map_location=device)
    model.load_state_dict(state_dict)

    # Calculate the validation loss
    ticker_variance = val_on_stock(model, dataloader, nn.MSELoss(), device)

    # Make a prediction for the specified start date
    X = dataset.get_X_from_date(start_date).to(device)
    ticker_pred = model(X).item()

    return {"var":ticker_variance, "pred":ticker_pred} 

def predict(timestep, start_date, progress_bar=None):
    """
    Predict stock prices using pre-trained LSTM models.

    Parameters:
    timestep (str): Time step for prediction ('d' for day, 'w' for week, 'm' for month, 'y' for year).
    start_date (str): Start date in the format 'yyyy-mm-dd hh:mm:00-05:00' for daily predictions,
                      or 'yyyy-mm-dd' for other time steps. Make sure that the date/time is present in the 
                      stock data

    Returns:
    dict: Dictionary containing final price, predictions and variances for each stock.
    """
    # Hyperparams we used to train the model
    num_layers, input_size, hidden_size = 4, 17, 100

    # Get the hardware to run torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Changing the start date to a pandas datetime
    start_date = pd.to_datetime(start_date)

    # Set up times based one which timestep input was chosen
    if timestep=="d":
        n_lags = 8 * 30 * 1
        forecast_horizon = 8 * 1
        interval = "1h"
        
        stock_end_date = start_date + timedelta(days=1)
        stock_start_date = start_date - relativedelta(years=1)
        
        end = str(stock_end_date.date())
        start = str(stock_start_date.date())

        means = pd.read_csv('data_means_stds/train_hourly_mean.csv', index_col=0)
        std_devs = pd.read_csv('data_means_stds/train_hourly_std_dev.csv', index_col=0)

        model_dir = 'LSTMs/LSTM_weights_hourly_day/LSTM_weights'
    elif timestep=="w":
        n_lags = 12*21
        forecast_horizon = 5
        interval = "1d"
        
        stock_end_date = start_date + timedelta(days=1)
        stock_start_date = start_date - relativedelta(years=10)
        
        end = str(stock_end_date.date())
        start = str(stock_start_date.date())

        means = pd.read_csv('data_means_stds/train_daily_mean.csv', index_col=0)
        std_devs = pd.read_csv('data_means_stds/train_daily_std_dev.csv', index_col=0)

        model_dir = 'LSTMs/LSTM_weights_day_week/LSTM_weights'
    elif timestep=="m":
        n_lags = 12*21
        forecast_horizon = 21
        interval = "1d"
        
        stock_end_date = start_date + timedelta(days=1)
        stock_start_date = start_date - relativedelta(years=10)
        
        end = str(stock_end_date.date())
        start = str(stock_start_date.date())

        means = pd.read_csv('data_means_stds/train_daily_mean.csv', index_col=0)
        std_devs = pd.read_csv('data_means_stds/train_daily_std_dev.csv', index_col=0)

        model_dir = 'LSTMs/LSTM_weights_day_month/LSTM_weights'
    elif timestep=="y":
        n_lags = 52
        forecast_horizon = 52
        interval = "1wk"
        
        stock_end_date = start_date + timedelta(days=1)
        stock_start_date = start_date - relativedelta(years=10)
        
        end = str(stock_end_date.date())
        start = str(stock_start_date.date())

        means = pd.read_csv('data_means_stds/train_weekly_mean.csv', index_col=0)
        std_devs = pd.read_csv('data_means_stds/train_weekly_std_dev.csv', index_col=0)

        model_dir = 'LSTMs/LSTM_weights_week_year/LSTM_weights'
    else:
        raise ValueError("This timestep is not supported")
    
    # Get path to models 
    models = os.listdir(model_dir)
    tickers = list(map(lambda x:x.split(".")[0], models))

    # Get data
    full_stock_data = pull_stock_data(tickers, start, end, interval)
    full_stock_data = pull_stock_indicators(full_stock_data)
    full_stock_data = cleaning_data(full_stock_data)

    # Get the divident yeilds and the sectors
    stock_sectors = sectors(tickers)
    stock_dividend_yields = dividend_yield(tickers, start_date)

    output = {}

    # Iterate through all tickers and obtain predictions as well as variance
    i = 0
    for ticker, df in tqdm(full_stock_data.items(), desc="Predicting"):
        try:
            df = (df-means[ticker])/std_devs[ticker] # Normalize
            
            # Find date in df as close as possible to start_date
            stock_start_date = df.index[find_closest_date_index(start_date, df)]

            # Create dataset
            dataset = ValTimeSeriesDataset(ticker, df, n_lags, forecast_horizon)
            dataloader = DataLoader(dataset, batch_size=20, shuffle=False)

            ticker_stats = evaluate(ticker, dataloader, dataset, model_dir, stock_start_date, device, num_layers, input_size, hidden_size)
            
            # Denormalize and get percentage change
            init_price = df.loc[stock_start_date]['Close']*std_devs[ticker]['Close']+means[ticker]['Close']
            pred_price = ticker_stats['pred']*std_devs[ticker]['Close']+means[ticker]['Close']
            
            ticker_stats['expected_returns'] = (pred_price - init_price)/init_price
            ticker_stats['risk'] = math.sqrt(ticker_stats['var'])
            del ticker_stats['var']
            del ticker_stats['pred']
            ticker_stats['price'] = init_price
            ticker_stats['sector'] = stock_sectors[ticker]
            ticker_stats['dividend_yield'] = stock_dividend_yields[ticker]/init_price

            output[ticker] = ticker_stats
        except Exception as e:
            print(e)
            print(f"Failed to predict {ticker}")
        i += 1
        if progress_bar:
            progress_bar.progress(min(1, (i + 1) / len(full_stock_data)))

    # Return all predictions and variances
    return output