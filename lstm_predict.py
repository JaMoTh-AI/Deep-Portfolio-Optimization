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


""" While there are 3 functions in this file, the main one is predict which takes
    a start date and a timestep and predicts using the LSTM models we have trained.
    It makes predictions on all stocks with LSTM available"""

# The validation loop given a criterion and dataloader
def val_on_stock(model, val_dataloader, criterion, device):
  # Get val loss
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

def evaluate(ticker, dataloader, dataset, model_dir, start_date, device, num_layers, input_size, hidden_size):
    # Get path to model first
    path_to_model = os.path.join(model_dir, ticker +".pt")
    
    # Load the model for the ticker
    model = LSTM(num_layers, input_size, hidden_size, device).to(device)
    state_dict = torch.load(path_to_model, map_location=device)
    model.load_state_dict(state_dict)

    # Get the variance 
    ticker_variance = val_on_stock(model, dataloader, nn.MSELoss(), device)

    # Get the prediction one away
    X = dataset.get_X_from_date(start_date).to(device)
    ticker_pred = model(X).item()

    return {"var":ticker_variance, "pred":ticker_pred} 

def predict(timestep, start_date):
    """
    Inputs:
        timestep: d, w, m, y for day, week, month and year
              - note that the year predictor performs very poorely
        start_date: a string that needs to be in the form of 'yyyy-mm-dd hh:mm:00-05:00'
        when timestep input is d. It needs to be in 'yyyy-mm-dd' when timestep 
        input is anything else.
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
        stock_start_date = start_date - relativedelta(years=2) + relativedelta(months=5)
        
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

    output = {}

    # Iterate through all tickers and obtain predictions as well as variance
    for ticker, df in tqdm(full_stock_data.items(), desc="Predicting"):
        try:
            df = (df-means[ticker])/std_devs[ticker] # Normalize

            # Create dataset
            dataset = ValTimeSeriesDataset(ticker, df, n_lags, forecast_horizon)
            dataloader = DataLoader(dataset, batch_size=20, shuffle=False)

            ticker_stats = evaluate(ticker, dataloader, dataset, model_dir, start_date, device, num_layers, input_size, hidden_size)
            
            # Denormalize and get percentage change
            init_price = df.loc[start_date]['Close']*std_devs[ticker]['Close']+means[ticker]['Close']
            pred_price = ticker_stats['pred']*std_devs[ticker]['Close']+means[ticker]['Close']
            
            ticker_stats['pred'] = pred_price/init_price
            output[ticker] = ticker_stats
        except:
            print(f"Failed to predict {ticker}")

    # Return all predictions and variances
    return output