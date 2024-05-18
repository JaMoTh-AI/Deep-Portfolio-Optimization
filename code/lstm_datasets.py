# Importing relevant libraries for the datasets
import sklearn.preprocessing as sklp
import torch
from torch.utils.data import Dataset
import numpy as np

"""
This file contains two dataset classes:
    - TimeSeriesDataset: Used for training the LSTM model.
    - ValTimeSeriesDataset: Used for making predictions (e.g., charts) and for the portfolio optimizer.
"""

class TimeSeriesDataset(Dataset):
    """
    Dataset class for training the LSTM models.

    Parameters:
    ticker (str): Stock ticker symbol.
    data (pd.DataFrame): DataFrame containing the stock data.
    n_lags (int): Number of lagged observations to use as input features.
    forecast_horizon (int, optional): Forecast horizon for the target variable. Defaults to 1.
    feature_cols (list, optional): List of feature columns to use. Defaults to None, using all columns.
    label_col (str, optional): Name of the label column. Defaults to 'Close'.
    normalize (bool, optional): Whether to normalize the data. Defaults to False.
    indices (list): List of indices to use for the dataset.
    """

    def __init__(self, ticker, data, n_lags, forecast_horizon=1, feature_cols=None, label_col='Close', normalize=False, indices=None):
        if indices==None:
          raise ValueError("Invalid argument: indices cannot be None.")

        self.ticker = ticker
        self.n_lags = n_lags
        self.forecast_horizon = forecast_horizon
        self.label_col = label_col
        self.indices = indices

        #allow selection of features
        if feature_cols:
          data = data[feature_cols]
        else:
          data = data[data.columns]

        #normalize features
        if normalize:
          self.scaler = sklp.MinMaxScaler()
          data_scaled = self.scaler.fit_transform(data)

        #create usable data from scaled df
        self.data = data

    def create_sequences(self, i):
        """
        Create sequences of features and labels.

        Parameters:
        i (int): Index to start the sequence.

        Returns:
        np.array: Array of features.
        np.array: Label value.
        """
        X = self.data[i:i + self.n_lags]
        y = self.data.iloc[i + self.n_lags + self.forecast_horizon - 1, 3]

        return np.array(X), np.array(y)

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
        int: Length of the dataset.
        """
        return len(self.indices)

    def get_predictions_dates(self):
        """
        Get the dates for which predictions will be made.

        Returns:
        list: List of prediction dates.
        """
        return [i + self.n_lags + self.forecast_horizon - 1 for i in self.indices]

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Parameters:
        idx (int): Index of the sample.

        Returns:
        torch.Tensor: Features tensor.
        torch.Tensor: Label tensor.
        """

        i = self.indices[idx]
        X, y = self.create_sequences(i)
        return torch.tensor(X, dtype=torch.float), torch.tensor(y, dtype=torch.float)
    
class ValTimeSeriesDataset(Dataset):
    """
    Dataset class for making predictions with LSTM models.

    Parameters:
    ticker (str): Stock ticker symbol.
    data (pd.DataFrame): DataFrame containing the stock data.
    n_lags (int): Number of lagged observations to use as input features.
    forecast_horizon (int, optional): Forecast horizon for the target variable. Defaults to 1.
    feature_cols (list, optional): List of feature columns to use. Defaults to None, using all columns.
    label_col (str, optional): Name of the label column. Defaults to 'Close'.
    indices (list, optional): List of indices to use for the dataset. If None, uses the entire range of the data.
    """
    def __init__(self, ticker, data, n_lags, forecast_horizon=1, feature_cols=None, label_col='Close', indices=None):
        if indices==None:
          indices = list(range(len(data)-n_lags-forecast_horizon+1))

        self.ticker = ticker
        self.n_lags = n_lags
        self.forecast_horizon = forecast_horizon
        self.label_col = label_col
        self.indices = indices

        #allow selection of features
        if feature_cols:
          data = data[feature_cols]
        else:
          data = data[data.columns]

        #create usable data from scaled df
        self.data = data

    def create_sequences(self, i):
        """
        Create sequences of features and labels.

        Parameters:
        i (int): Index to start the sequence.

        Returns:
        np.array: Array of features.
        np.array: Label value.
        """
        X = self.data[i:i + self.n_lags]
        y = self.data.iloc[i + self.n_lags + self.forecast_horizon - 1, 3]

        return np.array(X), np.array(y)

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
        int: Length of the dataset.
        """
        return len(self.indices)
        
    def get_dates(self, i):
        """
        Get the start and end dates for a sequence.

        Parameters:
        i (int): Index of the sequence.

        Returns:
        tuple: Start and end dates of the sequence.
        """
        from_date = self.data.index[i + self.n_lags-1]
        to_date = self.data.index[i + self.n_lags + self.forecast_horizon - 1]
        return from_date, to_date

    def get_X_from_date(self, date):
        """
        Get the feature sequence for a given date.

        Parameters:
        date (str): Date to get the feature sequence for.

        Returns:
        torch.Tensor: Features tensor.
        """
        row_number = self.data.index.get_loc(date)
        rows = self.data.iloc[row_number-self.n_lags+1:row_number + 1]
        X = np.array(rows)
        return torch.tensor(X, dtype=torch.float).unsqueeze(0)
        
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Parameters:
        idx (int): Index of the sample.

        Returns:
        torch.Tensor: Features tensor.
        torch.Tensor: Label tensor.
        """
        i = self.indices[idx]
        X, y = self.create_sequences(i)
        return torch.tensor(X, dtype=torch.float), torch.tensor(y, dtype=torch.float)