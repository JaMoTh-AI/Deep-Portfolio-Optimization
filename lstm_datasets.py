# Importing relevant libraries for the datasets
import sklearn.preprocessing as sklp
import torch
from torch.utils.data import Dataset
import numpy as np

""" This files contains two datasets:
        - TimeSeriesDataset used for the training of our LSTM model
        - ValTimeSeriesDataset used for making predictions, ie. charts and for 
        the portfolio optimizer
"""

# The main dataset class used for training the lSTMs
class TimeSeriesDataset(Dataset):
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
        X = self.data[i:i + self.n_lags]
        y = self.data.iloc[i + self.n_lags + self.forecast_horizon - 1, 3]

        return np.array(X), np.array(y)

    def __len__(self):
        return len(self.indices)

    def get_predictions_dates(self):
       return [i + self.n_lags + self.forecast_horizon - 1 for i in self.indices]

    def __getitem__(self, idx):
        i = self.indices[idx]
        X, y = self.create_sequences(i)
        return torch.tensor(X, dtype=torch.float), torch.tensor(y, dtype=torch.float)
    
# The dataset class used for our stocks predictions
class ValTimeSeriesDataset(Dataset):
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
        X = self.data[i:i + self.n_lags]
        y = self.data.iloc[i + self.n_lags + self.forecast_horizon - 1, 3]

        return np.array(X), np.array(y)

    def __len__(self):
        return len(self.indices)
        
    def get_dates(self, i):
        from_date = self.data.index[i + self.n_lags-1]
        to_date = self.data.index[i + self.n_lags + self.forecast_horizon - 1]
        return from_date, to_date

    def get_X_from_date(self, date):
        row_number = self.data.index.get_loc(date)
        rows = self.data.iloc[row_number-self.n_lags+1:row_number + 1]
        X = np.array(rows)
        return torch.tensor(X, dtype=torch.float).unsqueeze(0)
        
    def __getitem__(self, idx):
        i = self.indices[idx]
        X, y = self.create_sequences(i)
        return torch.tensor(X, dtype=torch.float), torch.tensor(y, dtype=torch.float)