import torch
import torch.nn as nn

""" This file contains the LSTM model"""

class LSTM(nn.Module):
    def __init__(self, num_layers, input_size, hidden_size, device=None, dropout_prob=0.2):
        """
        Inputs:
        num_layers: Number of recurrent layers
        input_size: Number of features for input
        hidden_size: Number of features in hidden state

        Outputs: 1
        """
        super(LSTM, self).__init__()

        if device:
          self.device = device
        else:
           self.device = torch.device("cpu")

        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.lstm_layers = nn.ModuleList([nn.LSTM(input_size if i == 0 else hidden_size,
                                                   hidden_size,
                                                   batch_first=True)
                                          for i in range(num_layers)])

        self.dropout_layers = nn.ModuleList([nn.Dropout(dropout_prob) for _ in range(num_layers)])

        self.dense = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
      '''
      Inputs:
      x: input data

      Outputs:
      out: output of forward pass
      '''
      # Normalizing
      h = torch.zeros(1, x.size(0), self.hidden_size).to(self.device)
      c = torch.zeros(1, x.size(0), self.hidden_size).to(self.device)

      out = x
      for i in range(self.num_layers):
        out, (h, c) = self.lstm_layers[i](out, (h, c))
        out = self.dropout_layers[i](out)

      out = out[:, -1, :]
      x = self.dense(out)

      return x