import torch
import torch.nn as nn

"""This file contains the implementation of an LSTM model."""

class LSTM(nn.Module):
    def __init__(self, num_layers, input_size, hidden_size, device=None, dropout_prob=0.2):
        """
        Initialize the LSTM model.

        Parameters:
        num_layers (int): Number of recurrent layers.
        input_size (int): Number of features for the input.
        hidden_size (int): Number of features in the hidden state.
        device (torch.device, optional): Device to run the model on (CPU or CUDA). Defaults to CPU if not specified.
        dropout_prob (float, optional): Dropout probability for regularization. Defaults to 0.2.
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
      """
      Forward pass of the LSTM model.

      Parameters:
      x (torch.Tensor): Input data tensor.

      Returns:
      torch.Tensor: Output of the forward pass.
      """
      # Initialize hidden and cell states
      h = torch.zeros(1, x.size(0), self.hidden_size).to(self.device)
      c = torch.zeros(1, x.size(0), self.hidden_size).to(self.device)

      # Take the output of the last time step
      out = x
      for i in range(self.num_layers):
        out, (h, c) = self.lstm_layers[i](out, (h, c))
        out = self.dropout_layers[i](out)

      # Pass through the dense layers
      out = out[:, -1, :]
      x = self.dense(out)

      return x