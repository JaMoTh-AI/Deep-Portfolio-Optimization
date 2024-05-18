# Deep Portfolio Optimization
Check out our app [here](https://deep-portfolio-optimization.streamlit.app/)!

## Downloading the libraries
First, you need to import the following libraries:
```
yfinance
ta
scikit-learn
torch
numpy
pandas
tqdm
streamlit
cvxpy
gurobipy
torchvision
matplotlib
```
or just run `pip install requirements.txt` after pulling the repo. If your goal is to run the following jupyter notebooks, `model_training.ipynb` or `plotter.ipynb`, you can also simply import them to google colab and run them from top to bottom.

## Running the code

### LSTM Code

If you want to train the LSTM model, open the `model_training.ipynb` and read through the notebook and run the cells. This notebook isn't as interactive as the next one, so in order to train different models make sure you understand the code!

If your goal is to visualize our LSTMs predictions or our training statistics, open the `plotter.ipynb` jupyter notebook. This notebook is much simpler to use than the previous one. Change the `ticker` variable in cell 5 to any other stock ticker such as `META` or `GOOGL` to see that stock's training and validation loss throughout the epochs. In order to make predictions using our LSTMs, verify that the inputs you are giving in cell 9 and 10 are valid. In cell 10, the `stock` can be any stock ticker in `all_stocks`. Make sure that the inputs for `n_lags`, `forecast_horizon`, `model_path`, `data_timeinterval`(cell 10) and `interval`(cell 9) are compatible. If you are wondering which combination are combatible, take a look at the code for `lstm_predict.py` or simply run the combination present in the notebook.