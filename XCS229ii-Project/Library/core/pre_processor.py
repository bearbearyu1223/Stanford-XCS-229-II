# Data Manipulation
import numpy as np
import pandas as pd
from pandas_datareader import data as pdr
import seaborn as sns
# Finance
import talib as ta
import yfinance as yf
# Machine Learning
from sklearn import preprocessing
from sklearn import metrics

# Plotting
import matplotlib.pyplot as plt

# Utils
import os

plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['font.size'] = 18
plt.rcParams['legend.fontsize'] = 'large'
plt.rcParams['figure.titlesize'] = 'medium'

STOCK_SYMBOL = 'GOOG'
START_DATE = '2010-01-01'
END_DATE = '2021-04-01'
ROLLING_X = 3


def get_raw_data(stock_symbol: str, start_date: str, end_date: str, plot_data=False) -> (np.ndarray, pd.DataFrame):
    df = pdr.get_data_yahoo(stock_symbol, start_date, end_date)
    df = df.dropna()
    df.reset_index(inplace=True)
    df['Date'] = pd.to_datetime(df['Date'], unit='ms')
    df['Date'] = [d.strftime('%Y-%m-%d') for d in df['Date']]

    if plot_data:
        fig, axs = plt.subplots(2, 1, figsize=(20, 20))

        axs[0].plot(range(df.shape[0]), df['Close'], label='Close')
        axs[0].set_xticks(range(0, df.shape[0], 28 * 3))
        axs[0].set_xticklabels(df['Date'].loc[::28 * 3])
        axs[0].set_xlabel('Date')
        axs[0].set_ylabel(stock_symbol + "[price]")
        axs[0].legend(loc=0)
        plt.setp(axs[0].xaxis.get_majorticklabels(), rotation=45)

        axs[1].plot(range(df.shape[0]), df['Volume'], label='Volume')
        axs[1].set_xticks(range(0, df.shape[0], 28 * 3))
        axs[1].set_xticklabels(df['Date'].loc[::28 * 3])
        axs[1].set_xlabel('Date')
        axs[1].set_ylabel(stock_symbol + "[volume]")
        axs[1].legend(loc=0)
        plt.setp(axs[1].xaxis.get_majorticklabels(), rotation=45)

        plt.savefig("./plots/stock_price_and_volume[%s]" % stock_symbol)
        plt.close()

        cols = ['Volume', 'Close']
        var_corr = df.get(cols).corr()
        sns_plot = sns.heatmap(var_corr, xticklabels=var_corr.columns, yticklabels=var_corr.columns, annot=True)
        sns_plot.figure.savefig("./plots/var_corr_heatmap_%s" % cols + "<%s>" % stock_symbol)
        plt.close()
        return df


def get_technical_indicator_features(df: pd.DataFrame, stock_symbol: str, rolling_x=14, plot_data=False) \
        -> (np.ndarray, pd.DataFrame):
    # OBV: on-balance volume
    obv = ta.OBV(df['Close'].values, df['Volume'].values)
    obv_ema = ta.EMA(obv, timeperiod=rolling_x)
    df['obv_ema'] = obv_ema.tolist()

    # ATR: Average True Range
    atr = ta.ATR(df['High'], df['Low'], df['Close'], timeperiod=rolling_x)
    df['atr'] = atr.tolist()

    # WILLR: Willian's %R
    willr = ta.WILLR(df['High'], df['Low'], df['Close'], timeperiod=rolling_x)
    df['willr'] = willr.tolist()

    # BBANDS: Bollinger Bands
    ub, mb, lb = ta.BBANDS(df['Close'], timeperiod=rolling_x, nbdevup=2, nbdevdn=2, matype=0)
    df['bb_up'] = ub.tolist()
    df['bb_mb'] = mb.tolist()
    df['bb_lb'] = lb.tolist()

    cols = ['obv_ema', 'atr', 'willr', 'bb_mb']
    if plot_data:
        fig, axs = plt.subplots(4, 1, figsize=(20, 40))

        axs[0].plot(range(df.shape[0]), df['obv_ema'], label='obv_ema')
        axs[0].set_xticks(range(0, df.shape[0], 28 * 3))
        axs[0].set_xticklabels(df['Date'].loc[::28 * 3])
        axs[0].set_xlabel('Date')
        axs[0].set_ylabel(stock_symbol + "[obv_ema]")
        axs[0].legend(loc=0)
        plt.setp(axs[0].xaxis.get_majorticklabels(), rotation=45)

        axs[1].plot(range(df.shape[0]), df['atr'], label='atr')
        axs[1].set_xticks(range(0, df.shape[0], 28 * 3))
        axs[1].set_xticklabels(df['Date'].loc[::28 * 3])
        axs[1].set_xlabel('Date')
        axs[1].set_ylabel(stock_symbol + "[atr]")
        axs[1].legend(loc=0)
        plt.setp(axs[1].xaxis.get_majorticklabels(), rotation=45)

        axs[2].plot(range(df.shape[0]), df['willr'], label='willr')
        axs[2].set_xticks(range(0, df.shape[0], 28 * 3))
        axs[2].set_xticklabels(df['Date'].loc[::28 * 3])
        axs[2].set_xlabel('Date')
        axs[2].set_ylabel(stock_symbol + "[willr]")
        axs[2].legend(loc=0)
        plt.setp(axs[2].xaxis.get_majorticklabels(), rotation=45)

        axs[3].plot(range(df.shape[0]), df['bb_up'], label='bb_ub', linestyle='-', color='b')
        axs[3].plot(range(df.shape[0]), df['bb_lb'], label='bb_lb', linestyle='-', color='g')
        axs[3].plot(range(df.shape[0]), df['bb_mb'], label='bb_mb', linestyle='--', color='r')
        axs[3].set_xticks(range(0, df.shape[0], 28 * 3))
        axs[3].set_xticklabels(df['Date'].loc[::28 * 3])
        axs[3].set_xlabel('Date')
        axs[3].set_ylabel(stock_symbol + "[bbs]")
        axs[3].legend(loc=0)
        plt.setp(axs[3].xaxis.get_majorticklabels(), rotation=45)

        plt.savefig("./plots/technical_indicator_features[%s]" % stock_symbol)
        plt.close()

        var_corr = df.get(cols).corr()
        sns_plot = sns.heatmap(var_corr, xticklabels=var_corr.columns, yticklabels=var_corr.columns, annot=True)
        sns_plot.figure.savefig("./plots/var_corr_heatmap_%s" % cols + "<%s>" % stock_symbol)
        plt.close()

    return cols, df


def derive_simple_features(df: pd.DataFrame, stock_symbol: str, rolling_x=3, plot_data=False) \
        -> (np.ndarray, pd.DataFrame):
    # daily return (%)
    df['daily_return_pct'] = df['Close'].pct_change().values
    # daily return (%) rolling x days average
    df['roc_ma'] = df['daily_return_pct'].rolling(rolling_x).mean()
    df['target'] = np.where(df['roc_ma'] > 0.0, 1.0, -1.0)
    # daily volume delta (%)
    df['daily_volume_delta_pct'] = df['Volume'].pct_change().values
    # daily volume delta (%) rolling x days average
    df['vol_roc_ma'] = df['daily_volume_delta_pct'].rolling(rolling_x).mean()
    cols = ['roc_ma', 'vol_roc_ma']

    if plot_data:
        fig, axs = plt.subplots(2, 1, figsize=(20, 20))

        axs[0].plot(range(df.shape[0]), df['roc_ma'], label='roc_ma')
        axs[0].set_xticks(range(0, df.shape[0], 28 * 3))
        axs[0].set_xticklabels(df['Date'].loc[::28 * 3])
        axs[0].set_xlabel('Date')
        axs[0].set_ylabel(stock_symbol + "[roc_ma]")
        axs[0].legend(loc=0)
        plt.setp(axs[0].xaxis.get_majorticklabels(), rotation=45)

        axs[1].plot(range(df.shape[0]), df['vol_roc_ma'],
                    label='vol_roc_ma')
        axs[1].set_xticks(range(0, df.shape[0], 28 * 3))
        axs[1].set_xticklabels(df['Date'].loc[::28 * 3])
        axs[1].set_xlabel('Date')
        axs[1].set_ylabel(stock_symbol + "[vol_roc_ma]")
        axs[1].legend(loc=0)
        plt.setp(axs[1].xaxis.get_majorticklabels(), rotation=45)

        plt.savefig("./plots/simple_data_features[%s]" % stock_symbol)
        plt.close()
        var_corr = df.get(cols).corr()
        sns_plot = sns.heatmap(var_corr, xticklabels=var_corr.columns, yticklabels=var_corr.columns, annot=True)
        sns_plot.figure.savefig("./plots/var_corr_heatmap_%s" % cols + "<%s>" % stock_symbol)
        plt.close()
    return cols, df


def train_test_split(input_df: pd.DataFrame, features: list, target: str,
                     split_ratio: float, seq_len_for_x=7, seq_len_for_y=1, normalize=True) -> (np.ndarray, np.ndarray):
    train_indices = int(input_df.shape[0] * split_ratio)
    test_indices = input_df.shape[0] - train_indices
    cols = features + [target]
    train_dataset = input_df.get(cols).values[:train_indices, :]
    print('shape of input dataset: {}'.format(train_dataset.shape))
    test_dataset = input_df.get(cols).values[train_indices:, :]

    data_windows = []
    for i in range(0, train_dataset.shape[0], seq_len_for_y):
        if (i + 1) * seq_len_for_y + seq_len_for_x <= train_dataset.shape[0]:
            data_windows.append(train_dataset[i * seq_len_for_y:(i + 1) * seq_len_for_y + seq_len_for_x, :])
    data_windows_arrays = np.array(data_windows).astype(float)
    assert data_windows_arrays.shape[1] == (seq_len_for_x + seq_len_for_y), 'get {}'.format(
        data_windows_arrays.shape[1])
    assert data_windows_arrays.shape[-1] == len(cols), 'get {}'.format(data_windows_arrays.shape[-1])

    x_train = data_windows_arrays[:, 0:seq_len_for_x, 0:-1]
    assert x_train.shape[1] == seq_len_for_x
    assert x_train.shape[-1] == len(features)

    y_train = data_windows_arrays[:, seq_len_for_x:, -1]
    assert y_train.shape[1] == seq_len_for_y

    data_windows = []
    for i in range(0, test_dataset.shape[0], seq_len_for_y):
        if (i + 1) * seq_len_for_y + seq_len_for_x <= test_dataset.shape[0]:
            data_windows.append(test_dataset[i * seq_len_for_y:(i + 1) * seq_len_for_y + seq_len_for_x, :])
    data_windows_arrays = np.array(data_windows).astype(float)
    assert data_windows_arrays.shape[1] == (seq_len_for_x + seq_len_for_y), 'get {}'.format(
        data_windows_arrays.shape[1])
    assert data_windows_arrays.shape[-1] == len(cols), 'get {}'.format(data_windows_arrays.shape[-1])

    x_test = data_windows_arrays[:, 0:seq_len_for_x, 0:-1]
    assert x_test.shape[1] == seq_len_for_x
    assert x_test.shape[-1] == len(features)

    y_test = data_windows_arrays[:, seq_len_for_x:, -1]
    assert y_test.shape[1] == seq_len_for_y

    return x_train, y_train, x_test, y_test


if __name__ == '__main__':
    df_input_data = get_raw_data(stock_symbol=STOCK_SYMBOL, start_date=START_DATE, end_date=END_DATE, plot_data=True)
    simple_features, df = derive_simple_features(df_input_data, stock_symbol=STOCK_SYMBOL, rolling_x=3,
                                                 plot_data=True)
    technical_indicator_features, df = get_technical_indicator_features(df, stock_symbol=STOCK_SYMBOL,
                                                                        rolling_x=14, plot_data=True)
    selected_features = simple_features + technical_indicator_features
    X_train, Y_train, X_test, Y_test = train_test_split(df, features=selected_features, target='target', split_ratio=0.9,
                                                        seq_len_for_x=7, seq_len_for_y=1)
    print('Shape of X_train: {}'.format(X_train.shape))
    print('Shape of Y_train: {}'.format(Y_train.shape))
    print('Shape of X_test: {}'.format(X_test.shape))
    print('Shape of Y_test: {}'.format(Y_test.shape))
