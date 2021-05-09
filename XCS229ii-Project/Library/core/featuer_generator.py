# Data Manipulation
import numpy as np
import pandas as pd
from pandas_datareader import data as pdr
import seaborn as sns
# Finance
import talib as ta
# Plotting
import matplotlib.pyplot as plt

plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['font.size'] = 14
plt.rcParams['legend.fontsize'] = 'medium'
plt.rcParams['figure.titlesize'] = 'medium'

STOCK_SYMBOL = 'GOOG'
START_DATE = '2010-01-01'
END_DATE = '2021-04-01'
ROLLING_X = 3


def plot_time_series_charts(figsize: tuple, xlabels: list, ylabels: list, data: pd.DataFrame, fig_name: str,
                            rotation=45, num_xticks=20, save_fig=True, use_subplots=True):
    if not use_subplots:
        plt.gcf().set_size_inches(figsize[0], figsize[1], forward=True)
        for i in range(len(ylabels)):
            plt.plot(range(data.shape[0]), data[ylabels[i]], label=ylabels[i])
        plt.xticks(ticks=range(0, data.shape[0], int(data.shape[0] / num_xticks)),
                   labels=data[xlabels[-1]].loc[::int(data.shape[0] / num_xticks)], rotation=rotation)
        plt.ylabel(ylabels[-1])
        plt.title(fig_name)
        plt.legend()

    else:
        num_rows = len(ylabels)
        num_cols = 1
        fig, axs = plt.subplots(num_rows, num_cols, figsize=figsize)
        for i in range(len(axs)):
            axs[i].plot(range(data.shape[0]), data[ylabels[i]], label=ylabels[i])
            axs[i].set_xticks(range(0, data.shape[0], int(data.shape[0] / num_xticks)))
            axs[i].set_xticklabels(data[xlabels[i]].loc[::int(data.shape[0] / num_xticks)])
            axs[i].set_xlabel(xlabels[i])
            axs[i].set_ylabel(ylabels[i])
            axs[i].legend(loc=0)
            plt.setp(axs[i].xaxis.get_majorticklabels(), rotation=rotation)
    if save_fig:
        plt.savefig("./plots/" + fig_name)
    plt.close()


def plot_corr_heatmap(fig_name: str, cols: list, data: pd.DataFrame, save_fig=True):
    var_corr = data.get(cols).corr()
    sns_plot = sns.heatmap(var_corr, xticklabels=var_corr.columns, yticklabels=var_corr.columns, annot=True)
    if save_fig:
        sns_plot.figure.savefig("./plots/" + fig_name + "_%s" % cols)
    plt.close()


def get_raw_data(stock_symbol: str, start_date: str, end_date: str, plot_data=False) -> (np.ndarray, pd.DataFrame):
    data = pdr.get_data_yahoo(stock_symbol, start_date, end_date)
    data = data.dropna()
    data.reset_index(inplace=True)
    data['Date'] = pd.to_datetime(data['Date'], unit='ms')
    data['Date'] = [d.strftime('%Y-%m-%d') for d in data['Date']]

    if plot_data:
        cols = ['Volume', 'Adj Close']
        plot_time_series_charts(figsize=(40, 20), xlabels=['Date', 'Date'], ylabels=cols, data=data,
                                fig_name=stock_symbol)
        plot_corr_heatmap(cols=cols, data=data, fig_name="corr_heatmap")
    return data


def get_technical_indicator_features(data: pd.DataFrame, rolling_x=14, plot_data=False) \
        -> (np.ndarray, pd.DataFrame):
    # OBV: on-balance volume
    obv = ta.OBV(data['Close'].values, data['Volume'].values)
    obv_ema = ta.EMA(obv, timeperiod=rolling_x)
    data['obv_ema'] = obv_ema.tolist()

    # ATR: Average True Range
    atr = ta.ATR(data['High'], data['Low'], data['Close'], timeperiod=rolling_x)
    data['atr'] = atr.tolist()

    # WILLR: Willian's %R
    willr = ta.WILLR(data['High'], data['Low'], data['Close'], timeperiod=rolling_x)
    data['willr'] = willr.tolist()

    # BBANDS: Bollinger Bands
    ub, mb, lb = ta.BBANDS(data['Close'], timeperiod=rolling_x, nbdevup=2, nbdevdn=2, matype=0)
    data['bb_up'] = ub.tolist()
    data['bb_mb'] = mb.tolist()
    data['bb_lb'] = lb.tolist()

    cols = ['obv_ema', 'atr', 'willr', 'bb_mb']
    if plot_data:
        plot_time_series_charts(figsize=(40, 80), xlabels=['Date', 'Date', 'Date', 'Date'], ylabels=cols, data=data,
                                fig_name="tech_features")
        plot_corr_heatmap(cols=cols, data=data, fig_name="corr_heatmap")
    return cols, data


def derive_simple_features(data: pd.DataFrame, rolling_x=3, plot_data=False) \
        -> (np.ndarray, pd.DataFrame):
    # daily return pct change based on adj closed price
    data['return_roc'] = data['Adj Close'].pct_change().values
    # daily return pct change rolling x days exponential moving average
    data['return_roc_ema'] = ta.EMA(data['return_roc'], rolling_x)
    # daily volume pct change
    data['vol_roc'] = data['Volume'].pct_change().values
    # daily volume pct change rolling x days exponential moving average
    data['vol_roc_ema'] = ta.EMA(data['vol_roc'], rolling_x)
    cols = ['return_roc_ema', 'vol_roc_ema']
    if plot_data:
        plot_time_series_charts(figsize=(40, 20), xlabels=['Date', 'Date'], ylabels=cols, data=data,
                                fig_name="simple_features")
        plot_corr_heatmap(cols=cols, data=data, fig_name="corr_heatmap")
    return cols, data


if __name__ == '__main__':
    df_input_data = get_raw_data(stock_symbol=STOCK_SYMBOL, start_date=START_DATE, end_date=END_DATE, plot_data=True)
    simple_features, df_input_data = derive_simple_features(data=df_input_data, rolling_x=3,
                                                            plot_data=True)
    technical_indicator_features, df_input_data = get_technical_indicator_features(data=df_input_data,
                                                                                   rolling_x=14, plot_data=True)
