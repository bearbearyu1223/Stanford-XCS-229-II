import numpy as np
from pandas_datareader import data as pdr
import talib as ta

from utils import *

STOCK_SYMBOL = 'GOOG'
START_DATE = '2010-01-01'
END_DATE = '2021-04-30'
ROLLING_X = 3

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
    data['ret_pct'] = data['Adj Close'].pct_change().values
    # daily return pct change rolling x days exponential moving average
    data['ret_ema'] = ta.EMA(data['ret_pct'], rolling_x)
    # daily volume pct change
    data['vol_pct'] = data['Volume'].pct_change().values
    # daily volume pct change rolling x days exponential moving average
    data['vol_ema'] = ta.EMA(data['vol_pct'], rolling_x)
    cols = ['ret_ema', 'vol_ema']
    if plot_data:
        plot_time_series_charts(figsize=(40, 20), xlabels=['Date', 'Date'], ylabels=cols, data=data,
                                fig_name="simple_features")
        plot_corr_heatmap(cols=cols, data=data, fig_name="corr_heatmap")
    return cols, data
