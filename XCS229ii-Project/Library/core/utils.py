import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing


def plot_time_series_charts(figsize: tuple, xlabels: list, ylabels: list, data: pd.DataFrame, fig_name: str,
                            rotation=45, num_xticks=50, num_annotations=10, save_fig=True, use_subplots=True,
                            model_name=None):
    if model_name is not None:
        fig_name = fig_name + "_" + model_name
    if not use_subplots:
        plt.gcf().set_size_inches(figsize[0], figsize[1], forward=True)
        for i in range(len(ylabels)):
            plt.plot(range(data.shape[0]), data[ylabels[i]], marker='.', label=ylabels[i])
            xs = np.arange(0, data.shape[0], int(data.shape[0] / num_annotations))
            ys = [data[ylabels[i]].values.tolist()[ts] for ts in xs]
            for x, y in zip(xs, ys):
                label = "{:.1f}".format(y)
                plt.annotate(label,  # this is the text
                             (x, y),  # this is the point to label
                             textcoords="offset points",  # how to position the text
                             xytext=(0, 20),  # distance from text to points (x,y)
                             va='top',
                             bbox=dict(boxstyle="round", fc="cyan"),
                             ha='center')  # horizontal alignment can be left, right or center
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
    g = sns_plot = sns.heatmap(var_corr, xticklabels=var_corr.columns, yticklabels=False, annot=True)
    g.set_xticklabels(g.get_xticklabels(), rotation=30, horizontalalignment='right')
    if save_fig:
        sns_plot.figure.savefig("./plots/" + fig_name + "_%s" % cols)
    plt.close()


def plot_trade_action(model_name: str, figsize: tuple, fig_name: str, xlabels: list, ylabels: list,
                      number_stock: int, initial_invest: float, asset: float, passive_trade_result: float,
                      data: pd.DataFrame, rotation=45, num_xticks=50, num_annotations=20, save_fig=True):
    plt.gcf().set_size_inches(figsize[0], figsize[1], forward=True)
    for i in range(len(ylabels)):
        plt.plot(range(data.shape[0]), data[ylabels[i]], marker='.', label=ylabels[i])
        if i == len(ylabels) - 1:
            xs = np.arange(0, data.shape[0], int(data.shape[0] / num_annotations))
            ys = [data[ylabels[i]].values.tolist()[ts] for ts in xs]
            ls = [data['trade_action'].values.tolist()[ts] for ts in xs]
            for x, y, l in zip(xs, ys, ls):
                if l == -1:
                    label = "S"
                    plt.annotate(label,  # this is the text
                                 (x, y),  # this is the point to label
                                 textcoords="offset points",  # how to position the text
                                 xytext=(0, 50),  # distance from text to points (x,y)
                                 va='top',
                                 bbox=dict(boxstyle="round", fc="red"),
                                 arrowprops=dict(
                                     arrowstyle="-", fc="lightgrey"),
                                 ha='center')  # horizontal alignment can be left, right or center
                    # plt.scatter(x, y, marker='v', color='red', s=100)
                elif l == 1:
                    label = "B"
                    plt.annotate(label,  # this is the text
                                 (x, y),  # this is the point to label
                                 textcoords="offset points",  # how to position the text
                                 xytext=(0, 50),  # distance from text to points (x,y)
                                 va='top',
                                 bbox=dict(boxstyle="round", fc="cyan"),
                                 arrowprops=dict(
                                     arrowstyle="-", fc="lightgrey"),
                                 ha='center')  # horizontal alignment can be left, right or center
                    # plt.scatter(x, y, marker='^', color='green', s=100)
                # else:
                #     label = "H"
                #     plt.annotate(label,  # this is the text
                #                  (x, y),  # this is the point to label
                #                  textcoords="offset points",  # how to position the text
                #                  xytext=(0, 50),  # distance from text to points (x,y)
                #                  va='top',
                #                  bbox=dict(boxstyle="round", fc="lightgrey"),
                #                  arrowprops=dict(
                #                      arrowstyle="-", fc="lightgrey"),
                #                  ha='center')  # horizontal alignment can be left, right or center
    plt.xticks(ticks=range(0, data.shape[0], max(int(data.shape[0] / num_xticks), 1)),
               labels=data[xlabels[-1]].loc[::max(int(data.shape[0] / num_xticks), 1)], rotation=rotation)
    plt.ylabel(ylabels[-1])
    title = "Model : {}; Total Assets via Algorithmic Trading : {:,}USD;  " \
            "Total # of Stocks via Algorithmic Trading: {}  " \
            "Total return via Algorithmic Trading: {:.2%}  \n" \
            "Total Assets via Passive Trading : {:,}USD;   " \
            "Total # of Stocks via Passive Trading:{}  " \
            "Total return via Passive Trading: {:.2%}".format(model_name, int(asset), number_stock,
                                                              (asset / initial_invest), int(passive_trade_result),
                                                              int(initial_invest / data['real_price'].values.tolist()[
                                                                  0]),
                                                              (passive_trade_result / initial_invest))
    plt.title(title)
    plt.legend()
    if save_fig:
        plt.savefig("./plots/" + fig_name + '_' + model_name)
    plt.close()


def train_test_split(data: pd.DataFrame, train_test_split_ratio: float, features: list, targets: list,
                     history_points: int, pred_points: int):
    data = data.dropna(axis=0)

    train_indices = int(data.shape[0] * train_test_split_ratio)
    train_data_x = data.get(features)[:train_indices]
    test_data_x = data.get(features)[train_indices:]
    train_data_y = data.get(targets)[:train_indices]
    test_data_y_date = data.get(['Date'])[train_indices:]
    test_data_y = data.get(targets)[train_indices:]

    train_data_x.reset_index(inplace=True)
    train_data_x = train_data_x.drop(columns=['index'])
    test_data_x.reset_index(inplace=True)
    test_data_x = test_data_x.drop(columns=['index'])
    train_data_y.reset_index(inplace=True)
    train_data_y = train_data_y.drop(columns=['index'])
    test_data_y_date.reset_index(inplace=True)
    test_data_y_date = test_data_y_date.drop(columns=['index'])
    test_data_y.reset_index(inplace=True)
    test_data_y = test_data_y.drop(columns=['index'])

    scaler_x = preprocessing.MinMaxScaler()
    train_data_normalized_x = scaler_x.fit_transform(train_data_x)
    test_data_normalized_x = scaler_x.transform(test_data_x)

    scaler_y = preprocessing.MinMaxScaler()
    train_data_normalized_y = scaler_y.fit_transform(train_data_y)
    test_data_normalized_y = scaler_y.transform(test_data_y)

    data_windows = []
    for i in range(0, train_data_normalized_x.shape[0] - history_points - pred_points, pred_points):
        data_windows.append(train_data_normalized_x[i:i + history_points])
    X_train = np.array(data_windows).astype(float)

    data_windows = []
    for i in range(0, test_data_normalized_x.shape[0] - history_points - pred_points, pred_points):
        data_windows.append(test_data_normalized_x[i:i + history_points])
    X_test = np.array(data_windows).astype(float)

    data_windows = []
    for i in range(history_points, train_data_normalized_y.shape[0] - pred_points, pred_points):
        data_windows.append(train_data_normalized_y[i:i + pred_points])
    y_train = np.array(data_windows).astype(float)

    data_windows = []
    for i in range(history_points, test_data_normalized_y.shape[0] - pred_points, pred_points):
        data_windows.append(test_data_normalized_y[i:i + pred_points])
    y_test = np.array(data_windows).astype(float)

    y_test_date = []
    for i in range(history_points, y_test.shape[0] * pred_points + history_points, 1):
        y_test_date.append(test_data_y_date['Date'].values.tolist()[i])

    assert len(y_test_date) == y_test.shape[0] * pred_points

    return X_train, X_test, y_train, y_test, scaler_y, y_test_date
