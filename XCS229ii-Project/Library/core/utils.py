import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


def plot_time_series_charts(figsize: tuple, xlabels: list, ylabels: list, data: pd.DataFrame, fig_name: str,
                            rotation=45, num_xticks=50, num_annotations=10, save_fig=True, use_subplots=True):
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
