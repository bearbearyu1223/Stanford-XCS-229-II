import numpy as np
import pandas as pd

def normalised_windows(window_data, single_window=False):
    """
    :param window_data: np.array
    :param single_window: bool
    :return: normalised_data
    """
    normalised_data = []
    window_data = [window_data] if single_window else window_data
    for window in window_data:
        normalised_window = []
        for col_i in range(window.shape[1]):
            normalised_col = [((float(p) / float(window[0, col_i])) - 1) for p in window[:, col_i]]
            normalised_window.append(normalised_col)
        normalised_window = np.array(
            normalised_window).T  # reshape and transpose array back into original multidimensional format
        normalised_data.append(normalised_window)
    normalised_data = np.array(normalised_data).astype(float)
    return normalised_data


class DataLoader:
    """
    A class for loading and transforming data for the LSTM model
    """

    def __init__(self, filename: str, split: float, cols: list):
        """
        :param filename: name of the cvs file
        :param split: train_size / test_size
        :param cols: list of columns' names to be used in the train and test dataset
        """
        dataframe = pd.read_csv(filename)
        index_split = int(len(dataframe) * split)
        self.data_train = dataframe.get(cols).values[:index_split]
        self.data_test = dataframe.get(cols).values[index_split:]
        self.len_train = len(self.data_train)
        self.len_test = len(self.data_test)
        self.len_train_window = None

    def get_test_data(self, seq_len: int, normalised: bool):
        """
        :param seq_len:
        :param normalised:
        :return: data_x, data_y
        """
        data_windows = []
        assert self.len_test > seq_len, "ERR:length of test dataset should be larger than the length of the data window"
        for i in range(self.len_test - seq_len):
            data_windows.append(self.data_test[i:i + seq_len])

        data_windows = np.array(data_windows).astype(float)
        data_windows = normalised_windows(window_data=data_windows,
                                              single_window=False) if normalised else data_windows

        data_x = data_windows[:, :-1]
        data_y = data_windows[:, -1, [0]]
        return data_x, data_y

    def get_train_data(self, seq_len: int, normalised: bool):
        """
        :param seq_len:
        :param normalised:
        :return: x, y
        """
        data_x = []
        data_y = []
        assert self.len_train > seq_len, "ERR:length of train dataset should be larger than the length of the data " \
                                         "window "
        for i in range(self.len_train - seq_len):
            x, y = self._next_window(i, seq_len, normalised)
            data_x.append(x)
            data_y.append(y)
        data_x = np.array(data_x).astype(float)
        data_y = np.array(data_y).astype(float)
        return data_x, data_y

    def generate_train_batch(self, seq_len: int, batch_size: int, normalised: bool):
        """
        :param seq_len:
        :param batch_size:
        :param normalised:
        :return: data_x_batch, data_y_batch
        """
        i = 0
        assert self.len_train > seq_len, "ERR:length of train dataset should be larger than the length of the data " \
                                         "window "
        while i < (self.len_train - seq_len):
            x_batch = []
            y_batch = []
            for b in range(batch_size):
                if i >= (self.len_train - seq_len):
                    # stop-condition for a smaller final batch if data doesn't divide evenly
                    data_x_batch = np.array(x_batch).astype(float)
                    data_y_batch = np.array(y_batch).astype(float)
                    yield data_x_batch, data_y_batch
                    i = 0
                x, y = self._next_window(i, seq_len, normalised)
                x_batch.append(x)
                y_batch.append(y)
                i += 1
            data_x_batch = np.array(x_batch).astype(float)
            data_y_batch = np.array(y_batch).astype(float)
            yield data_x_batch, data_y_batch

    def _next_window(self, i: int, seq_len: int, normalised: bool):
        """
        :param i:
        :param seq_len:
        :param normalised:
        :return: x, y
        """
        window = self.data_train[i:i + seq_len]
        window = normalised_windows(window, single_window=True)[0] if normalised else window
        x = window[:-1]
        y = window[-1, [0]]
        return x, y
