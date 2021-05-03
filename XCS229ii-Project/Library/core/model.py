import os
import numpy as np
import datetime as dt
from numpy import newaxis
from core_utils import Timer
from keras.layers import Dense, Activation, Dropout, LSTM
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint


class Model:
    """Class for building the LSTM model"""

    def __init__(self):
        self.model = Sequential()

    def load_saved_model(self, filepath: str):
        """load saved model"""
        print("[Phase::Load Model]=======> Load Model from {} ...".format(filepath))
        self.model = load_model(filepath=filepath)

    def build_model(self, config: dict):
        """use keras basic building blocks and parameters in the config to instantiate the LSTM model"""
        for layer in config["model"]["layers"]:
            assert isinstance(layer, dict)
            neurons = layer["neurons"] if "neurons" in layer.keys() else None
            dropout_rate = layer["rate"] if "rate" in layer.keys() else None
            activation = layer["activation"] if "activation" in layer.keys() else None
            return_seq = layer["return_seq"] if "return_seq" in layer.keys() else None
            input_time_steps = layer["input_time_steps"] if "input_time_steps" in layer.keys() else None
            input_dim = layer["input_dim"] if "input_dim" in layer.keys() else None

            if layer["type"] == "dense":
                self.model.add(Dense(neurons, activation=activation))
            if layer["type"] == "lstm":
                self.model.add(LSTM(neurons, input_shape=(input_time_steps, input_dim), return_sequences=return_seq))
            if layer["type"] == "dropout":
                self.model.add(Dropout(dropout_rate))

        self.model.compile(loss=config["model"]["loss"], optimizer=config["model"]["optimizer"])
        print("[Phase::Build Model]=======> Compiled the Model ...")
        print(self.model.summary())

    def train(self, x: np.array, y: np.array, epochs: int, batch_size: int, save_dir: str):
        """train the model"""
        timer = Timer()
        timer.start()
        print("[Phase::Train Model]=======> Start Training...")

        file_name = os.path.join(save_dir, '%s-e%s.h5' % (dt.datetime.now().strftime('%d%m%Y'), str(epochs)))
        callbacks = [
            EarlyStopping(monitor="loss", patience=2),
            ModelCheckpoint(filepath=file_name, monitor="loss", save_best_only=True)
        ]
        self.model.fit(
            x,
            y,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks
        )
        self.model.save(file_name)
        print("[Phase::Train Model]=======> Training completed. Model saved as {}".format(file_name))
        timer.stop()

    def predict_one_step_ahead(self, data):
        """predict 1 time step ahead"""
        print("[Phase::Prediction]=======> Predict one step ahead ...")
        predicted = self.model.predict(x=data)
        predicted = np.reshape(predicted, (predicted.size,))
        return predicted

    def predict_multiple_steps_head(self, data, window_size, prediction_len):
        """predict N time steps ahead"""
        print("[Phase::Prediction]=======> Predict multiple steps ahead ...")
        prediction_seqs = []
        for i in range(int(len(data) / window_size)):
            curr_frame = data[i * window_size]
            predicted = []
            for j in range(prediction_len):
                predicted.append(self.model.predict(curr_frame[newaxis, :, :])[0, 0])
                curr_frame = curr_frame[1:]
                curr_frame = np.insert(curr_frame, [window_size - 2], predicted[-1], axis=0)
            prediction_seqs.append(predicted)
        return prediction_seqs

    def predict_full_sequence(self, data, window_size):
        """predict a full sequence """
        print("[Phase::Prediction]=======>Predict a full sequence ahead ...")
        curr_frame = data[0]
        predicted = []
        for i in range(len(data)):
            predicted.append(self.model.predict(curr_frame[newaxis, :, :])[0, 0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size - 2], predicted[-1], axis=0)
        return predicted
