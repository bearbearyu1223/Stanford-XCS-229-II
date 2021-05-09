import os
import numpy as np
import datetime as dt
import tensorflow as tf
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint


class Model:
    """Class for building the LSTM model"""
    def __init__(self, input_time_steps, input_dim, output_dim, dropout_rate=0.1, loss='mse',
                 optimizer="adam"):
        self.model = Sequential()
        self.input_time_steps = input_time_steps
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.loss = loss
        self.optimizer = optimizer

    def load_saved_model(self, filepath: str):
        """Load saved model"""
        self.model = load_model(filepath=filepath)

    def build_model(self):
        """Build the model"""
        print("[Phase::Build]")
        tf.random.set_seed(20)
        np.random.seed(10)
        self.model.add(LSTM(units=200, input_shape=(self.input_time_steps, self.input_dim), return_sequences=True))
        self.model.add(Dropout(self.dropout_rate))
        self.model.add(LSTM(units=100, return_sequences=False))
        self.model.add(Dropout(self.dropout_rate))
        self.model.add(Dense(units=self.output_dim, activation='linear'))

        self.model.compile(loss=self.loss, optimizer=self.optimizer)
        print(self.model.summary())

    def train(self, x: np.array, y: np.array, epochs: int, batch_size: int, save_dir: str):
        """Train the model"""
        print("[Phase::Train]")
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

    def predict(self, data):
        """predict one time step ahead"""
        print("[Phase::Prediction]")
        predicted = self.model.predict(x=data)
        predicted = np.reshape(predicted, (predicted.size,))
        return predicted


