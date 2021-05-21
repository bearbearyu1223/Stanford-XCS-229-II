import os
import numpy as np
import datetime as dt
import tensorflow as tf
from keras.layers import Dense, Dropout, LSTM, RepeatVector, TimeDistributed, Input
from keras.models import Sequential, load_model, Model
from keras.callbacks import EarlyStopping, ModelCheckpoint


class Encoder_Decoder_Model:
    """Class for building the LSTM model"""

    def __init__(self, input_time_steps, input_dim, output_time_steps, output_dim, dropout_rate=0.1, loss='mse',
                 optimizer="adam"):
        self.model = None
        self.input_time_steps = input_time_steps
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.output_time_steps = output_time_steps
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
        # encoder_inputs = Input(shape=(self.input_time_steps, self.input_dim))
        # encoder_l1 = LSTM(100, return_state=True)
        # encoder_outputs1 = encoder_l1(encoder_inputs)
        # encoder_states1 = encoder_outputs1[1:]
        # decoder_inputs = RepeatVector(self.output_time_steps)(encoder_outputs1[0])
        # decoder_l1 = LSTM(100, return_sequences=True)(decoder_inputs, initial_state=encoder_states1)
        # decoder_outputs1 = TimeDistributed(Dense(self.output_dim))(decoder_l1)
        # self.model = Model(encoder_inputs, decoder_outputs1)
        self.model = Sequential()
        self.model.add(LSTM(200, activation='relu', input_shape=(self.input_time_steps, self.input_dim)))
        self.model.add(Dropout(self.dropout_rate))
        self.model.add(RepeatVector(self.output_time_steps))
        self.model.add(LSTM(100, activation='relu', return_sequences=True))
        self.model.add(Dropout(self.dropout_rate))
        self.model.add(TimeDistributed(Dense(self.output_dim, activation='linear')))
        self.model.compile(loss=self.loss, optimizer=self.optimizer)

    def train(self, x: np.array, y: np.array, epochs: int, batch_size: int, save_dir: str):
        """Train the model"""
        print("[Phase::Train]")
        file_name = os.path.join(save_dir, '%s-%s-e%s-h%s-p%s.h5' % ("encoder-decoder",
            dt.datetime.now().strftime('%d%m%Y'), str(epochs), str(self.input_time_steps), str(self.output_time_steps)))
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
        print(self.model.summary())
        self.model.save(file_name)

    def predict(self, data):
        """predict one time step ahead"""
        print("[Phase::Prediction]")
        predicted = self.model.predict(x=data)
        predicted = np.reshape(predicted, (predicted.size,))
        return predicted
