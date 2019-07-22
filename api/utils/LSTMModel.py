from keras import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import LSTM, Dropout, Dense


class LSTMModel(object):

    def __init__(self, x_train):
        self.model = None

        self.model = Sequential()
        self.model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        self.model.add(Dropout(0.2))

        self.model.add(LSTM(units=100, return_sequences=True))
        self.model.add(Dropout(0.2))

        self.model.add(LSTM(units=100, return_sequences=True))
        self.model.add(Dropout(0.2))

        self.model.add(LSTM(units=100))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(1))

        self.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

    def fit(self, features, labels, config):
        self.model.fit(features, labels, epochs=config['epochs'], batch_size=config['batch_size'],
                       verbose=config['verbose'], callbacks=config['callbacks'])

    def predict(self, data):
        return self.model.predict(data)
