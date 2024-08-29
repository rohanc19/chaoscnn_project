import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense

class ChaosCNN:
    def __init__(self):
        self.model = Sequential([
            Conv1D(32, 3, activation='relu', input_shape=(30, 4)),
            MaxPooling1D(2),
            Conv1D(64, 3, activation='relu'),
            MaxPooling1D(2),
            Conv1D(64, 3, activation='relu'),
            Flatten(),
            Dense(64, activation='relu'),
            Dense(1)
        ])
        self.model.compile(optimizer='adam', loss='mse')

    def fit(self, X, y, epochs=50, batch_size=32):
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)

    def predict(self, X):
        return self.model.predict(X).flatten()