"""LSTM and GRU architectures (identical except the recurrent cell)."""
from tensorflow import keras
from tensorflow.keras import layers

import config


def build_model(kind: str, window: int, n_features: int,
                units=config.UNITS, dropout=config.DROPOUT,
                lr=config.LEARNING_RATE) -> keras.Model:
    kind = kind.upper()
    Cell = {"LSTM": layers.LSTM, "GRU": layers.GRU}[kind]

    model = keras.Sequential(name=f"{kind}_model")
    model.add(layers.Input(shape=(window, n_features)))
    for i, u in enumerate(units):
        last = i == len(units) - 1
        model.add(Cell(u, return_sequences=not last))
        model.add(layers.Dropout(dropout))
    model.add(layers.Dense(16, activation="relu"))
    model.add(layers.Dense(1))
    model.compile(optimizer=keras.optimizers.Adam(lr), loss="mse", metrics=["mae"])
    return model
