"""Training routine with early stopping and timing."""
import time
import numpy as np
from tensorflow import keras

import config
from src.models import build_model


def set_seed(seed=config.SEED):
    import random
    import tensorflow as tf
    random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)


def train_model(kind, ds, epochs=config.EPOCHS, batch_size=config.BATCH_SIZE,
                verbose=2):
    set_seed()
    model = build_model(kind, ds.window, ds.n_features)
    cbs = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=config.PATIENCE,
                                      restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                          patience=3, min_lr=1e-5),
    ]
    t0 = time.time()
    history = model.fit(ds.X_train, ds.y_train, epochs=epochs,
                        batch_size=batch_size, validation_split=config.VAL_SPLIT,
                        shuffle=False, callbacks=cbs, verbose=verbose)
    return model, history.history, time.time() - t0
