"""Scaling, windowing and chronological train/test split."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import config


@dataclass
class Dataset:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    test_dates: pd.Index
    scaler: MinMaxScaler
    target_idx: int
    n_features: int
    scaled_all: np.ndarray
    window: int


def make_sequences(data: np.ndarray, target_idx: int, window: int):
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i - window:i])
        y.append(data[i, target_idx])
    return np.array(X), np.array(y)


def prepare(df: pd.DataFrame, window=config.WINDOW, features=config.FEATURES,
            target=config.TARGET, train_ratio=config.TRAIN_RATIO) -> Dataset:
    values = df[features].values.astype("float32")
    split = int(len(values) * train_ratio)

    # Fit scaler ONLY on the training part -> no data leakage
    scaler = MinMaxScaler()
    scaler.fit(values[:split])
    scaled = scaler.transform(values)
    t_idx = features.index(target)

    X_all, y_all = make_sequences(scaled, t_idx, window)
    seq_split = split - window          # first test target is day `split`
    return Dataset(
        X_train=X_all[:seq_split], y_train=y_all[:seq_split],
        X_test=X_all[seq_split:], y_test=y_all[seq_split:],
        test_dates=df.index[split:], scaler=scaler, target_idx=t_idx,
        n_features=len(features), scaled_all=scaled, window=window)


def inverse_target(ds: Dataset, arr: np.ndarray) -> np.ndarray:
    """Inverse-scale a 1-D array of target values back to price units."""
    dummy = np.zeros((len(arr), ds.n_features), dtype="float32")
    dummy[:, ds.target_idx] = np.asarray(arr).ravel()
    return ds.scaler.inverse_transform(dummy)[:, ds.target_idx]
