import numpy as np
from src import data_loader, preprocessing, evaluate


def test_data_and_windows():
    df = data_loader.load_data("TEST", source="synthetic",
                               start="2020-01-01", end="2021-12-31")
    ds = preprocessing.prepare(df, window=30)
    assert ds.X_train.shape[1:] == (30, 1)
    assert len(ds.X_test) == len(ds.test_dates)


def test_metrics_perfect():
    y = np.linspace(10, 20, 50)
    m = evaluate.metrics(y, y)
    assert m["RMSE"] == 0 and m["R2"] == 1
