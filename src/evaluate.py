"""Metrics + recursive multi-day forecasting."""
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.preprocessing import inverse_target


def metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
    r2 = float(r2_score(y_true, y_pred))
    # Directional accuracy: was the up/down move predicted correctly?
    da = float(np.mean(np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred))) * 100)
    return {"RMSE": rmse, "MAE": mae, "MAPE(%)": mape, "R2": r2, "DirAcc(%)": da}


def predict_test(model, ds):
    pred = model.predict(ds.X_test, verbose=0).ravel()
    return inverse_target(ds, ds.y_test), inverse_target(ds, pred)


def forecast_future(model, ds, days=30):
    """Recursive forecast. For multivariate input, non-target features are
    held at their last known value."""
    window = ds.scaled_all[-ds.window:].copy()
    out = []
    for _ in range(days):
        p = model.predict(window[np.newaxis], verbose=0)[0, 0]
        out.append(p)
        nxt = window[-1].copy()
        nxt[ds.target_idx] = p
        window = np.vstack([window[1:], nxt])
    return inverse_target(ds, np.array(out))
