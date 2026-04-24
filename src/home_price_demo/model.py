"""Train and run a linear regression on (sqft, price)."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression


def train_model(sqft: np.ndarray, price: np.ndarray) -> LinearRegression:
    X = np.asarray(sqft, dtype=np.float64).reshape(-1, 1)
    y = np.asarray(price, dtype=np.float64)
    model = LinearRegression()
    model.fit(X, y)
    return model


def predict_price(model: LinearRegression, sqft: float) -> float:
    X = np.array([[float(sqft)]], dtype=np.float64)
    return float(model.predict(X)[0])
