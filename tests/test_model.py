import math

import numpy as np

from home_price_demo.data import generate_dataset
from home_price_demo.model import predict_price, train_model


def test_predict_scalar_shape():
    sqft, price = generate_dataset(n=100, seed=3)
    model = train_model(sqft, price)
    p = predict_price(model, 2000.0)
    assert isinstance(p, float)
    assert math.isfinite(p)


def test_positive_slope_implies_monotonic_predictions():
    sqft, price = generate_dataset(n=100, seed=11)
    model = train_model(sqft, price)
    assert model.coef_[0] > 0
    assert predict_price(model, 1000.0) < predict_price(model, 4000.0)


def test_reproducible_predictions():
    sqft, price = generate_dataset(n=100, seed=99)
    model = train_model(sqft, price)
    a = predict_price(model, 2222.0)
    sqft2, price2 = generate_dataset(n=100, seed=99)
    model2 = train_model(sqft2, price2)
    b = predict_price(model2, 2222.0)
    assert a == b
