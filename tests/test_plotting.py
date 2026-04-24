import matplotlib.colors as mcolors
import numpy as np
from matplotlib.figure import Figure

from home_price_demo.data import generate_dataset
from home_price_demo.model import predict_price, train_model
from home_price_demo.plotting import ACCENT, make_plot


def test_make_plot_returns_figure():
    sqft, price = generate_dataset(n=100, seed=5)
    model = train_model(sqft, price)
    u_sq = 2100.0
    u_pr = predict_price(model, u_sq)
    fig = make_plot(model, sqft, price, u_sq, u_pr)
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 1


def test_plot_contains_regression_line_and_highlight():
    sqft, price = generate_dataset(n=100, seed=8)
    model = train_model(sqft, price)
    u_sq = 1800.0
    u_pr = predict_price(model, u_sq)
    fig = make_plot(model, sqft, price, u_sq, u_pr)
    ax = fig.axes[0]
    assert len(ax.lines) >= 1, "expected regression line"
    assert len(ax.collections) >= 2, "expected training scatter + user marker"


def test_user_point_uses_accent_color():
    sqft, price = generate_dataset(n=100, seed=2)
    model = train_model(sqft, price)
    u_sq = 2500.0
    u_pr = predict_price(model, u_sq)
    fig = make_plot(model, sqft, price, u_sq, u_pr)
    ax = fig.axes[0]
    user_collection = ax.collections[-1]
    expected = np.array(mcolors.to_rgba(ACCENT))
    actual = user_collection.get_facecolor()[0]
    np.testing.assert_allclose(actual, expected, atol=0.06)
