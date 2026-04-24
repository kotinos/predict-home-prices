"""Matplotlib figure: training scatter, regression line, highlighted user point."""

from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure
from sklearn.linear_model import LinearRegression

from home_price_demo import constants

ACCENT = "#00e5ff"


def make_plot(
    model: LinearRegression,
    train_sqft: np.ndarray,
    train_price: np.ndarray,
    user_sqft: float,
    user_price: float,
) -> Figure:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.scatter(
        train_sqft,
        train_price,
        alpha=0.5,
        color="#4a5568",
        s=42,
        label="Training (synthetic)",
        zorder=2,
    )
    xs = np.linspace(constants.SQFT_MIN, constants.SQFT_MAX, 200)
    ys = model.predict(xs.reshape(-1, 1))
    ax.plot(xs, ys, color="#6b46c1", linewidth=2.4, label="Regression line", zorder=3)
    ax.scatter(
        [user_sqft],
        [user_price],
        s=220,
        color=ACCENT,
        edgecolors="#1a202c",
        linewidths=1.8,
        zorder=6,
        label="Your selection",
    )
    ax.set_xlabel("Square feet", fontsize=11)
    ax.set_ylabel("Price (USD)", fontsize=11)
    ax.set_title("Home size vs. price (demo model)", fontsize=13, fontweight="600")
    ax.legend(loc="upper left", framealpha=0.92)
    ax.grid(True, alpha=0.28)
    fig.tight_layout()
    return fig
