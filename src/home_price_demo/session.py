"""Train-once demo state: prediction + plot + HTML for Gradio."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LinearRegression

from home_price_demo import constants
from home_price_demo.data import generate_dataset
from home_price_demo.model import predict_price, train_model
from home_price_demo.plotting import make_plot


@dataclass(frozen=True)
class DemoState:
    train_sqft: np.ndarray
    train_price: np.ndarray
    model: LinearRegression

    @classmethod
    def create(cls, seed: int = constants.DEFAULT_SEED, n: int = constants.N_SAMPLES) -> DemoState:
        sqft, price = generate_dataset(n=n, seed=seed)
        model = train_model(sqft, price)
        return cls(train_sqft=sqft, train_price=price, model=model)

    @staticmethod
    def format_price(value: float) -> str:
        return f"${value:,.0f}"

    def update(self, sqft: float) -> tuple[str, object]:
        sq = float(sqft)
        pred = predict_price(self.model, sq)
        html = (
            '<div style="text-align:center;margin:8px 0 4px;">'
            f'<span style="font-size:2.85rem;font-weight:800;line-height:1.1;'
            f'font-family:system-ui,Segoe UI,sans-serif;color:#1a202c;">'
            f"{self.format_price(pred)}</span></div>"
            f'<div style="text-align:center;color:#718096;font-size:1rem;">'
            f"Predicted price at <strong>{sq:,.0f}</strong> sq ft</div>"
        )
        fig = make_plot(self.model, self.train_sqft, self.train_price, sq, pred)
        return html, fig
