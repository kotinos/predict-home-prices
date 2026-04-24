"""Synthetic square-footage and price data for the demo model."""

from __future__ import annotations

import numpy as np

from home_price_demo import constants


def generate_dataset(
    n: int = constants.N_SAMPLES,
    seed: int | None = constants.DEFAULT_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(sqft, price)`` arrays with ``n`` realistic synthetic samples.

    Uses a linear base with Gaussian noise and a price floor so all values stay
    positive and roughly house-like. Square footage spans the demo slider range
    so the fitted model does not extrapolate far outside training support.
    """
    if n < 2:
        raise ValueError("n must be at least 2 for a regression demo")
    rng = np.random.default_rng(seed)
    sqft = rng.uniform(constants.SQFT_MIN, constants.SQFT_MAX, size=n)
    base_price_per_sqft = 125.0
    intercept = 35_000.0
    noise = rng.normal(0.0, 18_000.0, size=n)
    price = intercept + base_price_per_sqft * sqft + noise
    price = np.maximum(price, 45_000.0)
    return sqft.astype(np.float64), price.astype(np.float64)
