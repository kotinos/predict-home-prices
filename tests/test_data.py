import numpy as np
import pytest

from home_price_demo.data import generate_dataset


def test_default_length_is_100():
    sqft, price = generate_dataset()
    assert len(sqft) == len(price) == 100


def test_reproducible_with_seed():
    a1, p1 = generate_dataset(n=50, seed=7)
    a2, p2 = generate_dataset(n=50, seed=7)
    np.testing.assert_array_equal(a1, a2)
    np.testing.assert_array_equal(p1, p2)


def test_all_prices_positive():
    _, price = generate_dataset(n=100, seed=0)
    assert np.all(price > 0)


def test_sqft_within_slider_support():
    sqft, _ = generate_dataset(n=500, seed=1)
    assert np.all(sqft >= 500.0)
    assert np.all(sqft <= 5000.0)


def test_n_must_be_at_least_2():
    with pytest.raises(ValueError, match="at least 2"):
        generate_dataset(n=1, seed=0)
