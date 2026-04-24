import math
import re

from matplotlib.figure import Figure

from home_price_demo import constants
from home_price_demo.session import DemoState


def _predicted_from_html(html: str) -> float:
    m = re.search(r"\$[\d,]+", html)
    assert m, "expected a dollar amount in HTML"
    return float(m.group(0).replace("$", "").replace(",", ""))


def test_update_returns_html_and_figure():
    state = DemoState.create(seed=7)
    html, fig = state.update(2000.0)
    assert isinstance(fig, Figure)
    assert "$" in html
    assert "sq ft" in html.lower()


def test_update_at_slider_extremes():
    state = DemoState.create(seed=3)
    for sq in (constants.SQFT_MIN, constants.SQFT_MAX):
        html, fig = state.update(sq)
        assert isinstance(fig, Figure)
        assert math.isfinite(_predicted_from_html(html))
