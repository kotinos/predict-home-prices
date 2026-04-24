# Predict home prices (demo)

Interactive **Gradio** app that estimates a listing price from **square footage** using **scikit-learn linear regression**. Training data is **100 synthetic** points generated at startup—no CSV, no real market data.

## Stack

Python 3.10+ · Gradio · scikit-learn · NumPy · Matplotlib · pytest

## Quick start

```bash
python -m pip install -r requirements.txt
python run.py
```

Open the URL Gradio prints (usually `http://127.0.0.1:7860`). Drag the **house size** slider (500–5,000 sq ft); the **predicted price** updates immediately with a plot of the regression line and your point highlighted.

### Run tests

```bash
python -m pytest tests -q
```

## Project layout

| Path | Role |
|------|------|
| `run.py` | Adds `src` to the path and launches the app |
| `src/home_price_demo/` | Data, model, plotting, session, Gradio UI |
| `tests/` | Unit tests (matplotlib uses `Agg` in `conftest.py`) |

## Notes

- The model is a **toy illustration**; do not use for real pricing decisions.
- Dependencies are pinned loosely in `requirements.txt`; pin tighter for production deployments.
