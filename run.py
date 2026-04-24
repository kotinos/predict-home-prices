"""Launch the Gradio demo from the repo root (sets import path)."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from home_price_demo.app import main  # noqa: E402

if __name__ == "__main__":
    main()
