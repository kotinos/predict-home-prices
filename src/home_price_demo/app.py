"""Gradio entrypoint: slider, predicted price, and regression plot."""

from __future__ import annotations

import gradio as gr

from home_price_demo import constants
from home_price_demo.session import DemoState

_SLIDER_CSS = """
#sqft-slider-card {
  border: 2px solid #805ad5;
  border-radius: 14px;
  padding: 18px 20px 22px;
  background: linear-gradient(180deg, #faf5ff 0%, #f7fafc 100%);
  box-shadow: 0 4px 14px rgba(107, 70, 193, 0.12);
}
#sqft-slider-card label { font-weight: 700 !important; font-size: 1.05rem !important; }
"""


def build_demo(seed: int = constants.DEFAULT_SEED) -> gr.Blocks:
    state = DemoState.create(seed=seed)
    initial_sq = constants.DEFAULT_SLIDER_VALUE
    with gr.Blocks(theme=gr.themes.Soft(primary_hue="purple"), css=_SLIDER_CSS) as demo:
        gr.Markdown(
            "# Home price explorer\n"
            "Interactive **demo**: linear regression on **100 synthetic** listings "
            "(not real market data)."
        )
        with gr.Column(elem_id="sqft-slider-card"):
            sq_slider = gr.Slider(
                minimum=int(constants.SQFT_MIN),
                maximum=int(constants.SQFT_MAX),
                value=int(initial_sq),
                step=constants.SLIDER_STEP,
                label="House size (square feet)",
            )
        price_html = gr.HTML()
        plot = gr.Plot(label="Model & your selection", container=True)
        sq_slider.change(state.update, inputs=sq_slider, outputs=[price_html, plot])
        demo.load(lambda: state.update(initial_sq), outputs=[price_html, plot])
    return demo


def main() -> None:
    build_demo().launch()


if __name__ == "__main__":
    main()
