from __future__ import annotations

import plotly.graph_objects as go

from src.plotting.plot_config import load_plot_config
from src.plotting.plotting_export import write_html


def _swatch_text_color(color: str) -> str:
    red = int(color[1:3], 16)
    green = int(color[3:5], 16)
    blue = int(color[5:7], 16)
    luminance = (0.299 * red) + (0.587 * green) + (0.114 * blue)
    return "#111111" if luminance >= 150 else "#FFFFFF"


def build_palette_preview(color_cycle: list[str], expanded_palette: list[str]) -> go.Figure:
    """Build a labeled swatch plot for the standard and expanded palettes."""
    palettes = [("Standardpalette", color_cycle), ("Erweiterte Palette", expanded_palette)]
    fig = go.Figure()

    for row, (label, palette) in enumerate(palettes):
        for index, color in enumerate(palette):
            fig.add_shape(
                type="rect",
                x0=index,
                x1=index + 1,
                y0=row - 0.35,
                y1=row + 0.35,
                line=dict(color="#FFFFFF", width=1),
                fillcolor=color,
            )
            fig.add_annotation(
                x=index + 0.5,
                y=row,
                text=f"{index + 1}<br>{color}",
                showarrow=False,
                font=dict(size=10, color=_swatch_text_color(color)),
            )

    max_length = max(len(palette) for _, palette in palettes)
    fig.update_layout(
        title="Plotfarben",
        height=230,
        width=max(900, 42 * max_length),
        margin=dict(l=150, r=30, t=60, b=35),
        shapes=list(fig.layout.shapes),
        annotations=list(fig.layout.annotations),
        xaxis=dict(
            range=[0, max_length],
            visible=False,
            fixedrange=True,
        ),
        yaxis=dict(
            range=[-0.7, len(palettes) - 0.3],
            tickvals=list(range(len(palettes))),
            ticktext=[label for label, _ in palettes],
            fixedrange=True,
            showgrid=False,
            zeroline=False,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    return fig


def main() -> None:
    params, params_global, out_path = load_plot_config("palette_preview")
    del params
    fig = build_palette_preview(
        params_global["color_cycle"],
        params_global["color_palette_expanded"],
    )
    write_html(fig, out_path, trace_map=None, plot_type="bar")


if __name__ == "__main__":
    main()
