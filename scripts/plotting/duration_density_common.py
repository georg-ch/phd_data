from __future__ import annotations

from typing import Sequence

import pandas as pd
import plotly.graph_objects as go


DurationPlotSpec = tuple[str, pd.Series, str, float]


def duration_years_from_dates(later: pd.Series, earlier: pd.Series) -> pd.Series:
    """Return a duration in fractional years from two day-month-year string columns."""
    return (
        pd.to_datetime(later, errors="coerce", format="%d-%m-%Y")
        - pd.to_datetime(earlier, errors="coerce", format="%d-%m-%Y")
    ).dt.days / 365


def add_violin_trace(
    fig: go.Figure,
    values: pd.Series,
    x_position: float,
    name: str,
    color: str,
    width: float,
    opacity: float,
    side: str | None = None,
) -> None:
    """Add one styled violin trace centered on a numeric x-position."""
    clean = values.dropna()
    violin_kwargs = dict(
        x=[x_position] * len(clean),
        y=clean,
        name=name,
        line=dict(color=color, width=2),
        fillcolor=color,
        opacity=opacity,
        width=width,
        box_visible=False,
        meanline_visible=False,
        points=False,
        spanmode="hard",
        scalemode="width",
        hoveron="kde",
        hoverinfo="y",
        yhoverformat=".2f",
    )
    if side is not None:
        violin_kwargs["side"] = side
    fig.add_trace(go.Violin(**violin_kwargs))


def add_box_trace(
    fig: go.Figure,
    values: pd.Series,
    x_position: float,
    name: str,
    color: str,
    width: float,
    line_width: float,
) -> None:
    """Overlay a non-hoverable box trace so median and mean remain visible."""
    clean = values.dropna()
    if clean.empty:
        return

    fig.add_trace(
        go.Box(
            x=[x_position] * len(clean),
            y=clean,
            name=name,
            width=width,
            fillcolor="rgba(255,255,255,0.0)",
            line=dict(color=color, width=line_width),
            marker=dict(opacity=0),
            showlegend=False,
            boxpoints=False,
            boxmean=True,
            hoverinfo="skip",
        )
    )


def build_duration_violin_figure(
    *,
    plot_specs: Sequence[DurationPlotSpec],
    x_positions: Sequence[float],
    x_tickvals: Sequence[float],
    x_ticktext: Sequence[str],
    x_padding: float,
    yaxis_title: str,
    yaxis_range: Sequence[float] | None,
    always_dense_violin_style: bool,
    dense_violin_width: float,
    dense_box_width: float,
    show_linkedin_and_model: bool,
    flip_linkedin_violin: bool,
    linkedin_violin_scale: float,
    model_box_index: int = 2,
) -> go.Figure:
    """Render the shared duration-density violin layout without changing trace ordering."""
    if always_dense_violin_style:
        violin_width = dense_violin_width
        box_width = dense_box_width
        violin_side = "positive"
    elif show_linkedin_and_model:
        violin_width = 0.34
        box_width = 0.08
        violin_side = None
    else:
        violin_width = 0.5
        box_width = 0.12
        violin_side = None

    if show_linkedin_and_model:
        violin_width *= linkedin_violin_scale
        box_width *= linkedin_violin_scale

    fig = go.Figure()
    for x_position, (label, values, color, opacity) in zip(
        x_positions, plot_specs, strict=True
    ):
        side = violin_side
        if flip_linkedin_violin and label == "LinkedIn-Dauer":
            side = "negative"
        add_violin_trace(
            fig,
            values,
            x_position,
            label,
            color,
            violin_width,
            opacity,
            side=side,
        )
    for idx, (x_position, (label, values, color, _opacity)) in enumerate(
        zip(x_positions, plot_specs, strict=True)
    ):
        box_line_width = 2.5 if show_linkedin_and_model and idx == model_box_index else 2
        add_box_trace(fig, values, x_position, label, color, box_width, box_line_width)

    if yaxis_range is None:
        yaxis = dict(
            title=yaxis_title,
            showgrid=True,
            gridcolor="rgba(0,0,0,0.15)",
            zeroline=False,
            showspikes=True,
            spikemode="across",
            spikesnap="cursor",
            spikethickness=1,
            automargin=True,
        )
    else:
        yaxis = dict(
            title=yaxis_title,
            showgrid=True,
            gridcolor="rgba(0,0,0,0.15)",
            zeroline=False,
            range=list(yaxis_range),
            showspikes=True,
            spikemode="across",
            spikesnap="cursor",
            spikethickness=1,
            automargin=True,
        )

    layout_kwargs = dict(
        violinmode="group",
        showlegend=False,
        hovermode="closest",
        violingap=0.02,
        violingroupgap=0.02,
        margin=dict(l=60, r=30, t=40, b=90),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=yaxis,
        xaxis=dict(
            title="",
            tickmode="array",
            tickvals=x_tickvals,
            ticktext=x_ticktext,
            range=[x_positions[0] - x_padding, x_positions[-1] + x_padding],
            automargin=True,
        ),
    )

    fig.update_layout(**layout_kwargs)
    return fig
