from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from scripts.paths import DATA_DIR
from src.plotting.plot_config import load_plot_config
from src.plotting.plotting_export import write_html
from scripts.plotting.duration_density_common import configured_duration_colors


def _yearmonth_to_year(years: pd.Series, months: pd.Series) -> pd.Series:
    return pd.to_numeric(years, errors="coerce") + (pd.to_numeric(months, errors="coerce") - 1) / 12


def _add_violin_trace(
    fig: go.Figure,
    values: pd.Series,
    x_position: float,
    name: str,
    color: str,
    width: float,
    opacity: float,
    side: str | None = None,
) -> None:
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


def _add_box_trace(
    fig: go.Figure,
    values: pd.Series,
    x_position: float,
    name: str,
    color: str,
    width: float,
    line_width: float,
) -> None:
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


def build_start_violin(feature_set: str) -> None:
    params, params_global, out_path = load_plot_config(f"start_densities_{feature_set}")
    duration_colors = configured_duration_colors(params_global, start=True)
    always_dense_violin_style = bool(params.get("always_dense_violin_style", False))
    dense_violin_width = float(params.get("dense_violin_width", 0.5))
    dense_box_width = float(params.get("dense_box_width", 0.12))

    data = pd.read_csv(DATA_DIR / f"start_predictions_{feature_set}.csv")
    required_cols = [
        "start_year",
        "start_month",
        "start_year_est",
        "start_month_est",
        "start_year_est_clipped",
        "start_month_est_clipped",
    ]
    data = data.loc[data[required_cols].notna().all(axis=1)].copy()

    actual_start = _yearmonth_to_year(data["start_year"], data["start_month"])
    model_start = _yearmonth_to_year(data["start_year_est"], data["start_month_est"])
    model_start_clipped = _yearmonth_to_year(
        data["start_year_est_clipped"], data["start_month_est_clipped"]
    )

    plot_specs = [
        ("LinkedIn-Start", actual_start, duration_colors["linkedin_start"], 0.42),
        ("Modell-Start", model_start, duration_colors["model_start"], 0.58),
        ("Modell-Start (geclippt)", model_start_clipped, duration_colors["model_start_clipped"], 0.62),
    ]

    x_positions = [0.0, 0.48, 0.96]
    x_tickvals = x_positions
    x_ticktext = [label for label, _values, _color, _opacity in plot_specs]
    x_padding = 0.24

    if always_dense_violin_style:
        violin_width = dense_violin_width
        box_width = dense_box_width
        violin_side = "positive"
    else:
        violin_width = 0.34
        box_width = 0.08
        violin_side = None

    fig = go.Figure()
    for x_position, (label, values, color, opacity) in zip(
        x_positions, plot_specs, strict=True
    ):
        _add_violin_trace(
            fig,
            values,
            x_position,
            label,
            color,
            violin_width,
            opacity,
            side=violin_side,
        )
    for idx, (x_position, (label, values, color, _opacity)) in enumerate(
        zip(x_positions, plot_specs, strict=True)
    ):
        box_line_width = 2.5 if idx == 2 else 2
        _add_box_trace(fig, values, x_position, label, color, box_width, box_line_width)

    y_min = float(pd.concat([actual_start, model_start, model_start_clipped]).min())
    y_max = float(pd.concat([actual_start, model_start, model_start_clipped]).max())

    fig.update_layout(
        violinmode="group",
        showlegend=False,
        hovermode="closest",
        violingap=0.02,
        violingroupgap=0.02,
        margin=dict(l=60, r=30, t=40, b=90),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(
            title="Startjahr",
            showgrid=True,
            gridcolor="rgba(0,0,0,0.15)",
            zeroline=False,
            range=[y_min - 0.75, y_max + 0.75],
            showspikes=True,
            spikemode="across",
            spikesnap="cursor",
            spikethickness=1,
            automargin=True,
        ),
        xaxis=dict(
            title="",
            tickmode="array",
            tickvals=x_tickvals,
            ticktext=x_ticktext,
            range=[x_positions[0] - x_padding, x_positions[-1] + x_padding],
            automargin=True,
        ),
    )

    write_html(fig, out_path, trace_map=None, plot_type="violin")
