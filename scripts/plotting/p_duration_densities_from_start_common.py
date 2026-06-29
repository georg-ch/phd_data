from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from scripts.paths import DATA_DIR
from src.plotting.distribution_similarity import compare_distribution_similarity
from src.plotting.plot_config import load_plot_config
from src.plotting.plotting_export import write_html


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
) -> None:
    clean = values.dropna()
    fig.add_trace(
        go.Violin(
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
    )


def add_box_trace(
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


def _compute_double_long_side_from_start(start_data: pd.DataFrame) -> pd.Series:
    left_side = (
        start_data["duration_estimate_clipped_s"] - start_data["lower_bound_adjusted_s"]
    )
    right_side = (
        start_data["upper_bound_adjusted_s"] - start_data["duration_estimate_clipped_s"]
    )
    return 2 * pd.concat([left_side, right_side], axis=1).max(axis=1)


def get_combined_start_duration(
    start_data: pd.DataFrame, ci_threshold: float, mode: str = "adjusted_ci"
) -> pd.Series:
    duration_combined = start_data["duration_estimate_clipped_s"].copy()
    if mode == "adjusted_ci":
        ci_mask = (start_data["ci_size_adjusted_s"] > ci_threshold) | start_data[
            "ci_size_adjusted_s"
        ].isna()
    elif mode == "double_long_side":
        double_long_side = _compute_double_long_side_from_start(start_data)
        ci_mask = (double_long_side > ci_threshold) | double_long_side.isna()
    else:
        raise ValueError(f"Unknown mode: {mode}")

    duration_combined[ci_mask] = pd.NA
    duration_combined[start_data["duration_computed"].notna()] = start_data[
        "duration_computed"
    ]
    return duration_combined


def build_duration_densities_from_start(feature_set: str) -> None:
    params, params_global, out_path = load_plot_config(
        f"duration_densities_from_start_{feature_set}"
    )
    color_cycle = params_global["color_cycle"]

    show_linkedin_and_model = params["show_linkedin_and_model"]
    use_model = params["use_model"]
    ci_threshold = params["ci_threshold"]
    ci_mode = params["ci_mode"]

    start_data = pd.read_csv(DATA_DIR / f"start_predictions_{feature_set}.csv")
    combined_duration = get_combined_start_duration(
        start_data, ci_threshold, mode=ci_mode
    )

    duration_linkedin = start_data["duration_computed"]
    duration_combined = combined_duration
    combined_label = f"LinkedIn + Startmodell (CI <= {ci_threshold})"

    similarity = compare_distribution_similarity(duration_linkedin, duration_combined)
    ln_median = duration_linkedin.dropna().median()

    print(
        f"Duration distribution comparison (LinkedIn vs start({feature_set})-model): "
        f"p={similarity['p_value']:.4g} (n={similarity['n_x']}), indicating a detectable difference. "
        f"However, the effect size is small: the distributions differ by only "
        f"{12 * similarity['wasserstein']:.2f} months on average "
        f"({similarity['wasserstein'] / ln_median * 100:.2f}% of a typical duration)."
    )

    data_defended = start_data[~start_data["defense_date"].isna()]
    duration_accest = duration_years_from_dates(
        data_defended["defense_date"], data_defended["acceptance_date"]
    )
    duration_pbaest = duration_years_from_dates(
        data_defended["defense_date"], data_defended["pba_date"]
    )

    plot_specs = [
        ("Verteidigung - Annahme", duration_accest, color_cycle[1], 0.55),
        ("LinkedIn-Dauer", duration_linkedin, color_cycle[0], 0.38),
    ]

    if show_linkedin_and_model:
        plot_specs.append((combined_label, duration_combined, color_cycle[2], 0.62))
    elif use_model:
        plot_specs[-1] = (combined_label, duration_combined, color_cycle[0], 0.55)

    plot_specs.append(("Verteidigung - PBA", duration_pbaest, color_cycle[3], 0.55))

    if show_linkedin_and_model:
        x_positions = [0.0, 0.44, 0.52, 0.96]
        x_tickvals = [0.0, 0.48, 0.96]
        x_ticktext = [
            "Verteidigung - Annahme",
            "LinkedIn / Modell",
            "Verteidigung - PBA",
        ]
        violin_width = 0.34
        box_width = 0.08
        x_padding = 0.26
    else:
        x_step = 0.42
        x_positions = [i * x_step for i in range(len(plot_specs))]
        x_tickvals = x_positions
        x_ticktext = [label for label, _values, _color, _opacity in plot_specs]
        violin_width = 0.5
        box_width = 0.12
        x_padding = 0.24

    fig = go.Figure()
    for x_position, (label, values, color, opacity) in zip(
        x_positions, plot_specs, strict=True
    ):
        add_violin_trace(fig, values, x_position, label, color, violin_width, opacity)
    for idx, (x_position, (label, values, color, _opacity)) in enumerate(
        zip(x_positions, plot_specs, strict=True)
    ):
        box_line_width = 2.5 if show_linkedin_and_model and idx == 2 else 2
        add_box_trace(fig, values, x_position, label, color, box_width, box_line_width)

    y_min = float(
        pd.concat(
            [duration_accest, duration_linkedin, duration_combined, duration_pbaest]
        ).min()
    )
    y_max = float(
        pd.concat(
            [duration_accest, duration_linkedin, duration_combined, duration_pbaest]
        ).max()
    )

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
            title="Dauer (Jahre)",
            showgrid=True,
            gridcolor="rgba(0,0,0,0.15)",
            zeroline=False,
            range=[-0.5, 15],
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
