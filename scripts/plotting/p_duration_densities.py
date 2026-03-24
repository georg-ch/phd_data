from src.data_loading.load_data import load_data
from src.plotting.plot_config import load_plot_config
from src.plotting.plotting_export import write_html
from scripts.paths import DATA_DIR
import pandas as pd
import plotly.graph_objects as go
from src.data_processing.duration_estimation import get_combined_duration


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
    """Add one styled violin trace centered on a numeric x-position."""
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


def main():
    """Write a violin-plot comparison of duration distributions to the configured HTML path."""
    params, params_global, out_path = load_plot_config("duration_densities")

    color_cycle = params_global["color_cycle"]

    use_model = params["use_model"]
    show_linkedin_and_model = params["show_linkedin_and_model"]
    ci_threshold = params["ci_threshold"]

    data = load_data(DATA_DIR / "clean_data.csv", DATA_DIR / "cleaned_data_dtypes.json")
    duration_data = pd.read_csv(DATA_DIR / "duration_predictions.csv")
    combined_duration = get_combined_duration(duration_data, ci_threshold)

    duration_linkedin = duration_data["duration_computed"]
    duration_combined = combined_duration
    combined_label = f"LinkedIn + Modell (CI <= {ci_threshold})"

    data_defended = data[~data["defense_date"].isna()]
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


if __name__ == "__main__":
    main()
