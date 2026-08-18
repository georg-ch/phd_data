from __future__ import annotations

import pandas as pd

from scripts.paths import DATA_DIR
from src.plotting.distribution_similarity import compare_distribution_similarity
from src.plotting.plot_config import load_plot_config
from src.plotting.plotting_export import write_html

from scripts.plotting.duration_density_common import (
    build_duration_violin_figure,
    configured_duration_colors,
    duration_years_from_dates,
)


def yearmonth_to_year(years: pd.Series, months: pd.Series) -> pd.Series:
    return pd.to_numeric(years, errors="coerce") + (
        pd.to_numeric(months, errors="coerce") - 1
    ) / 12


def compute_double_long_side_from_start(start_data: pd.DataFrame) -> pd.Series:
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
        double_long_side = compute_double_long_side_from_start(start_data)
        ci_mask = (double_long_side > ci_threshold) | double_long_side.isna()
    else:
        raise ValueError(f"Unknown mode: {mode}")

    duration_combined[ci_mask] = pd.NA
    duration_combined[start_data["duration_computed"].notna()] = start_data[
        "duration_computed"
    ]
    return duration_combined


def build_duration_densities_from_start(feature_set: str) -> None:
    """Write the start-based duration-density plot for one source feature set."""
    params, params_global, out_path = load_plot_config(
        f"duration_densities_from_start_{feature_set}"
    )
    duration_colors = configured_duration_colors(params_global)

    show_linkedin_and_model = params["show_linkedin_and_model"]
    use_model = params["use_model"]
    ci_threshold = params["ci_threshold"]
    ci_mode = params["ci_mode"]
    always_dense_violin_style = bool(params.get("always_dense_violin_style", False))
    dense_violin_width = float(params.get("dense_violin_width", 0.5))
    dense_box_width = float(params.get("dense_box_width", 0.12))
    flip_linkedin_violin = bool(params.get("flip_linkedin_violin", False))
    linkedin_violin_scale = float(params.get("linkedin_violin_scale", 0.9))

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
        ("Verteidigung - Annahme", duration_accest, duration_colors["defense_minus_acceptance"], 0.55),
        ("LinkedIn-Dauer", duration_linkedin, duration_colors["linkedin"], 0.38),
    ]

    if show_linkedin_and_model:
        plot_specs.append((combined_label, duration_combined, duration_colors["model"], 0.62))
    elif use_model:
        plot_specs[-1] = (combined_label, duration_combined, duration_colors["linkedin"], 0.55)

    plot_specs.append(("Verteidigung - PBA", duration_pbaest, duration_colors["defense_minus_pba"], 0.55))

    if show_linkedin_and_model:
        x_positions = [0.0, 0.46, 0.50, 0.96]
        x_tickvals = [0.0, 0.48, 0.96]
        x_ticktext = [
            "Verteidigung - Annahme",
            "LinkedIn / Modell",
            "Verteidigung - PBA",
        ]
        x_padding = 0.26
    else:
        x_step = 0.42
        x_positions = [i * x_step for i in range(len(plot_specs))]
        x_tickvals = x_positions
        x_ticktext = [label for label, _values, _color, _opacity in plot_specs]
        x_padding = 0.24

    fig = build_duration_violin_figure(
        plot_specs=plot_specs,
        x_positions=x_positions,
        x_tickvals=x_tickvals,
        x_ticktext=x_ticktext,
        x_padding=x_padding,
        yaxis_title="Dauer (Jahre)",
        yaxis_range=(-0.5, 15),
        always_dense_violin_style=always_dense_violin_style,
        dense_violin_width=dense_violin_width,
        dense_box_width=dense_box_width,
        show_linkedin_and_model=show_linkedin_and_model,
        flip_linkedin_violin=flip_linkedin_violin,
        linkedin_violin_scale=linkedin_violin_scale,
    )

    write_html(fig, out_path, trace_map=None, plot_type="violin")
