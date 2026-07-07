from __future__ import annotations

import pandas as pd

from scripts.paths import DATA_DIR
from src.data_loading.load_data import load_data
from src.data_processing.duration_estimation import get_combined_duration
from src.plotting.distribution_similarity import compare_distribution_similarity
from src.plotting.plot_config import load_plot_config
from src.plotting.plotting_export import write_html

from scripts.plotting.duration_density_common import (
    build_duration_violin_figure,
    duration_years_from_dates,
)


def main():
    """Write the duration-density comparison using the simple duration model."""
    params, params_global, out_path = load_plot_config("duration_densities_simple")
    color_cycle = params_global["color_cycle"]

    show_linkedin_and_model = params["show_linkedin_and_model"]
    use_model = params["use_model"]
    ci_threshold = params["ci_threshold"]
    ci_mode = params["ci_mode"]
    always_dense_violin_style = bool(params.get("always_dense_violin_style", False))
    dense_violin_width = float(params.get("dense_violin_width", 0.5))
    dense_box_width = float(params.get("dense_box_width", 0.12))
    flip_linkedin_violin = bool(params.get("flip_linkedin_violin", False))
    linkedin_violin_scale = float(params.get("linkedin_violin_scale", 0.9))

    data = load_data(DATA_DIR / "clean_data.csv", DATA_DIR / "cleaned_data_dtypes.json")
    duration_data = pd.read_csv(DATA_DIR / "duration_predictions_simple_model.csv")
    duration_combined = get_combined_duration(duration_data, ci_threshold, mode=ci_mode)

    duration_linkedin = duration_data["duration_computed"]
    simple_label = f"LinkedIn + Einfaches Modell (CI <= {ci_threshold})"

    similarity = compare_distribution_similarity(duration_linkedin, duration_combined)
    ln_median = duration_linkedin.dropna().median()

    print(
        "Duration distribution comparison (LinkedIn vs simple model): "
        f"p={similarity['p_value']:.4g} (n={similarity['n_x']}), indicating a detectable difference. "
        f"However, the effect size is small: the distributions differ by only "
        f"{12 * similarity['wasserstein']:.2f} months on average "
        f"({similarity['wasserstein'] / ln_median * 100:.2f}% of a typical duration)."
    )

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
        plot_specs.append((simple_label, duration_combined, color_cycle[2], 0.62))
    elif use_model:
        plot_specs[-1] = (simple_label, duration_combined, color_cycle[0], 0.55)

    plot_specs.append(("Verteidigung - PBA", duration_pbaest, color_cycle[3], 0.55))

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


if __name__ == "__main__":
    main()
