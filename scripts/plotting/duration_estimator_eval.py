import numpy as np
import json
import tomllib
from pathlib import Path

from scripts.paths import DATA_DIR, CONFIG_DIR
from src.data_processing.duration_estimation import (
    engineer_features,
    cross_validated_model_estimates,
    calculate_errors,
    summarize_ci_thresholds,
    visualize_binned_errors,
)
from src.data_loading.load_data import load_data
import matplotlib.pyplot as plt
from aquarel import load_theme


def main():
    """Run cross-validated duration evaluation, save plots, and write CI summaries."""
    theme = load_theme("scientific")
    config_path = CONFIG_DIR / "duration_estimation_params.toml"
    with open(config_path, "rb") as f:
        config = tomllib.load(f)
    plot_dir = Path(config["eval_plot_dir"])

    data = load_data(DATA_DIR / "clean_data.csv", DATA_DIR / "cleaned_data_dtypes.json")
    data, feature_col_info = engineer_features(data)
    with open(DATA_DIR / "best_params_RMSEWithUncertainty.json", "r") as f:
        params_unc = json.load(f)

    cv_result = cross_validated_model_estimates(
        data, "duration_computed", params_unc, feature_col_info
    )
    cv_result = calculate_errors(cv_result)

    bin_edges = [0, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 10, 15]
    bin_labels = [
        "0-0.5",
        "0.5-1",
        "1-2",
        "2-3",
        "3-4",
        "4-5",
        "5-6",
        "6-7",
        "7-8",
        "8-10",
        "10-15",
    ]

    visualize_binned_errors(
        cv_result, bin_edges, bin_labels, plot_dir, theme, bin_var="gap_size"
    )
    visualize_binned_errors(
        cv_result, bin_edges, bin_labels, plot_dir, theme, bin_var="ci_size_adjusted"
    )
    ci_summary = summarize_ci_thresholds(
        cv_result,
        out_path=DATA_DIR / "duration_estimator_ci_threshold_summary.csv",
    )
    print(ci_summary.to_string(index=False))


if __name__ == "__main__":
    main()
