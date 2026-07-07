import numpy as np
import json
import tomllib
from pathlib import Path
import pandas as pd

from scripts.paths import DATA_DIR, CONFIG_DIR
from src.data_processing.duration_estimation import (
    engineer_features,
    cross_validated_model_estimates,
    calculate_errors,
    summarize_ci_thresholds,
    visualize_binned_errors,
    scatter_gap_ci,
    get_combined_duration,
)
from src.data_loading.load_data import load_data
import matplotlib.pyplot as plt
from aquarel import load_theme


def plot_duration_comparison(
    duration_data,
    ci_thereshold,
    out_path,
    out_fname,
    theme,
    bin_range=(0, 20),
    transform_func=None,
):
    duration_combined = get_combined_duration(duration_data, ci_thereshold)
    duration_original = duration_data["duration_computed"].copy()
    if transform_func is not None:
        duration_combined = transform_func(duration_combined)
        duration_original = transform_func(duration_original)

    fig, ax = plt.subplots(figsize=(10, 6))

    bin_edges = np.linspace(bin_range[0], bin_range[1], 50)
    ax.hist(
        duration_original,
        bins=bin_edges,
        alpha=0.5,
        label=f"Duration LinkedIn (mean: {duration_original.mean():.2f} yrs, median: {duration_data['duration_computed'].median():.2f} yrs)",
        density=True,
    )
    ax.hist(
        duration_combined,
        bins=bin_edges,
        alpha=0.5,
        label=f"Duration w/ model (mean: {duration_combined.mean():.2f} yrs, median: {duration_combined.median():.2f} yrs)",
        density=True,
    )
    ax.legend()
    theme.apply_transforms()

    plt.savefig(out_path / out_fname, dpi=300)


def main():
    theme = load_theme("scientific")

    data = load_data(DATA_DIR / "clean_data.csv", DATA_DIR / "cleaned_data_dtypes.json")
    duration_data = pd.read_csv(DATA_DIR / "duration_predictions.csv")

    print(duration_data.columns.values)

    ci_thereshold = 1.5
    out_path = DATA_DIR / "duration_eval_plots"
    out_fname = "duration_density_comparison.png"

    plot_duration_comparison(
        duration_data,
        ci_thereshold,
        out_path,
        out_fname,
        theme,
        transform_func=np.log,
        bin_range=(0, 3),
    )
    # duration_combined = get_combined_duration(duration_data, ci_thereshold)


if __name__ == "__main__":
    main()
