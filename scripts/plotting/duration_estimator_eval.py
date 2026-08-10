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
)
from src.data_loading.load_data import load_data
import matplotlib.pyplot as plt
from aquarel import load_theme


def _load_params(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def _evaluate_model(
    *,
    data,
    params_unc,
    feature_col_info,
    plot_dir: Path,
    theme,
    label: str,
    target_col_name: str = "duration_computed",
):
    cv_result = cross_validated_model_estimates(
        data, target_col_name, params_unc, feature_col_info
    )
    cv_result = calculate_errors(cv_result)

    label_dir = plot_dir / label
    label_dir.mkdir(parents=True, exist_ok=True)

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
        cv_result, bin_edges, bin_labels, label_dir, theme, bin_var="gap_size"
    )
    visualize_binned_errors(
        cv_result, bin_edges, bin_labels, label_dir, theme, bin_var="double_long_side"
    )
    scatter_gap_ci(cv_result, label_dir, theme, ci_col="ci_size_adjusted")
    scatter_gap_ci(cv_result, label_dir, theme, ci_col="double_long_side")
    scatter_gap_ci(cv_result, label_dir, theme, ci_col="ci_size_raw")

    ci_summary = summarize_ci_thresholds(
        cv_result,
        out_path=DATA_DIR / f"duration_estimator_ci_threshold_summary_{label}.csv",
        ci_col="double_long_side",
    )

    summary = pd.Series(
        {
            "model": label,
            "n_rows": len(cv_result),
            "n_labeled": int(cv_result[target_col_name].notna().sum()),
            "rmse_estimate": float(np.sqrt(np.mean(cv_result["estimate_error"]))),
            "rmse_clipped": float(np.sqrt(np.mean(cv_result["estimate_error_clipped"]))),
            "rmse_naive": float(np.sqrt(np.mean(cv_result["naive_error"]))),
            "rmse_acc": float(np.sqrt(np.mean(cv_result["acc_error"]))),
            "mean_ci_adjusted": float(cv_result["ci_size_adjusted"].mean()),
            "mean_ci_raw": float(cv_result["ci_size_raw"].mean()),
        }
    )
    return summary, ci_summary


def main():
    """Run cross-validated duration evaluation, save plots, and write CI summaries."""
    theme = load_theme("scientific")
    config_path = CONFIG_DIR / "duration_estimation_params.toml"
    with open(config_path, "rb") as f:
        config = tomllib.load(f)
    plot_dir = Path(config["eval_plot_dir"])
    full_params_path = DATA_DIR / "best_params_RMSEWithUncertainty.json"
    simple_params_path = DATA_DIR / config["params_fname_for_simplified_model"]

    data = load_data(DATA_DIR / "clean_data.csv", DATA_DIR / "cleaned_data_dtypes.json")
    full_data, full_feature_col_info = engineer_features(data)
    simple_data, simple_feature_col_info = engineer_features(data, simplify=True)

    full_params_unc = _load_params(full_params_path)
    simple_params_unc = _load_params(simple_params_path)

    full_summary, full_ci_summary = _evaluate_model(
        data=full_data,
        params_unc=full_params_unc,
        feature_col_info=full_feature_col_info,
        plot_dir=plot_dir,
        theme=theme,
        label="full",
    )
    simple_summary, simple_ci_summary = _evaluate_model(
        data=simple_data,
        params_unc=simple_params_unc,
        feature_col_info=simple_feature_col_info,
        plot_dir=plot_dir,
        theme=theme,
        label="simple_temporal",
    )

    comparison = pd.DataFrame([full_summary, simple_summary])
    comparison.to_csv(DATA_DIR / "duration_estimator_model_comparison.csv", index=False)
    print(comparison.to_string(index=False))
    print()
    print("Full-model CI summary:")
    print(full_ci_summary.to_string(index=False))
    print()
    print("Simple temporal-model CI summary:")
    print(simple_ci_summary.to_string(index=False))


if __name__ == "__main__":
    main()
