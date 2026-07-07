from __future__ import annotations

import pandas as pd

from scripts.paths import DATA_DIR, PROJECT_ROOT
from src.data_loading.load_data import load_data
from src.data_processing.regression import plot_category_contrast_violins
from src.plotting.plot_config import load_plot_config
from src.plotting.plotting_data import limit_year_range


TARGET_COLS = ("duration_computed", "duration_estimate_clipped_simple")


def _prepare_data_for_target(data, target_col: str):
    if target_col == "duration_estimate_clipped_simple":
        duration_df = pd.read_csv(DATA_DIR / "duration_predictions_simple_model.csv")
        return data.merge(
            duration_df[["pagination_nr", "duration_estimate_clipped"]].rename(
                columns={"duration_estimate_clipped": "duration_estimate_clipped_simple"}
            ),
            on="pagination_nr",
            how="left",
        )
    if target_col == "duration_estimate_clipped":
        duration_df = pd.read_csv(DATA_DIR / "duration_predictions.csv")
        return data.merge(
            duration_df[["pagination_nr", "duration_estimate_clipped"]],
            on="pagination_nr",
            how="left",
        )
    return data


def main():
    params, _, _ = load_plot_config("regression")
    plot_params, _, _ = load_plot_config("catcontrast_violins")

    base_data = load_data(
        DATA_DIR / "clean_data.csv", DATA_DIR / "cleaned_data_dtypes.json"
    )

    for target_col in TARGET_COLS:
        data = _prepare_data_for_target(base_data, target_col)
        data = limit_year_range(
            data,
            year_col="defense_year",
            start_year=int(params["start_year"]),
            end_year=int(params["end_year"]),
        )

        plot_category_contrast_violins(
            data,
            target_col=target_col,
            target_transform=plot_params["target_transform"],
            min_count=int(params["small_group_cutoff"]),
            merge_cols=params["merge_cols"],
            cov_type="HC3",
            output_dir=DATA_DIR / "regression_contrasts",
            plot_output_dir=PROJECT_ROOT / "int_plots" / f"regr_dur_dens_{target_col}",
        )


if __name__ == "__main__":
    main()
