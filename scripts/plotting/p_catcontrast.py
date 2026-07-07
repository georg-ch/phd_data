from __future__ import annotations

import tomllib

import pandas as pd

from scripts.paths import CONFIG_DIR, DATA_DIR
from src.data_loading.load_data import load_data
from src.data_processing.regression import category_contrast_analysis
from src.plotting.plot_config import load_plot_config
from src.plotting.plotting_data import limit_year_range


def main():
    params, _, _ = load_plot_config("regression")
    with open(CONFIG_DIR / "regression.toml", "rb") as f:
        regression_config = tomllib.load(f)

    data = load_data(DATA_DIR / "clean_data.csv", DATA_DIR / "cleaned_data_dtypes.json")
    duration_data_simplified = pd.read_csv(
        DATA_DIR / "duration_predictions_simple_model.csv"
    )
    data = pd.merge(
        data,
        duration_data_simplified[
            ["pagination_nr", "duration_estimate", "duration_estimate_clipped"]
        ].rename(
            columns={
                "duration_estimate": "duration_estimate_simple",
                "duration_estimate_clipped": "duration_estimate_clipped_simple",
            }
        ),
        on="pagination_nr",
        how="left",
    )
    data = limit_year_range(
        data,
        year_col="defense_year",
        start_year=int(params["start_year"]),
        end_year=int(params["end_year"]),
    )

    category_contrast_analysis(
        data,
        target_col=regression_config["target_column"],
        target_transform=regression_config["target_transform"],
        min_count=int(params["small_group_cutoff"]),
        merge_cols=params["merge_cols"],
        cov_type="HC3",
        output_dir=DATA_DIR / "regression_contrasts",
    )


if __name__ == "__main__":
    main()
