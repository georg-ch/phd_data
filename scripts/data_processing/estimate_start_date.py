from src.data_processing.duration_estimation import (
    find_parameters_and_predict_start,
    engineer_features_startdate,
)
from src.data_loading.load_data import load_data
from scripts.paths import DATA_DIR, CONFIG_DIR
import tomllib
from pathlib import Path


def main():
    config_path = CONFIG_DIR / "startdate_estimation_params.toml"
    with open(config_path, "rb") as f:
        config = tomllib.load(f)
    FEATURE_SET = config.get("feature_set", "defense")
    ORIG_COLS = config["orig_cols"]
    OUT_PATH = config["out_path"]
    OUT_FNAME = config["out_fname"].format(feature_set=FEATURE_SET)
    PARAMS_FNAME = f"best_params_start_{FEATURE_SET}.json"
    TARGET_COL_NAME = config["target_col_name"]
    NAIVE_ESTIMATOR_COLS = config["naive_est_cols"]
    data = load_data(DATA_DIR / "clean_data.csv", DATA_DIR / "cleaned_data_dtypes.json")

    data, feature_col_info = engineer_features_startdate(data, feature_set=FEATURE_SET)

    # data, feature_col_info = engineer_features(data)
    find_parameters_and_predict_start(
        data,
        TARGET_COL_NAME,
        feature_col_info,
        ORIG_COLS,
        OUT_PATH,
        OUT_FNAME,
        dump_folder=DATA_DIR,
        dump_fname=PARAMS_FNAME,
        naive_estimator_for_small_gap=NAIVE_ESTIMATOR_COLS,
    )


if __name__ == "__main__":
    main()
