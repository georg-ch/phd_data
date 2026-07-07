from pathlib import Path
import argparse
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.paths import DATA_DIR
from src.data_loading.load_data import load_data
from src.data_processing.gender_regression import (
    LOCATION_FEATURES,
    ensure_dirs,
    run_gender_regression,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--location-feature", default=None)
    parser.add_argument("--context-feature", default=None)
    args = parser.parse_args()

    ensure_dirs()
    data = load_data(DATA_DIR / "clean_data.csv", DATA_DIR / "cleaned_data_dtypes.json")
    location_features = [args.location_feature] if args.location_feature else list(LOCATION_FEATURES)
    for location_feature in location_features:
        run_gender_regression(
            data,
            location_feature=location_feature,
            context_feature=args.context_feature,
        )


if __name__ == "__main__":
    main()
