from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd
import json
from data_cleaning.cleaning_functions import *


def main():
    df = pd.read_csv(
        PROJECT_ROOT / "data/20260113_PDB.csv",
        dtype={
            "Betr_Kostst": str,
            "Gutacht1_Kostst": str,
            "Gutacht2_Kostst": str,
            "Gutacht3_Kostst": str,
            "Vorsitz_Kostst": str,
        },
    )
    cleaned_df, dtypes_dict = build_cleaned_table(
        df,
        keep_original=True,
        resource_path=PROJECT_ROOT / "resources",
        data_path=PROJECT_ROOT / "data",
    )

    # save cleaned DataFrame to CSV
    cleaned_df.to_csv(PROJECT_ROOT / "data/clean_data.csv", index=False)
    with open(PROJECT_ROOT / "data/cleaned_data_dtypes.json", "w") as f:
        json.dump(dtypes_dict, f)


if __name__ == "__main__":
    main()
