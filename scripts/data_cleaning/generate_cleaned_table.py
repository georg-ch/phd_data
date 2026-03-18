import pandas as pd
import json

from scripts.paths import PROJECT_ROOT, DATA_DIR, RESOURCES_DIR
from src.data_cleaning.cleaning_functions import build_cleaned_table


def main():
    df = pd.read_csv(
        DATA_DIR / "20260113_PDB.csv",
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
        resource_path=RESOURCES_DIR,
        data_path=DATA_DIR,
    )

    # save cleaned DataFrame to CSV
    cleaned_df.to_csv(DATA_DIR / "clean_data.csv", index=False)
    with open(RESOURCES_DIR / "cleaned_data_dtypes.json", "w") as f:
        json.dump(dtypes_dict, f)


if __name__ == "__main__":
    main()
