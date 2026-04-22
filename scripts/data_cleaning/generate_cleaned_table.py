import pandas as pd
import json

from scripts.paths import PROJECT_ROOT, DATA_DIR, RESOURCES_DIR
from src.data_cleaning.cleaning_functions import build_cleaned_table
import argparse
from pathlib import Path


def main(input_file: str):
    input_path = Path(input_file)

    if input_path.suffix:
        stem = input_path.with_suffix("")
        excel_path = input_path
    else:
        stem = DATA_DIR / input_path
        excel_path = stem.with_suffix(".xls")

    csv_path = stem.with_suffix(".csv")

    # fname = DATA_DIR / "20260113_PDB"
    xls = pd.ExcelFile(excel_path)

    if len(xls.sheet_names) != 1:
        raise ValueError(
            f"Expected exactly 1 sheet in {excel_path}, found {len(xls.sheet_names)}: {xls.sheet_names}"
        )

    df_excel = pd.read_excel(xls, sheet_name=xls.sheet_names[0])
    df_excel.to_csv(csv_path, index=False)

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
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="Excel file path or file stem")
    args = parser.parse_args()
    main(args.input_file)
