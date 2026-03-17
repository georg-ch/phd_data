import json
import pandas as pd


def load_data(data_csv, dtypes_json):
    with open(dtypes_json) as f:
        dtypes = json.load(f)
    return pd.read_csv(
        data_csv,
        dtype=dtypes,
        keep_default_na=False,
        na_values=[""],
        dtype_backend="pyarrow",
    )
