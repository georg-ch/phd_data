# Repo Layout

The repo now separates runnable scripts from shared library code:

- `scripts/` contains entry scripts
- `src/` contains reusable modules
- `config/` contains TOML configuration
- `data/` contains the cleaned data inputs used by plotting scripts
- `resources/` contains auxiliary lookup tables used during data cleaning

## Plotting

Interactive plotting code is split into:

- `scripts/plotting/` for the thin plot entry modules
- `src/plotting/` for shared plotting logic
- `src/data_loading/load_data.py` for reading the cleaned dataset
- `config/plotting_params.toml` for plot thresholds, output folders, and style mappings

The plotting config uses separate sections for the different workflows, including
`[regression]` for duration regression and `[gender_regression]` for gender
regression.

The gender regression plots are driven by `scripts/plotting/p_gender_regression.py`
and use the gender-specific section in the same TOML file.

The plotting scripts expect these files under `data/`:

- `clean_data.csv`
- `cleaned_data_dtypes.json`

and some additional files under `resources/`.

Example for running one plot module from the repo root:

```bash
python -m scripts.plotting.p_acceptance_by_year
```

Run the full interactive plotting batch from the repo root with:

```bash
python -m scripts.plotting.all_int_plots
```

## Data Cleaning

The cleaned plotting input can be regenerated from:

- `scripts/data_cleaning/generate_cleaned_table.py`
- `src/data_cleaning/cleaning_functions.py`
- `src/data_processing/duration_preprocessing.py`

The cleaning pipeline expects raw input files in `data/` and lookup tables in
`resources/`.

To run the data cleaning pipeline, simply execute

```bash
python -m scripts.pdata_cleaning.generate_cleaned_table [file_name]
```

where `file_name` is expected to be an `.xls` file located in `data/` (the suffix is optional in the script argument).
