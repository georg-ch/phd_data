# Interactive Plotting

## Purpose

This repo generates the interactive HTML plots configured in
`config/plotting_params.toml`.

The plotting code is now split into:

- `scripts/plotting/` for runnable entry modules
- `src/plotting/` for shared plotting helpers and writers
- `src/data_loading/` for loading the cleaned data input

## Data And Config Location

The plotting scripts expect:

- `data/clean_data.csv`
- `data/cleaned_data_dtypes.json`
- `config/plotting_params.toml`

`src/plotting/plot_config.py` resolves the config file relative to the project
root and derives output HTML paths from the selected TOML section.

## Reading Order

1. `scripts/plotting/p_acceptance_by_year.py`
2. `src/data_loading/load_data.py`
3. `src/plotting/plot_config.py`
4. `src/plotting/plotting_data.py`
5. `src/plotting/plotting_export.py`
6. `src/plotting/year_barplot_interactive.py`
7. `src/plotting/year_lineplot_interactive.py`
8. `src/plotting/pie_charts.py`
9. `src/plotting/sankey_interactive.py`
10. one of the other `scripts/plotting/p_*.py` entry modules

## Entrypoints

These files are the thin plotting entry modules:

- `scripts/plotting/p_acceptance_by_year.py`
- `scripts/plotting/p_defense_by_year.py`
- `scripts/plotting/p_pba_category_by_year.py`
- `scripts/plotting/p_pba_state_by_year.py`
- `scripts/plotting/p_cship_by_year.py`
- `scripts/plotting/p_gender_acceptance_by_year.py`
- `scripts/plotting/p_gender_defense_by_year.py`
- `scripts/plotting/p_sankey_institutes.py`
- `scripts/plotting/p_sankey_faculties.py`
- `scripts/plotting/p_pba_category_pie.py`
- `scripts/plotting/p_pba_state_pie.py`
- `scripts/plotting/p_cship_pie.py`
- `scripts/plotting/p_region_matrix_barplot.py`

The batch entrypoint lives in `scripts/plotting/all_int_plots.py`.

Run it from the repo root with:

```bash
python -m scripts.plotting.all_int_plots
```

Each entry module should only:

- load config via `src/plotting/plot_config.py`
- load and shape input data
- call one shared plot writer

The entry modules should not contain plotting logic beyond lightweight
dataframe preparation.

## Shared Data Contracts

Prefer generic column names in shared plotting code.

Yearly count plots expect:

- `year`
- `faculty`
- `count`
- optional grouping column such as `institute_name` or a category column

Category pie plots expect:

- `year`
- `faculty`
- `count`
- the configured category column passed as `category_col`

Yearly percentage line plots expect:

- `year`
- `faculty`
- `institute_name`
- a metric column passed as `value_col`

Sankey plots do not use the generic `year`/`count` contract. They expect the
subject/faculty columns required by the respective writer after year filtering
and institute cleanup.

## Important Invariants

- Shared helpers should prefer `year` and `count`.
- Plot-specific thresholds and output folders live in `config/plotting_params.toml`.
- Entry modules should use `load_plot_config(...)` instead of reimplementing TOML
  loading and output path derivation.
- Shared modules under `src/plotting/` are imported via `src.plotting...` paths.
