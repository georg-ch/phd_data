# Interactive Plotting

## Purpose

This repo generates the interactive HTML plots in `int_plots/`.

`all_int_plots.py` is the batch entrypoint. It calls the individual plot modules in a fixed order and writes the current HTML outputs.

# data location

The code expects a folder `data/` with the following files:

- `clean_data.csv` - the cleaned database
- `cleaned_data_dtypes.json` - dtype-info for the cleaned database

## Reading Order

1. `all_int_plots.py`
2. `plotting_data.py`
3. `plot_config.py`
4. `plotting_export.py`
5. `year_barplot_interactive.py`
6. `year_lineplot_interactive.py`
7. `pie_charts.py`
8. `sankey_interactive.py`
9. One of the small `*_int.py` or `*_pie.py` entry modules

## Entrypoints

These files are the thin entry modules used by `all_int_plots.py`:

- `p_acceptance_by_year.py`
- `p_defense_by_year.py`
- `p_pba_category_by_year.py`
- `p_pba_state_by_year.py`
- `p_cship_by_year.py`
- `p_gender_acceptance_by_year.py`
- `p_gender_defense_by_year.py`
- `p_sankey_institutes.py`
- `p_sankey_faculties.py`
- `p_pba_category_pie.py`
- `p_pba_state_pie.py`
- `p_cship_pie.py`
- `p_region_matrix_barplot.py`

Each module exposes `main()` and is intended to do only:

- load config via `plot_config.py`
- load and shape input data
- call one shared plot writer

The entry modules should not contain plotting logic beyond lightweight dataframe preparation.

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

Sankey plots do not use the generic `year`/`count` contract. They expect the subject/faculty columns required by the respective writer after year filtering and institute cleanup.

## Important Invariants

- Shared helpers should prefer `year` and `count`.
- `params.toml` defines output locations and plot-specific thresholds.
- Entry modules should use `plot_config.load_plot_config(...)` instead of reimplementing `params.toml` loading and output path derivation.

## Generated Outputs

The current batch run writes HTML files under `int_plots/`:
