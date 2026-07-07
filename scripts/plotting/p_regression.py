from src.data_loading.load_data import load_data
from src.plotting.plot_config import load_plot_config
from src.plotting.plotting_data import limit_year_range
from scripts.paths import DATA_DIR
import pandas as pd
from src.data_processing.regression import regression_analysis
from scripts.paths import DATA_DIR, CONFIG_DIR
import tomllib


def main():
    params, params_global, out_path = load_plot_config("regression")

    config_path = CONFIG_DIR / "regression.toml"
    with open(config_path, "rb") as f:
        config = tomllib.load(f)
    target_column = config["target_column"]
    target_transform = config["target_transform"]
    feature_columns = config["feature_columns"]

    merge_cols = params["merge_cols"]

    time_period = (params["start_year"], params["end_year"])
    min_count = params["small_group_cutoff"]

    data = load_data(DATA_DIR / "clean_data.csv", DATA_DIR / "cleaned_data_dtypes.json")
    duration_data = pd.read_csv(DATA_DIR / "duration_predictions.csv")
    duration_data_simplified = pd.read_csv(
        DATA_DIR / "duration_predictions_simple_model.csv"
    )

    df = pd.merge(
        data,
        duration_data[
            ["pagination_nr", "duration_estimate", "duration_estimate_clipped"]
        ],
        on="pagination_nr",
        how="left",
    )

    df = pd.merge(
        df,
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

    df = limit_year_range(
        df, year_col="defense_year", start_year=time_period[0], end_year=time_period[1]
    )

    if target_column != "duration_computed":
        df = df.copy()
        df.loc[df["duration_computed"].notna(), target_column] = df["duration_computed"]
    df = df[df[target_column].notna()]

    print(df["pba_grade"].value_counts())

    regression_analysis(
        df,
        target_column,
        target_transform=target_transform,
        min_count=min_count,
        merge_cols=merge_cols,
    )

    # df = data[["acceptance_year", "faculty", "institute_name"]].copy()
    # df = df.groupby(
    #     ["acceptance_year", "faculty", "institute_name"], as_index=False
    # ).size()
    # df = df.rename(columns={"acceptance_year": "year", "size": "count"})
    # df = limit_year_range(
    #     df, year_col="year", start_year=time_period[0], end_year=time_period[1]
    # )


if __name__ == "__main__":
    main()

# - degree_gap = (defense_year - duration) - pba_year
# - institute (alternativelyfaculty/lecode/subject)
# - cship_category
# - pba_category
# - age_at_start = (defense_year - duration) - birth_year
# - gender
# - pba_grade
# - pba_type
# - local student/studed abroad/int_early/international
# - covid_overlap
# - start_year (no correction) / no start_year / start_year (filtered 90%)
# - or defense_year
#
# also do not forget the transform
#
# First, also residual plots:
# * age_at_start
# * degree_gap
# * start_year (if included)
# * pba_grade (depending on the grading system)
#
# potentially use simplified estimation model (with only acceptance/pba gap as features)
#
# features to potentially get better duration estimate:
# degree_gap, age_at_start, local-cat
#
#
# 1) engineer all the features
# 2) first model
# 3) diagnostics suite
# 4) try all feature combos and transofrmations
#
#     Cook’s distance
# * leverage
# and other diagnostics
# vif
#
#
# 1) decide on parameter set x
# 2) decide what is worth further investigating x
# 3) train extra model and compare outcome x
# 4) decide on data to use - do both, first check a little bit if on other dataset, other params are good x
# 4a) Different feature sets for both
# 5) decide on contrast tests: institute, gender (m vs w), gender per institute (m vs w), type, grade, cship
# 6) save contrast results and model results
# 7) decide on how to show regression results - key driver analysis (hat is NOT a driver?)
# 8) more detailed in contrasts
#
#   - institute_name
# - gender
# - cship_category
# # - pba_type
# - pba_grade
# - pba_grade_missing / or grad_num
#
# trade mobility category
# look for interactions
#
# different regression models to use:
# - computed set
# - estimated set
# - only institutes
# - only everything else -raw
#
# computed-set:
# institute, cship, male, grade2x, pba_type
# augmented set: remove cship, maybe pba type
#
# 1) decide on ols strings to use, rewrite code to directly inject
# 2) write saving code
# 3) run for all strings
# 4) plotting code
#
# within-category comparison
#
# gender effect in choice of subject/institute. /
# explain gender by institute/subject group / pba_category, grade
#
# - git stuff (4)
# - implement other requests (5)
