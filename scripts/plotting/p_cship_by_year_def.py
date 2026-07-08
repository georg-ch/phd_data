from src.data_loading.load_data import load_data
from src.plotting.plot_config import load_plot_config
from src.plotting.plotting_data import (
    PBA_CATEGORY_LABELS as category_code_labels,
    prepare_yearly_category_counts,
)
from src.plotting.year_barplot_interactive import (
    write_year_barplot_interactive_by_category,
)
from scripts.paths import DATA_DIR


def main():
    params, params_global, out_path = load_plot_config("cship_by_year_def")

    time_period = (params["start_year"], params["end_year"])
    min_category_count = params["n_min_categories"]
    color_cycle = params_global["color_cycle"]
    restcolor = params_global["restcolor"]
    faculty_colormap = params_global["faculty_institute_mappings"]

    data = load_data(DATA_DIR / "clean_data.csv", DATA_DIR / "cleaned_data_dtypes.json")
    df = prepare_yearly_category_counts(
        data,
        year_col="defense_year",
        faculty_col="faculty",
        category_col="citizenship",
        start_year=time_period[0],
        end_year=time_period[1],
        category_label_map=category_code_labels,
    )

    write_year_barplot_interactive_by_category(
        df=df,
        out_path=out_path,
        n_min=min_category_count,
        count_name="Promotionen",
        color_cycle=color_cycle,
        restcolor=restcolor,
        faculty_colormap=faculty_colormap,
        category_col="citizenship",
        category_name="Land",
        category_label_map=None,
        rest_aggregation=True,
    )


if __name__ == "__main__":
    main()
