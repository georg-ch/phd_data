from src.data_loading.load_data import load_data
from src.plotting.plot_config import load_plot_config
from src.plotting.plotting_data import (
    PBA_CATEGORY_LABELS as pba_category_labels,
    prepare_yearly_category_counts,
)
from src.plotting.year_barplot_interactive import (
    write_year_barplot_interactive_by_category,
)
from scripts.paths import DATA_DIR


def main():
    params, params_global, out_path = load_plot_config("pbacat_by_year_def")

    time_period = (params["start_year"], params["end_year"])
    min_category_count = params["n_min_categories"]
    stack_order = params.get("stack_order")
    color_order = params.get("color_order")
    color_cycle = params_global["color_palette_expanded"]
    restcolor = params_global["restcolor"]
    faculty_colormap = params_global["faculty_institute_mappings"]

    data = load_data(DATA_DIR / "clean_data.csv", DATA_DIR / "cleaned_data_dtypes.json")
    df = prepare_yearly_category_counts(
        data,
        year_col="defense_year",
        faculty_col="faculty",
        category_col="pba_category",
        start_year=time_period[0],
        end_year=time_period[1],
        category_label_map=pba_category_labels,
    )

    write_year_barplot_interactive_by_category(
        df=df,
        out_path=out_path,
        n_min=min_category_count,
        count_name="Promotionen",
        color_cycle=color_cycle,
        restcolor=restcolor,
        faculty_colormap=faculty_colormap,
        category_col="pba_category",
        category_name="Abschluss",
        category_label_map=None,
        rest_aggregation=False,
        stack_order=stack_order,
        color_order=color_order,
    )


if __name__ == "__main__":
    main()
