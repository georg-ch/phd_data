from src.data_loading.load_data import load_data
from src.plotting.plot_config import load_plot_config
from src.plotting.pie_charts import write_category_pie_interactive
from src.plotting.plotting_data import prepare_yearly_category_counts
from scripts.paths import DATA_DIR


def main():
    params, params_global, out_path = load_plot_config("cship_pie", pie_output=True)

    time_period = (params["start_year"], params["end_year"])
    min_country_count = params["n_min_country"]
    max_named_countries = params.get("n_biggest_countries")
    color_order = params.get("color_order")
    stack_order = params.get("stack_order")
    category_labels = params["category_label_names"]
    color_cycle = params_global["color_palette_expanded"]
    restcolor = params_global["restcolor"]
    faculty_colormap = params_global["faculty_institute_mappings"]

    data = load_data(DATA_DIR / "clean_data.csv", DATA_DIR / "cleaned_data_dtypes.json")
    df = prepare_yearly_category_counts(
        data,
        year_col="acceptance_year",
        faculty_col="faculty",
        category_col="citizenship",
        start_year=time_period[0],
        end_year=time_period[1],
        category_label_map=category_labels,
    )

    write_category_pie_interactive(
        df=df,
        out_path=out_path,
        n_min=min_country_count,
        count_name="Anmeldungen",
        color_cycle=color_cycle,
        restcolor=restcolor,
        faculty_colormap=faculty_colormap,
        category_col="citizenship",
        category_name="Staatsangehörigkeit",
        category_label_map=None,
        rest_aggregation=False,
        top_n=max_named_countries,
        rotation=90,
        order_by_size=True,
        color_order=color_order,
        stack_order=stack_order,
    )


if __name__ == "__main__":
    main()
