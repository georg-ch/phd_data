from load_data import load_data
from plot_config import load_plot_config
from pie_charts import write_category_pie_interactive
from plotting_data import prepare_yearly_category_counts


def main():
    params, params_global, out_path = load_plot_config("pbacat_pie", pie_output=True)

    time_period = (params["start_year"], params["end_year"])
    min_category_count = params["n_min_institute"]
    pba_category_labels = params["category_label_names"]
    color_cycle = params_global["color_cycle"]
    restcolor = params_global["restcolor"]
    faculty_colormap = params_global["faculty_institute_mappings"]

    data = load_data("data/clean_data.csv", "data/cleaned_data_dtypes.json")
    df = prepare_yearly_category_counts(
        data,
        year_col="acceptance_year",
        faculty_col="faculty",
        category_col="pba_category",
        start_year=time_period[0],
        end_year=time_period[1],
        category_label_map=pba_category_labels,
    )

    write_category_pie_interactive(
        df=df,
        out_path=out_path,
        n_min=min_category_count,
        count_name="Anmeldungen",
        color_cycle=color_cycle,
        restcolor=restcolor,
        faculty_colormap=faculty_colormap,
        category_col="pba_category",
        category_name="Abschluss",
        category_label_map=None,
        rest_aggregation=False,
    )


if __name__ == "__main__":
    main()
