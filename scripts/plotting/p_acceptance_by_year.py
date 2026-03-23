from src.data_loading.load_data import load_data
from src.plotting.plot_config import load_plot_config
from src.plotting.plotting_data import limit_year_range
from src.plotting.year_barplot_interactive import write_year_barplot_interactive
from scripts.paths import DATA_DIR


def main():
    params, params_global, out_path = load_plot_config("acc_by_year")

    time_period = (params["start_year"], params["end_year"])
    n_min = params["n_min_institute"]
    color_cycle = params_global["color_cycle"]
    restcolor = params_global["restcolor"]
    faculty_colormap = params_global["faculty_institute_mappings"]

    data = load_data(DATA_DIR / "clean_data.csv", DATA_DIR / "cleaned_data_dtypes.json")
    df = data[["acceptance_year", "faculty", "institute_name"]].copy()
    df = df.groupby(
        ["acceptance_year", "faculty", "institute_name"], as_index=False
    ).size()
    df = df.rename(columns={"acceptance_year": "year", "size": "count"})
    df = limit_year_range(
        df, year_col="year", start_year=time_period[0], end_year=time_period[1]
    )

    write_year_barplot_interactive(
        df=df,
        out_path=out_path,
        n_min=n_min,
        count_name="Anmeldungen",
        params_global=params_global,
        color_cycle=color_cycle,
        restcolor=restcolor,
        faculty_colormap=faculty_colormap,
    )


if __name__ == "__main__":
    main()
