from load_data import load_data
from plot_config import load_plot_config
from plotting_data import limit_year_range
from year_lineplot_interactive import write_year_lineplot_interactive


def main():
    params, params_global, out_path = load_plot_config("gender_by_year")

    time_period = (params["start_year"], params["end_year"])
    color_cycle = params_global["color_cycle"]
    restcolor = params_global["restcolor"]
    fac_total_color = params_global["fac_in_inst_color"]
    faculty_colormap = params_global["faculty_institute_mappings"]

    marker_fac = dict(size=params["marker_fac_size"], symbol="circle", line=dict(width=1))
    marker_total = dict(
        size=params["marker_total_size"],
        symbol="circle",
        line=dict(width=0),
        color=restcolor,
    )
    marker_inst = dict(size=params["marker_inst_size"], symbol="circle", line=dict(width=1))
    marker_fac_total = dict(
        size=params["marker_fac_in_inst"],
        symbol="diamond",
        line=dict(width=1),
        color=fac_total_color,
    )

    data = load_data("data/clean_data.csv", "data/cleaned_data_dtypes.json")
    df = data[["acceptance_year", "faculty", "institute_name", "gender"]].copy()
    df["is_female"] = ~df["gender"].isin(["M", "m"])
    df = df.rename(columns={"acceptance_year": "year"})
    df = limit_year_range(
        df, year_col="year", start_year=time_period[0], end_year=time_period[1]
    )

    write_year_lineplot_interactive(
        df=df[["year", "faculty", "institute_name", "is_female"]],
        out_path=out_path,
        value_col="is_female",
        n_min_year_prune=params["min_year_prune"],
        n_min_inst_abs=params["n_min_institutes_abs"],
        n_min_inst_mean=params["n_min_institutes_mean"],
        params_global=params_global,
        color_cycle=color_cycle,
        restcolor=restcolor,
        faculty_colormap=faculty_colormap,
        yaxis_title="Anteil Weiblich/Divers",
        total_label="Gesamt",
        all_faculties_label="Fakultäten (alle)",
        faculty_total_suffix=" (Fakultät gesamt)",
        marker_fac=marker_fac,
        marker_total=marker_total,
        marker_inst=marker_inst,
        marker_fac_total=marker_fac_total,
        width_total=params["width_total"],
        width_fac=params["width_fac"],
        width_inst=params["width_inst"],
        width_fac_total=params["width_fac_in_inst"],
        fac_total_color=fac_total_color,
    )


if __name__ == "__main__":
    main()
