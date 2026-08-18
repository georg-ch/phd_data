from src.data_loading.load_data import load_data
from src.plotting.plot_config import load_plot_config
from src.plotting.plotting_data import limit_year_range
from src.plotting.sankey_interactive import write_sankey_subject_groups_interactive
from scripts.paths import DATA_DIR


def main():
    params, params_global, out_path = load_plot_config("subject_overview")

    time_period = (params["start_year"], params["end_year"])
    color_cycle = params_global["color_cycle"]
    restcolor = params_global["restcolor"]
    link_alpha = params["link_alpha"]
    node_thickness = params["node_thickness"]
    faculty_colormap = params_global["faculty_institute_mappings"]
    subject_groups = list(params_global["subject_groups"])
    category_label_map = params.get("category_label_names", {})

    data = load_data(DATA_DIR / "clean_data.csv", DATA_DIR / "cleaned_data_dtypes.json")
    df = data[
        [
            "acceptance_year",
            "subject_group_category_stala",
            "faculty",
        ]
    ]
    df = limit_year_range(
        df,
        year_col="acceptance_year",
        start_year=time_period[0],
        end_year=time_period[1],
    )
    df = df.drop("acceptance_year", axis=1)

    write_sankey_subject_groups_interactive(
        df=df,
        out_path=out_path,
        color_cycle=color_cycle,
        restcolor=restcolor,
        link_alpha=link_alpha,
        node_thickness=node_thickness,
        faculty_colormap=faculty_colormap,
        subject_groups=subject_groups,
        category_label_map=category_label_map,
    )


if __name__ == "__main__":
    main()
