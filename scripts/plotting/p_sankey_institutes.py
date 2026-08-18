from src.data_loading.load_data import load_data
from src.plotting.plot_config import load_plot_config
from src.plotting.plotting_data import (
    build_faculties,
    limit_year_range,
    remap_institutes,
    remove_dubious_institute_assignments,
)
from src.plotting.sankey_interactive import write_sankey_diag_interactive
from scripts.paths import DATA_DIR


def main():
    params, params_global, out_path = load_plot_config("subject")

    time_period = (params["start_year"], params["end_year"])
    color_cycle = params_global["color_cycle"]
    restcolor = params_global["restcolor"]
    link_alpha = params["link_alpha"]
    node_thickness = params["node_thickness"]
    faculty_colormap = params_global["faculty_institute_mappings"]
    faculties = build_faculties(params_global["faculty_institute_mappings"])
    category_label_map = params.get("category_label_names", {})

    data = load_data(DATA_DIR / "clean_data.csv", DATA_DIR / "cleaned_data_dtypes.json")
    df = data[
        [
            "acceptance_year",
            "institute_name",
            "subject_group_stala",
            "subject_group_category_stala",
            "faculty",
        ]
    ]
    df = remap_institutes(df, params_global)
    df = remove_dubious_institute_assignments(df, faculties)
    df = limit_year_range(
        df,
        year_col="acceptance_year",
        start_year=time_period[0],
        end_year=time_period[1],
    )
    df = df.drop("acceptance_year", axis=1)

    write_sankey_diag_interactive(
        df=df,
        out_path=out_path,
        color_cycle=color_cycle,
        restcolor=restcolor,
        link_alpha=link_alpha,
        node_thickness=node_thickness,
        faculty_colormap=faculty_colormap,
        category_label_map=category_label_map,
    )


if __name__ == "__main__":
    main()
