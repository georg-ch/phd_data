from src.data_loading.load_data import load_data
from src.plotting.plot_config import load_plot_config
from src.plotting.plotting_data import (
    PBA_CATEGORY_LABELS as pba_category_labels,
    prepare_country_cat_matrix,
    prep_cship_and_pba_cats,
)
from src.plotting.year_barplot_interactive import (
    write_category_barplot_interactive_by_category,
)
from scripts.paths import DATA_DIR


def main():
    params, params_global, out_path = load_plot_config("study_matrix_def")

    time_period = (params["start_year"], params["end_year"])
    color_cycle = params_global["color_cycle"]
    restcolor = params_global["restcolor"]
    faculty_colormap = params_global["faculty_institute_mappings"]
    citizenship_category_labels = {
        "GER": "Deutschland",
        "EU": "EU-Ausland",
        "NEU": "Nicht-EU-Ausland",
    }

    data = load_data(DATA_DIR / "clean_data.csv", DATA_DIR / "cleaned_data_dtypes.json")

    df = prepare_country_cat_matrix(
        data,
        start_year=time_period[0],
        end_year=time_period[1],
        year_col="defense_year",
        fac_col="faculty",
        category_1="citizenship_category",
        category_2="pba_category",
        label_map_1=citizenship_category_labels,
        label_map_2=pba_category_labels,
        df_transform_func=prep_cship_and_pba_cats,
    )

    write_category_barplot_interactive_by_category(
        df=df,
        out_path=out_path,
        count_name="Promotionen",
        color_cycle=color_cycle,
        restcolor=restcolor,
        faculty_colormap=faculty_colormap,
        x_col="pba_category",
        stack_col="citizenship_category",
        xaxis_title="Hochschule Masterstudium",
        stack_name="Staatsangehörigkeit",
        x_order=[
            pba_category_labels[key]
            for key in ["TUB", "GER", "EU", "NEU"]
            if pba_category_labels[key] in set(df["pba_category"])
        ],
        legend_title="Staatsangehörigkeit<br>",
    )


if __name__ == "__main__":
    main()
