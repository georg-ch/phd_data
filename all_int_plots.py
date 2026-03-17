from p_acceptance_by_year import main as acceptance_by_year_main
from p_cship_by_year import main as cship_by_year_main
from p_pba_state_by_year import main as pba_state_by_year_main
from p_cship_pie import main as cship_pie_main
from p_pba_state_pie import main as pba_state_pie_main
from p_defense_by_year import main as defense_by_year_main
from p_gender_acceptance_by_year import main as gender_acceptance_by_year_main
from p_gender_defense_by_year import main as gender_defense_by_year_main
from p_sankey_institutes import main as sankey_institutes_main
from p_sankey_faculties import main as sankey_faculties_main
from p_region_matrix_barplot import main as region_matrix_barplot_main
from p_pba_category_pie import main as pba_category_pie_main
from p_pba_category_by_year import main as pba_category_by_year_main


PLOT_RUNNERS = [
    acceptance_by_year_main,
    defense_by_year_main,
    pba_category_by_year_main,
    pba_state_by_year_main,
    cship_by_year_main,
    gender_acceptance_by_year_main,
    gender_defense_by_year_main,
    sankey_institutes_main,
    sankey_faculties_main,
    pba_category_pie_main,
    pba_state_pie_main,
    cship_pie_main,
    region_matrix_barplot_main,
]


def main():
    for run_plot in PLOT_RUNNERS:
        run_plot()


if __name__ == "__main__":
    main()
