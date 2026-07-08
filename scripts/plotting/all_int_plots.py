import warnings
from pathlib import Path
import textwrap

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r".*pkg_resources is deprecated as an API.*",
    module=r"hyperopt\.atpe",
)

from scripts.plotting.p_acceptance_by_year import main as acceptance_by_year_main
from scripts.plotting.p_cship_by_year import main as cship_by_year_main
from scripts.plotting.p_cship_by_year_def import main as cship_by_year_def_main
from scripts.plotting.p_pba_state_by_year import main as pba_state_by_year_main
from scripts.plotting.p_pba_state_by_year_def import main as pba_state_by_year_def_main
from scripts.plotting.p_cship_pie import main as cship_pie_main
from scripts.plotting.p_cship_pie_def import main as cship_pie_def_main
from scripts.plotting.p_pba_state_pie import main as pba_state_pie_main
from scripts.plotting.p_defense_by_year import main as defense_by_year_main
from scripts.plotting.p_gender_acceptance_by_year import (
    main as gender_acceptance_by_year_main,
)
from scripts.plotting.p_gender_defense_by_year import (
    main as gender_defense_by_year_main,
)
from scripts.plotting.p_sankey_institutes import main as sankey_institutes_main
from scripts.plotting.p_sankey_faculties import main as sankey_faculties_main
from scripts.plotting.p_region_matrix_barplot import main as region_matrix_barplot_main
from scripts.plotting.p_region_matrix_barplot_def import (
    main as region_matrix_barplot_def_main,
)
from scripts.plotting.p_duration_densities import main as duration_densities_main
from scripts.plotting.p_duration_densities_simple import (
    main as duration_densities_simple_main,
)
from scripts.plotting.p_duration_densities_from_start_anchor import (
    main as duration_densities_from_start_anchor_main,
)
from scripts.plotting.p_duration_densities_from_start_defense import (
    main as duration_densities_from_start_defense_main,
)
from scripts.plotting.p_catcontrast_violins import main as catcontrast_violins_main
from scripts.plotting.p_pba_category_pie import main as pba_category_pie_main
from scripts.plotting.p_pba_category_by_year import main as pba_category_by_year_main
from scripts.plotting.p_pba_category_by_year_def import (
    main as pba_category_by_year_def_main,
)
from scripts.plotting.p_gender_regression import main as gender_regression_main
from scripts.plotting.p_regression_features import main as regression_features_main
from src.plotting.plot_config import load_plot_config


PLOT_RUNNERS = [
    acceptance_by_year_main,
    defense_by_year_main,
    pba_category_by_year_main,
    pba_state_by_year_main,
    pba_category_by_year_def_main,
    pba_state_by_year_def_main,
    cship_by_year_main,
    cship_by_year_def_main,
    gender_acceptance_by_year_main,
    gender_defense_by_year_main,
    sankey_institutes_main,
    sankey_faculties_main,
    pba_category_pie_main,
    pba_state_pie_main,
    cship_pie_main,
    cship_pie_def_main,
    gender_regression_main,
    region_matrix_barplot_main,
    region_matrix_barplot_def_main,
    duration_densities_main,
    duration_densities_simple_main,
    duration_densities_from_start_anchor_main,
    duration_densities_from_start_defense_main,
    catcontrast_violins_main,
    regression_features_main,
]


def _warning_summary_key(warning: warnings.WarningMessage) -> tuple[str, str]:
    message = textwrap.shorten(
        str(warning.message).replace("\n", " "), width=72, placeholder="..."
    )
    return warning.category.__name__, message


def _print_warning_summary(
    run_name: str, caught: list[warnings.WarningMessage]
) -> None:
    if not caught:
        return
    summary: dict[tuple[str, str], dict[str, object]] = {}
    for warning in caught:
        key = _warning_summary_key(warning)
        entry = summary.setdefault(
            key,
            {
                "count": 0,
                "filename": Path(warning.filename).name,
                "lineno": warning.lineno,
            },
        )
        entry["count"] += 1
    for (category, message), entry in sorted(summary.items()):
        filename = entry["filename"]
        lineno = entry["lineno"]
        count = entry["count"]
        suffix = f" x{count}" if count > 1 else ""
        print(f"[{run_name}] {category} {filename}:{lineno}{suffix}: {message}")


def main():
    params_global = load_plot_config("regression")[1]
    verbose = bool(params_global.get("verbose", False))
    for run_plot in PLOT_RUNNERS:
        if verbose:
            run_plot()
            continue
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            run_plot()
        _print_warning_summary(run_plot.__module__.rsplit(".", 1)[-1], caught)


if __name__ == "__main__":
    main()
