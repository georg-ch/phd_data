import pandas as pd

from plotting_style import faculty_color_for, short_faculty_label


def reindex_years(df: pd.DataFrame, years_sorted, year_col: str = "year") -> pd.DataFrame:
    """Return df reindexed to years_sorted, inserting missing years as empty rows."""
    return df.set_index(year_col).reindex(years_sorted).reset_index()


def prepare_plot_frame(
    df: pd.DataFrame,
    years_sorted,
    *,
    year_col: str = "year",
    zero_fill_cols: list[str] | None = None,
    hover_col: str | None = None,
) -> pd.DataFrame:
    """Return a year-complete plotting frame with selected numeric columns zero-filled and hover text filled with ''. """
    out = reindex_years(df, years_sorted, year_col=year_col)
    if zero_fill_cols:
        out[zero_fill_cols] = out[zero_fill_cols].fillna(0)
    if hover_col is not None:
        out[hover_col] = out.get(hover_col, "").fillna("")
    return out


def add_count_text(
    df: pd.DataFrame,
    *,
    count_col: str,
    singular_name: str,
    plural_name: str,
    out_col: str = "count_text",
) -> pd.DataFrame:
    """Return a copy with out_col added, formatting count_col using the provided singular/plural labels."""
    out = df.copy()
    out[out_col] = [
        f"{int(value):,} {singular_name}" if value == 1 else f"{int(value):,} {plural_name}"
        for value in out[count_col]
    ]
    return out


def sorted_with_rest(values, *, rest_label: str = "Rest") -> list:
    """Return the sorted labels with rest_label moved to the end when present."""
    labels = sorted(x for x in values if x != rest_label)
    if rest_label in set(values):
        labels.append(rest_label)
    return labels


def build_trace_visibility(
    n_traces: int,
    visible_indices,
    *,
    legend_indices=None,
    hidden_value="legendonly",
) -> tuple[list, list[bool]]:
    """Return the Plotly visible and showlegend arrays for one button state."""
    vis = [hidden_value] * n_traces
    showleg = [False] * n_traces

    for idx in visible_indices:
        vis[idx] = True

    for idx in (visible_indices if legend_indices is None else legend_indices):
        showleg[idx] = True

    return vis, showleg


def build_faculty_visibility_maps(
    *,
    n_traces: int,
    faculty_list,
    trace_indices_by_faculty: dict,
    always_visible_indices=None,
    always_legend_indices=None,
    hidden_value="legendonly",
) -> tuple[dict, dict]:
    """Return per-faculty visible/showlegend maps, including any always-visible traces."""
    vis_by_faculty = {}
    showleg_by_faculty = {}

    for faculty in faculty_list:
        visible_indices = list(always_visible_indices or [])
        visible_indices.extend(trace_indices_by_faculty.get(faculty, []))

        legend_indices = list(always_legend_indices or [])
        legend_indices.extend(trace_indices_by_faculty.get(faculty, []))

        vis_by_faculty[faculty], showleg_by_faculty[faculty] = build_trace_visibility(
            n_traces,
            visible_indices,
            legend_indices=legend_indices,
            hidden_value=hidden_value,
        )

    return vis_by_faculty, showleg_by_faculty


def build_faculty_trace_state(
    faculty_list,
    *,
    color_cycle,
    faculty_colormap,
) -> tuple[dict, dict, list]:
    """Return faculty colors plus empty per-faculty and global institute trace-index containers."""
    faculty_color = {
        faculty: faculty_color_for(faculty, color_cycle, faculty_colormap)
        for faculty in faculty_list
    }
    institute_trace_indices_by_faculty = {faculty: [] for faculty in faculty_list}
    all_institute_trace_indices = []
    return faculty_color, institute_trace_indices_by_faculty, all_institute_trace_indices


def build_faculty_buttons(
    *,
    all_label: str,
    faculty_list,
    vis_default,
    showleg_default,
    vis_by_faculty: dict,
    showleg_by_faculty: dict,
    traceorder_default: str,
    traceorder_faculty: str,
    extra_default_args: list | None = None,
    extra_faculty_args: list | None = None,
) -> tuple[list[dict], dict[str, int]]:
    """Return the Plotly update-menu button specs and the faculty-to-button-index lookup."""
    button_faculty_order = list(reversed(faculty_list))
    buttons = [
        dict(
            label=all_label,
            method="update",
            args=[
                {"visible": vis_default, "showlegend": showleg_default},
                {"legend.traceorder": traceorder_default},
            ]
            + (extra_default_args or []),
        )
    ]

    for faculty in button_faculty_order:
        buttons.append(
            dict(
                label=short_faculty_label(faculty),
                method="update",
                args=[
                    {
                        "visible": vis_by_faculty[faculty],
                        "showlegend": showleg_by_faculty[faculty],
                    },
                    {"legend.traceorder": traceorder_faculty},
                ]
                + (extra_faculty_args or []),
            )
        )

    button_index_by_faculty = {
        faculty: i + 1 for i, faculty in enumerate(button_faculty_order)
    }
    return buttons, button_index_by_faculty
