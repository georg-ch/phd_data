import numpy as np
import plotly.graph_objects as go
import pandas as pd

from src.plotting.plotting_data import (
    aggregate_small_groups,
    build_faculties,
    fac_by_year_agg,
    finalize_group_percentages,
    get_singular_count_name,
    inst_by_year_agg,
    normalize_institute_names,
    prepare_category_frame,
    remap_institutes,
    year_total_agg,
)
from src.plotting.plotting_export import write_html
from src.plotting.plotting_style import (
    apply_bar_layout,
    institute_color_for,
    short_institute_label,
)
from src.plotting.plotting_traces import (
    add_count_text,
    build_faculty_trace_state,
    build_faculty_visibility_maps,
    build_faculty_buttons,
    build_trace_visibility,
    prepare_plot_frame,
    sorted_with_rest,
)


def add_bar_trace(fig, **kwargs) -> int:
    """Mutate fig by adding one bar trace and return the new trace index."""
    fig.add_trace(go.Bar(**kwargs))
    return len(fig.data) - 1


def prepare_count_bar_frame(
    df: pd.DataFrame,
    years_sorted,
    *,
    zero_fill_cols: list[str],
    singular_count_name: str,
    count_name: str,
    hover_col: str = "rest_hover",
) -> pd.DataFrame:
    """Return a year-complete count frame with selected columns filled and count_text added."""
    d = prepare_plot_frame(
        df,
        years_sorted,
        zero_fill_cols=zero_fill_cols,
        hover_col=hover_col,
    )
    return add_count_text(
        d,
        count_col="count",
        singular_name=singular_count_name,
        plural_name=count_name,
    )


def prepare_count_bar_frame_for_order(
    df: pd.DataFrame,
    ordered_values,
    *,
    axis_col: str,
    zero_fill_cols: list[str],
    singular_count_name: str,
    count_name: str,
    hover_col: str = "rest_hover",
) -> pd.DataFrame:
    """Return an axis-complete count frame for ordered categorical bars."""
    d = df.set_index(axis_col).reindex(ordered_values).reset_index()
    d = d.rename(columns={"index": axis_col})
    if zero_fill_cols:
        d[zero_fill_cols] = d[zero_fill_cols].fillna(0)
    if hover_col is not None:
        if hover_col in d.columns:
            d[hover_col] = d[hover_col].fillna("")
        else:
            d[hover_col] = ""
    return add_count_text(
        d,
        count_col="count",
        singular_name=singular_count_name,
        plural_name=count_name,
    )


def add_count_bar_trace(
    fig,
    d: pd.DataFrame,
    *,
    customdata_cols: list[str],
    **kwargs,
) -> int:
    """Mutate fig by adding one count bar trace after building customdata from d."""
    customdata = np.c_[*[d[col].to_numpy() for col in customdata_cols]]
    return add_bar_trace(fig, customdata=customdata, **kwargs)


def aggregate_small_categories_total(cat, n_min):
    """Return the all-faculties category frame with small categories merged into Rest."""
    return aggregate_small_groups(
        cat,
        group_cols=["year"],
        label_col="category_name",
        count_col="count",
        total_cols=["year_total"],
        pct_specs=[("pct_of_year_total", "year_total")],
        n_min=n_min,
        singular_name="Anmeldung",
        plural_name="Anmeldungen",
    )


def finalize_categories_total(cat):
    """Return the all-faculties category frame with percentages recomputed and no Rest merge."""
    return finalize_group_percentages(
        cat,
        count_col="count",
        pct_specs=[("pct_of_year_total", "year_total")],
    )


def aggregate_small_categories_faculty(cat, n_min):
    """Return the per-faculty category frame with small categories merged into Rest."""
    return aggregate_small_groups(
        cat,
        group_cols=["year", "faculty"],
        label_col="category_name",
        count_col="count",
        total_cols=["faculty_year_total", "year_total"],
        pct_specs=[
            ("pct_of_faculty", "faculty_year_total"),
            ("pct_of_year_total", "year_total"),
        ],
        n_min=n_min,
        singular_name="Anmeldung",
        plural_name="Anmeldungen",
    )


def finalize_categories_faculty(cat):
    """Return the per-faculty category frame with percentages recomputed and no Rest merge."""
    return finalize_group_percentages(
        cat,
        count_col="count",
        pct_specs=[
            ("pct_of_faculty", "faculty_year_total"),
            ("pct_of_year_total", "year_total"),
        ],
    )


def write_year_barplot_interactive_by_category(
    df,
    out_path,
    n_min: int,
    count_name: str,
    color_cycle,
    restcolor,
    faculty_colormap,
    category_col: str,
    category_name: str,
    category_label_map: dict | None = None,
    rest_aggregation: bool = True,
):
    """Write a category-based yearly stacked-bar HTML plot with faculty drilldown to out_path."""
    if category_label_map is None:
        category_label_map = {}
    if category_col not in df.columns:
        raise KeyError(f"Column '{category_col}' not found in dataframe.")

    singular_count_name = get_singular_count_name(count_name)
    df = prepare_category_frame(df, category_col, category_label_map)

    year_total = year_total_agg(df, year_col="year", value_col="count")
    years_sorted = sorted(df["year"].unique())
    faculty_list = sorted(df["faculty"].unique(), reverse=True)

    category_total = (
        df.groupby(["year", "category_name"], as_index=False)["count"]
        .sum()
        .merge(year_total, on="year", how="left")
    )
    if rest_aggregation:
        category_total = aggregate_small_categories_total(category_total, n_min)
    else:
        category_total = finalize_categories_total(category_total)

    fac_year_total = (
        df.groupby(["year", "faculty"], as_index=False)["count"]
        .sum()
        .rename(columns={"count": "faculty_year_total"})
    )
    category_faculty = (
        df.groupby(["year", "faculty", "category_name"], as_index=False)["count"]
        .sum()
        .merge(year_total, on="year", how="left")
        .merge(fac_year_total, on=["year", "faculty"], how="left")
    )
    if rest_aggregation:
        category_faculty = aggregate_small_categories_faculty(category_faculty, n_min)
    else:
        category_faculty = finalize_categories_faculty(category_faculty)

    category_names = sorted_with_rest(category_total["category_name"].unique())

    category_colors = {
        name: (restcolor if name == "Rest" else color_cycle[i % len(color_cycle)])
        for i, name in enumerate(category_names)
    }

    fig = go.Figure()
    total_trace_indices = []
    category_trace_indices_by_faculty = {f: [] for f in faculty_list}

    for category in category_names:
        d = prepare_count_bar_frame(
            category_total[category_total["category_name"] == category],
            years_sorted,
            zero_fill_cols=["count", "pct_of_year_total"],
            singular_count_name=singular_count_name,
            count_name=count_name,
        )

        idx = add_count_bar_trace(
            fig,
            d,
            customdata_cols=["pct_of_year_total", "rest_hover", "count_text"],
            x=d["year"],
            y=d["count"],
            name=category,
            meta=[category],
            visible=True,
            showlegend=True,
            marker=dict(color=category_colors[category]),
            hovertemplate=(
                "%{x}<br>"
                f"{category_name}: "
                "%{meta[0]}<br>"
                "%{customdata[2]}<br>"
                f"%{{customdata[0]:.1%}} aller {count_name} des Jahres"
                "<br>%{customdata[1]}"
                "<extra></extra>"
            ),
        )
        total_trace_indices.append(idx)

    for faculty in faculty_list:
        dff = category_faculty[category_faculty["faculty"] == faculty]
        faculty_categories = sorted_with_rest(dff["category_name"].unique())

        for category in faculty_categories:
            d = prepare_count_bar_frame(
                dff[dff["category_name"] == category],
                years_sorted,
                zero_fill_cols=["count", "pct_of_faculty", "pct_of_year_total"],
                singular_count_name=singular_count_name,
                count_name=count_name,
            )

            idx = add_count_bar_trace(
                fig,
                d,
                customdata_cols=[
                    "pct_of_faculty",
                    "pct_of_year_total",
                    "rest_hover",
                    "count_text",
                ],
                x=d["year"],
                y=d["count"],
                name=category,
                meta=[category],
                legendgroup=faculty,
                visible="legendonly",
                showlegend=False,
                marker=dict(color=category_colors.get(category, restcolor)),
                hovertemplate=(
                    "%{x}<br>"
                    f"{category_name}: "
                    "%{meta[0]}<br>"
                    "%{customdata[3]}<br>"
                    f"%{{customdata[0]:.1%}} % aller {count_name} in "
                    + faculty
                    + "<br>"
                    f"%{{customdata[1]:.1%}} % aller {count_name} des Jahres"
                    "<br>%{customdata[2]}"
                    "<extra></extra>"
                ),
            )
            category_trace_indices_by_faculty[faculty].append(idx)

    n_traces = len(fig.data)
    vis_total, showleg_total = build_trace_visibility(n_traces, total_trace_indices)
    vis_cat_by_fac, showleg_cat_by_fac = build_faculty_visibility_maps(
        n_traces=n_traces,
        faculty_list=faculty_list,
        trace_indices_by_faculty=category_trace_indices_by_faculty,
    )

    buttons_top, button_index_by_faculty = build_faculty_buttons(
        all_label="Fakultät: alle",
        faculty_list=faculty_list,
        vis_default=vis_total,
        showleg_default=showleg_total,
        vis_by_faculty=vis_cat_by_fac,
        showleg_by_faculty=showleg_cat_by_fac,
        traceorder_default="normal",
        traceorder_faculty="normal",
        extra_default_args=[{}],
    )

    apply_bar_layout(
        fig,
        yaxis_title=count_name,
        legend_title_text="",
        legend_traceorder="normal",
        buttons_top=buttons_top,
    )

    trace_map = {
        "faculty_trace_indices": [],
        "institute_trace_indices_by_faculty": category_trace_indices_by_faculty,
        "n_traces": n_traces,
        "button_index_by_faculty": button_index_by_faculty,
    }
    write_html(fig, out_path, trace_map, "bar")


def write_category_barplot_interactive_by_category(
    df,
    out_path,
    count_name: str,
    color_cycle,
    restcolor,
    faculty_colormap,
    x_col: str,
    stack_col: str,
    xaxis_title: str,
    stack_name: str,
    x_order: list[str] | None = None,
    legend_title: str = "",
):
    """Write a categorical stacked-bar HTML plot with faculty drilldown to out_path."""
    missing = [
        col for col in [x_col, stack_col, "faculty", "count"] if col not in df.columns
    ]
    if missing:
        raise KeyError(f"Missing required columns: {', '.join(missing)}")

    singular_count_name = get_singular_count_name(count_name)
    x_values = (
        list(x_order) if x_order is not None else sorted(df[x_col].dropna().unique())
    )
    faculty_list = sorted(df["faculty"].unique(), reverse=True)

    total_counts = df.groupby([x_col, stack_col], as_index=False)["count"].sum()
    faculty_counts = df
    stack_names = sorted_with_rest(total_counts[stack_col].unique())

    stack_colors = {
        name: (restcolor if name == "Rest" else color_cycle[i % len(color_cycle)])
        for i, name in enumerate(stack_names)
    }

    fig = go.Figure()
    total_trace_indices = []
    stack_trace_indices_by_faculty = {faculty: [] for faculty in faculty_list}

    for stack_value in stack_names:
        d = prepare_count_bar_frame_for_order(
            total_counts[total_counts[stack_col] == stack_value],
            x_values,
            axis_col=x_col,
            zero_fill_cols=["count"],
            singular_count_name=singular_count_name,
            count_name=count_name,
        )

        idx = add_count_bar_trace(
            fig,
            d,
            customdata_cols=["count_text"],
            x=d[x_col],
            y=d["count"],
            name=stack_value,
            meta=[stack_value],
            visible=True,
            showlegend=True,
            marker=dict(color=stack_colors[stack_value]),
            hovertemplate=(
                f"{stack_name}: %{{meta[0]}}<br>%{{customdata[0]}}<extra></extra>"
            ),
        )
        total_trace_indices.append(idx)

    for faculty in faculty_list:
        dff = faculty_counts[faculty_counts["faculty"] == faculty]
        faculty_stack_names = sorted_with_rest(dff[stack_col].unique())

        for stack_value in faculty_stack_names:
            d = prepare_count_bar_frame_for_order(
                dff[dff[stack_col] == stack_value],
                x_values,
                axis_col=x_col,
                zero_fill_cols=["count"],
                singular_count_name=singular_count_name,
                count_name=count_name,
            )

            idx = add_count_bar_trace(
                fig,
                d,
                customdata_cols=["count_text"],
                x=d[x_col],
                y=d["count"],
                name=stack_value,
                meta=[stack_value],
                legendgroup=faculty,
                visible="legendonly",
                showlegend=False,
                marker=dict(color=stack_colors.get(stack_value, restcolor)),
                hovertemplate=(
                    f"{stack_name}: %{{meta[0]}}<br>%{{customdata[0]}}<extra></extra>"
                ),
            )
            stack_trace_indices_by_faculty[faculty].append(idx)

    n_traces = len(fig.data)
    vis_total, showleg_total = build_trace_visibility(n_traces, total_trace_indices)
    vis_by_faculty, showleg_by_faculty = build_faculty_visibility_maps(
        n_traces=n_traces,
        faculty_list=faculty_list,
        trace_indices_by_faculty=stack_trace_indices_by_faculty,
    )

    buttons_top, button_index_by_faculty = build_faculty_buttons(
        all_label="Fakultät: alle",
        faculty_list=faculty_list,
        vis_default=vis_total,
        showleg_default=showleg_total,
        vis_by_faculty=vis_by_faculty,
        showleg_by_faculty=showleg_by_faculty,
        traceorder_default="normal",
        traceorder_faculty="normal",
        extra_default_args=[{}],
    )

    apply_bar_layout(
        fig,
        yaxis_title=count_name,
        legend_title_text=legend_title,
        legend_traceorder="normal",
        buttons_top=buttons_top,
    )
    fig.update_xaxes(
        title_text=xaxis_title, categoryorder="array", categoryarray=x_values
    )

    trace_map = {
        "faculty_trace_indices": [],
        "institute_trace_indices_by_faculty": stack_trace_indices_by_faculty,
        "n_traces": n_traces,
        "button_index_by_faculty": button_index_by_faculty,
    }
    write_html(fig, out_path, trace_map, "bar")


def write_year_barplot_interactive(
    df,
    out_path,
    n_min: int,
    count_name: str,
    params_global,
    color_cycle,
    restcolor,
    faculty_colormap,
    remap_inst=True,
    normalize_inst=True,
):
    """Write an institute-based yearly stacked-bar HTML plot with faculty drilldown to out_path."""
    faculties = build_faculties(faculty_colormap)
    singular_count_name = get_singular_count_name(count_name)

    if remap_inst:
        df = remap_institutes(df, params_global)
    if normalize_inst:
        df = normalize_institute_names(df, faculties)

    year_total = year_total_agg(df, year_col="year", value_col="count")
    fac = fac_by_year_agg(
        df,
        year_total,
        year_col="year",
        faculty_col="faculty",
        value_col="count",
        pct_col="pct_of_year_total",
    )
    inst = inst_by_year_agg(
        df,
        year_total,
        year_col="year",
        faculty_col="faculty",
        institute_col="institute_name",
        value_col="count",
        pct_of_faculty_col="pct_of_faculty",
        pct_of_year_total_col="pct_of_year_total",
    )
    inst = aggregate_small_groups(
        inst,
        group_cols=["year", "faculty"],
        label_col="institute_name",
        count_col="count",
        total_cols=["faculty_year_total", "year_total"],
        pct_specs=[
            ("pct_of_faculty", "faculty_year_total"),
            ("pct_of_year_total", "year_total"),
        ],
        n_min=n_min,
        singular_name="Anmeldung",
        plural_name="Anmeldungen",
    )

    years_sorted = sorted(df["year"].unique())
    faculty_list = sorted(df["faculty"].unique(), reverse=True)

    fig = go.Figure()

    faculty_trace_indices = []
    (
        faculty_color,
        institute_trace_indices_by_faculty,
        all_institute_trace_indices,
    ) = build_faculty_trace_state(
        faculty_list,
        color_cycle=color_cycle,
        faculty_colormap=faculty_colormap,
    )

    for f in faculty_list:
        d = prepare_plot_frame(fac[fac["faculty"] == f], years_sorted)

        idx = add_count_bar_trace(
            fig,
            d,
            customdata_cols=["pct_of_year_total"],
            x=d["year"],
            y=d["count"],
            name=f,
            visible=True,
            showlegend=True,
            marker=dict(color=faculty_color[f]),
            hovertemplate=(
                "%{x}<br>"
                "%{fullData.name}<br>"
                f"%{{y:,}} {count_name}<br>"
                f"%{{customdata[0]:.1%}} % aller {count_name} des Jahres"
                "<extra></extra>"
            ),
        )
        faculty_trace_indices.append(idx)

    for f in faculty_list:
        dff = inst[inst["faculty"] == f]
        institutes_sorted = sorted_with_rest(dff["institute_name"].unique())

        for i_name in institutes_sorted:
            d = prepare_count_bar_frame(
                dff[dff["institute_name"] == i_name],
                years_sorted,
                zero_fill_cols=["count", "pct_of_faculty", "pct_of_year_total"],
                singular_count_name=singular_count_name,
                count_name=count_name,
            )

            idx = add_count_bar_trace(
                fig,
                d,
                customdata_cols=[
                    "pct_of_faculty",
                    "pct_of_year_total",
                    "rest_hover",
                    "count_text",
                ],
                x=d["year"],
                y=d["count"],
                name=short_institute_label(i_name),
                meta=[i_name],
                legendgroup=f,
                visible="legendonly",
                showlegend=False,
                marker=dict(
                    color=institute_color_for(
                        f, i_name, color_cycle, faculty_colormap, restcolor
                    )
                ),
                hovertemplate=(
                    "%{x}<br>"
                    "%{meta[0]}<br>"
                    "%{customdata[3]}<br>"
                    f"%{{customdata[0]:.1%}} % aller {count_name} in " + f + "<br>"
                    f"%{{customdata[1]:.1%}} % aller {count_name} des Jahres"
                    "<br>%{customdata[2]}"
                    "<extra></extra>"
                ),
            )
            institute_trace_indices_by_faculty[f].append(idx)
            all_institute_trace_indices.append(idx)

    n_traces = len(fig.data)

    vis_faculty, showleg_faculty = build_trace_visibility(
        n_traces, faculty_trace_indices
    )
    vis_inst_by_fac, showleg_inst_by_fac = build_faculty_visibility_maps(
        n_traces=n_traces,
        faculty_list=faculty_list,
        trace_indices_by_faculty=institute_trace_indices_by_faculty,
    )

    buttons_top, button_index_by_faculty = build_faculty_buttons(
        all_label="Fakultät: alle",
        faculty_list=faculty_list,
        vis_default=vis_faculty,
        showleg_default=showleg_faculty,
        vis_by_faculty=vis_inst_by_fac,
        showleg_by_faculty=showleg_inst_by_fac,
        traceorder_default="reversed",
        traceorder_faculty="normal",
        extra_default_args=[{}],
    )

    apply_bar_layout(
        fig,
        yaxis_title=count_name,
        legend_title_text="",
        legend_traceorder="reversed",
        buttons_top=buttons_top,
    )

    trace_map = {
        "faculty_trace_indices": faculty_trace_indices,
        "institute_trace_indices_by_faculty": institute_trace_indices_by_faculty,
        "n_traces": n_traces,
        "button_index_by_faculty": button_index_by_faculty,
    }
    write_html(fig, out_path, trace_map, "bar_category")
