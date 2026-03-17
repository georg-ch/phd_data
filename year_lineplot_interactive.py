import numpy as np
import plotly.graph_objects as go

from plotting_data import (
    build_faculties,
    fac_by_year_agg,
    normalize_institute_names,
    prune_groups_for_plot,
    remap_institutes,
    series_with_rest_from_keys,
    year_total_agg,
)
from plotting_export import write_html
from plotting_style import (
    apply_line_layout,
    institute_color_for,
    short_institute_label,
)
from plotting_traces import (
    build_faculty_trace_state,
    build_faculty_visibility_maps,
    build_faculty_buttons,
    build_trace_visibility,
    prepare_plot_frame,
    sorted_with_rest,
)


def add_scatter_trace(fig, **kwargs) -> int:
    """Mutate fig by adding one scatter trace and return the new trace index."""
    fig.add_trace(go.Scatter(**kwargs))
    return len(fig.data) - 1


def add_percentage_line_trace(
    fig,
    df,
    years_sorted,
    *,
    y_col: str,
    **kwargs,
) -> int:
    """Reindex df to years_sorted, add one line trace to fig, and return its trace index."""
    d = prepare_plot_frame(df, years_sorted)
    return add_scatter_trace(fig, x=d["year"], y=d[y_col], **kwargs)


def add_institute_percentage_trace(
    fig,
    df,
    years_sorted,
    *,
    faculty: str,
    institute_name: str,
    color_cycle,
    faculty_colormap,
    restcolor: str,
    width_inst: float,
    marker_inst: dict,
) -> int:
    """Add one institute percentage line trace to fig and return its trace index, including Rest styling."""
    d = prepare_plot_frame(
        df,
        years_sorted,
        zero_fill_cols=["female_count", "total_count"],
        hover_col="rest_hover",
    )
    customdata = np.c_[
        d["female_count"].to_numpy(),
        d["total_count"].to_numpy(),
        d["rest_hover"].to_numpy(),
    ]

    is_rest = institute_name == "Rest"
    hovertemplate = "%{x}<br>%{meta[0]}<br>%{y:.1%}<br>"
    if is_rest:
        hovertemplate += "<br>%{customdata[2]}"
    hovertemplate += "<extra></extra>"

    return add_scatter_trace(
        fig,
        x=d["year"],
        y=d["female_percentage"],
        name=short_institute_label(institute_name),
        meta=[institute_name],
        legendgroup=faculty,
        visible="legendonly",
        showlegend=False,
        mode="lines+markers",
        line=dict(
            color=institute_color_for(
                faculty,
                institute_name,
                color_cycle,
                faculty_colormap,
                restcolor,
            ),
            width=width_inst,
            dash="dash" if is_rest else "solid",
        ),
        marker=(dict(**marker_inst, color=restcolor) if is_rest else marker_inst),
        opacity=(0.4 if is_rest else 1.0),
        customdata=customdata,
        hovertemplate=hovertemplate,
    )


def write_year_lineplot_interactive(
    df,
    out_path,
    value_col: str,
    n_min_year_prune: int,
    n_min_inst_abs: int,
    n_min_inst_mean: int,
    params_global,
    color_cycle,
    restcolor,
    faculty_colormap,
    yaxis_title: str = "Anteil Weiblich/Divers",
    total_label: str = "Gesamt",
    all_faculties_label: str = "Fakultäten (alle)",
    faculty_total_suffix: str = " (Fakultät gesamt)",
    marker_fac: dict | None = None,
    marker_total: dict | None = None,
    marker_inst: dict | None = None,
    marker_fac_total: dict | None = None,
    width_total: float = 12,
    width_fac: float = 4,
    width_inst: float = 4,
    width_fac_total: float = 4,
    fac_total_color: str = "#000000",
):
    """Write a yearly percentage line-chart HTML plot with faculty and institute drilldown to out_path."""
    if value_col not in df.columns:
        raise KeyError(f"Column '{value_col}' not found in dataframe.")

    marker_fac = marker_fac or dict(size=12, symbol="circle", line=dict(width=1))
    marker_total = marker_total or dict(
        size=22,
        symbol="circle",
        line=dict(width=0),
        color=restcolor,
    )
    marker_inst = marker_inst or dict(size=12, symbol="circle", line=dict(width=1))
    marker_fac_total = marker_fac_total or dict(
        size=16,
        symbol="diamond",
        line=dict(width=1),
        color=fac_total_color,
    )

    df = df.copy()
    df = remap_institutes(df, params_global)
    faculties = build_faculties(faculty_colormap)
    df = normalize_institute_names(df, faculties)

    years_sorted = sorted(df["year"].dropna().unique())
    faculty_list = sorted(df["faculty"].dropna().unique(), reverse=True)

    metric_df = df.rename(columns={value_col: "is_female"})
    year_total = year_total_agg(
        metric_df,
        year_col="year",
        value_col="is_female",
        total_col="year_total_percentage",
        agg="mean",
    )
    fac_by_year = fac_by_year_agg(
        metric_df,
        year_total,
        year_col="year",
        faculty_col="faculty",
        value_col="is_female",
        total_col="year_total_percentage",
        out_value_col="female_percentage",
        agg="mean",
    )

    fac_year_total = (
        df.groupby(["year", "faculty"], as_index=False)[value_col]
        .mean()
        .rename(columns={value_col: "faculty_year_total"})
    )

    survivors_keys, remainder_keys, _inst_stats = prune_groups_for_plot(
        df,
        year_col="year",
        group_cols=["faculty", "institute_name"],
        years_sorted=years_sorted,
        n_min_year_prune=n_min_year_prune,
        n_min_abs=n_min_inst_abs,
        n_min_mean=n_min_inst_mean,
    )

    inst_by_year_plot = series_with_rest_from_keys(
        metric_df,
        year_total,
        fac_year_total,
        survivors_keys,
        remainder_keys,
        year_col="year",
        faculty_col="faculty",
        label_col="institute_name",
        value_col="is_female",
    )

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

    fig.add_shape(
        type="line",
        x0=years_sorted[0] - 0.2,
        x1=years_sorted[-1] + 0.2,
        y0=0.5,
        y1=0.5,
        line=dict(color="rgba(0,0,0,0.5)", dash="dot"),
        layer="below",
    )

    year_total_trace_idx = add_percentage_line_trace(
        fig,
        year_total,
        years_sorted,
        y_col="year_total_percentage",
        name=total_label,
        visible=True,
        showlegend=True,
        mode="lines+markers",
        marker=marker_total,
        line=dict(width=width_total, color=restcolor),
        opacity=0.6,
        hovertemplate=f"%{{x}}<br>{total_label}<br>%{{y:.1%}}<extra></extra>",
    )

    for faculty in faculty_list:
        idx = add_percentage_line_trace(
            fig,
            fac_by_year[fac_by_year["faculty"] == faculty],
            years_sorted,
            y_col="female_percentage",
            name=faculty,
            visible=True,
            showlegend=True,
            mode="lines+markers",
            line=dict(color=faculty_color[faculty], width=width_fac),
            marker=marker_fac,
            hovertemplate="%{x}<br>%{fullData.name}<br>%{y:.1%}<extra></extra>",
        )
        faculty_trace_indices.append(idx)

    fac_total_trace_idx_by_faculty = {}
    for faculty in faculty_list:
        fac_total_trace_idx_by_faculty[faculty] = add_percentage_line_trace(
            fig,
            fac_year_total[fac_year_total["faculty"] == faculty],
            years_sorted,
            y_col="faculty_year_total",
            name=f"{faculty}{faculty_total_suffix}",
            visible="legendonly",
            showlegend=False,
            mode="lines+markers",
            line=dict(color=fac_total_color, width=width_fac_total, dash="dot"),
            marker=marker_fac_total,
            hovertemplate="%{x}<br>%{fullData.name}<br>%{y:.1%}<extra></extra>",
        )

    for faculty in faculty_list:
        dff = inst_by_year_plot[inst_by_year_plot["faculty"] == faculty].copy()

        institutes_sorted = sorted_with_rest(dff["institute_name"].unique())

        for institute_name in institutes_sorted:
            idx = add_institute_percentage_trace(
                fig,
                dff[dff["institute_name"] == institute_name],
                years_sorted,
                faculty=faculty,
                institute_name=institute_name,
                color_cycle=color_cycle,
                faculty_colormap=faculty_colormap,
                restcolor=restcolor,
                width_inst=width_inst,
                marker_inst=marker_inst,
            )
            institute_trace_indices_by_faculty[faculty].append(idx)
            all_institute_trace_indices.append(idx)

    n_traces = len(fig.data)

    vis_faculty, showleg_faculty = build_trace_visibility(
        n_traces, [year_total_trace_idx, *faculty_trace_indices]
    )
    vis_inst_by_fac, showleg_inst_by_fac = build_faculty_visibility_maps(
        n_traces=n_traces,
        faculty_list=faculty_list,
        trace_indices_by_faculty={
            faculty: [
                *( [fac_total_trace_idx_by_faculty[faculty]] if faculty in fac_total_trace_idx_by_faculty else [] ),
                *institute_trace_indices_by_faculty.get(faculty, []),
            ]
            for faculty in faculty_list
        },
        always_visible_indices=[year_total_trace_idx],
        always_legend_indices=[year_total_trace_idx],
    )

    buttons_top, button_index_by_faculty = build_faculty_buttons(
        all_label=all_faculties_label,
        faculty_list=faculty_list,
        vis_default=vis_faculty,
        showleg_default=showleg_faculty,
        vis_by_faculty=vis_inst_by_fac,
        showleg_by_faculty=showleg_inst_by_fac,
        traceorder_default="reversed",
        traceorder_faculty="normal",
    )

    apply_line_layout(
        fig,
        yaxis_title=yaxis_title,
        buttons_top=buttons_top,
    )

    trace_map = {
        "faculty_trace_indices": faculty_trace_indices,
        "institute_trace_indices_by_faculty": institute_trace_indices_by_faculty,
        "n_traces": n_traces,
        "button_index_by_faculty": button_index_by_faculty,
        "year_total_trace_idx": year_total_trace_idx,
        "fac_total_trace_idx_by_faculty": fac_total_trace_idx_by_faculty,
    }

    write_html(fig, out_path, trace_map, "line")
