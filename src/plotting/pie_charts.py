import plotly.graph_objects as go

from src.plotting.plotting_data import (
    aggregate_top_groups,
    aggregate_small_groups,
    finalize_group_percentages,
    get_singular_count_name,
    prepare_category_frame,
)
from src.plotting.plotting_export import write_html
from src.plotting.plotting_style import short_faculty_label
from src.plotting.plotting_traces import sorted_with_rest


def build_hovertext_total(d, category_name, count_name):
    """Return hover labels for the all-faculties pie view."""
    lines = []
    for label, count_text, pct_total, rest_hover in zip(
        d["category_name"],
        d["count_text"],
        d["pct_of_total"],
        d["rest_hover"],
    ):
        text = (
            f"{category_name}: {label}<br>"
            f"{count_text}<br>"
            f"{pct_total:.1%} % aller {count_name} im gewählten Zeitraum"
        )
        if rest_hover:
            text += f"<br>{rest_hover}"
        lines.append(text)
    return lines


def build_hovertext_faculty(d, category_name, count_name, faculty):
    """Return hover labels for one faculty-specific pie view."""
    lines = []
    for label, count_text, pct_fac, pct_total, rest_hover in zip(
        d["category_name"],
        d["count_text"],
        d["pct_of_faculty"],
        d["pct_of_total"],
        d["rest_hover"],
    ):
        text = (
            f"{category_name}: {label}<br>"
            f"{count_text}<br>"
            f"{pct_fac:.1%} % aller {count_name} in {faculty}<br>"
            f"{pct_total:.1%} % aller {count_name} im gewählten Zeitraum"
        )
        if rest_hover:
            text += f"<br>{rest_hover}"
        lines.append(text)
    return lines


def order_pie_categories(d):
    """Return a copy ordered alphabetically, with Rest forced to the end."""
    order = {
        name: i for i, name in enumerate(sorted_with_rest(d["category_name"].unique()))
    }
    return (
        d.assign(_order=d["category_name"].map(order))
        .sort_values(["_order", "category_name"])
        .drop(columns="_order")
    )


def order_pie_categories_by_size(d):
    """Return a copy ordered by descending size, with Rest forced to the end."""
    out = d.copy()
    out["_is_rest"] = out["category_name"].eq("Rest")
    return (
        out.sort_values(
            ["_is_rest", "count", "category_name"],
            ascending=[True, False, True],
        )
        .drop(columns="_is_rest")
        .reset_index(drop=True)
    )


def build_category_colors(category_total, category_faculty, color_cycle, restcolor):
    """Return a stable label-to-color mapping across the total and faculty pies."""
    category_names = sorted_with_rest(
        set(category_total["category_name"]).union(category_faculty["category_name"])
    )
    return {
        name: (restcolor if name == "Rest" else color_cycle[i % len(color_cycle)])
        for i, name in enumerate(category_names)
    }


def aggregate_small_categories_total(cat, n_min):
    """Return all-faculties category counts with small groups merged into ``Rest``."""
    return aggregate_small_groups(
        cat,
        group_cols=[],
        label_col="category_name",
        count_col="count",
        total_cols=["total_count"],
        pct_specs=[("pct_of_total", "total_count")],
        n_min=n_min,
        singular_name="Anmeldung",
        plural_name="Anmeldungen",
    )


def finalize_categories_total(cat):
    """Return all-faculties category counts with percentages recomputed and no rest merge."""
    return finalize_group_percentages(
        cat,
        count_col="count",
        pct_specs=[("pct_of_total", "total_count")],
    )


def aggregate_top_categories_total(cat, top_n, singular_count_name, count_name):
    """Return all-faculties category counts with only the top_n groups kept separate."""
    return aggregate_top_groups(
        cat,
        group_cols=[],
        label_col="category_name",
        count_col="count",
        total_cols=["total_count"],
        pct_specs=[("pct_of_total", "total_count")],
        top_n=top_n,
        singular_name=singular_count_name,
        plural_name=count_name,
    )


def aggregate_small_categories_faculty(cat, n_min):
    """Return per-faculty category counts with small groups merged into ``Rest``."""
    return aggregate_small_groups(
        cat,
        group_cols=["faculty"],
        label_col="category_name",
        count_col="count",
        total_cols=["faculty_total", "total_count"],
        pct_specs=[
            ("pct_of_faculty", "faculty_total"),
            ("pct_of_total", "total_count"),
        ],
        n_min=n_min,
        singular_name="Anmeldung",
        plural_name="Anmeldungen",
    )


def finalize_categories_faculty(cat):
    """Return per-faculty category counts with percentages recomputed and no rest merge."""
    return finalize_group_percentages(
        cat,
        count_col="count",
        pct_specs=[
            ("pct_of_faculty", "faculty_total"),
            ("pct_of_total", "total_count"),
        ],
    )


def aggregate_top_categories_faculty(cat, top_n, singular_count_name, count_name):
    """Return per-faculty category counts with only the top_n groups kept separate."""
    return aggregate_top_groups(
        cat,
        group_cols=["faculty"],
        label_col="category_name",
        count_col="count",
        total_cols=["faculty_total", "total_count"],
        pct_specs=[
            ("pct_of_faculty", "faculty_total"),
            ("pct_of_total", "total_count"),
        ],
        top_n=top_n,
        singular_name=singular_count_name,
        plural_name=count_name,
    )


def write_category_pie_interactive(
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
    top_n: int | None = None,
    rotation: float = 0,
    order_by_size: bool = False,
):
    """Write a category-based pie chart to ``out_path``."""
    del faculty_colormap

    if category_label_map is None:
        category_label_map = {}
    if category_col not in df.columns:
        raise KeyError(f"Column '{category_col}' not found in dataframe.")
    if top_n is not None and top_n < 1:
        raise ValueError("top_n must be at least 1 when provided.")

    singular_count_name = get_singular_count_name(count_name)
    df = prepare_category_frame(df, category_col, category_label_map)
    df = df.copy()
    faculty_list = sorted(df["faculty"].unique(), reverse=True)

    total_count = int(df["count"].sum())
    category_total = (
        df.groupby("category_name", as_index=False)["count"]
        .sum()
        .assign(total_count=total_count)
    )
    if top_n is not None:
        category_total = aggregate_top_categories_total(
            category_total, top_n, singular_count_name, count_name
        )
    elif rest_aggregation:
        category_total = aggregate_small_categories_total(category_total, n_min)
    else:
        category_total = finalize_categories_total(category_total)

    faculty_total = (
        df.groupby("faculty", as_index=False)["count"]
        .sum()
        .rename(columns={"count": "faculty_total"})
    )
    category_faculty = (
        df.groupby(["faculty", "category_name"], as_index=False)["count"]
        .sum()
        .merge(faculty_total, on="faculty", how="left")
        .assign(total_count=total_count)
    )
    if top_n is not None:
        category_faculty = aggregate_top_categories_faculty(
            category_faculty, top_n, singular_count_name, count_name
        )
    elif rest_aggregation:
        category_faculty = aggregate_small_categories_faculty(category_faculty, n_min)
    else:
        category_faculty = finalize_categories_faculty(category_faculty)

    category_colors = build_category_colors(
        category_total, category_faculty, color_cycle, restcolor
    )
    order_categories = (
        order_pie_categories_by_size if order_by_size else order_pie_categories
    )

    fig = go.Figure()
    total_trace_indices = []
    category_trace_indices_by_faculty = {f: [] for f in faculty_list}

    d = order_categories(category_total.copy())
    d["count_text"] = [
        f"{int(y):,} {singular_count_name}" if y == 1 else f"{int(y):,} {count_name}"
        for y in d["count"]
    ]
    d["hovertext"] = build_hovertext_total(d, category_name, count_name)
    fig.add_trace(
        go.Pie(
            labels=d["category_name"],
            values=d["count"],
            visible=True,
            showlegend=True,
            sort=False,
            marker=dict(
                colors=[category_colors.get(x, restcolor) for x in d["category_name"]],
                line=dict(color="rgba(0,0,0,0)", width=0),
            ),
            hovertext=d["hovertext"],
            hovertemplate="%{hovertext}<extra></extra>",
            textinfo="percent",
            rotation=rotation,
        )
    )
    total_trace_indices.append(0)

    for faculty in faculty_list:
        dff = order_categories(
            category_faculty[category_faculty["faculty"] == faculty].copy()
        )
        dff["count_text"] = [
            f"{int(y):,} {singular_count_name}"
            if y == 1
            else f"{int(y):,} {count_name}"
            for y in dff["count"]
        ]
        dff["hovertext"] = build_hovertext_faculty(
            dff, category_name, count_name, faculty
        )
        fig.add_trace(
            go.Pie(
                labels=dff["category_name"],
                values=dff["count"],
                visible=False,
                showlegend=False,
                sort=False,
                marker=dict(
                    colors=[
                        category_colors.get(x, restcolor) for x in dff["category_name"]
                    ],
                    line=dict(color="rgba(0,0,0,0)", width=0),
                ),
                hovertext=dff["hovertext"],
                hovertemplate="%{hovertext}<extra></extra>",
                textinfo="percent",
                rotation=rotation,
            )
        )
        category_trace_indices_by_faculty[faculty].append(len(fig.data) - 1)

    n_traces = len(fig.data)
    vis_total = [False] * n_traces
    showleg_total = [False] * n_traces
    for idx in total_trace_indices:
        vis_total[idx] = True
        showleg_total[idx] = True

    vis_cat_by_fac = {}
    showleg_cat_by_fac = {}
    for faculty in faculty_list:
        vis = [False] * n_traces
        showleg = [False] * n_traces
        for idx in category_trace_indices_by_faculty.get(faculty, []):
            vis[idx] = True
            showleg[idx] = True
        vis_cat_by_fac[faculty] = vis
        showleg_cat_by_fac[faculty] = showleg

    buttons_top = [
        dict(
            label="Fakultät: alle",
            method="update",
            args=[{"visible": vis_total, "showlegend": showleg_total}],
        )
    ]

    for faculty in reversed(faculty_list):
        buttons_top.append(
            dict(
                label=short_faculty_label(faculty),
                method="update",
                args=[
                    {
                        "visible": vis_cat_by_fac[faculty],
                        "showlegend": showleg_cat_by_fac[faculty],
                    }
                ],
            )
        )

    fig.update_layout(
        margin=dict(t=88, r=24, b=110, l=24),
        legend_title_text="",
        legend=dict(
            orientation="h",
            x=0.5,
            xanchor="center",
            y=-0.15,
            yanchor="top",
            groupclick="toggleitem",
            tracegroupgap=0,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                x=0.5,
                y=1.05,
                xanchor="center",
                yanchor="bottom",
                buttons=buttons_top,
                pad=dict(t=6, r=6, b=6, l=6),
            )
        ],
    )

    write_html(fig, out_path, trace_map=None, plot_type="bar_category")
