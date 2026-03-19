import numpy as np
import plotly.graph_objects as go

from src.plotting.plotting_data import build_faculties
from src.plotting.plotting_export import write_html
from src.plotting.plotting_style import (
    faculty_color_for,
    institute_color_for,
    rgba_with_alpha,
    short_faculty_label,
    short_institute_label,
)


wrapping_replacements_subject_groups = {
    "Sozial- und Verhaltenswissenschaften, Sport": "Sozial- und Verhaltens-<br>wissenschaften, Sport",
    "Wirtschaftswissenschaften": "Wirtschaftswissenschaften",
    "Ingenieurwissenschaften": " <br>Ingenieurwissenschaften",
    "Mathematik, Naturwissenschaften": "<br> <br>Mathematik,<br>Naturwissenschaften",
}

wrapping_replacements_by_faculty = {
    "Philosophie, Literatur-, Wissenschafts- und Technikgeschichte": "Philosophie, Literatur-<br>Wissenschafts- und<br>Technikgeschichte",
    "Interdisziplinäre Studien (Schwerpunkt Naturwissenschaften)": " <br>Interdisziplinäre Studien<br>(Schwerpunkt Naturwissenschaften)",
    "Interdisziplinäre Studien (Schwerpunkt Ingenieurwissenschaften)": "Interdisziplinäre Studien<br>(Schwerpunkt Ingenieurwissenschaften)",
    "Interdisziplinäre Studien (Schwerpunkt Rechts-, Wirtschafts- und Sozialwissenschaften)": "<br> <br> <br> <br>Interdisziplinäre Studien<br>(Rechts-, Wirtschafts- und<br>Sozialwissenschaften)",
    "Wirtschaftsingenieurwesen mit ingenieurwissenschaftlichem Schwerpunkt": "<br> <br> <br> <br>Wirtschaftsingenieurwesen mit<br>ingenieurwissenschaftlichem<br>Schwerpunkt",
    "Angewandte Sprachwissenschaft": "Angewandte<br>Sprachwissenschaft",
    "Allgemeine Sprachwissenschaft": "Allgemeine<br>Sprachwissenschaft",
    "Sozial- und Verhaltenswissenschaften, Sport": "Sozial- und Verhaltens-<br>wissenschaften, Sport<br> <br>",
    "Arbeitslehre/Wirtschaftslehre": "<br> <br> <br>Arbeitslehre/<br>Wirtschaftslehre",
    "Erziehungswissenschaft (Pädagogik)": " <br> <br> <br>Erziehungswissenschaft<br>(Pädagogik)",
    "Wirtschaftswissenschaften": "Wirtschaftswissenschaften<br> <br>",
    "Energietechnik (ohne Elektrotechnik)": "Energietechnik<br>(ohne Elektrotechnik)",
    "Umwelttechnik (einschließlich Recycling)": "Umwelttechnik<br>(einschließlich Recycling)",
    "Lebensmittelchemie": "<br> <br>Lebensmittelchemie",
    "Wirtschaftsinformatik": "Wirtschafts-<br>informatik<br>",
    "Physikalische Technik/Mechanische Verfahrenstechnik": "Physikalische Technik/<br>Mechanische<br>Verfahrenstechnik",
    "Bauingenieurwesen/Ingenieurbau": "Bauingenieurwesen/<br>Ingenieurbau",
    "Vermessungswesen (Geodäsie)": "Vermessungswesen<br>(Geodäsie)",
    "Angewandte Geowissenschaften": "Angewandte<br>Geowissenschaften",
}


def wrap_label(
    label: str, explicit_replacements: dict[str, str], wrap_cutoff: int = 18
) -> str:
    if label in explicit_replacements:
        return explicit_replacements[label]

    replacements = [
        (" (Schwerpunkt ", "<br>(Schwerpunkt "),
        (", ", ",<br>"),
        (" und ", "<br>und "),
        ("wissenschaften", "wissenschaften<br>"),
    ]
    wrapped = label
    for old, new in replacements:
        if len(wrapped) > wrap_cutoff and old in wrapped:
            wrapped = wrapped.replace(old, new, 1)
            break
    return wrapped


def write_sankey_subject_groups_interactive(
    df,
    out_path,
    color_cycle,
    restcolor,
    link_alpha: float,
    node_thickness: int,
    faculty_colormap,
    subject_groups,
):
    faculties = build_faculties(faculty_colormap)
    faculty_names = list(faculties.keys())

    nodes = ["TU gesamt"] + faculty_names + list(subject_groups)
    node_labels = [wrap_label(x, wrapping_replacements_subject_groups) for x in nodes]
    nodes_dict = dict(
        label=node_labels,
        pad=15,
        thickness=node_thickness,
        x=[0.001] + [0.5] * 7 + [0.999] * 5,
        y=[0.5]
        + [0.001, 1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6, 1 - 1 / 12]
        + [0.001, 1 / 4 - 1 / 8, 2 / 4, 3 / 4, 1 - 1 / 8],
        color=[restcolor]
        + [faculty_color_for(f, color_cycle, faculty_colormap) for f in faculty_names]
        + [restcolor] * len(subject_groups),
        line=dict(color="rgba(0,0,0,0.15)", width=0.5),
        customdata=nodes,
        hovertemplate="%{customdata}<extra></extra>",
    )

    node_index = {label: i for i, label in enumerate(nodes)}

    faculty_counts = df["faculty"].value_counts()
    links_tu_to_faculty = {
        "source": [node_index["TU gesamt"]] * len(faculties),
        "target": [node_index[faculty] for faculty in faculty_names],
        "value": [faculty_counts.get(faculty, 0) for faculty in faculty_names],
    }

    faculty_subject_counts = (
        df.groupby(["faculty", "subject_group_category_stala"])
        .size()
        .reset_index(name="count")
    )

    links_faculty_to_subjgroups = {
        "source": [
            node_index[faculty] for faculty in faculty_subject_counts["faculty"]
        ],
        "target": [
            node_index[group]
            for group in faculty_subject_counts["subject_group_category_stala"]
        ],
        "value": faculty_subject_counts["count"].tolist(),
    }

    link_source = links_tu_to_faculty["source"] + links_faculty_to_subjgroups["source"]
    link_target = links_tu_to_faculty["target"] + links_faculty_to_subjgroups["target"]
    link_value = links_tu_to_faculty["value"] + links_faculty_to_subjgroups["value"]
    color_value = [
        rgba_with_alpha(
            faculty_color_for(faculty_names[f_idx - 1], color_cycle, faculty_colormap),
            link_alpha,
        )
        for f_idx in links_tu_to_faculty["target"]
    ] + [
        rgba_with_alpha(
            faculty_color_for(faculty_names[f_idx - 1], color_cycle, faculty_colormap),
            link_alpha,
        )
        for f_idx in links_faculty_to_subjgroups["source"]
    ]

    link_labels = []
    tu_total = sum(links_tu_to_faculty["value"])
    for source, target, value in zip(link_source, link_target, link_value):
        source_label = nodes[source]
        target_label = nodes[target]

        if source_label in faculty_names and target_label in subject_groups:
            link_labels.append(
                f"{value} Promotionen in {target_label} in {source_label}"
            )
        else:
            link_labels.append(f"{100 * value / tu_total:.2f} % aller Promotionen")

    link_dict = dict(
        source=link_source,
        target=link_target,
        value=link_value,
        color=color_value,
        label=link_labels,
        hovertemplate="%{label}<extra></extra>",
    )

    fig = go.Figure(
        data=[
            go.Sankey(
                node=nodes_dict,
                link=link_dict,
                arrangement="snap",
            )
        ]
    )

    fig.update_layout(
        margin=dict(l=10, r=10, t=110, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12),
        annotations=[
            dict(
                x=0.5,
                y=1.0,
                text="<b>Fakultäten</b>",
                showarrow=False,
                xref="paper",
                yref="paper",
                xanchor="center",
                yanchor="bottom",
                align="center",
                font=dict(size=15),
            ),
            dict(
                x=0.999,
                y=1.0,
                text="<b>Fächergruppen</b>",
                showarrow=False,
                xref="paper",
                yref="paper",
                xanchor="right",
                yanchor="bottom",
                align="right",
                font=dict(size=15),
            ),
        ],
    )

    write_html(fig, out_path, {}, plot_type="sankey")


def write_sankey_diag_interactive(
    df,
    out_path,
    color_cycle,
    restcolor,
    link_alpha: float,
    node_thickness: int,
    faculty_colormap,
    default_faculty: str = "Fakultät I",
):
    faculty_list = sorted(df["faculty"].unique(), reverse=True)

    fig = go.Figure()
    trace_index_by_faculty = {}

    for faculty in faculty_list:
        dff = df[df["faculty"] == faculty].copy()

        inst_sub = (
            dff.groupby(["institute_name", "subject_group_stala"], as_index=False)
            .size()
            .rename(columns={"size": "count"})
        )
        sub_sta = (
            dff.groupby(
                ["subject_group_stala", "subject_group_category_stala"], as_index=False
            )
            .size()
            .rename(columns={"size": "count"})
        )
        category_totals = (
            sub_sta.groupby("subject_group_category_stala")["count"].sum().to_dict()
        )

        institutes = sorted(inst_sub["institute_name"].unique().tolist())
        subjects = sorted(
            sorted(
                set(inst_sub["subject_group_stala"].unique().tolist())
                | set(sub_sta["subject_group_stala"].unique().tolist())
            )
        )
        stalas = sorted(sub_sta["subject_group_category_stala"].unique().tolist())

        inst_labels = [
            wrap_label(short_institute_label(x), wrapping_replacements_by_faculty)
            for x in institutes
        ]
        subj_labels = [
            wrap_label(x, wrapping_replacements_by_faculty) for x in subjects
        ]
        stala_labels = [wrap_label(x, wrapping_replacements_by_faculty) for x in stalas]

        labels = inst_labels + subj_labels + stala_labels
        hover_labels = (
            [short_institute_label(x) for x in institutes] + subjects + stalas
        )
        node_x = (
            [0.001] * len(institutes) + [0.5] * len(subjects) + [0.999] * len(stalas)
        )

        inst_idx = {name: i for i, name in enumerate(institutes)}
        subj_idx = {name: i + len(institutes) for i, name in enumerate(subjects)}
        stala_idx = {
            name: i + len(institutes) + len(subjects) for i, name in enumerate(stalas)
        }

        node_colors = []
        for inst in institutes:
            node_colors.append(
                institute_color_for(
                    faculty, inst, color_cycle, faculty_colormap, restcolor
                )
            )
        node_colors += [restcolor] * len(subjects)
        node_colors += [restcolor] * len(stalas)

        sources = []
        targets = []
        values = []
        link_colors = []
        link_hover = []

        for row in inst_sub.itertuples(index=False):
            inst = row.institute_name
            subj = row.subject_group_stala
            cnt = int(row.count)

            sources.append(inst_idx[inst])
            targets.append(subj_idx[subj])
            values.append(cnt)

            color = institute_color_for(
                faculty, inst, color_cycle, faculty_colormap, restcolor
            )
            link_colors.append(rgba_with_alpha(color, link_alpha))
            link_hover.append(
                f"{cnt} Promotionen<br>am {inst}<br>in {subj}<extra></extra>"
            )

        for row in sub_sta.itertuples(index=False):
            subj = row.subject_group_stala
            sta = row.subject_group_category_stala
            cnt = int(row.count)
            category_total = int(category_totals[sta])

            sources.append(subj_idx[subj])
            targets.append(stala_idx[sta])
            values.append(cnt)

            link_colors.append(rgba_with_alpha(restcolor, link_alpha))
            link_hover.append(
                f"{cnt} Promotionen<br>"
                f"in {subj}<br>"
                f"von {category_total} Promotionen<br>"
                f"in {sta}<extra></extra>"
            )

        visible = faculty == default_faculty

        sankey = go.Sankey(
            arrangement="fixed",
            node=dict(
                label=labels,
                customdata=hover_labels,
                x=node_x,
                color=node_colors,
                pad=12,
                thickness=node_thickness,
                line=dict(color="rgba(0,0,0,0.15)", width=0.5),
                hovertemplate="%{customdata}<extra></extra>",
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values,
                color=link_colors,
                customdata=np.array(link_hover, dtype=object),
                hovertemplate="%{customdata}",
            ),
            valueformat=",",
        )

        trace_index_by_faculty[faculty] = len(fig.data)
        fig.add_trace(sankey)
        fig.data[-1].visible = visible

    n_traces = len(fig.data)

    def vis_only(idx: int) -> list[bool]:
        vis = [False] * n_traces
        vis[idx] = True
        return vis

    ordered_faculties = [
        faculty
        for faculty in faculty_list
        if faculty != default_faculty and faculty in trace_index_by_faculty
    ] + [default_faculty]

    buttons = []
    for faculty in reversed(ordered_faculties):
        idx = trace_index_by_faculty[faculty]
        buttons.append(
            dict(
                label=short_faculty_label(faculty),
                method="update",
                args=[
                    {"visible": vis_only(idx)},
                    {
                        "title": {
                            "text": f"<u>Fakultät {short_faculty_label(faculty)}</u>",
                            "x": 0.12,
                            "y": 0.895,
                            "xanchor": "center",
                            "yanchor": "top",
                        }
                    },
                ],
            )
        )

    fig.update_layout(
        title=dict(
            text=f"<u>Fakultät {short_faculty_label(default_faculty)}</u>",
            x=0.12,
            y=0.895,
            xanchor="center",
            yanchor="top",
        ),
        margin=dict(l=10, r=10, t=125, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12),
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                x=0.5,
                y=1.16,
                xanchor="center",
                yanchor="top",
                buttons=buttons,
                pad=dict(t=6, r=6, b=6, l=6),
            )
        ],
        annotations=[
            dict(
                x=0.001,
                y=1.0,
                text="<b>Institute</b>",
                showarrow=False,
                xref="paper",
                yref="paper",
                xanchor="left",
                yanchor="bottom",
                align="left",
                font=dict(size=15),
            ),
            dict(
                x=0.5,
                y=1.0,
                text="<b>Fächer</b>",
                showarrow=False,
                xref="paper",
                yref="paper",
                xanchor="center",
                yanchor="bottom",
                align="center",
                font=dict(size=15),
            ),
            dict(
                x=0.999,
                y=1.0,
                text="<b>Fächergruppen</b>",
                showarrow=False,
                xref="paper",
                yref="paper",
                xanchor="right",
                yanchor="bottom",
                align="right",
                font=dict(size=15),
            ),
        ],
    )

    write_html(fig, out_path, {}, plot_type="sankey")
