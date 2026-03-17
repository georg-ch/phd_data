def cycle_color(i: int, color_cycle: list) -> str:
    """Return the i-th color from color_cycle, wrapping around for large indices."""
    return color_cycle[i % len(color_cycle)]


def faculty_color_for(
    faculty: str, color_cycle: list[str], faculty_colormap: dict
) -> str:
    """Return the configured display color for a faculty, falling back to the first cycle color."""
    idx = faculty_colormap.get(faculty, {}).get("color", 0)
    return cycle_color(idx, color_cycle)


def institute_color_for(
    faculty: str,
    institute_name: str,
    color_cycle: list[str],
    faculty_colormap: dict,
    restcolor: str,
) -> str:
    """Return the configured display color for an institute, or restcolor for the Rest bucket."""
    if institute_name == "Rest":
        return restcolor
    idx = (
        faculty_colormap.get(faculty, {})
        .get("institute_colors", {})
        .get(institute_name, 0)
    )
    return cycle_color(idx, color_cycle)


def short_institute_label(name: str) -> str:
    """Return a shortened institute label by stripping common leading prefixes."""
    prefixes = ("Institut für ", "Zentrum für ")
    for p in prefixes:
        if name.startswith(p):
            return name[len(p) :].strip()
    return name


def short_faculty_label(name: str) -> str:
    """Return a shortened faculty label by stripping the leading 'Fakultät' prefix."""
    prefix = "Fakultät "
    return name[len(prefix) :].strip() if name.startswith(prefix) else name


def rgba_with_alpha(color_str: str, alpha: float) -> str:
    """Return color_str converted from '#rrggbb' form to an rgba() string with alpha."""
    s = color_str.strip()
    r = int(s[1:3], 16)
    g = int(s[3:5], 16)
    b = int(s[5:7], 16)
    return f"rgba({r},{g},{b},{alpha})"


def apply_bar_layout(
    fig,
    *,
    yaxis_title: str,
    legend_title_text: str,
    legend_traceorder: str,
    buttons_top,
) -> None:
    """Mutate fig in place with the shared stacked-bar layout, legend, grid, and top buttons."""

    if legend_title_text:
        title_dict = dict(
            text=legend_title_text,
            side="top",
        )
    else:
        title_dict = None

    fig.update_layout(
        barmode="stack",
        hovermode="closest",
        xaxis_title="Jahr",
        yaxis_title=yaxis_title,
        legend_title_text=legend_title_text,
        margin=dict(autoexpand=False),
        legend=dict(
            orientation="h",
            x=0.5,
            xanchor="center",
            y=-0.15,
            yanchor="top",
            groupclick="toggleitem",
            traceorder=legend_traceorder,
            tracegroupgap=0,
            title=title_dict,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(0,0,0,0.15)",
            zeroline=False,
            title=dict(standoff=10),
            automargin=True,
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(0,0,0,0.15)",
            zeroline=False,
            domain=[0.0, 1.0],
            automargin=True,
        ),
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                x=0.5,
                y=1.0,
                xanchor="center",
                yanchor="bottom",
                buttons=buttons_top,
                pad=dict(t=6, r=6, b=6, l=6),
            )
        ],
    )


def apply_line_layout(
    fig,
    *,
    yaxis_title: str,
    buttons_top,
) -> None:
    """Mutate fig in place with the shared line-plot layout, axes, legend, and top buttons."""
    fig.update_layout(
        autosize=True,
        hovermode="closest",
        xaxis_title="Jahr",
        yaxis_title=yaxis_title,
        yaxis=dict(
            tickformat=".0%",
            showgrid=True,
            gridcolor="rgba(0,0,0,0.15)",
            zeroline=False,
            domain=[0.1, 1.0],
            range=[0, 1],
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(0,0,0,0.15)",
            zeroline=False,
        ),
        legend=dict(
            orientation="h",
            x=0.5,
            xanchor="center",
            y=-0.02,
            yanchor="top",
            groupclick="toggleitem",
            traceorder="reversed",
        ),
        margin=dict(autoexpand=False),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                x=0.5,
                y=1.06,
                xanchor="center",
                yanchor="bottom",
                buttons=buttons_top,
            )
        ],
    )
