import plotly.io as pio

from src.plotting.js_css import (
    css,
    generate_js,
    js_resize_barplot,
    js_resize_forest,
    js_resize_lineplot,
    js_resize_sankey,
    js_resize_violin,
    js_resize_violin_contrast,
)


def write_html(fig, out_path, trace_map, plot_type):
    """Write fig to out_path using the shared responsive HTML wrapper and plot-type-specific resize JS."""
    match plot_type:
        case "bar":
            js = generate_js(trace_map) + js_resize_barplot
        case "line":
            js = generate_js(trace_map) + js_resize_lineplot
        case "sankey":
            js = generate_js(trace_map) + js_resize_sankey
        case "bar_category":
            js = js_resize_barplot
        case "forest":
            js = js_resize_forest
        case "violin":
            js = js_resize_violin
        case "violin_contrast":
            js = js_resize_violin + js_resize_violin_contrast
        case _:
            raise ValueError(f"Unknown plot type: {plot_type}")

    plot_div = pio.to_html(
        fig,
        full_html=False,
        include_plotlyjs="cdn",
        div_id="phd_plot",
        post_script=js,
        config={"responsive": True, "displayModeBar": False},
    )

    html = f"""
    {css}
<div class="plot-wrap plot-16x9">
    {plot_div}
</div>
"""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Wrote {out_path}")
