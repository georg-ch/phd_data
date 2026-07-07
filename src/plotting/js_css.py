import json


def generate_js(trace_map):
    js = f"""
    (function() {{
    var TRACE_MAP = {json.dumps(trace_map)};
    var facultyIdx = TRACE_MAP.faculty_trace_indices;
    var instIdxByFac = TRACE_MAP.institute_trace_indices_by_faculty;
    var nTraces = TRACE_MAP.n_traces;
    var btnIdxByFac = TRACE_MAP.button_index_by_faculty || {{}};

    var gd = document.getElementById("phd_plot");
    if (!gd) return;

    function showInstitutes(facultyName) {{
        var idx = instIdxByFac[facultyName];
        if (!idx) return;

        var vis = new Array(nTraces).fill("legendonly");
        var showleg = new Array(nTraces).fill(false);

            // --- Optional reference traces (line plot)
        var yearTotalIdx = TRACE_MAP.year_total_trace_idx;
        if (yearTotalIdx !== undefined && yearTotalIdx !== null) {{
            vis[yearTotalIdx] = true;
            showleg[yearTotalIdx] = true;
        }}

        var facTotalIdxByFac = TRACE_MAP.fac_total_trace_idx_by_faculty || {{}};
        var facTotalIdx = facTotalIdxByFac[facultyName];
        if (facTotalIdx !== undefined && facTotalIdx !== null) {{
            vis[facTotalIdx] = true;
            showleg[facTotalIdx] = true;
        }}

        idx.forEach(function(i) {{
        vis[i] = true;
        showleg[i] = true;
        }});

        Plotly.restyle(gd, {{ visible: vis, showlegend: showleg }});

        Plotly.relayout(gd, {{
    'legend.traceorder': 'normal'
    }});

    var activeIdx = btnIdxByFac[facultyName];
    if (activeIdx !== undefined) {{
    var activeUpdate = {{'updatemenus[0].active': activeIdx}};
    if (gd.layout.updatemenus && gd.layout.updatemenus.length > 1) {{
        activeUpdate['updatemenus[1].active'] = activeIdx;
    }}
    Plotly.relayout(gd, activeUpdate);
    }}

        Plotly.relayout(gd, {{
        title: ""
        }});
    }}

    gd.on('plotly_click', function(ev) {{
        if (window.matchMedia && window.matchMedia("(max-width: 449px)").matches) return;
        if (!ev || !ev.points || !ev.points.length) return;
        var pt = ev.points[0];
        if (facultyIdx.indexOf(pt.curveNumber) !== -1) {{
        showInstitutes(pt.data.name);
        }}
    }});
    }})();
    """
    return js


js_resize_barplot = r"""
(function () {
  //var legend_scale = 25;
  var legend_scale_desktop = 27;
  var legend_scale_mobile = 20;

  var MOBILE_BREAKPOINT = 450;

  var gd = document.getElementById("phd_plot");
  if (!gd) return;

  var RATIO = 16 / 9;

  function clamp(x, lo, hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
  }

  function getWrapper() {
    var wrap = null;
    if (gd.closest) wrap = gd.closest(".plot-wrap");
    return wrap || gd.parentElement || gd;
  }

  function ensureTwoMenus(gd, fontSize) {
    if (!gd || !gd.layout) return false;
    if (!gd.layout.updatemenus || !gd.layout.updatemenus.length) return false;

    // Already installed?
    if (gd.layout.updatemenus.length >= 2) {
      Plotly.relayout(gd, {
        "updatemenus[0].font.size": fontSize,
        "updatemenus[1].font.size": fontSize
      });
      return true;
    }

    // Clone original buttons menu
    var menuButtons = JSON.parse(JSON.stringify(gd.layout.updatemenus[0]));
    menuButtons.type = "buttons";
    menuButtons.visible = true;
    menuButtons.direction = menuButtons.direction || "right";
    menuButtons.font = menuButtons.font || {};
    menuButtons.font.size = fontSize;

    // Create dropdown clone
    var menuDropdown = JSON.parse(JSON.stringify(menuButtons));
    menuDropdown.type = "dropdown";
    menuDropdown.direction = "down";
    menuDropdown.showactive = true;
    menuDropdown.visible = false;

    // Make dropdown readable on transparent background
    menuDropdown.bgcolor = "rgba(255,255,255,0.95)";
    menuDropdown.bordercolor = "rgba(0,0,0,0.20)";
    menuDropdown.borderwidth = 1;

    // Slightly tighter padding for dropdown
    menuDropdown.pad = {t: 2, r: 2, b: 2, l: 2};

    Plotly.relayout(gd, {
      updatemenus: [menuButtons, menuDropdown]
    });

    return true;
  }
  gd.on("plotly_afterplot", function () {
  applyMobileLegendTweaks();
});


  function toggleMenus(gd, isMobile, fontSize) {
    if (!ensureTwoMenus(gd, fontSize)) return;

    var mode = isMobile ? "mobile" : "desktop";
    if (gd._menuMode === mode) return;
    gd._menuMode = mode;

    // Preserve active selection
    var active = 0;
    if (gd.layout.updatemenus && gd.layout.updatemenus[0] &&
        typeof gd.layout.updatemenus[0].active === "number") {
      active = gd.layout.updatemenus[0].active;
    }

    Plotly.relayout(gd, {
      "updatemenus[0].visible": isMobile ? false : true,
      "updatemenus[1].visible": isMobile ? true : false,
      "updatemenus[0].active": active,
      "updatemenus[1].active": active,
      "updatemenus[0].font.size": fontSize,
      "updatemenus[1].font.size": fontSize
    });
  }

function tightenLegendRowSpacing(gd, factor) {
  try {
    var traces = gd.querySelectorAll(".legend g.traces");
    if (!traces || !traces.length) return;

    traces.forEach(function (g) {
      var tr = g.getAttribute("transform");
      if (!tr) return;

      var m = tr.match(/translate\(\s*([-\d.]+)[,\s]+([-\d.]+)\s*\)/);
      if (!m) return;

      var x = parseFloat(m[1]);
      var y = parseFloat(m[2]);

      if (g.dataset.origX === undefined) g.dataset.origX = String(x);
      if (g.dataset.origY === undefined) g.dataset.origY = String(y);

      var ox = parseFloat(g.dataset.origX);
      var oy = parseFloat(g.dataset.origY);

      g.setAttribute("transform", "translate(" + ox + "," + (oy * factor) + ")");
    });
  } catch (e) {}
}

    function applyMobileLegendTweaks() {
  var w = getWrapper().clientWidth;
  if (!w) return;

  var isMobile = (w < MOBILE_BREAKPOINT);
  if (!isMobile) return;

  // wait until Plotly has actually laid out the legend DOM
  requestAnimationFrame(function () {
    requestAnimationFrame(function () {
      tightenLegendRowSpacing(gd, 0.7);
    });
  });
}


  function resizeToWrapper() {
    var wrap = getWrapper();
    var w = wrap.clientWidth;
    if (!w) return;

    var LEGEND_WRAP_BREAKPOINT = 545;
    var isNarrowLegend = (w <= LEGEND_WRAP_BREAKPOINT);

    var base = clamp(w / 55, 7, 18);
    var legend = clamp(w / 80, 8, 16);
    var buttonsFont = clamp(w / 70, 8, 16);

    var isMobile = (w < MOBILE_BREAKPOINT);
    var legend_scale = isMobile ? legend_scale_mobile : legend_scale_desktop;

    var marginBottom = Math.round(legend * legend_scale);

    var marginTop = Math.round(buttonsFont * 5.0);

    var h = Math.round(w / RATIO) + marginTop + marginBottom;

    var extra = {};
    if (isMobile) {
      extra = {
        "margin.l": 18,
        "margin.r": 6,
        "yaxis.ticklabelposition": "inside",
        "yaxis.ticks": "",
        "yaxis.ticklen": 0,
        "yaxis.title.standoff": 2
      };
    } else {
      // restore defaults when leaving mobile
      extra = {
        "margin.l": null,
        "margin.r": null,
        "yaxis.ticklabelposition": null,
        "yaxis.ticks": null,
        "yaxis.ticklen": null,
        "yaxis.title.standoff": null
      };
    }
    var legendExtra = {};
    if (isNarrowLegend) {
    legendExtra = {
        // Force stable 2-column layout (no weird wrapping)
        "legend.entrywidthmode": "fraction",
        "legend.entrywidth": 1.0,

        // Negative indentation can produce bad layout in narrow widths
        "legend.indentation": 0,

        // Optional: a bit of breathing room between groups
        "legend.tracegroupgap": 6
    };
    } else {
    legendExtra = {
        "legend.entrywidthmode": null,
        "legend.entrywidth": null,
        "legend.indentation": null,   // restore your desktop preference
        "legend.tracegroupgap": 0
    };
    }

    Plotly.relayout(gd, Object.assign({
      width: w,
      height: h,
      "font.size": base,
      "legend.font.size": legend,
      "legend.title.font.size": legend,
      "margin.b": marginBottom,
      "margin.t": marginTop,
      "hovermode": isMobile ? false : "closest",
      "xaxis.automargin": true,
      "yaxis.automargin": true
    }, extra, legendExtra));


    Plotly.Plots.resize(gd);


setTimeout(function () {
  toggleMenus(gd, isMobile, buttonsFont);

applyMobileLegendTweaks();
}, 0);
  }

  if (window.ResizeObserver) {
    var ro = new ResizeObserver(function () { resizeToWrapper(); });
    ro.observe(getWrapper());
  }

  window.addEventListener("load", resizeToWrapper);
  window.addEventListener("orientationchange", resizeToWrapper);

  setTimeout(resizeToWrapper, 50);
  setTimeout(resizeToWrapper, 250);
})();
"""


js_resize_lineplot = r"""
(function () {
  //var legend_scale = 25;
  var legend_scale_desktop = 27;
  var legend_scale_mobile = 20;

  var MOBILE_BREAKPOINT = 450;

  var gd = document.getElementById("phd_plot");
  if (!gd) return;

  var RATIO = 16 / 9;

  function clamp(x, lo, hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
  }

  function getWrapper() {
    var wrap = null;
    if (gd.closest) wrap = gd.closest(".plot-wrap");
    return wrap || gd.parentElement || gd;
  }

  function ensureTwoMenus(gd, fontSize) {
    if (!gd || !gd.layout) return false;
    if (!gd.layout.updatemenus || !gd.layout.updatemenus.length) return false;

    // Already installed?
    if (gd.layout.updatemenus.length >= 2) {
      Plotly.relayout(gd, {
        "updatemenus[0].font.size": fontSize,
        "updatemenus[1].font.size": fontSize
      });
      return true;
    }

    // Clone original buttons menu
    var menuButtons = JSON.parse(JSON.stringify(gd.layout.updatemenus[0]));
    menuButtons.type = "buttons";
    menuButtons.visible = true;
    menuButtons.direction = menuButtons.direction || "right";
    menuButtons.font = menuButtons.font || {};
    menuButtons.font.size = fontSize;

    // Create dropdown clone
    var menuDropdown = JSON.parse(JSON.stringify(menuButtons));
    menuDropdown.type = "dropdown";
    menuDropdown.direction = "down";
    menuDropdown.showactive = true;
    menuDropdown.visible = false;

    // Make dropdown readable on transparent background
    menuDropdown.bgcolor = "rgba(255,255,255,0.95)";
    menuDropdown.bordercolor = "rgba(0,0,0,0.20)";
    menuDropdown.borderwidth = 1;

    // Slightly tighter padding for dropdown
    menuDropdown.pad = {t: 2, r: 2, b: 2, l: 2};

    Plotly.relayout(gd, {
      updatemenus: [menuButtons, menuDropdown]
    });

    return true;
  }
  gd.on("plotly_afterplot", function () {
  applyMobileLegendTweaks();
});


  function toggleMenus(gd, isMobile, fontSize) {
    if (!ensureTwoMenus(gd, fontSize)) return;

    var mode = isMobile ? "mobile" : "desktop";
    if (gd._menuMode === mode) return;
    gd._menuMode = mode;

    // Preserve active selection
    var active = 0;
    if (gd.layout.updatemenus && gd.layout.updatemenus[0] &&
        typeof gd.layout.updatemenus[0].active === "number") {
      active = gd.layout.updatemenus[0].active;
    }

    Plotly.relayout(gd, {
      "updatemenus[0].visible": isMobile ? false : true,
      "updatemenus[1].visible": isMobile ? true : false,
      "updatemenus[0].active": active,
      "updatemenus[1].active": active,
      "updatemenus[0].font.size": fontSize,
      "updatemenus[1].font.size": fontSize
    });
  }

function tightenLegendRowSpacing(gd, factor) {
  try {
    var traces = gd.querySelectorAll(".legend g.traces");
    if (!traces || !traces.length) return;

    traces.forEach(function (g) {
      var tr = g.getAttribute("transform");
      if (!tr) return;

      var m = tr.match(/translate\(\s*([-\d.]+)[,\s]+([-\d.]+)\s*\)/);
      if (!m) return;

      var x = parseFloat(m[1]);
      var y = parseFloat(m[2]);

      if (g.dataset.origX === undefined) g.dataset.origX = String(x);
      if (g.dataset.origY === undefined) g.dataset.origY = String(y);

      var ox = parseFloat(g.dataset.origX);
      var oy = parseFloat(g.dataset.origY);

      g.setAttribute("transform", "translate(" + ox + "," + (oy * factor) + ")");
    });
  } catch (e) {}
}

    function applyMobileLegendTweaks() {
  var w = getWrapper().clientWidth;
  if (!w) return;

  var isMobile = (w < MOBILE_BREAKPOINT);
  if (!isMobile) return;

  // wait until Plotly has actually laid out the legend DOM
  requestAnimationFrame(function () {
    requestAnimationFrame(function () {
      tightenLegendRowSpacing(gd, 0.7);
    });
  });
}


  function resizeToWrapper() {
    var wrap = getWrapper();
    var w = wrap.clientWidth;
    if (!w) return;

    var LEGEND_WRAP_BREAKPOINT = 545;
    var isNarrowLegend = (w <= LEGEND_WRAP_BREAKPOINT);

    var base = clamp(w / 55, 7, 18);
    var legend = clamp(w / 80, 8, 16);
    var buttonsFont = clamp(w / 70, 8, 16);

    var isMobile = (w < MOBILE_BREAKPOINT);
    var legend_scale = isMobile ? legend_scale_mobile : legend_scale_desktop;

    var marginBottom = Math.round(legend * legend_scale);

    var marginTop = Math.round(buttonsFont * 6.5);

    var h = Math.round(w / RATIO) + marginTop + marginBottom;

    var extra = {};
    if (isMobile) {
      extra = {
        "margin.l": 18,
        "margin.r": 6,
        "yaxis.ticklabelposition": "inside",
        "yaxis.ticks": "",
        "yaxis.ticklen": 0,
        "yaxis.title.standoff": 2
      };
    } else {
      // restore defaults when leaving mobile
      extra = {
        "margin.l": null,
        "margin.r": null,
        "yaxis.ticklabelposition": null,
        "yaxis.ticks": null,
        "yaxis.ticklen": null,
        "yaxis.title.standoff": null
      };
    }
    var legendExtra = {};
    if (isNarrowLegend) {
    legendExtra = {
        // Force stable 2-column layout (no weird wrapping)
        "legend.entrywidthmode": "fraction",
        "legend.entrywidth": 1.0,

        // Negative indentation can produce bad layout in narrow widths
        "legend.indentation": 0,

        // Optional: a bit of breathing room between groups
        "legend.tracegroupgap": 6
    };
    } else {
    legendExtra = {
        "legend.entrywidthmode": null,
        "legend.entrywidth": null,
        "legend.indentation": null,   // restore your desktop preference
        "legend.tracegroupgap": 0
    };
    }

function applyMobileStyling(isMobile) {
  cacheOriginalStyling();

  var lw = 1.0 - 0.35 * (+isMobile); // 0.65 on mobile else 1.0
  var ms = 1.0 - 0.50 * (+isMobile); // 0.50 on mobile else 1.0

  var lineWidths = gd._origLineWidth.map(function (w0) { return w0 * lw; });
  var markerSizes = gd._origMarkerSize.map(function (s0) { return s0 * ms; });

  Plotly.restyle(gd, {
    "line.width": lineWidths,
    "marker.size": markerSizes
  });
}

function cacheOriginalStyling() {
  if (gd._origStyleCached) return;
  gd._origStyleCached = true;

  gd._origLineWidth = gd.data.map(function (tr) {
    var w0 = ((tr || {}).line || {}).width;
    w0 = (w0 === undefined || w0 === null) ? 2 : w0;
    return w0;
  });

  gd._origMarkerSize = gd.data.map(function (tr) {
    var s0 = ((tr || {}).marker || {}).size;
    s0 = (s0 === undefined || s0 === null) ? 6 : s0;
    return s0;
  });
}

    Plotly.relayout(gd, Object.assign({
      width: w,
      height: h,
      "font.size": base,
      "legend.font.size": legend,
      "legend.title.font.size": legend,
      "margin.b": marginBottom,
      "margin.t": marginTop,
      "hovermode": isMobile ? false : "closest",
      "xaxis.automargin": true,
      "yaxis.automargin": true
    }, extra, legendExtra));


    Plotly.Plots.resize(gd);


setTimeout(function () {
  toggleMenus(gd, isMobile, buttonsFont);
  applyMobileStyling(isMobile);
  applyMobileLegendTweaks();
}, 0);
  }

  if (window.ResizeObserver) {
    var ro = new ResizeObserver(function () { resizeToWrapper(); });
    ro.observe(getWrapper());
  }

  window.addEventListener("load", resizeToWrapper);
  window.addEventListener("orientationchange", resizeToWrapper);

  setTimeout(resizeToWrapper, 50);
  setTimeout(resizeToWrapper, 250);
})();
"""


js_resize_sankey = r"""
(function () {
  var MOBILE_BREAKPOINT = 450;

  // Font sizing controls. Lower scale values make fonts grow faster.
  var font_scale_desktop = 80;
  var font_scale_mobile = 65;
  var min_font_size_desktop = 7;
  var min_font_size_mobile = 6;
  var max_font_size = 16;

  // Sankey-specific sizing controls.
  var node_scale_width_desktop = 1700;
  var node_scale_width_mobile = 1200;
  var min_node_scale_desktop = 0.35;
  var min_node_scale_mobile = 0.22;
  var max_node_scale = 1.1;
  var min_node_thickness = 12;
  var min_node_pad = 8;
  var annotation_font_factor = 1.25;
  var button_font_factor = 1.0;

  var gd = document.getElementById("phd_plot");
  if (!gd) return;

  function clamp(x, lo, hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
  }

  function getWrapper() {
    var wrap = null;
    if (gd.closest) wrap = gd.closest(".plot-wrap");
    return wrap || gd.parentElement || gd;
  }

  function cacheOriginalSizing() {
    if (gd._sankeyResizeCache) return gd._sankeyResizeCache;

    var layout = gd.layout || {};
    var annotations = layout.annotations || [];
    var menus = layout.updatemenus || [];

    gd._sankeyResizeCache = {
      baseFontSize: (((layout || {}).font || {}).size) || 12,
      annotationFontSizes: annotations.map(function (ann) {
        var font = ann && ann.font ? ann.font : {};
        return font.size || 15;
      }),
      menuFontSizes: menus.map(function (menu) {
        var font = menu && menu.font ? menu.font : {};
        return font.size || 12;
      }),
      traceNodeSizes: (gd.data || []).map(function (tr) {
        var node = (tr || {}).node || {};
        return {
          thickness: (node.thickness === undefined || node.thickness === null) ? 20 : node.thickness,
          pad: (node.pad === undefined || node.pad === null) ? 15 : node.pad
        };
      })
    };

    return gd._sankeyResizeCache;
  }

  function updateNodeSizing(nodeScale) {
    var cache = cacheOriginalSizing();

    cache.traceNodeSizes.forEach(function (nodeSize, idx) {
      if (!gd.data || !gd.data[idx] || gd.data[idx].type !== "sankey") return;

      var nextThickness = Math.max(
        min_node_thickness,
        Math.round(nodeSize.thickness * nodeScale)
      );
      var nextPad = Math.max(
        min_node_pad,
        Math.round(nodeSize.pad * Math.max(0.7, nodeScale))
      );

      Plotly.restyle(gd, {
        "node.thickness": [nextThickness],
        "node.pad": [nextPad]
      }, [idx]);
    });
  }

  function resizeToWrapper() {
    var wrap = getWrapper();
    var w = wrap.clientWidth;
    if (!w) return;

    var isMobile = (w < MOBILE_BREAKPOINT);
    var fontScale = isMobile ? font_scale_mobile : font_scale_desktop;
    var minFontSize = isMobile ? min_font_size_mobile : min_font_size_desktop;
    var nodeScaleWidth = isMobile ? node_scale_width_mobile : node_scale_width_desktop;
    var minNodeScale = isMobile ? min_node_scale_mobile : min_node_scale_desktop;

    var baseFont = clamp(w / fontScale, minFontSize, max_font_size);
    var annotationFont = Math.max(minFontSize, Math.round(baseFont * annotation_font_factor));
    var buttonFont = Math.max(minFontSize, Math.round(baseFont * button_font_factor));
    var nodeScale = clamp(w / nodeScaleWidth, minNodeScale, max_node_scale);

    var cache = cacheOriginalSizing();
    var hasMenus = !!((gd.layout || {}).updatemenus || []).length;

    var marginTop = hasMenus
      ? Math.round(buttonFont * 8.0)
      : Math.round(annotationFont * 3.2);
    var marginBottom = Math.round(baseFont * 1.8);
    var contentHeight = isMobile
      ? Math.round(clamp(w * 1.10, 440, 1020))
      : Math.round(clamp(w / 1.40, 440, 940));
    var resizeKey = [
      w,
      baseFont,
      annotationFont,
      buttonFont,
      nodeScale,
      contentHeight,
      marginTop,
      marginBottom,
      isMobile ? 1 : 0
    ].join("|");

    if (gd._lastSankeyResizeKey === resizeKey) return;
    gd._lastSankeyResizeKey = resizeKey;

    var relayoutUpdate = {
      width: w,
      height: contentHeight + marginTop + marginBottom,
      "font.size": baseFont,
      "margin.t": marginTop,
      "margin.b": marginBottom,
      "margin.l": isMobile ? 6 : 10,
      "margin.r": isMobile ? 6 : 10
    };

    cache.annotationFontSizes.forEach(function (size, idx) {
      relayoutUpdate["annotations[" + idx + "].font.size"] = Math.max(
        minFontSize,
        Math.round(size * (annotationFont / Math.max(1, 15)))
      );
    });

    cache.menuFontSizes.forEach(function (size, idx) {
      relayoutUpdate["updatemenus[" + idx + "].font.size"] = Math.max(
        minFontSize,
        Math.round(size * (buttonFont / Math.max(1, cache.baseFontSize)))
      );
    });

    Plotly.relayout(gd, relayoutUpdate).then(function () {
      return updateNodeSizing(nodeScale);
    }).then(function () {
      Plotly.Plots.resize(gd);
    });
  }

  function queueResize() {
    if (gd._sankeyResizeQueued) return;
    gd._sankeyResizeQueued = true;

    requestAnimationFrame(function () {
      gd._sankeyResizeQueued = false;
      resizeToWrapper();
    });
  }

  if (window.ResizeObserver) {
    var ro = new ResizeObserver(function () { queueResize(); });
    ro.observe(getWrapper());
  }

  gd.on("plotly_afterplot", queueResize);
  window.addEventListener("load", queueResize);
  window.addEventListener("orientationchange", queueResize);

  setTimeout(queueResize, 50);
  setTimeout(queueResize, 250);
})();
"""


js_resize_violin = r"""
(function () {
  var gd = document.getElementById("phd_plot");
  if (!gd) return;

  var RATIO = 16 / 9;
  var MOBILE_BREAKPOINT = 540;

  function clamp(x, lo, hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
  }

  function getWrapper() {
    var wrap = null;
    if (gd.closest) wrap = gd.closest(".plot-wrap");
    return wrap || gd.parentElement || gd;
  }

  function resizeToWrapper() {
    var wrap = getWrapper();
    var w = wrap.clientWidth;
    if (!w) return;

    var isMobile = (w < MOBILE_BREAKPOINT);
    var currentTickangle = null;
    if (gd.layout && gd.layout.xaxis && typeof gd.layout.xaxis.tickangle === "number") {
      currentTickangle = gd.layout.xaxis.tickangle;
    }
    var base = clamp(w / 52, 9, 18);
    var tick = clamp(w / 65, 8, 15);
    var title = clamp(w / 42, 10, 20);
    var hover = clamp(w / 70, 9, 14);

    var marginBottom = isMobile ? Math.round(base * 6.5) : Math.round(base * 4.8);
    var marginLeft = isMobile ? 42 : 60;
    var marginRight = isMobile ? 10 : 24;
    var marginTop = 28;
    var h = Math.round(w / RATIO) + marginBottom + marginTop;

    Plotly.relayout(gd, {
      width: w,
      height: h,
      "font.size": base,
      "hoverlabel.font.size": hover,
      "margin.l": marginLeft,
      "margin.r": marginRight,
      "margin.t": marginTop,
      "margin.b": marginBottom,
      "xaxis.tickfont.size": tick,
      "yaxis.tickfont.size": tick,
      "xaxis.title.font.size": title,
      "yaxis.title.font.size": title,
      "xaxis.tickangle": isMobile
        ? (currentTickangle !== null ? currentTickangle : -25)
        : (currentTickangle !== null ? currentTickangle : 0),
      "xaxis.automargin": true,
      "yaxis.automargin": true
    });

    Plotly.Plots.resize(gd);
  }

  if (window.ResizeObserver) {
    var ro = new ResizeObserver(function () { resizeToWrapper(); });
    ro.observe(getWrapper());
  }

  window.addEventListener("load", resizeToWrapper);
  window.addEventListener("orientationchange", resizeToWrapper);

  setTimeout(resizeToWrapper, 50);
  setTimeout(resizeToWrapper, 250);
})();
"""


js_resize_forest = r"""
(function () {
  var gd = document.getElementById("phd_plot");
  if (!gd) return;

  function clamp(x, lo, hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
  }

  function getWrapper() {
    var wrap = null;
    if (gd.closest) wrap = gd.closest(".plot-wrap");
    return wrap || gd.parentElement || gd;
  }

  function cacheOriginalMarkerSizes() {
    if (gd._forestOrigMarkerSize) return gd._forestOrigMarkerSize;

    gd._forestOrigMarkerSize = (gd.data || []).map(function (tr) {
      var marker = tr && tr.marker ? tr.marker : {};
      var size = marker.size;
      if (size === undefined || size === null) size = 10;
      return size;
    });
    return gd._forestOrigMarkerSize;
  }

  function cacheOriginalErrorThicknesses() {
    if (gd._forestOrigErrorThickness) return gd._forestOrigErrorThickness;

    gd._forestOrigErrorThickness = (gd.data || []).map(function (tr) {
      var errorX = tr && tr.error_x ? tr.error_x : {};
      var thickness = errorX.thickness;
      if (thickness === undefined || thickness === null) thickness = 2;
      return thickness;
    });
    return gd._forestOrigErrorThickness;
  }

  function getRowCount() {
    if (!gd.data || !gd.data.length) return 0;
    var first = gd.data[0] || {};
    if (Array.isArray(first.y)) return first.y.length;
    if (gd.layout && gd.layout.yaxis && Array.isArray(gd.layout.yaxis.categoryarray)) {
      return gd.layout.yaxis.categoryarray.length;
    }
    return 0;
  }

  function resizeToWrapper() {
    var wrap = getWrapper();
    var w = wrap.clientWidth;
    if (!w) return;

    var origMarkerSizes = cacheOriginalMarkerSizes();
    var origErrorThicknesses = cacheOriginalErrorThicknesses();
    var rowCount = getRowCount();

    var base = clamp(w / 72, 8, 14);
    var tick = clamp(w / 82, 7, 13);
    var title = clamp(w / 62, 9, 15);
    var hover = clamp(w / 85, 8, 13);
    var marginLeft = w < 520 ? 96 : 140;
    var marginRight = w < 520 ? 12 : 40;
    var rowPitch = clamp(w / 36, 13, 22);
    var height = Math.max(150, Math.round(rowPitch * Math.max(rowCount, 1) + 44));

    var resizeKey = [
      w,
      base,
      tick,
      title,
      hover,
      marginLeft,
      marginRight,
      rowPitch,
      rowCount
    ].join("|");

    if (gd._forestResizeKey === resizeKey) return;
    gd._forestResizeKey = resizeKey;

    Plotly.relayout(gd, {
      width: w,
      height: height,
      "font.size": base,
      "hoverlabel.font.size": hover,
      "xaxis.title.font.size": title,
      "xaxis.tickfont.size": tick,
      "yaxis.tickfont.size": tick,
      "margin.l": marginLeft,
      "margin.r": marginRight,
      "xaxis.automargin": true,
      "yaxis.automargin": true
    });

    (gd.data || []).forEach(function (tr, idx) {
      var marker = tr && tr.marker ? tr.marker : null;
      if (!marker) return;

      var nextSize = origMarkerSizes[idx];
      if (w < 650) {
        nextSize = Math.max(4, Math.round(nextSize * 0.5));
      }

      if (marker.size !== nextSize) {
        Plotly.restyle(gd, { "marker.size": [nextSize] }, [idx]);
      }
    });

    (gd.data || []).forEach(function (tr, idx) {
      var errorX = tr && tr.error_x ? tr.error_x : null;
      if (!errorX) return;

      var nextThickness = origErrorThicknesses[idx];
      if (w < 700) {
        nextThickness = Math.max(1, Math.round(nextThickness * 0.7));
      }

      if (errorX.thickness !== nextThickness) {
        Plotly.restyle(gd, { "error_x.thickness": [nextThickness] }, [idx]);
      }
    });
  }

  function queueResize() {
    if (gd._forestResizeQueued) return;
    gd._forestResizeQueued = true;

    requestAnimationFrame(function () {
      gd._forestResizeQueued = false;
      resizeToWrapper();
    });
  }

  if (window.ResizeObserver) {
    var ro = new ResizeObserver(function () { queueResize(); });
    ro.observe(getWrapper());
  }

  window.addEventListener("load", queueResize);
  window.addEventListener("orientationchange", queueResize);

  setTimeout(queueResize, 50);
  setTimeout(queueResize, 250);
})();
"""


js_resize_violin_contrast = r"""
(function () {
  var gd = document.getElementById("phd_plot");
  if (!gd) return;

  function clamp(x, lo, hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
  }

  function getWrapper() {
    var wrap = null;
    if (gd.closest) wrap = gd.closest(".plot-wrap");
    return wrap || gd.parentElement || gd;
  }

  function cacheForestTraceSizes() {
    if (gd._contrastResizeCache) return gd._contrastResizeCache;

    var traceSizes = [];
    (gd.data || []).forEach(function (tr, idx) {
      if (!tr || !tr.error_y || !tr.error_y.visible) return;
      var marker = tr.marker || {};
      var size = marker.size;
      if (size === undefined || size === null) size = 7;
      traceSizes.push({ idx: idx, size: size });
    });

    gd._contrastResizeCache = { traceSizes: traceSizes };
    return gd._contrastResizeCache;
  }

  function resizeToWrapper() {
    var wrap = getWrapper();
    var w = wrap.clientWidth;
    if (!w) return;

    var cache = cacheForestTraceSizes();
    var xaxis = gd.layout && gd.layout.xaxis ? gd.layout.xaxis : {};
    var ticktext = xaxis.ticktext;
    var nCategories = Array.isArray(ticktext) ? ticktext.length : 0;
    var manyCategories = nCategories >= 8;

    var base = clamp(w / 52, 9, 18);
    var tick = clamp(w / 65, 8, 15);
    var title = clamp(w / 42, 10, 20);
    var hover = clamp(w / 70, 9, 14);

    if (w < 550 && manyCategories) {
      base = Math.max(7, base - 1);
      tick = Math.max(7, tick - 1);
      title = Math.max(9, title - 1);
      hover = Math.max(8, hover - 1);
    }

    if (w < 450 && manyCategories) {
      base = Math.max(5, base - 2);
      tick = Math.max(5, tick - 2);
      title = Math.max(7, title - 2);
      hover = Math.max(6, hover - 2);
    }

    Plotly.relayout(gd, {
      "font.size": base,
      "hoverlabel.font.size": hover,
      "xaxis.tickfont.size": tick,
      "yaxis.tickfont.size": tick,
      "xaxis.title.font.size": title,
      "yaxis.title.font.size": title
    });

    cache.traceSizes.forEach(function (traceSize) {
      var nextSize = traceSize.size;
      if (w < 650) {
        nextSize = Math.max(4, Math.round(traceSize.size * 0.5));
      }

      var currentSize =
        gd.data &&
        gd.data[traceSize.idx] &&
        gd.data[traceSize.idx].marker &&
        gd.data[traceSize.idx].marker.size;

      if (currentSize !== nextSize) {
        Plotly.restyle(gd, { "marker.size": [nextSize] }, [traceSize.idx]);
      }
    });
  }

  if (window.ResizeObserver) {
    var ro = new ResizeObserver(function () {
      resizeToWrapper();
    });
    ro.observe(getWrapper());
  }

  window.addEventListener("load", resizeToWrapper);
  window.addEventListener("orientationchange", resizeToWrapper);

  setTimeout(resizeToWrapper, 50);
  setTimeout(resizeToWrapper, 250);
})();
"""


css = """
<style>
@media (max-width: 450px) {

  /* Shrink legend symbol group */
  #phd_plot .legend g.legendpoints {
    transform-box: fill-box;
    transform-origin: center;
    transform: scale(0.6);   /* try 0.55–0.7 */
  }

  /* Pull text closer after scaling */
  //#phd_plot .legend g.legendtext {
   // transform: translateX(-14px);
  }

  /* Optional: reduce legend vertical spacing slightly */
  //#phd_plot .legend .traces {
   // transform: translateY(-2px);
  }
}
</style>
"""
