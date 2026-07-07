from pathlib import Path
import tomllib

import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from scripts.paths import CONFIG_DIR, DATA_DIR
from src.data_loading.load_data import load_data
from src.data_processing.regression import (
    compare_current_model_feature_evaluation_grid,
    compare_feature_blocks_grid,
    visualize_feature_effects,
)
from src.plotting.plot_config import load_plot_config
from src.plotting.plotting_data import limit_year_range
from src.plotting.plotting_export import write_html


TARGET_COLS = ("duration_computed", "duration_estimate_clipped_simple")

COMPARISON_BLOCKS = [
    (
        "location_family",
        (
            "institute_name",
            "faculty",
            "lecode",
            "phd_subject",
            "subject_group_stala",
            "subject_group_category_stala",
        ),
        # ("cship_category", "is_male", "pba_grade", "pba_grade_missing", "pba_type"),
        ("cship_category", "is_male"),
    ),
    (
        "context_family",
        ("cship_category", "pba_category", "mobility_category"),
        ("institute_name", "is_male", "pba_grade", "pba_grade_missing", "pba_type"),
    ),
]

METRICS = [
    ("p_adj", "Adjusted p-value", "log", "#E15759"),
    ("max_vif", "Max VIF", "linear", "#4E79A7"),
    ("partial_r2", "Partial R²", "linear", "#59A14F"),
]


def _safe_series_for_plot(s: pd.Series, *, floor: float = 1e-300) -> pd.Series:
    out = pd.to_numeric(s, errors="coerce").copy()
    return out.clip(lower=floor)


def _build_effect_figure(df_by_block: dict[str, pd.DataFrame], target_col: str):
    n_rows = len(METRICS)
    n_cols = len(df_by_block)
    block_names = list(df_by_block.keys())

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        shared_xaxes=False,
        shared_yaxes=False,
        subplot_titles=[
            *(
                block if row_idx == 0 else ""
                for row_idx in range(n_rows)
                for block in block_names
            ),
        ],
        vertical_spacing=0.08,
        horizontal_spacing=0.06,
    )

    for col_idx, (block_name, df) in enumerate(df_by_block.items(), start=1):
        ordered = df.sort_values(
            ["p_adj", "partial_r2", "feature"], ascending=[True, False, True]
        ).copy()
        x = ordered["feature"].astype(str)

        for row_idx, (metric_col, metric_label, scale, color) in enumerate(
            METRICS, start=1
        ):
            y = _safe_series_for_plot(ordered[metric_col])
            trace = go.Bar(
                x=x,
                y=y,
                marker=dict(color=color),
                showlegend=False,
                hovertemplate=(
                    f"Feature: %{{x}}<br>{metric_label}: %{{y:.4g}}<extra></extra>"
                ),
            )
            fig.add_trace(trace, row=row_idx, col=col_idx)

            if metric_col == "p_adj":
                sig = ordered[metric_col] < 0.05
                if sig.any():
                    sig_y = y.where(sig)
                    sig_text = ["*" if v else "" for v in sig]
                    fig.add_trace(
                        go.Scatter(
                            x=x,
                            y=sig_y,
                            mode="text",
                            text=sig_text,
                            textposition="top center",
                            textfont=dict(color="#111111", size=16),
                            hoverinfo="skip",
                            showlegend=False,
                        ),
                        row=row_idx,
                        col=col_idx,
                    )

            if scale == "log":
                fig.update_yaxes(type="log", row=row_idx, col=col_idx)

            fig.update_xaxes(
                tickangle=-35,
                automargin=True,
                showticklabels=row_idx == n_rows,
                title_text="Feature" if row_idx == n_rows else None,
                row=row_idx,
                col=col_idx,
            )
            fig.update_yaxes(
                title_text=metric_label if col_idx == 1 else None,
                showticklabels=col_idx == 1,
                row=row_idx,
                col=col_idx,
            )

    fig.update_layout(
        barmode="group",
        height=340 * n_rows,
        width=max(1200, 420 * n_cols),
        margin=dict(l=60, r=30, t=80, b=80),
    )
    return fig


def main():
    params, _, _ = load_plot_config("regression")
    with open(CONFIG_DIR / "regression.toml", "rb") as f:
        regression_config = tomllib.load(f)

    data = load_data(DATA_DIR / "clean_data.csv", DATA_DIR / "cleaned_data_dtypes.json")
    duration_data_simplified = pd.read_csv(
        DATA_DIR / "duration_predictions_simple_model.csv"
    )
    data = pd.merge(
        data,
        duration_data_simplified[
            ["pagination_nr", "duration_estimate", "duration_estimate_clipped"]
        ].rename(
            columns={
                "duration_estimate": "duration_estimate_simple",
                "duration_estimate_clipped": "duration_estimate_clipped_simple",
            }
        ),
        on="pagination_nr",
        how="left",
    )
    data = limit_year_range(
        data,
        year_col="defense_year",
        start_year=int(params["start_year"]),
        end_year=int(params["end_year"]),
    )

    compare_feature_blocks_grid(
        data,
        target_cols=TARGET_COLS,
        comparison_blocks=COMPARISON_BLOCKS,
        target_transform=regression_config["target_transform"],
        min_count=int(params["small_group_cutoff"]),
        merge_cols=params["merge_cols"],
        cov_type="HC3",
        output_dir=DATA_DIR / "regression_feature_selection",
    )

    compare_current_model_feature_evaluation_grid(
        data,
        target_cols=TARGET_COLS,
        target_transform=regression_config["target_transform"],
        min_count=int(params["small_group_cutoff"]),
        merge_cols=params["merge_cols"],
        cov_type="HC3",
        output_dir=DATA_DIR / "feature_evaluation",
    )

    tables = visualize_feature_effects(
        target_cols=TARGET_COLS,
        comparison_blocks=COMPARISON_BLOCKS,
    )

    out_dir = DATA_DIR / "regression_feature_selection" / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    for target_col, block_tables in tables.items():
        fig = _build_effect_figure(block_tables, target_col)
        out_path = out_dir / f"{target_col}_feature_effects.html"
        write_html(fig, out_path, trace_map=None, plot_type="bar_category")


if __name__ == "__main__":
    main()
