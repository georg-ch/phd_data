from __future__ import annotations

import tomllib
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from scripts.paths import CONFIG_DIR, DATA_DIR
from src.data_loading.load_data import load_data
from src.data_processing.regression import compare_current_model_feature_evaluation_grid
from src.plotting.plot_config import load_plot_config
from src.plotting.plotting_data import limit_year_range
from src.plotting.plotting_export import write_html


TARGET_COLS = ("duration_computed", "duration_estimate_clipped_simple")
TABLE_KIND = ("heldout", "solo")


def main():
    params, _, _ = load_plot_config("regression")
    plot_params, _, _ = load_plot_config("regression_features")
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

    compare_current_model_feature_evaluation_grid(
        data,
        target_cols=TARGET_COLS,
        target_transform=regression_config["target_transform"],
        min_count=int(params["small_group_cutoff"]),
        merge_cols=params["merge_cols"],
        cov_type="HC3",
        output_dir=DATA_DIR / "feature_evaluation",
    )

    tables = {}
    for target_col in TARGET_COLS:
        tables[target_col] = pd.read_csv(
            DATA_DIR / "feature_evaluation" / f"{target_col}__current_model_features.csv"
        )

    out_dir = Path(__file__).resolve().parents[2] / "int_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    for target_col, df in tables.items():
        for kind in TABLE_KIND:
            _write_feature_barplot(
                df,
                target_col=target_col,
                kind=kind,
                x_label=plot_params["x_label"],
                y_label=plot_params["y_label"],
                feature_label_map=plot_params.get("feature_label_names", {}),
                omitted_features=set(plot_params.get("omitted_features", [])),
                out_path=out_dir / f"regression_features__{target_col}__{kind}_feature_effects.html",
            )


def _sig_label(p_adj: float) -> str:
    if pd.isna(p_adj):
        return "n.s."
    if p_adj < 0.001:
        return "***"
    if p_adj < 0.01:
        return "**"
    if p_adj < 0.05:
        return "*"
    return "n.s."


def _feature_display_label(feature: str, feature_label_map: dict) -> str:
    return str(feature_label_map.get(feature, feature))


def _write_feature_barplot(
    df: pd.DataFrame,
    *,
    target_col: str,
    kind: str,
    x_label: str,
    y_label: str,
    feature_label_map: dict,
    omitted_features: set[str],
    out_path,
):
    effect_col = f"{kind}_partial_r2"
    p_col = f"{kind}_p_adj"

    ordered = df.loc[~df["feature"].isin(omitted_features)].copy()
    ordered = ordered.sort_values([p_col, effect_col, "feature"], ascending=[True, False, True]).copy()
    ordered["feature_label"] = ordered["feature"].map(
        lambda feature: _feature_display_label(feature, feature_label_map)
    )
    ordered["sig_label"] = ordered[p_col].apply(_sig_label)
    ordered["hover_label"] = ordered.apply(
        lambda row: (
            f"Feature: {row['feature_label']}<br>"
            f"Raw feature: {row['feature']}<br>"
            f"Partial R²: {row[effect_col]:.4g}<br>"
            f"Adj. p: {row[p_col]:.4g}<br>"
            f"Significance: {row['sig_label']}"
        ),
        axis=1,
    )

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=ordered["feature_label"].astype(str),
            y=ordered[effect_col],
            marker=dict(color="#4E79A7"),
            text=ordered["sig_label"],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{customdata}<extra></extra>",
            customdata=ordered["hover_label"],
        )
    )
    fig.update_layout(
        xaxis_title=x_label,
        yaxis_title=y_label,
        bargap=0.25,
        margin=dict(l=80, r=30, t=25, b=110),
        autosize=True,
    )
    fig.update_xaxes(tickangle=-35, automargin=True)
    fig.update_yaxes(automargin=True)

    write_html(fig, out_path, trace_map=None, plot_type="bar")


if __name__ == "__main__":
    main()
