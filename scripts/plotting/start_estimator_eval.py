from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from scripts.paths import DATA_DIR
from src.data_loading.load_data import load_data
from src.data_processing.duration_estimation import (
    engineer_features,
    engineer_features_startdate,
    prepare_data_for_prediction,
    prepare_data_raw,
    postprocess_predictions,
    postprocess_start,
    train_catboost,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from aquarel import load_theme
except ModuleNotFoundError:

    class _NoTheme:
        def apply_transforms(self):
            return None

    def load_theme(_name):
        return _NoTheme()


RMSE_BIN_EDGES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 15, 20]
N_RUNS = 30
TEST_SIZE = 0.5
RANDOM_STATE = 42

DISPLAY_LABELS = {
    "start_anchor": "start / anchor",
    "start_defense": "start / defense",
    "duration_direct": "direct duration",
    "duration_simple_temporal": "direct duration / temporal only",
}

MODEL_ORDER = {
    "start_anchor": 0,
    "start_defense": 1,
    "duration_direct": 2,
    "duration_simple_temporal": 3,
}


def _bin_labels(bin_edges) -> list[str]:
    return [f"{bin_edges[i]:g}-{bin_edges[i + 1]:g}" for i in range(len(bin_edges) - 1)]


def _load_params(fname: str) -> dict:
    with open(DATA_DIR / fname, "r") as f:
        return json.load(f)


def _train_test_split_df(df: pd.DataFrame, *, test_size: float, random_state: int):
    rng = np.random.default_rng(random_state)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    test_n = int(round(len(df) * test_size))
    test_idx = idx[:test_n]
    train_idx = idx[test_n:]
    return df.iloc[train_idx].copy(), df.iloc[test_idx].copy()


def _annotate(df: pd.DataFrame, model_label: str) -> pd.DataFrame:
    df = df.copy()
    df["model_label"] = model_label
    df["display_label"] = DISPLAY_LABELS[model_label]
    df["model_order"] = MODEL_ORDER[model_label]
    return df


def _predicted_start_val(df: pd.DataFrame, year_col: str, month_col: str) -> pd.Series:
    return df[year_col] + (df[month_col] - 1) / 12 - 2000


def _start_eval_from_splits(feature_set: str, n_runs=N_RUNS) -> pd.DataFrame:
    data = load_data(DATA_DIR / "clean_data.csv", DATA_DIR / "cleaned_data_dtypes.json")
    data, feature_col_info = engineer_features_startdate(data, feature_set=feature_set)
    target_col = "start_val"
    labeled_data, cat_features = prepare_data_raw(
        data, target_col, feature_col_info, dropna=("d_pba", "d_acceptance")
    )
    params = _load_params(f"best_params_start_{feature_set}.json")

    outputs = []
    for run in range(n_runs):
        train_df, test_df = _train_test_split_df(
            labeled_data, test_size=TEST_SIZE, random_state=RANDOM_STATE + run
        )
        test_full = data.loc[test_df.index].copy()
        model = train_catboost(
            train_df,
            target_col,
            cat_features,
            params,
            loss_function="RMSEWithUncertainty",
        )
        X_test = prepare_data_for_prediction(test_df, feature_col_info, target_col)
        pred_mean_var = model.predict(X_test, prediction_type="RMSEWithUncertainty")
        fold_out = postprocess_start(
            test_full,
            predicted_start_unc=pred_mean_var[:, 0],
            predicted_start_var=pred_mean_var[:, 1],
            carry_cols=list(test_full.columns),
            naive_estimator_for_small_gap=(
                "duration_estimate_s",
                "duration_estimate_clipped_s",
            ),
        )
        fold_out["run"] = run
        fold_out["feature_set"] = feature_set
        fold_out["gap_size"] = fold_out["acceptance_val"] - fold_out["pba_val"]
        fold_out["pred_start_val"] = _predicted_start_val(
            fold_out, "start_year_est", "start_month_est"
        )
        fold_out["pred_start_val_clipped"] = _predicted_start_val(
            fold_out, "start_year_est_clipped", "start_month_est_clipped"
        )
        fold_out["midpoint_start_val"] = (
            fold_out["pba_val"] + fold_out["acceptance_val"]
        ) / 2
        fold_out["start_error_raw"] = fold_out["pred_start_val"] - fold_out["start_val"]
        fold_out["start_error_clipped"] = (
            fold_out["pred_start_val_clipped"] - fold_out["start_val"]
        )
        fold_out["start_error_midpoint"] = (
            fold_out["midpoint_start_val"] - fold_out["start_val"]
        )
        fold_out["duration_error_raw"] = (
            fold_out["duration_estimate_s"] - fold_out["duration_computed"]
        )
        fold_out["duration_error_clipped"] = (
            fold_out["duration_estimate_clipped_s"] - fold_out["duration_computed"]
        )
        outputs.append(_annotate(fold_out, f"start_{feature_set}"))

    return pd.concat(outputs, ignore_index=True)


def _duration_eval_from_splits(n_runs=N_RUNS, *, simplify: bool = False) -> pd.DataFrame:
    data = load_data(DATA_DIR / "clean_data.csv", DATA_DIR / "cleaned_data_dtypes.json")
    data, feature_col_info = engineer_features(data, simplify=simplify)
    target_col = "duration_computed"
    labeled_data, cat_features = prepare_data_raw(
        data, target_col, feature_col_info, dropna=("d_pba", "d_acceptance")
    )
    params = _load_params(
        "best_params_RMSEWithUncertainty_simple.json"
        if simplify
        else "best_params_RMSEWithUncertainty.json"
    )

    outputs = []
    for run in range(n_runs):
        train_df, test_df = _train_test_split_df(
            labeled_data, test_size=TEST_SIZE, random_state=RANDOM_STATE + run
        )
        test_full = data.loc[test_df.index].copy()
        model = train_catboost(
            train_df,
            target_col,
            cat_features,
            params,
            loss_function="RMSEWithUncertainty",
        )
        X_test = prepare_data_for_prediction(test_df, feature_col_info, target_col)
        pred_mean_var = model.predict(X_test, prediction_type="RMSEWithUncertainty")
        fold_out = postprocess_predictions(
            test_full,
            predicted_duration_unc=pred_mean_var[:, 0],
            predicted_duration_var=pred_mean_var[:, 1],
            carry_cols=list(test_full.columns),
            naive_estimator_for_small_gap=("duration_estimate", "duration_estimate_clipped"),
        )
        fold_out["run"] = run
        fold_out["feature_set"] = "temporal_only" if simplify else "direct"
        fold_out["gap_size"] = fold_out["d_pba"] - fold_out["d_acceptance"]
        fold_out["duration_error_raw"] = (
            fold_out["duration_estimate"] - fold_out["duration_computed"]
        )
        fold_out["duration_error_clipped"] = (
            fold_out["duration_estimate_clipped"] - fold_out["duration_computed"]
        )
        outputs.append(
            _annotate(
                fold_out,
                "duration_simple_temporal" if simplify else "duration_direct",
            )
        )

    return pd.concat(outputs, ignore_index=True)


def _bin_stats(df: pd.DataFrame, bin_col: str, error_col: str, bin_edges=RMSE_BIN_EDGES) -> pd.DataFrame:
    rows = []
    valid = df[bin_col].notna() & df[error_col].notna()
    for low, high in zip(bin_edges[:-1], bin_edges[1:]):
        if high == bin_edges[-1]:
            mask = valid & (df[bin_col] >= low) & (df[bin_col] <= high)
        else:
            mask = valid & (df[bin_col] >= low) & (df[bin_col] < high)
        sub = df.loc[mask]
        rows.append(
            {
                "model_label": df["model_label"].iloc[0],
                "display_label": df["display_label"].iloc[0],
                "model_order": int(df["model_order"].iloc[0]),
                "bin_low": low,
                "bin_high": high,
                "bin_label": f"{low:g}-{high:g}",
                "count": int(len(sub)),
                "rmse": float(np.sqrt(np.mean(sub[error_col] ** 2))) if len(sub) else float("nan"),
                "mae": float(np.mean(np.abs(sub[error_col]))) if len(sub) else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def _plot_legacy_comparison(
    stats: pd.DataFrame,
    out_file: Path,
    *,
    xlabel: str,
    metric_col: str,
    theme,
    title: str | None = None,
) -> None:
    bins = _bin_labels(RMSE_BIN_EDGES)
    ref_model = stats.sort_values("model_order").iloc[0]["model_label"]
    counts = (
        stats.loc[stats["model_label"] == ref_model]
        .set_index("bin_label")["count"]
        .reindex(bins)
        .fillna(0)
    )
    if metric_col == "fraction":
        counts_scaled = counts / counts.sum() * 10 + 1 if counts.sum() else counts + 1
    else:
        counts_scaled = counts / counts.sum() * 10 if counts.sum() else counts

    fig, ax = plt.subplots(figsize=(10, 6))
    avg_guess_err = np.array(RMSE_BIN_EDGES)[1:] / 2
    for _, gdf in stats.sort_values(["model_order", "bin_low"]).groupby("display_label", sort=False):
        series = gdf.set_index("bin_label")["rmse"].reindex(bins).to_numpy()
        if metric_col == "fraction":
            metric = 1 / (series / avg_guess_err)
        else:
            metric = series
        ax.plot(bins, metric, label=gdf["display_label"].iloc[0], marker="o")
    ax.bar(bins, counts_scaled, alpha=0.3, label="Count")
    ax.set_xlabel(xlabel)
    if metric_col == "fraction":
        ax.set_ylabel("fraction")
    else:
        ax.set_ylabel("rmse")
    if metric_col == "fraction":
        ax.set_ylim(1, 10)
    else:
        ax.set_ylim(0, 10)
    if title is not None:
        ax.set_title(title)
    ax.legend()
    theme.apply_transforms()
    fig.tight_layout()
    fig.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    theme = load_theme("scientific")
    plot_dir = Path(__file__).resolve().parents[2] / "eval_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    start_anchor = _start_eval_from_splits("anchor")
    start_defense = _start_eval_from_splits("defense")
    duration_direct = _duration_eval_from_splits()
    duration_simple_temporal = _duration_eval_from_splits(simplify=True)

    start_df = pd.concat([start_anchor, start_defense], ignore_index=True)
    duration_df = pd.concat(
        [start_anchor, start_defense, duration_direct, duration_simple_temporal],
        ignore_index=True,
    )

    start_gap_stats = pd.concat(
        [_bin_stats(gdf, "gap_size", "start_error_clipped") for _, gdf in start_df.groupby("model_label", sort=False)],
        ignore_index=True,
    )
    duration_gap_stats = pd.concat(
        [_bin_stats(gdf, "gap_size", "duration_error_clipped") for _, gdf in duration_df.groupby("model_label", sort=False)],
        ignore_index=True,
    )

    _plot_legacy_comparison(
        duration_gap_stats,
        plot_dir / "duration_gap_fraction_comparison.png",
        xlabel="Acceptance minus PBA gap (years)",
        metric_col="fraction",
        theme=theme,
    )
    _plot_legacy_comparison(
        duration_gap_stats,
        plot_dir / "duration_gap_rmse_comparison.png",
        xlabel="Acceptance minus PBA gap (years)",
        metric_col="rmse",
        theme=theme,
    )
    _plot_legacy_comparison(
        start_gap_stats,
        plot_dir / "start_gap_fraction_comparison.png",
        xlabel="Acceptance minus PBA gap (years)",
        metric_col="fraction",
        theme=theme,
    )
    _plot_legacy_comparison(
        start_gap_stats,
        plot_dir / "start_gap_rmse_comparison.png",
        xlabel="Acceptance minus PBA gap (years)",
        metric_col="rmse",
        theme=theme,
    )


if __name__ == "__main__":
    main()
