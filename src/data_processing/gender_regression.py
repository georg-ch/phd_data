from __future__ import annotations

from pathlib import Path
import re
import textwrap

import numpy as np
import pandas as pd
import patsy
import plotly.graph_objects as go
import statsmodels.api as sm
import statsmodels.formula.api as smf
from plotly.subplots import make_subplots
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor

from scripts.paths import DATA_DIR
from src.data_loading.load_data import load_data
from src.plotting.plot_config import load_plot_config
from src.plotting.plotting_data import limit_year_range
from src.plotting.plotting_export import write_html
from src.plotting.plotting_style import short_institute_label


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TABLE_ROOT = DATA_DIR / "gender_regression"
PLOT_ROOT = PROJECT_ROOT / "int_plots" / "gender_regression"
TABLE_DIR = TABLE_ROOT
PLOT_DIR = PLOT_ROOT

LOCATION_FEATURES = (
    "institute_name",
    "subject_group_stala",
    "lecode",
    "subject_group_category_stala",
)
CONTEXT_FEATURES = ("cship_category", "pba_category", "mobility_category")
CATEGORICAL_FEATURES = LOCATION_FEATURES + CONTEXT_FEATURES
FEATURE_MERGE_CUTOFFS = {
    "institute_name": 10,
    "subject_group_category_stala": 10,
    "subject_group_stala": 30,
    "lecode": 30,
}
FEATURE_LABELS = {
    "institute_name": "Institut",
    "subject_group_stala": "Fachgruppe",
    "lecode": "LE-Code",
    "subject_group_category_stala": "Fachgruppenkategorie",
    "cship_category": "Staatsbuergerschaft",
    "pba_category": "PBA-Kategorie",
    "mobility_category": "Mobilitaet",
    "grade_num": "Note",
}

PBA_CATEGORY_LABELS = {
    "FH": "Fachhochschule",
    "TUB": "TU Berlin",
    "GER": "Deutschland (ohne FH, TUB)",
    "EU": "EU-Ausland",
    "NEU": "nicht-EU-Ausland",
}


def ensure_dirs() -> None:
    """Create the gender-regression output directories."""
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    (TABLE_DIR / "selection").mkdir(parents=True, exist_ok=True)
    (TABLE_DIR / "final").mkdir(parents=True, exist_ok=True)
    (TABLE_DIR / "feature_shares").mkdir(parents=True, exist_ok=True)


def slugify(text: str) -> str:
    """Return a filesystem-safe run label."""
    slug = re.sub(r"[^A-Za-z0-9]+", "_", text.strip())
    slug = slug.strip("_")
    return slug or "run"


def configure_output_scope(run_label: str) -> None:
    """Switch the module-level output roots to a run-specific scope."""
    global TABLE_DIR, PLOT_DIR
    TABLE_DIR = TABLE_ROOT / run_label
    PLOT_DIR = PLOT_ROOT / run_label
    ensure_dirs()


def feature_label(feature: str) -> str:
    """Return the display label for a feature."""
    return FEATURE_LABELS.get(feature, feature)


def share_axis_label() -> str:
    """Return the x-axis label used on share plots."""
    return "Anteil weibl./div."


def grand_mean_label() -> str:
    """Return the label used for the grand-mean reference bar."""
    return "Gesamtdurchschnitt"


def odds_ratio_label() -> str:
    """Return the label used for odds-ratio plots."""
    return "Quotenverhältnis (männl. - weibl./div.)"


def category_tickangle(n_categories: int) -> int:
    """Return the tick angle for categorical axes."""
    return 0


def display_category_label(feature: str, label: str) -> str:
    """Return a human-readable label for a category level."""
    if feature == "institute_name":
        return short_institute_label(label)
    if feature == "pba_category":
        return PBA_CATEGORY_LABELS.get(label, str(label))
    return str(label)


def feature_merge_cutoff(feature: str, default_min_count: int) -> int:
    """Return the feature-specific cutoff for merging rare levels."""
    return FEATURE_MERGE_CUTOFFS.get(feature, default_min_count)


def preferred_feature_dtype(feature: str, series: pd.Series) -> pd.Series:
    """Normalize feature dtypes before modeling."""
    if feature == "pba_category":
        levels = list(pd.Series(series).dropna().astype(str).unique())
        if "GER" in levels:
            levels = [level for level in levels if level != "GER"] + ["GER"]
        return pd.Categorical(series.astype("object"), categories=levels, ordered=True)
    return series.astype("object")


def maybe_merge_rare_categories(series: pd.Series, feature: str, *, min_count: int) -> pd.Series:
    """Merge rare levels using the feature-specific cutoff."""
    return merge_rare_categories(series, min_count=feature_merge_cutoff(feature, min_count), other_label="Other")


def prepare_feature_column(series: pd.Series, feature: str, *, min_count: int, merge_cols: set[str]) -> pd.Series:
    """Apply the merge and dtype normalization used for model features."""
    if feature in merge_cols or feature in FEATURE_MERGE_CUTOFFS:
        series = maybe_merge_rare_categories(series, feature, min_count=min_count)
    return preferred_feature_dtype(feature, series)


def contrast_level_from_label(contrast: str) -> str | None:
    """Extract the level label from a patsy contrast name."""
    match = re.search(r"\[(?:T|S)\.(.*)\]$", contrast)
    return match.group(1) if match else None


def merge_rare_categories(s: pd.Series, min_count: int = 10, other_label: str = "Other") -> pd.Series:
    s = s.astype("object")
    counts = s.dropna().value_counts()
    rare = counts[counts < min_count].index
    return s.where(s.isna() | ~s.isin(rare), other_label)


def prepare_gender_frame(data: pd.DataFrame, *, min_count: int, merge_cols: set[str]) -> pd.DataFrame:
    """Prepare the input data frame for the gender regression models."""
    df = data.copy()
    if "gender" not in df.columns:
        raise KeyError("Missing required column: gender")
    if "birth_country" not in df.columns or "pba_uni_is_german" not in df.columns:
        raise KeyError("Missing required columns for mobility_category")
    if "grade_num" not in df.columns:
        raise KeyError("Missing required column: grade_num")

    df["is_male"] = (df["gender"] == "m").astype("int64")
    born_in_germany = df["birth_country"].eq("Deutschland")
    pba_in_germany = df["pba_uni_is_german"].fillna(False)
    df["mobility_category"] = pd.Series("unknown", index=df.index, dtype="object")
    df.loc[born_in_germany & pba_in_germany, "mobility_category"] = "local"
    df.loc[born_in_germany & ~pba_in_germany, "mobility_category"] = "ger_studied_abroad"
    df.loc[~born_in_germany & pba_in_germany, "mobility_category"] = "int_early_mover"
    df.loc[~born_in_germany & ~pba_in_germany, "mobility_category"] = "international"
    df["grade_num"] = pd.to_numeric(df["grade_num"], errors="coerce")

    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = prepare_feature_column(df[col], col, min_count=min_count, merge_cols=merge_cols)

    drop_cols = ["is_male", "grade_num", *CATEGORICAL_FEATURES]
    drop_cols = [col for col in drop_cols if col in df.columns]
    df = df.dropna(subset=drop_cols).copy()
    return df


def feature_formula_term(data: pd.DataFrame, feature: str) -> str:
    """Return the Patsy term for a feature."""
    series = data[feature]
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        return feature
    return f"C({feature}, Sum)"


def model_formula(target_col: str, features: tuple[str, ...], data: pd.DataFrame) -> str:
    """Build a formula string for a set of features."""
    if not features:
        return f"{target_col} ~ 1"
    rhs = " + ".join(feature_formula_term(data, feature) for feature in features)
    return f"{target_col} ~ {rhs}"


def fit_glm(formula: str, data: pd.DataFrame, *, cov_type: str = "HC3"):
    """Fit the binomial GLM used throughout the module."""
    return smf.glm(formula, data=data, family=sm.families.Binomial(), missing="drop").fit(
        cov_type=cov_type, maxiter=200
    )


def wald_chi2_from_cov(model, term_cols: list[int]) -> tuple[float, float]:
    """Compute a robust Wald chi-square statistic from the model covariance."""
    params = np.asarray(model.params, dtype=float)
    cov = np.asarray(model.cov_params(), dtype=float)
    if not term_cols:
        return np.nan, np.nan
    r = np.zeros((len(term_cols), len(params)))
    for i, col_idx in enumerate(term_cols):
        r[i, col_idx] = 1.0
    term_params = params[term_cols]
    term_cov = np.asarray(r @ cov @ r.T, dtype=float)
    if not np.isfinite(term_cov).all():
        return np.nan, np.nan
    term_cov = (term_cov + term_cov.T) / 2.0
    try:
        eigvals, eigvecs = np.linalg.eigh(term_cov)
        max_eig = np.max(np.abs(eigvals)) if eigvals.size else 0.0
        tol = np.finfo(float).eps * max(term_cov.shape) * max_eig
        inv_eigvals = np.where(eigvals > tol, 1.0 / eigvals, 0.0)
        term_cov_inv = (eigvecs * inv_eigvals) @ eigvecs.T
    except np.linalg.LinAlgError:
        term_cov_inv = np.linalg.pinv(term_cov)
    if not np.isfinite(term_cov_inv).all():
        return np.nan, np.nan
    statistic = float(term_params.T @ term_cov_inv @ term_params)
    return statistic, float(stats.chi2.sf(statistic, df=len(term_cols)))


def max_vif_for_formula(formula: str, data: pd.DataFrame) -> tuple[float, str | None]:
    """Return the maximum VIF and the term that produced it."""
    try:
        _, design = patsy.dmatrices(formula, data=data, return_type="dataframe")
    except Exception:
        return np.nan, None
    if design.shape[1] == 0:
        return np.nan, None
    values = design.to_numpy(dtype=float)
    vifs = []
    for i in range(values.shape[1]):
        try:
            vifs.append(float(variance_inflation_factor(values, i)))
        except Exception:
            vifs.append(np.nan)
    vif_series = pd.Series(vifs, index=design.columns, dtype="float64")
    finite = vif_series[np.isfinite(vif_series)]
    if "Intercept" in finite.index and len(finite) > 1:
        finite = finite.drop("Intercept")
    if finite.empty:
        return np.nan, None
    return float(finite.max()), str(finite.idxmax())


def feature_level_table(model, data: pd.DataFrame | None = None) -> pd.DataFrame:
    """Summarize term-level Wald tests and partial pseudo R² values."""
    if data is None:
        data = getattr(model.model.data, "frame", None)
    if data is None:
        raise ValueError("feature_level_table needs the original data frame")
    formula = getattr(model.model, "formula", None)
    if formula is None:
        raise ValueError("feature_level_table needs a formula-based model")
    design_info = getattr(getattr(model.model, "data", None), "design_info", None)
    if design_info is None:
        raise ValueError("feature_level_table needs patsy design information")

    lhs, rhs = formula.split("~", 1)
    rhs_terms = [term.strip() for term in rhs.split("+") if term.strip()]
    rows = []
    for feature, col_slice in design_info.term_name_slices.items():
        if feature == "Intercept":
            continue
        term_cols = list(range(col_slice.start, col_slice.stop))
        statistic, p_value = wald_chi2_from_cov(model, term_cols)
        reduced_terms = [term for term in rhs_terms if term != feature]
        reduced_formula = lhs.strip() + (" ~ " + " + ".join(reduced_terms) if reduced_terms else " ~ 1")
        reduced = fit_glm(reduced_formula, data)
        dev_full = float(model.deviance)
        dev_reduced = float(reduced.deviance)
        partial_pseudo_r2 = np.nan
        if np.isfinite(dev_reduced) and dev_reduced > 0:
            partial_pseudo_r2 = max(0.0, (dev_reduced - dev_full) / dev_reduced)
        rows.append(
            {
                "feature": feature,
                "statistic": statistic,
                "p_value": p_value,
                "df_constraint": float(len(term_cols)),
                "partial_pseudo_r2": partial_pseudo_r2,
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_adj"] = multipletests(out["p_value"], method="fdr_bh")[1]
    return out


def contrast_table(model, terms: dict[str, str]) -> pd.DataFrame:
    """Evaluate arbitrary contrast expressions against a fitted model."""
    rows = []
    for label, expr in terms.items():
        test = model.t_test(expr)
        effect = float(np.asarray(test.effect).squeeze())
        se = float(np.asarray(test.sd).squeeze())
        z = float(np.asarray(test.tvalue).squeeze())
        p = float(np.asarray(test.pvalue).squeeze())
        rows.append(
            {
                "contrast": label,
                "effect": effect,
                "se": se,
                "z": z,
                "p": p,
                "odds_ratio": float(np.exp(effect)),
                "or_low": float(np.exp(effect - 1.96 * se)),
                "or_high": float(np.exp(effect + 1.96 * se)),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_adj"] = multipletests(out["p"], method="fdr_bh")[1]
    return out


def feature_level_contrasts(model, data: pd.DataFrame, feature: str) -> pd.DataFrame:
    """Build contrast rows for a single feature."""
    series = pd.Series(data[feature]).dropna()
    if isinstance(series.dtype, pd.CategoricalDtype):
        observed_levels = [str(level) for level in series.cat.categories]
    else:
        observed_levels = list(series.astype(str).unique())
    params = np.asarray(model.params, dtype=float)
    cov_params = getattr(model, "cov_params", None)
    cov = np.asarray(cov_params(), dtype=float) if callable(cov_params) else np.diag(np.zeros(len(params), dtype=float))
    param_index = {name: idx for idx, name in enumerate(model.params.index)}
    rows = []
    prefix = f"C({feature}, Sum)[S."
    param_names = [name for name in model.params.index if name.startswith(prefix)]
    if not param_names:
        return pd.DataFrame(columns=["feature", "level", "contrast", "effect", "se", "z", "p", "odds_ratio", "or_low", "or_high", "p_adj"])
    explicit_levels = [lvl for lvl in (contrast_level_from_label(name) for name in param_names) if lvl is not None]
    omitted_levels = [level for level in observed_levels if level not in explicit_levels]
    rows = []
    for level in explicit_levels:
        term_name = f"C({feature}, Sum)[S.{level}]"
        idx = param_index[term_name]
        effect = float(params[idx])
        se = float(np.sqrt(cov[idx, idx]))
        z = effect / se if se > 0 else np.nan
        p = float(2.0 * stats.norm.sf(abs(z))) if np.isfinite(z) else np.nan
        rows.append(
            {
                "feature": feature,
                "level": level,
                "contrast": term_name,
                "effect": effect,
                "se": se,
                "z": z,
                "p": p,
                "odds_ratio": float(np.exp(effect)),
                "or_low": float(np.exp(effect - 1.96 * se)),
                "or_high": float(np.exp(effect + 1.96 * se)),
            }
        )
    if len(omitted_levels) == 1:
        omitted = omitted_levels[0]
        idxs = [param_index[name] for name in param_names]
        l_vec = np.zeros(len(params))
        l_vec[idxs] = -1.0
        effect = float(l_vec @ params)
        se = float(np.sqrt(l_vec @ cov @ l_vec))
        z = effect / se if se > 0 else np.nan
        p = float(2.0 * stats.norm.sf(abs(z))) if np.isfinite(z) else np.nan
        rows.append(
            {
                "feature": feature,
                "level": omitted,
                "contrast": f"{feature}[omitted={omitted}]",
                "effect": effect,
                "se": se,
                "z": z,
                "p": p,
                "odds_ratio": float(np.exp(effect)),
                "or_low": float(np.exp(effect - 1.96 * se)),
                "or_high": float(np.exp(effect + 1.96 * se)),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_adj"] = multipletests(out["p"].fillna(1.0), method="fdr_bh")[1]
    return out


def compare_feature_blocks(
    data: pd.DataFrame,
    compare_features,
    common_features=(),
    *,
    target_col: str = "is_male",
    min_count: int = 10,
    merge_cols=(),
    cov_type: str = "HC3",
) -> pd.DataFrame:
    compare_features = tuple(compare_features)
    common_features = tuple(common_features)
    merge_cols = set(merge_cols)
    if set(compare_features) & set(common_features):
        raise ValueError("Features cannot be both common and compared")
    if not compare_features:
        return pd.DataFrame(columns=["feature", "nobs", "statistic", "p_value", "p_adj", "df_constraint", "partial_pseudo_r2", "max_vif", "max_vif_term", "baseline_max_vif", "baseline_max_vif_term", "delta_aic", "delta_bic", "delta_deviance"])

    needed_cols = [target_col, *common_features, *compare_features]
    complete = data.loc[:, needed_cols].dropna(subset=needed_cols).copy()
    for col in common_features + compare_features:
        if pd.api.types.is_numeric_dtype(complete[col]) and not pd.api.types.is_bool_dtype(complete[col]):
            complete[col] = pd.to_numeric(complete[col], errors="coerce").astype("float64")
        else:
            complete[col] = preferred_feature_dtype(col, complete[col])
    for col in merge_cols & set(common_features + compare_features):
        complete[col] = maybe_merge_rare_categories(complete[col], col, min_count=min_count)
    for col in set(common_features + compare_features) & set(FEATURE_MERGE_CUTOFFS):
        if col not in merge_cols:
            complete[col] = maybe_merge_rare_categories(complete[col], col, min_count=min_count)
    complete[target_col] = pd.to_numeric(complete[target_col], errors="coerce").astype("float64")
    complete = complete.dropna(subset=[target_col, *common_features, *compare_features]).copy()

    def build_formula(feature_terms):
        return model_formula(target_col, (*common_features, *feature_terms), complete)

    baseline_formula = build_formula(())
    baseline_model = fit_glm(baseline_formula, complete, cov_type=cov_type)
    baseline_max_vif, baseline_max_vif_term = max_vif_for_formula(baseline_formula, complete)

    rows = []
    for feature in compare_features:
        raw_formula = build_formula((feature,))
        candidate_model = fit_glm(raw_formula, complete, cov_type=cov_type)
        candidate_max_vif, candidate_max_vif_term = max_vif_for_formula(raw_formula, complete)
        candidate_table = feature_level_table(candidate_model, data=complete)
        feature_term = feature_formula_term(complete, feature)
        candidate_row = candidate_table[candidate_table["feature"] == feature_term]
        if candidate_row.empty:
            candidate_row = candidate_table[candidate_table["feature"] == feature]
        candidate_row = candidate_row.iloc[0]
        baseline_dev = float(baseline_model.deviance)
        candidate_dev = float(candidate_model.deviance)
        partial_pseudo_r2 = np.nan
        if np.isfinite(baseline_dev) and baseline_dev > 0:
            partial_pseudo_r2 = max(0.0, (baseline_dev - candidate_dev) / baseline_dev)
        rows.append(
            {
                "feature": feature,
                "nobs": int(candidate_model.nobs),
                "statistic": float(candidate_row["statistic"]),
                "p_value": float(candidate_row["p_value"]),
                "df_constraint": float(candidate_row["df_constraint"]),
                "partial_pseudo_r2": float(partial_pseudo_r2),
                "max_vif": candidate_max_vif,
                "max_vif_term": candidate_max_vif_term,
                "baseline_max_vif": baseline_max_vif,
                "baseline_max_vif_term": baseline_max_vif_term,
                "delta_aic": float(baseline_model.aic - candidate_model.aic),
                "delta_bic": float(baseline_model.bic - candidate_model.bic),
                "delta_deviance": float(baseline_dev - candidate_dev),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        adj = pd.Series(np.nan, index=out.index)
        mask = np.isfinite(out["p_value"].to_numpy(dtype=float))
        if mask.any():
            adj.loc[mask] = multipletests(out.loc[mask, "p_value"], method="fdr_bh")[1]
        out["p_adj"] = adj
        out = out.sort_values(["p_adj", "partial_pseudo_r2", "feature"], ascending=[True, False, True])
    return out


def selection_plot(tables: dict[str, pd.DataFrame]) -> go.Figure:
    """Render the feature-selection summary plot."""
    metric_specs = [
        ("p_adj", "Adjusted p-value", "log", "#E15759"),
        ("delta_aic", "Delta AIC", "linear", "#4E79A7"),
        ("partial_pseudo_r2", "Partial pseudo-R2", "linear", "#59A14F"),
    ]
    block_names = list(tables.keys())
    fig = make_subplots(rows=len(metric_specs), cols=len(block_names), shared_xaxes=False, shared_yaxes=False, subplot_titles=[block if row_idx == 0 else "" for row_idx in range(len(metric_specs)) for block in block_names], vertical_spacing=0.08, horizontal_spacing=0.06)
    for col_idx, (block_name, df) in enumerate(tables.items(), start=1):
        ordered = df.copy()
        for col in ["p_adj", "partial_pseudo_r2", "delta_aic"]:
            if col not in ordered.columns:
                ordered[col] = np.nan
        ordered = ordered.sort_values(["p_adj", "partial_pseudo_r2", "feature"], ascending=[True, False, True]).copy()
        x = ordered["feature"].astype(str)
        for row_idx, (metric_col, metric_label, scale, color) in enumerate(metric_specs, start=1):
            y = pd.to_numeric(ordered[metric_col], errors="coerce")
            fig.add_trace(go.Bar(x=x, y=y, marker=dict(color=color), showlegend=False, hovertemplate=f"Feature: %{{x}}<br>{metric_label}: %{{y:.4g}}<extra></extra>"), row=row_idx, col=col_idx)
            if metric_col == "p_adj":
                sig = y < 0.05
                if sig.any():
                    fig.add_trace(go.Scatter(x=x[sig], y=y[sig], mode="text", text=["*" for _ in range(int(sig.sum()))], textposition="top center", textfont=dict(color="#111111", size=16), hoverinfo="skip", showlegend=False), row=row_idx, col=col_idx)
            if scale == "log":
                fig.update_yaxes(type="log", row=row_idx, col=col_idx)
            fig.update_xaxes(tickangle=-35, automargin=True, showticklabels=row_idx == len(metric_specs), title_text="Feature" if row_idx == len(metric_specs) else None, row=row_idx, col=col_idx)
            fig.update_yaxes(title_text=metric_label if col_idx == 1 else None, showticklabels=col_idx == 1, row=row_idx, col=col_idx)
    fig.update_layout(barmode="group", height=320 * len(metric_specs), width=max(1200, 420 * len(block_names)), margin=dict(l=60, r=30, t=80, b=80))
    return fig


def compare_feature_blocks_grid(
    data: pd.DataFrame,
    *,
    target_col: str,
    comparison_blocks,
    common_features=(),
    min_count: int = 10,
    merge_cols=(),
    cov_type: str = "HC3",
    output_dir: Path | None = None,
    write_plot: bool = True,
    write_tables: bool = True,
) -> dict[str, pd.DataFrame]:
    output_path = Path(output_dir) if output_dir is not None else TABLE_DIR / "selection"
    output_path.mkdir(parents=True, exist_ok=True)
    results = {}
    for block in comparison_blocks:
        if len(block) == 2:
            table_name, compare_features = block
            block_common = common_features
        elif len(block) == 3:
            table_name, compare_features, block_common = block
        else:
            raise ValueError("comparison_blocks entries must have 2 or 3 items")
        out = compare_feature_blocks(data, compare_features=compare_features, common_features=block_common, target_col=target_col, min_count=min_count, merge_cols=merge_cols, cov_type=cov_type)
        results[table_name] = out
        if write_tables:
            out.to_csv(output_path / f"{target_col}__{table_name}.csv", index=False)
    if write_plot:
        write_html(selection_plot(results), PLOT_DIR / f"{target_col}__feature_selection.html", trace_map=None, plot_type="bar_category")
    return results


def pick_best_feature(table: pd.DataFrame) -> str:
    """Pick the best feature from a selection table."""
    ordered = table.sort_values(["p_adj", "partial_pseudo_r2", "delta_aic", "feature"], ascending=[True, False, False, True])
    return str(ordered.iloc[0]["feature"])


def write_selection_outputs(tables: dict[str, pd.DataFrame]) -> None:
    """Write the feature-selection outputs to disk."""
    selection_dir = TABLE_DIR / "selection"
    selection_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(selection_dir / f"is_male__{name}.csv", index=False)
    pd.concat(
        [df.assign(block=name) for name, df in tables.items()],
        ignore_index=True,
        sort=False,
    ).to_csv(selection_dir / "selection_summary.csv", index=False)
    write_html(
        selection_plot(tables),
        PLOT_DIR / "is_male__feature_selection.html",
        trace_map=None,
        plot_type="bar_category",
    )


def run_selection_sweep(data: pd.DataFrame, *, min_count: int, merge_cols: set[str], cov_type: str = "HC3") -> tuple[dict[str, pd.DataFrame], str, str]:
    """Run the selection sweep for location and context features."""
    location_blocks = compare_feature_blocks_grid(
        data,
        target_col="is_male",
        comparison_blocks=[("location_family", LOCATION_FEATURES)],
        min_count=min_count,
        merge_cols=merge_cols,
        cov_type=cov_type,
        output_dir=TABLE_DIR / "selection",
        write_plot=False,
        write_tables=False,
    )
    location_best = pick_best_feature(location_blocks["location_family"])
    context_blocks = compare_feature_blocks_grid(
        data,
        target_col="is_male",
        comparison_blocks=[("context_family", CONTEXT_FEATURES)],
        min_count=min_count,
        merge_cols=merge_cols,
        cov_type=cov_type,
        output_dir=TABLE_DIR / "selection",
        write_plot=False,
        write_tables=False,
    )
    context_best = pick_best_feature(context_blocks["context_family"])
    tables = {**location_blocks, **context_blocks}
    return tables, location_best, context_best


def forest_plot(contrasts: pd.DataFrame) -> go.Figure:
    """Render the odds-ratio forest plot."""
    if contrasts.empty:
        return go.Figure()
    filtered = contrasts.copy()
    if "contrast" in filtered.columns:
        filtered = filtered[~filtered["contrast"].astype(str).str.contains("Other", na=False)].copy()
    if filtered.empty:
        return go.Figure()
    ordered_features = list(dict.fromkeys(filtered["feature"].tolist()))
    fig = make_subplots(rows=len(ordered_features), cols=1, shared_xaxes=True, vertical_spacing=0.04, subplot_titles=[feature_label(feature) for feature in ordered_features])
    for row_idx, feature in enumerate(ordered_features, start=1):
        df = filtered[filtered["feature"] == feature].copy().sort_values("odds_ratio")
        x = df["odds_ratio"].to_numpy(dtype=float)
        err_plus = df["or_high"].to_numpy(dtype=float) - x
        err_minus = x - df["or_low"].to_numpy(dtype=float)
        sig = df["p_adj"].to_numpy(dtype=float) < 0.05
        colors = np.where(sig, "#E15759", "#9a9a9a")
        fig.add_trace(go.Scatter(x=x, y=df["contrast"].astype(str).tolist(), mode="markers", marker=dict(color=colors, size=10), error_x=dict(type="data", array=err_plus, arrayminus=err_minus, visible=True), hovertemplate=f"Kontrast: %{{y}}<br>{odds_ratio_label()}: %{{x:.3f}}<br>p_adj: %{{customdata[0]:.4g}}<extra></extra>", customdata=np.column_stack([df["p_adj"].to_numpy(dtype=float)]), showlegend=False), row=row_idx, col=1)
        fig.add_vline(x=1.0, line_dash="dash", line_color="#333333", row=row_idx, col=1)
        fig.update_xaxes(type="log", row=row_idx, col=1)
        fig.update_yaxes(autorange="reversed", tickangle=category_tickangle(len(df)), row=row_idx, col=1)
    fig.update_layout(height=max(380, 180 * len(ordered_features)), width=1100, margin=dict(l=80, r=30, t=70, b=60), xaxis_title=odds_ratio_label())
    return fig


def feature_level_plot(summary: pd.DataFrame, feature: str) -> go.Figure:
    """Render the female-share plot for a feature."""
    if summary.empty:
        return go.Figure()
    summary = summary[summary["level"] != "Other"].copy()
    if summary.empty:
        return go.Figure()
    summary = summary.sort_values(["female_share", "n"], ascending=[False, False]).copy()
    summary["label"] = summary["level"].map(lambda x: display_category_label(feature, x))
    grand_mean = float(summary["overall_female_share"].iloc[0])
    sig_summary = summary[summary["p_adj"].fillna(1.0).le(0.05)].copy()
    fig = go.Figure()
    if not sig_summary.empty:
        sig_summary = sig_summary.iloc[::-1]
        fig.add_trace(
            go.Bar(
                x=sig_summary["female_share"],
                y=sig_summary["label"],
                orientation="h",
                marker=dict(color="#E15759"),
                hovertemplate=(
                    "Kategorie: %{y}<br>"
                    f"{share_axis_label()}: %{{x:.1%}}<br>"
                    "n: %{customdata[0]}<br>"
                    "p_adj: %{customdata[1]:.4g}<extra></extra>"
                ),
                customdata=np.column_stack([sig_summary["n"], sig_summary["p_adj"].fillna(np.nan)]),
                name="Signifikanz",
                showlegend=False,
            )
        )
    fig.add_trace(
        go.Bar(
            x=[grand_mean],
            y=[grand_mean_label()],
            orientation="h",
            marker=dict(color="#8A8A8A"),
            hovertemplate=f"{grand_mean_label()}: %{{x:.1%}}<extra></extra>",
            name=grand_mean_label(),
            showlegend=False,
        )
    )
    fig.add_vline(x=grand_mean, line_dash="dash", line_color="#333333")
    fig.update_layout(title=f"{feature_label(feature)}: groesste signifikante Anteile weibl./div.", xaxis=dict(title=share_axis_label(), tickformat=".0%", range=[0, 1]), yaxis=dict(title="", tickangle=category_tickangle(len(summary))), margin=dict(l=90, r=30, t=60, b=50), height=max(320, 30 * len(summary) + 150), showlegend=False)
    return fig


def feature_share_table(
    data: pd.DataFrame,
    feature: str,
    *,
    min_count: int,
    merge_cols: set[str],
    cov_type: str = "HC3",
    model=None,
    apply_merge: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute the share summary and contrast table for one feature."""
    frame = data[[feature, "is_male"]].dropna().copy()
    if apply_merge:
        frame[feature] = prepare_feature_column(frame[feature], feature, min_count=min_count, merge_cols=merge_cols)
    else:
        frame[feature] = preferred_feature_dtype(feature, frame[feature])
    if frame[feature].nunique() < 2:
        return pd.DataFrame(), pd.DataFrame()
    if model is None:
        model = fit_glm(model_formula("is_male", (feature,), frame), frame, cov_type=cov_type)
    level_table = feature_level_contrasts(model, frame, feature)
    summary = frame.groupby(feature, as_index=False, observed=False)["is_male"].agg(n="count", male_share="mean").rename(columns={feature: "level"})
    summary["feature"] = feature
    summary["female_share"] = 1.0 - summary["male_share"]
    summary["overall_female_share"] = float(1.0 - frame["is_male"].mean())
    out = summary.merge(level_table, on=["feature", "level"], how="left")
    out["p_adj"] = out["p_adj"].fillna(1.0)
    out["sig"] = out["p_adj"] <= 0.05
    return out.sort_values(["female_share", "n"], ascending=[False, False]).reset_index(drop=True), level_table


def feature_from_param_name(name: str, *, location_feature: str, context_feature: str) -> str:
    """Map a parameter name back to its source feature."""
    if name == "grade_num":
        return "grade_num"
    if name.startswith(f"C({location_feature},"):
        return location_feature
    if name.startswith(f"C({context_feature},"):
        return context_feature
    return name


def final_feature_effect_plot(contrast_df: pd.DataFrame, feature: str) -> go.Figure:
    """Render the final feature-effect plot."""
    df = contrast_df[contrast_df["feature"] == feature].copy()
    if df.empty:
        return go.Figure()

    def pretty_label(text: str) -> str:
        level = contrast_level_from_label(text)
        if level is None and text.startswith(f"{feature}[omitted="):
            level = text.split("=", 1)[-1].rstrip("]")
        label = level if level is not None else text
        return textwrap.fill(display_category_label(feature, label), width=34)

    df["label"] = df["contrast"].astype(str).map(pretty_label)
    df = df[~df["label"].astype(str).str.contains("Other", na=False)].copy()
    if df.empty:
        return go.Figure()
    df = df.sort_values("odds_ratio")
    x = df["odds_ratio"].to_numpy(dtype=float)
    err_plus = df["or_high"].to_numpy(dtype=float) - x
    err_minus = x - df["or_low"].to_numpy(dtype=float)
    sig = df["p_adj"].to_numpy(dtype=float) < 0.05
    colors = np.where(sig, "#E15759", "#9a9a9a")

    fig = go.Figure(
        go.Scatter(
            x=x,
            y=df["label"].tolist(),
            mode="markers",
            marker=dict(color=colors, size=10),
            error_x=dict(type="data", array=err_plus, arrayminus=err_minus, visible=True),
            hovertemplate=(
                "Term: %{y}<br>"
                f"{odds_ratio_label()}: %{{x:.3f}}<br>"
                "p_adj: %{customdata:.4g}<extra></extra>"
            ),
            customdata=df["p_adj"].to_numpy(dtype=float),
            showlegend=False,
        )
    )
    fig.add_vline(x=1.0, line_dash="dash", line_color="#333333")
    fig.update_layout(
        title=feature_label(feature),
        xaxis=dict(title=odds_ratio_label(), type="log"),
        yaxis=dict(
            title="",
            automargin=True,
            tickangle=category_tickangle(len(df)),
            tickfont=dict(size=12),
            categoryorder="array",
            categoryarray=df["label"].tolist()[::-1],
        ),
        margin=dict(l=130, r=35, t=55, b=30),
        height=max(200, 28 * len(df) + 70),
        showlegend=False,
    )
    return fig


def run_final_model(data: pd.DataFrame, *, location_feature: str, context_feature: str, min_count: int, merge_cols: set[str], cov_type: str = "HC3") -> dict[str, object]:
    """Fit the final model and write its output tables and plots."""
    frame = data[["is_male", location_feature, context_feature]].dropna().copy()
    for col in (location_feature, context_feature):
        frame[col] = prepare_feature_column(frame[col], col, min_count=min_count, merge_cols=merge_cols)
    frame = frame.dropna(subset=["is_male", location_feature, context_feature]).copy()
    model = fit_glm(model_formula("is_male", (location_feature, context_feature), frame), frame, cov_type=cov_type)
    term_table = feature_level_table(model, data=frame)
    contrast_frames = []
    for feature in [location_feature, context_feature]:
        contrast_frames.append(feature_level_contrasts(model, frame, feature))
    contrast_df = pd.concat(contrast_frames, ignore_index=True, sort=False)
    term_table.to_csv(TABLE_DIR / "final" / "feature_terms.csv", index=False)
    contrast_df.to_csv(TABLE_DIR / "final" / "feature_contrasts.csv", index=False)
    final_plot_dir = PLOT_DIR / "final"
    final_plot_dir.mkdir(parents=True, exist_ok=True)
    for feature in [location_feature, context_feature]:
        write_html(
            final_feature_effect_plot(contrast_df, feature),
            final_plot_dir / f"{feature}_effects.html",
            trace_map=None,
            plot_type="forest",
        )
    return {"model": model, "formula": model.model.formula, "frame": frame, "term_table": term_table, "contrast_df": contrast_df}


def plot_feature_shares(
    data: pd.DataFrame,
    *,
    min_count: int,
    merge_cols: set[str],
    forest_feature: str | None = None,
    simple_forest_features: tuple[str, ...] = (),
    cov_type: str = "HC3",
) -> dict[str, pd.DataFrame]:
    """Write the simple-model share plots for all categorical features."""
    out_dir = TABLE_DIR / "feature_shares"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = PLOT_DIR / "simple_model"
    plot_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    forest_features = {feature for feature in (forest_feature, *simple_forest_features) if feature}
    for feature in CATEGORICAL_FEATURES:
        summary, level_table = feature_share_table(data, feature, min_count=min_count, merge_cols=merge_cols, cov_type=cov_type)
        if summary.empty:
            continue
        results[feature] = summary
        summary.to_csv(out_dir / f"{feature}.csv", index=False)
        write_html(feature_level_plot(summary, feature), PLOT_DIR / f"{feature}__female_share.html", trace_map=None, plot_type="bar_category")
        if feature in forest_features and not level_table.empty:
            write_html(
                final_feature_effect_plot(level_table, feature),
                plot_dir / f"{feature}_effects.html",
                trace_map=None,
                plot_type="forest",
            )
    return results


def plot_final_share_plots(
    model,
    frame: pd.DataFrame,
    *,
    features: tuple[str, ...],
) -> dict[str, pd.DataFrame]:
    """Write the share plots derived from the final model."""
    out_dir = PLOT_DIR / "final"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for feature in features:
        summary, level_table = feature_share_table(
            frame,
            feature,
            min_count=1,
            merge_cols=set(),
            cov_type="HC3",
            model=model,
            apply_merge=False,
        )
        if summary.empty or level_table.empty:
            continue
        results[feature] = summary
        write_html(
            feature_level_plot(summary, feature),
            out_dir / f"{feature}__female_share.html",
            trace_map=None,
            plot_type="bar_category",
        )
    return results


def run_gender_regression(
    data: pd.DataFrame,
    *,
    location_feature: str | None = None,
    context_feature: str | None = None,
) -> dict[str, object]:
    """Run the full gender-regression workflow."""
    params, _, _ = load_plot_config("gender_regression")
    min_count = int(params["small_group_cutoff"])
    merge_cols = set(params.get("merge_cols", []))
    prepared = prepare_gender_frame(data, min_count=min_count, merge_cols=merge_cols)
    prepared = limit_year_range(prepared, year_col="defense_year", start_year=int(params["start_year"]), end_year=int(params["end_year"]))
    prepared = prepared[prepared["is_male"].notna()].copy()
    selection_tables, location_best, context_best = run_selection_sweep(prepared, min_count=min_count, merge_cols=merge_cols)
    selection_tables = dict(selection_tables)
    final_location = location_feature or location_best
    final_context = context_feature or context_best
    run_label = slugify(final_location)
    configure_output_scope(run_label)
    write_selection_outputs(selection_tables)
    final = run_final_model(
        prepared,
        location_feature=final_location,
        context_feature=final_context,
        min_count=min_count,
        merge_cols=merge_cols,
    )
    share_tables = plot_feature_shares(
        prepared,
        min_count=min_count,
        merge_cols=merge_cols,
        forest_feature=final_location,
        simple_forest_features=(final_context,),
    )
    final_share_tables = plot_final_share_plots(
        final["model"],
        final["frame"],
        features=(final_location, final_context),
    )
    return {"selection_tables": selection_tables, "best_location_feature": location_best, "best_context_feature": context_best, "final": final, "share_tables": share_tables, "final_share_tables": final_share_tables}


def main() -> None:
    """Run the gender regression workflow as a script."""
    ensure_dirs()
    data = load_data(DATA_DIR / "clean_data.csv", DATA_DIR / "cleaned_data_dtypes.json")
    result = run_gender_regression(data)
    print("Best location feature:", result["best_location_feature"])
    print("Best context feature:", result["best_context_feature"])
    print(result["final"]["term_table"].to_string(index=False))
    print(result["final"]["contrast_df"].to_string(index=False))


if __name__ == "__main__":
    main()
