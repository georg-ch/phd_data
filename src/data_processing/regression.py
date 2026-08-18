import statsmodels.formula.api as smf
from pathlib import Path
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from scipy import stats
import numpy as np
import pandas as pd
import patsy
import re

import matplotlib.pyplot as plt
import plotly.graph_objects as go
from statsmodels.stats.outliers_influence import OLSInfluence
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multitest import multipletests
from scripts.paths import DATA_DIR
from src.plotting.plot_config import load_plot_config
from src.plotting.plotting_export import write_html
from src.plotting.plotting_traces import ordered_with_preference


def contrast_table(model, terms):
    """Evaluate arbitrary contrast expressions against a fitted model."""
    rows = []
    for label, expr in terms.items():
        test = model.t_test(expr)
        rows.append(
            {
                "contrast": label,
                "effect": float(test.effect),
                "se": float(test.sd),
                "t": float(test.tvalue),
                "p": float(test.pvalue),
            }
        )

    out = pd.DataFrame(rows)
    out["p_adj"] = multipletests(out["p"], method="fdr_bh")[1]
    return out


def wald_chi2_from_cov(model, term_cols):
    """Fallback robust Wald chi-square for a block of coefficients."""
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        params = np.asarray(model.params, dtype=float)
        cov = np.asarray(model.cov_params(), dtype=float)

        r_matrix = np.zeros((len(term_cols), len(params)))
        for i, col_idx in enumerate(term_cols):
            r_matrix[i, col_idx] = 1.0

        term_params = params[term_cols]
        term_cov = r_matrix @ cov @ r_matrix.T
        term_cov = np.asarray(term_cov, dtype=float)
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
            try:
                term_cov_inv = np.linalg.pinv(term_cov, hermitian=True)
            except TypeError:
                term_cov_inv = np.linalg.pinv(term_cov)

        if not np.isfinite(term_cov_inv).all():
            return np.nan, np.nan

        statistic = float(term_params.T @ term_cov_inv @ term_params)
        p_value = float(stats.chi2.sf(statistic, df=len(term_cols)))
        return statistic, p_value


def max_vif_for_formula(formula, data):
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
            vif = float(variance_inflation_factor(values, i))
        except Exception:
            vif = np.nan
        vifs.append(vif)

    vif_series = pd.Series(vifs, index=design.columns, dtype="float64")
    finite = vif_series[np.isfinite(vif_series)]
    if "Intercept" in finite.index and (len(finite) > 1):
        finite = finite.drop("Intercept")
    if finite.empty:
        return np.nan, None

    max_term = finite.idxmax()
    return float(finite.loc[max_term]), str(max_term)


def feature_level_table(model, data=None):
    """Return omnibus tests and partial R² values for each model term.

    The returned table has one row per model term, using the term names from
    ``wald_test_terms()``. For each term we report:
    - the omnibus Wald test statistic and p-value
    - the corresponding degrees of freedom
    - the partial R² from a nested-model comparison

    Parameters
    ----------
    model:
        A fitted statsmodels OLS result.
    data:
        Optional data frame to refit reduced models on. If omitted, the data
        attached to the fitted model is used.
    """
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
    fit_kw = {}
    cov_type = getattr(model, "cov_type", None)
    if cov_type:
        fit_kw["cov_type"] = cov_type

    rows = []
    for feature, col_slice in design_info.term_name_slices.items():
        if feature == "Intercept":
            continue

        term_cols = list(range(col_slice.start, col_slice.stop))
        if not term_cols:
            continue

        statistic, p_value = wald_chi2_from_cov(model, term_cols)

        reduced_terms = [term for term in rhs_terms if term != feature]
        if reduced_terms:
            reduced_formula = lhs.strip() + " ~ " + " + ".join(reduced_terms)
        else:
            reduced_formula = lhs.strip() + " ~ 1"
        reduced = smf.ols(reduced_formula, data=data, missing="drop").fit(**fit_kw)

        rss_full = float(model.ssr)
        rss_reduced = float(reduced.ssr)
        partial_r2 = np.nan
        if rss_reduced > 0:
            partial_r2 = max(0.0, (rss_reduced - rss_full) / rss_reduced)

        rows.append(
            {
                "feature": feature,
                "statistic": statistic,
                "p_value": p_value,
                "df_constraint": float(len(term_cols)),
                "df_denom": np.nan,
                "partial_r2": partial_r2,
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_adj"] = multipletests(out["p_value"], method="fdr_bh")[1]
    return out


def categorical_term(feature: str, *, coding: str, reference=None) -> str:
    """Build a Patsy term with the requested categorical coding."""
    if coding == "Sum":
        return f"C({feature}, Sum)"
    if coding == "Treatment":
        if reference is None:
            return f"C({feature}, Treatment)"
        if isinstance(reference, str):
            reference_repr = repr(reference)
        else:
            reference_repr = str(reference)
        return f"C({feature}, Treatment(reference={reference_repr}))"
    raise ValueError(f"Unknown coding: {coding}")


def contrast_rows_for_feature(model, feature: str) -> pd.DataFrame:
    """Build contrast rows for one feature."""
    design_info = getattr(getattr(model.model, "data", None), "design_info", None)
    if design_info is None:
        raise ValueError("contrast rows need patsy design information")

    term_slice = design_info.term_name_slices.get(feature)
    if term_slice is None:
        if feature in model.params.index:
            term_names = [feature]
        else:
            return pd.DataFrame(
                columns=["feature", "contrast", "effect", "se", "t", "p", "p_adj"]
            )
    else:
        term_names = list(model.params.index[term_slice.start : term_slice.stop])

    rows = []
    for term_name in term_names:
        if term_name == "Intercept":
            continue
        test = model.t_test(f"{term_name} = 0")
        rows.append(
            {
                "feature": feature,
                "contrast": term_name,
                "effect": float(np.asarray(test.effect).squeeze()),
                "se": float(np.asarray(test.sd).squeeze()),
                "t": float(np.asarray(test.tvalue).squeeze()),
                "p": float(np.asarray(test.pvalue).squeeze()),
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_adj"] = multipletests(out["p"], method="fdr_bh")[1]
    return out


def feature_formula_term(data: pd.DataFrame, feature: str) -> str:
    """Return the Patsy term for a feature using sum coding for categoricals."""
    if feature not in data.columns:
        raise KeyError(f"Unknown feature: {feature}")

    series = data[feature]
    if (
        pd.api.types.is_bool_dtype(series)
        or pd.api.types.is_object_dtype(series)
        or pd.api.types.is_categorical_dtype(series)
    ):
        return f"C({feature}, Sum)"
    if pd.api.types.is_numeric_dtype(series):
        return feature
    return f"C({feature}, Sum)"


def prepare_contrast_frame(
    frame: pd.DataFrame,
    *,
    target_col_trafo: str,
    spec_subset,
    merge_cols,
    min_count: int,
) -> tuple[pd.DataFrame, list[str]]:
    """Prepare a contrast-analysis frame and its RHS terms."""
    frame = frame.copy()
    rhs_terms = []
    used_cols = [target_col_trafo]

    for spec in spec_subset:
        feature = spec["feature"]
        used_cols.append(feature)
        if spec["kind"] == "num":
            frame[feature] = pd.to_numeric(frame[feature], errors="coerce").astype(
                "float64"
            )
        else:
            frame[feature] = frame[feature].astype("object")
            frame[feature] = frame[feature].where(frame[feature].notna(), "Missing")
            if feature in merge_cols:
                frame[feature] = merge_rare_categories(
                    frame[feature], min_count=min_count, other_label="Other"
                )
        rhs_terms.append(spec["term"])

    frame = frame.dropna(subset=used_cols).copy()
    return frame, rhs_terms


def fit_contrast_model(
    frame: pd.DataFrame,
    rhs_terms: list[str],
    *,
    target_col_trafo: str,
    cov_type: str,
):
    """Fit the contrast-analysis model for a prepared frame."""
    formula = f"{target_col_trafo} ~ " + " + ".join(rhs_terms)
    model = smf.ols(formula, data=frame, missing="drop").fit(cov_type=cov_type)
    return model, formula


def compare_feature_blocks(
    data,
    compare_features,
    common_features=(),
    target_col="duration_computed",
    target_transform="log",
    min_count=10,
    merge_cols=(),
    cov_type="HC3",
):
    """Compare candidate feature blocks against a common baseline model.

    The function fits one candidate model per feature on the same
    complete-case sample:

    - a baseline model with ``common_features`` only
    - a candidate model with ``common_features`` plus one compared feature

    It returns one row per candidate feature with:

    - omnibus tests for the candidate model
    - BH-adjusted p-values across candidate models
    - partial R^2 for the candidate model
    - max VIF for the candidate model
    - ``ΔAIC`` and ``ΔBIC`` against the common-only baseline

    Parameters
    ----------
    data:
        Raw input dataframe.
    compare_features:
        Candidate features to compare against each other.
    common_features:
        Features included in every model.
    target_col:
        Outcome column before transformation.
    target_transform:
        Passed to :func:`transform_target`.
    min_count:
        Minimum level count for categories listed in ``merge_cols``.
    merge_cols:
        Categorical features whose rare levels should be merged to ``Other``.
    cov_type:
        Covariance estimator passed to statsmodels ``fit``.
    """
    compare_features = tuple(compare_features)
    common_features = tuple(common_features)
    merge_cols = set(merge_cols)

    overlap = set(compare_features) & set(common_features)
    if overlap:
        raise ValueError(
            f"Features cannot be both common and compared: {sorted(overlap)}"
        )

    if not compare_features:
        return pd.DataFrame(
            columns=[
                "feature",
                "statistic",
                "p_value",
                "p_adj",
                "partial_r2",
                "df_constraint",
                "max_vif",
                "max_vif_term",
                "baseline_max_vif",
                "baseline_max_vif_term",
                "delta_aic",
                "delta_bic",
                "nobs",
            ]
        )

    data = prefer_duration_computed(data, target_col)
    data = engineer_features(data, target_col)
    needed_cols = [target_col, *common_features, *compare_features]
    missing_cols = [col for col in needed_cols if col not in data.columns]
    if missing_cols:
        raise KeyError(f"Missing columns for comparison: {missing_cols}")

    target_trafo_col = f"{target_col}_transformed"
    data[target_trafo_col], _ = transform_target(data[target_col], fun=target_transform)

    # Use one complete-case sample for every model so that comparisons are fair.
    complete_cols = [target_trafo_col, *common_features, *compare_features]
    complete_cols = list(dict.fromkeys(complete_cols))
    complete = data.loc[:, complete_cols].copy()
    complete = complete.dropna(subset=complete_cols).copy()

    # Normalize dtypes for Patsy.
    for col in common_features + compare_features:
        if pd.api.types.is_numeric_dtype(
            complete[col]
        ) and not pd.api.types.is_bool_dtype(complete[col]):
            complete[col] = pd.to_numeric(complete[col], errors="coerce").astype(
                "float64"
            )
        else:
            complete[col] = complete[col].astype("object")
            complete[col] = complete[col].where(complete[col].notna(), "Missing")

    complete[target_trafo_col] = pd.to_numeric(
        complete[target_trafo_col], errors="coerce"
    ).astype("float64")
    complete = complete.dropna(subset=[target_trafo_col]).copy()

    for col in merge_cols & set(common_features + compare_features):
        complete[col] = merge_rare_categories(
            complete[col], min_count=min_count, other_label="Other"
        )

    def build_formula(feature_terms):
        rhs_terms = [*common_features, *feature_terms]
        if not rhs_terms:
            return f"{target_trafo_col} ~ 1"
        rhs = " + ".join(
            feature_formula_term(complete, feature) for feature in rhs_terms
        )
        return f"{target_trafo_col} ~ {rhs}"

    baseline_formula = build_formula(())
    baseline_model = smf.ols(baseline_formula, data=complete, missing="drop").fit(
        cov_type=cov_type
    )
    baseline_max_vif, baseline_max_vif_term = max_vif_for_formula(
        baseline_formula, complete
    )

    rows = []
    for feature in compare_features:
        raw_formula = build_formula((feature,))
        candidate_model = smf.ols(raw_formula, data=complete, missing="drop").fit(
            cov_type=cov_type
        )
        candidate_max_vif, candidate_max_vif_term = max_vif_for_formula(
            raw_formula, complete
        )
        candidate_table = feature_level_table(candidate_model, data=complete)
        candidate_row = candidate_table[
            candidate_table["feature"] == feature_formula_term(complete, feature)
        ]
        if candidate_row.empty:
            candidate_row = candidate_table[candidate_table["feature"] == feature]
        if candidate_row.empty:
            raise KeyError(f"Could not find omnibus row for feature {feature!r}")
        candidate_row = candidate_row.iloc[0]

        rows.append(
            {
                "feature": feature,
                "statistic": float(candidate_row["statistic"]),
                "p_value": float(candidate_row["p_value"]),
                "df_constraint": float(candidate_row["df_constraint"]),
                "partial_r2": float(candidate_row["partial_r2"]),
                "max_vif": candidate_max_vif,
                "max_vif_term": candidate_max_vif_term,
                "baseline_max_vif": baseline_max_vif,
                "baseline_max_vif_term": baseline_max_vif_term,
                "delta_aic": float(baseline_model.aic - candidate_model.aic),
                "delta_bic": float(baseline_model.bic - candidate_model.bic),
                "nobs": int(candidate_model.nobs),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        adj = pd.Series(np.nan, index=out.index)
        mask = np.isfinite(out["p_value"].to_numpy(dtype=float))
        if mask.any():
            adj.loc[mask] = multipletests(out.loc[mask, "p_value"], method="fdr_bh")[1]
        out["p_adj"] = adj
        out = out[
            [
                "feature",
                "nobs",
                "statistic",
                "p_value",
                "p_adj",
                "df_constraint",
                "partial_r2",
                "max_vif",
                "max_vif_term",
                "baseline_max_vif",
                "baseline_max_vif_term",
                "delta_aic",
                "delta_bic",
            ]
        ].sort_values(["p_adj", "partial_r2", "feature"], ascending=[True, False, True])
    return out


def compare_feature_blocks_grid(
    data,
    *,
    target_cols,
    comparison_blocks,
    common_features=(),
    target_transform="log",
    min_count=10,
    merge_cols=(),
    cov_type="HC3",
    output_dir=None,
):
    """Run multiple feature-block comparisons and optionally save each table.

    Parameters
    ----------
    data:
        Input dataframe.
    target_cols:
        Iterable of target column names to compare.
    comparison_blocks:
        Iterable of ``(table_name, compare_features)`` or
        ``(table_name, compare_features, block_common_features)`` tuples.
    common_features:
        Features included in every comparison.
    output_dir:
        If provided, each table is written as CSV there.
    """

    results = {}
    output_path = (
        Path(output_dir)
        if output_dir is not None
        else DATA_DIR / "regression_feature_selection"
    )
    if output_path is not None:
        output_path.mkdir(parents=True, exist_ok=True)

    for target_col in target_cols:
        for block in comparison_blocks:
            if len(block) == 2:
                table_name, compare_features = block
                block_common = common_features
            elif len(block) == 3:
                table_name, compare_features, block_common = block
            else:
                raise ValueError("comparison_blocks entries must have 2 or 3 items")
            out = compare_feature_blocks(
                data,
                compare_features=compare_features,
                common_features=block_common,
                target_col=target_col,
                target_transform=target_transform,
                min_count=min_count,
                merge_cols=merge_cols,
                cov_type=cov_type,
            )
            key = (target_col, table_name)
            results[key] = out
            if output_path is not None:
                fname = f"{target_col}__{table_name}.csv"
                out.to_csv(output_path / fname, index=False)

    return results


def compare_current_model_feature_evaluation_grid(
    data,
    *,
    target_cols,
    target_transform="log",
    min_count=10,
    merge_cols=(),
    cov_type="HC3",
    output_dir=None,
):
    """Evaluate each current-model feature with and without the other features.

    For each target column we first infer the current model features from
    :func:`feature_selection`. Then, for every feature in that model, we fit:

    - a model with the other current-model features as common features
    - a univariate model with the feature as sole regressor

    The two rows are merged into a single table per target and written to
    ``DATA_DIR / "feature_evaluation"`` by default.
    """

    target_cols = tuple(target_cols)
    if not target_cols:
        return {}

    output_path = (
        Path(output_dir) if output_dir is not None else DATA_DIR / "feature_evaluation"
    )
    output_path.mkdir(parents=True, exist_ok=True)

    # Use the first target column to recover the current feature set. The
    # feature-selection step itself is target-agnostic in this pipeline.
    feature_probe_target = target_cols[0]
    probe_data = prefer_duration_computed(data, feature_probe_target)
    probe_data = engineer_features(probe_data, feature_probe_target)
    _, used_cols = feature_selection(
        probe_data,
        feature_probe_target,
        min_count=min_count,
        merge_cols=merge_cols,
    )
    current_features = [col for col, _ in used_cols]

    def prefix_values(row, prefix):
        return {
            f"{prefix}_{key}": value for key, value in row.items() if key != "feature"
        }

    results = {}
    for target_col in target_cols:
        rows = []
        for feature in current_features:
            common_features = tuple(
                other_feature
                for other_feature in current_features
                if other_feature != feature
            )

            heldout = compare_feature_blocks(
                data,
                compare_features=(feature,),
                common_features=common_features,
                target_col=target_col,
                target_transform=target_transform,
                min_count=min_count,
                merge_cols=merge_cols,
                cov_type=cov_type,
            )
            solo = compare_feature_blocks(
                data,
                compare_features=(feature,),
                common_features=(),
                target_col=target_col,
                target_transform=target_transform,
                min_count=min_count,
                merge_cols=merge_cols,
                cov_type=cov_type,
            )

            if heldout.empty or solo.empty:
                continue

            heldout_row = heldout.iloc[0].to_dict()
            solo_row = solo.iloc[0].to_dict()
            row = {"feature": feature}
            row.update(prefix_values(heldout_row, "heldout"))
            row.update(prefix_values(solo_row, "solo"))
            rows.append(row)

        out = pd.DataFrame(rows)
        if not out.empty:
            sort_cols = [
                col
                for col in ["heldout_p_adj", "heldout_partial_r2", "feature"]
                if col in out.columns
            ]
            ascending = [True, False, True][: len(sort_cols)]
            if sort_cols:
                out = out.sort_values(sort_cols, ascending=ascending)
            out.to_csv(
                output_path / f"{target_col}__current_model_features.csv", index=False
            )
        results[target_col] = out

    return results


def transform_target(col, fun="log"):
    if fun == "log":
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.log(col.replace(0, np.nan)), 0
    elif fun == "boxcox":
        out = pd.Series(np.nan, index=col.index)

        mask = col.notna() & (col > 0)

        out.loc[mask], lam = stats.boxcox(col.loc[mask])

        return out, lam
    else:
        raise ValueError(f"Unknown transformation function: {fun}")


def prefer_duration_computed(data: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Use duration_computed wherever it exists; otherwise keep target_col."""
    if target_col == "duration_computed" or "duration_computed" not in data.columns:
        return data.copy()
    if target_col not in data.columns:
        raise KeyError(f"Missing target column: {target_col}")

    out = data.copy()
    fallback_mask = out["duration_computed"].isna()
    out[target_col] = out[target_col].where(fallback_mask, out["duration_computed"])
    return out


def engineer_features(data, target_col):
    data = data.copy()

    # fix single birthday
    mean_birth_year = data.loc[data["pagination_nr"] != 8629, "birth_year"].mean()

    data.loc[data["pagination_nr"] == 8629, "birth_year"] = mean_birth_year

    data["degree_gap"] = data["defense_year"] - data[target_col] - data["pba_year"]
    data["age_at_start"] = data["defense_year"] - data[target_col] - data["birth_year"]
    data["is_male"] = data["gender"] == "m"
    data["pba_grade_missing"] = (
        data["pba_grade"] == "passed, no grade"
    )  # ALWAYS use this together with pba_grade
    data["pba_grade"] = (
        data["pba_grade"].replace({"passed, no grade": "2.0"}).astype(float)
    )

    born_in_germany_mask = data["birth_country"] == "Deutschland"
    pba_in_germany = data["pba_uni_is_german"]

    data["mobility_category"] = pd.Series("unknown", index=data.index)
    data.loc[born_in_germany_mask & pba_in_germany, "mobility_category"] = "local"
    data.loc[born_in_germany_mask & ~pba_in_germany, "mobility_category"] = (
        "ger_studied_abroad"
    )
    data.loc[~born_in_germany_mask & pba_in_germany, "mobility_category"] = (
        "int_early_mover"
    )
    data.loc[~born_in_germany_mask & ~pba_in_germany, "mobility_category"] = (
        "international"
    )
    defenses = pd.to_datetime(data["defense_date"], format="%d-%m-%Y")
    acceptances = pd.to_datetime(data["acceptance_date"], format="%d-%m-%Y")
    pbas = pd.to_datetime(data["pba_date"], format="%d-%m-%Y")

    data["d_acceptance"] = (defenses - acceptances).dt.days / 365.2425
    data["d_pba"] = (defenses - pbas).dt.days / 365.2425

    durations = data[target_col]

    starts = defenses - pd.to_timedelta(durations * 365.2425, unit="D")

    covid_start = pd.Timestamp("2020-03-01")
    covid_end = pd.Timestamp("2022-03-31")

    overlap_start = starts.clip(lower=covid_start)
    overlap_end = defenses.clip(upper=covid_end)
    overlap_years = (overlap_end - overlap_start).dt.days.clip(lower=0) / 365.2425

    data["covid_overlap"] = overlap_years

    return data


def merge_rare_categories(s, min_count=10, other_label="Other"):
    s = s.astype("object")
    counts = s.dropna().value_counts()

    rare = counts[counts < min_count].index

    return s.where(s.isna() | ~s.isin(rare), other_label)


def feature_selection(data, target_col, min_count=10, merge_cols={}):
    used_cols = [
        # ("degree_gap", "num"),  # XOR age_at_start
        # ("faculty", "cat"),  # XOR faculty/lecode/subject/subject_group
        # (
        #     "cship_category",
        #     "cat",
        # ),  # may overlap with pba_category and mobility_category
        # ("age_at_start", "num"),
        # ("grade_num", "num"),
        # ("mobility_category", "cat"),
        # ("covid_overlap", "num"),
        # ("birth_year", "num"),
        # ("defense_year", "num"),
        # ("pba_year", "num"),
        # ("d_acceptance", "num"),
        # ("d_pba", "num"),
        # ("birth_continent", "cat"),
        # ("birth_country", "cat"),
        ("institute_name", "cat"),  # XOR faculty/lecode/subject/subject_group # sum
        ("pba_category", "cat"),  # reference GER
        ("is_male", "cat"),  # reference True
        # ("pba_grade", "num"),
        # ("pba_grade_missing", "cat"),  # reference False
        # ("pba_type", "cat"),  # reference Konsekutives Mastersudium
    ]

    return_data = data[[c[0] for c in used_cols] + [target_col]].copy()

    for col in set(merge_cols) & set(return_data.columns):
        return_data[col] = merge_rare_categories(
            return_data[col],
            min_count=min_count,
            other_label="Other",
        )

    return return_data, used_cols


def explore_features(data, target_col, min_count=10, transform_function="log"):
    data = prefer_duration_computed(data, target_col)
    data = engineer_features(data, target_col)
    data = data[data[target_col].notna()]

    target_col_trafo = target_col + "_transformed"
    data[target_col_trafo], lam = transform_target(
        data[target_col], fun=transform_function
    )
    testing_cols = [
        ("degree_gap", "num"),
        # ("institute_name", "cat"),  # alternatively faculty/lecode/subject/subject_group
        ("faculty", "cat"),  # alternatively faculty/lecode/subject/subject_group
        ("lecode", "cat"),  # alternatively faculty/lecode/subject/subject_group
        ("phd_subject", "cat"),  # alternatively faculty/lecode/subject/subject_group
        (
            "subject_group_stala",
            "cat",
        ),  # alternatively faculty/lecode/subject/subject_group
        (
            "subject_group_category_stala",
            "cat",
        ),  # alternatively faculty/lecode/subject/subject_group
        ("cship_category", "cat"),
        ("pba_category", "cat"),
        ("age_at_start", "num"),
        ("is_male", "cat"),
        ("pba_grade", "num"),
        ("pba_grade_missing", "cat"),
        ("pba_type", "cat"),
        ("mobility_category", "cat"),
        ("covid_overlap", "num"),
        ("start_year", "num"),  # alternatively defense_year. different truncation modes
        (
            "defense_year",
            "num",
        ),  # alternatively defense_year. different truncation modes
    ]

    n_num_plots = len([c for c in testing_cols if c[1] == "num"])
    n_cat_plots = len([c for c in testing_cols if c[1] == "cat"])

    nnum_rows = int(np.ceil(n_num_plots / 2))
    nnum_cols = 2

    num_cols = [c[0] for c in testing_cols if c[1] == "num"]

    ncat_rows = 2
    ncat_cols = int(np.ceil(n_cat_plots / 2))

    cat_cols = [c[0] for c in testing_cols if c[1] == "cat"]

    num_corr = data[num_cols].corr()

    fig, axes = plt.subplots(
        nrows=nnum_rows, ncols=nnum_cols, figsize=(12, 4 * nnum_rows)
    )
    for ax, col in zip(axes.flatten(), num_cols):
        df = data[[col, target_col_trafo]].copy()

        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[target_col_trafo] = pd.to_numeric(df[target_col_trafo], errors="coerce")

        df = df.replace([np.inf, -np.inf], np.nan).dropna()

        x = df[col].astype(float).to_numpy()
        y = df[target_col_trafo].astype(float).to_numpy()

        pearson_r = df[col].corr(df[target_col_trafo], method="pearson")
        spearman_rho = df[col].corr(df[target_col_trafo], method="spearman")

        ols_object = smf.ols(f"{target_col_trafo} ~ {col}", data=df).fit()
        ols_coeff = ols_object.params[col]  # not params[1]
        r_squared = ols_object.rsquared

        ax.hexbin(x, y, gridsize=30, cmap="Blues")
        ax.set_xlabel(col)
        ax.set_ylabel(target_col_trafo)
        ax.set_title(
            f"{col} vs {target_col_trafo}\n"
            f"Pearson r={pearson_r:.3f}, Spearman ρ={spearman_rho:.3f}\n"
            f"OLS coeff={ols_coeff:.3f}, R²={r_squared:.3f}"
        )
        print(
            f"{col} vs {target_col_trafo}\n"
            f"Pearson r={pearson_r:.3f}, Spearman ρ={spearman_rho:.3f}\n"
            f"OLS coeff={ols_coeff:.3f}, R²={r_squared:.3f}"
        )

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.imshow(num_corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(num_cols)))
    ax.set_yticks(range(len(num_cols)))
    ax.set_xticklabels(num_cols, rotation=45, ha="right")
    ax.set_yticklabels(num_cols)
    ax.set_title("Correlation Matrix of Numerical Features")

    print("correlation matrix of numerical features:")
    print(num_corr.to_string())

    # hide unused subplot axes
    for ax in axes.flatten()[len(num_cols) :]:
        ax.axis("off")

    plt.tight_layout()

    fig, axes = plt.subplots(
        nrows=ncat_rows, ncols=ncat_cols, figsize=(12, 4 * ncat_rows)
    )

    for ax, col in zip(axes.flatten(), cat_cols):
        df = data[[col, target_col_trafo]].dropna().copy()
        df[col] = merge_rare_categories(
            df[col], min_count=min_count, other_label="Other"
        )

        df.boxplot(column=target_col_trafo, by=col, ax=ax)

        n_observations = df.groupby(col)[target_col_trafo].count()
        means = df.groupby(col)[target_col_trafo].mean()
        medians = df.groupby(col)[target_col_trafo].median()
        combined_stats = pd.DataFrame(
            {"mean": means, "median": medians, "n": n_observations}
        )
        print(combined_stats.to_string())

    for ax in axes.flatten()[len(cat_cols) :]:
        ax.axis("off")
    plt.tight_layout()

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.hist(data[target_col_trafo].dropna(), bins=30, color="blue", alpha=0.7)
    plt.show()


def regression_analysis(
    data, target_col, target_transform="log", min_count=10, merge_cols=[]
):
    data = data.copy()

    # data = data[data["institute_name"] == "Institut für Chemie"]

    data = prefer_duration_computed(data, target_col)
    data = engineer_features(data, target_col)
    data = data.loc[data[target_col].notna()].copy()

    data, used_cols = feature_selection(
        data, target_col, min_count=min_count, merge_cols=merge_cols
    )
    data = data.copy()

    target_col_trafo = target_col + "_transformed"
    data[target_col_trafo], lam = transform_target(
        data[target_col], fun=target_transform
    )

    num_cols = [c for c, typ in used_cols if typ == "num"]
    cat_cols = [c for c, typ in used_cols if typ == "cat"]

    for col in num_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce").astype("float64")

    data[target_col_trafo] = pd.to_numeric(
        data[target_col_trafo], errors="coerce"
    ).astype("float64")

    for col, typ in used_cols:
        if typ == "cat":
            data[col] = data[col].astype("object")
        else:
            data[col] = pd.to_numeric(data[col], errors="coerce").astype("float64")

    # Drop rows with any missing used feature so the design matrix matches the
    # strict complete-case model.
    data = data.dropna(subset=num_cols + cat_cols + [target_col_trafo]).copy()

    ols_string = f"{target_col_trafo} ~ " + " + ".join(
        [c if typ == "num" else f"C({c})" for c, typ in used_cols]
    )
    ols_string = f"{target_col_trafo} ~ " + " + ".join(
        [c if typ == "num" else f"C({c}, Sum)" for c, typ in used_cols]
    )

    # ols_string = f"{target_col_trafo} ~ "
    # ols_string = (
    #     f"{target_col_trafo} ~ " + "C(faculty, Treatment(reference='Fakultät I'))"
    # )
    # ols_string_with_interaction = ols_string + "+ C(institute_name):C(is_male)"
    #
    # ols_string += " + C(birth_year):C(pba_type)"

    model_hc3 = smf.ols(
        ols_string,
        data=data,
        missing="drop",
    ).fit(cov_type="HC3")
    #
    # # model_hc3_interaction = smf.ols(
    #     ols_string_with_interaction,
    #     data=data,
    #     missing="drop",
    # ).fit(cov_type="HC3")
    #
    # print("ANOVA")
    # print(anova_lm(model_hc3, model_hc3_interaction))
    #
    print(model_hc3.summary())
    print(feature_level_table(model_hc3, data=data).sort_values("p_adj"))

    inst_terms = {
        name: f"{name} = 0"
        for name in model_hc3.params.index
        if name.startswith("C(institute_name, Sum)[S.")
    }
    inst_ct = contrast_table(model_hc3, inst_terms)
    print(inst_ct.sort_values("p_adj"))

    regression_diagnostics(model_hc3, ols_string, data)


def regression_diagnostics(model, ols_string, data, plotting=False):
    # -------------------------
    # Cook's distance
    # -------------------------
    influence = OLSInfluence(model)
    cooks_d, cooks_p = influence.cooks_distance

    n = int(model.nobs)
    cooks_threshold = 4 / n

    cooks_df = pd.DataFrame(
        {
            "row_label": model.model.data.row_labels,
            "cooks_d": cooks_d,
            "cooks_p": cooks_p,
        }
    ).sort_values("cooks_d", ascending=False)

    print("\nCook's distance")
    print(f"Threshold 4/n = {cooks_threshold:.6f}")
    print(f"Number above threshold: {(cooks_d > cooks_threshold).sum()}")
    print(cooks_df.head(20).to_string(index=False))

    if plotting:
        plt.figure(figsize=(8, 4))
        plt.stem(np.arange(len(cooks_d)), cooks_d, markerfmt=",", basefmt=" ")
        plt.axhline(cooks_threshold, linestyle="--")
        plt.xlabel("Observation index in fitted model")
        plt.ylabel("Cook's distance")
        plt.title("Cook's distance")
        plt.tight_layout()
        plt.show()

    # -------------------------
    # Breusch-Pagan test
    # -------------------------
    bp_lm, bp_lm_pvalue, bp_fvalue, bp_f_pvalue = het_breuschpagan(
        model.resid,
        model.model.exog,
    )

    print("\nBreusch-Pagan test")
    print(f"LM statistic: {bp_lm:.3f}")
    print(f"LM p-value:   {bp_lm_pvalue:.4g}")
    print(f"F statistic:  {bp_fvalue:.3f}")
    print(f"F p-value:    {bp_f_pvalue:.4g}")

    if bp_lm_pvalue < 0.05:
        print("Evidence of heteroscedasticity; HC3 robust SEs are justified.")
    else:
        print("No strong evidence of heteroscedasticity from this test.")

    y, X = patsy.dmatrices(ols_string, data=data, return_type="dataframe")

    print("X shape:", X.shape)
    print("rank:", np.linalg.matrix_rank(X))
    print("deficiency:", X.shape[1] - np.linalg.matrix_rank(X))

    zero_cols = X.columns[(X == 0).all()]
    print("zero columns:")
    print(zero_cols.tolist())

    duplicate_cols = X.T.duplicated()
    print("duplicate columns:")
    print(X.columns[duplicate_cols].tolist())

    # design matrix VIF

    vif = pd.DataFrame(
        {
            "term": X.columns,
            "vif": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])],
        }
    )
    print(vif.sort_values("vif", ascending=False).head(20))

    resid = model.resid
    fitted = model.fittedvalues

    if plotting:
        plt.figure()

        plt.scatter(fitted, resid, alpha=0.4)
        plt.axhline(0)

        sm.qqplot(resid, line="s")
        plt.show()

    return cooks_df


def category_contrast_analysis(
    data,
    target_col,
    target_transform="log",
    min_count=10,
    merge_cols=(),
    cov_type="HC3",
    output_dir=None,
):
    """Fit the current categorical model and compare controlled vs solo contrasts.

    The full model uses the following terms:

    - ``institute_name`` with sum coding
    - ``pba_category`` with GER as reference
    - ``is_male`` with True as reference
    - ``pba_grade`` as numeric
    - ``pba_grade_missing`` with False as reference
    - ``pba_type`` with ``Konsekutives Masterstudium`` as reference

    For each feature, we fit:

    - a full model controlling for the other features
    - a solo model with only that feature

    The result is one combined contrast table with BH-adjusted p-values within
    each feature and model kind.
    """

    output_path = (
        Path(output_dir)
        if output_dir is not None
        else DATA_DIR / "regression_contrasts"
    )
    output_path.mkdir(parents=True, exist_ok=True)

    specs = [
        {
            "feature": "institute_name",
            "term": categorical_term("institute_name", coding="Sum"),
            "kind": "cat",
        },
        {
            "feature": "pba_category",
            "term": categorical_term(
                "pba_category", coding="Treatment", reference="GER"
            ),
            "kind": "cat",
        },
        {
            "feature": "is_male",
            "term": categorical_term("is_male", coding="Treatment", reference=True),
            "kind": "cat",
        },
        {"feature": "pba_grade", "term": "pba_grade", "kind": "num"},
        {
            "feature": "pba_grade_missing",
            "term": categorical_term(
                "pba_grade_missing", coding="Treatment", reference=False
            ),
            "kind": "cat",
        },
        # {
        #     "feature": "pba_type",
        #     "term": categorical_term(
        #         "pba_type",
        #         coding="Treatment",
        #         reference="Konsekutives Masterstudium",
        #     ),
        #     "kind": "cat",
        # },
    ]

    data = prefer_duration_computed(data, target_col)
    data = engineer_features(data, target_col)
    data = data.loc[data[target_col].notna()].copy()

    needed_cols = [target_col, *[spec["feature"] for spec in specs]]
    missing_cols = [col for col in needed_cols if col not in data.columns]
    if missing_cols:
        raise KeyError(f"Missing columns for contrast analysis: {missing_cols}")

    target_col_trafo = f"{target_col}_transformed"
    data[target_col_trafo], _ = transform_target(data[target_col], fun=target_transform)
    data[target_col_trafo] = pd.to_numeric(
        data[target_col_trafo], errors="coerce"
    ).astype("float64")

    full_needed_cols = [target_col_trafo, *[spec["feature"] for spec in specs]]
    for spec in specs:
        feature = spec["feature"]
        if spec["kind"] == "num":
            data[feature] = pd.to_numeric(data[feature], errors="coerce").astype(
                "float64"
            )
        else:
            data[feature] = data[feature].astype("object")
            if feature in merge_cols:
                data[feature] = merge_rare_categories(
                    data[feature], min_count=min_count, other_label="Other"
                )
    data = data.dropna(subset=full_needed_cols).copy()

    full_frame, full_rhs_terms = prepare_contrast_frame(
        data,
        target_col_trafo=target_col_trafo,
        spec_subset=specs,
        merge_cols=merge_cols,
        min_count=min_count,
    )
    full_model, full_formula = fit_contrast_model(
        full_frame,
        full_rhs_terms,
        target_col_trafo=target_col_trafo,
        cov_type=cov_type,
    )

    full_contrasts = []
    for spec in specs:
        contrast_df = contrast_rows_for_feature(full_model, spec["term"])
        if contrast_df.empty:
            continue
        contrast_df = contrast_df.copy()
        contrast_df["feature"] = spec["feature"]
        contrast_df["term"] = spec["term"]
        contrast_df["model_kind"] = "full"
        contrast_df["formula"] = full_formula
        contrast_df["nobs"] = int(full_model.nobs)
        contrast_df["feature_kind"] = spec["kind"]
        full_contrasts.append(contrast_df)

    solo_contrasts = []
    solo_summaries = []
    for spec in specs:
        solo_frame, solo_rhs_terms = prepare_contrast_frame(
            data,
            target_col_trafo=target_col_trafo,
            spec_subset=[spec],
            merge_cols=merge_cols,
            min_count=min_count,
        )
        solo_model, solo_formula = fit_contrast_model(
            solo_frame,
            solo_rhs_terms,
            target_col_trafo=target_col_trafo,
            cov_type=cov_type,
        )
        contrast_df = contrast_rows_for_feature(solo_model, spec["term"])
        if contrast_df.empty:
            continue
        contrast_df = contrast_df.copy()
        contrast_df["feature"] = spec["feature"]
        contrast_df["term"] = spec["term"]
        contrast_df["model_kind"] = "solo"
        contrast_df["formula"] = solo_formula
        contrast_df["nobs"] = int(solo_model.nobs)
        contrast_df["feature_kind"] = spec["kind"]
        solo_contrasts.append(contrast_df)
        solo_summaries.append(
            {
                "model_kind": "solo",
                "feature": spec["feature"],
                "formula": solo_formula,
                "nobs": int(solo_model.nobs),
                "r2": float(solo_model.rsquared),
                "adj_r2": float(solo_model.rsquared_adj),
                "aic": float(solo_model.aic),
                "bic": float(solo_model.bic),
            }
        )

    contrast_frames = [*full_contrasts, *solo_contrasts]
    contrasts = (
        pd.concat(contrast_frames, ignore_index=True)
        if contrast_frames
        else pd.DataFrame()
    )
    if not contrasts.empty:
        contrasts = contrasts.sort_values(
            ["model_kind", "feature", "p_adj", "contrast"],
            ascending=[True, True, True, True],
        )
        contrasts.to_csv(
            output_path / f"{target_col}__category_contrasts.csv", index=False
        )

    model_summary = pd.DataFrame(
        [
            {
                "model_kind": "full",
                "formula": full_formula,
                "nobs": int(full_model.nobs),
                "r2": float(full_model.rsquared),
                "adj_r2": float(full_model.rsquared_adj),
                "aic": float(full_model.aic),
                "bic": float(full_model.bic),
            }
        ]
        + solo_summaries
    )
    model_summary.to_csv(
        output_path / f"{target_col}__category_model_summary.csv", index=False
    )

    # print(full_model.summary())
    # print(contrasts.to_string(index=False))
    # print(model_summary.to_string(index=False))

    return {
        "full_model": full_model,
        "full_formula": full_formula,
        "contrasts": contrasts,
        "model_summary": model_summary,
        "output_dir": output_path,
    }


def contrast_level_from_label(contrast: str) -> str | None:
    """Extract a level label from a patsy contrast name."""
    match = re.search(r"\[(?:T|S)\.(.*)\]$", contrast)
    if not match:
        return None
    return match.group(1)


def percent_vs_reference(value: float, reference: float) -> float:
    """Return the percent delta relative to a reference value."""
    if not np.isfinite(value) or not np.isfinite(reference) or reference == 0:
        return np.nan
    return (value - reference) / abs(reference) * 100.0


def model_percent_from_effect(effect: float, target_transform: str) -> float:
    """Convert a transformed effect into a percent change."""
    if not np.isfinite(effect):
        return np.nan
    if target_transform == "log":
        return np.expm1(effect) * 100.0
    return np.nan


def model_estimate_and_ci_from_effect(
    effect: float,
    se: float,
    reference_mean: float,
    target_transform: str,
) -> tuple[float, float, float]:
    """Back-transform a contrast into an original-scale estimate and CI."""
    if (
        not np.isfinite(effect)
        or not np.isfinite(se)
        or not np.isfinite(reference_mean)
    ):
        return np.nan, np.nan, np.nan
    if target_transform != "log":
        return np.nan, np.nan, np.nan

    z = stats.norm.ppf(0.975)
    estimate = reference_mean * np.exp(effect)
    ci_low = reference_mean * np.exp(effect - z * se)
    ci_high = reference_mean * np.exp(effect + z * se)
    return float(estimate), float(ci_low), float(ci_high)


def slugify_for_filename(text: str) -> str:
    """Return a filesystem-safe filename stem."""
    slug = re.sub(r"[^A-Za-z0-9]+", "_", text.strip())
    slug = slug.strip("_")
    return slug or "feature"


def display_category_label(feature: str, label: str) -> str:
    """Return a readable display label for a category level."""
    if feature == "institute_name":
        label = re.sub(r"^(Institut für|Zentrum für)\s+", "", label)
    return label


def display_category_label_with_replacements(
    feature: str, label: str, replacements: dict | None = None
) -> str:
    """Apply feature-specific and caller-provided label replacements."""
    display = display_category_label(feature, label)
    if replacements:
        display = replacements.get(display, display)
    return display


def sig_stars(p_value: float) -> str:
    """Return standard significance stars for a p-value."""
    if not np.isfinite(p_value):
        return ""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def reference_line_label(feature: str, ref_display_label: str) -> str:
    """Return the label used for the reference line in a plot."""
    if ref_display_label == "Grand mean":
        return "Gesamtdurchschnitt"
    return f"Durchschnitt {ref_display_label}"


def prepare_plot_target(
    plot_data: pd.DataFrame,
    target_col: str,
    augment_for_plotting: bool,
) -> pd.DataFrame:
    """Prepare the target column used by the violin plots."""
    plot_data = plot_data.copy()
    if augment_for_plotting and target_col != "duration_computed":
        if target_col not in plot_data.columns:
            duration_source = pd.read_csv(DATA_DIR / "duration_predictions.csv")
            plot_data = plot_data.merge(
                duration_source[["pagination_nr", "duration_estimate_clipped"]],
                on="pagination_nr",
                how="left",
            )
        augmented_target = pd.to_numeric(plot_data[target_col], errors="coerce")
        if (
            target_col != "duration_estimate_clipped"
            and "duration_estimate_clipped" in plot_data.columns
        ):
            fill_mask = (
                augmented_target.isna() & plot_data["duration_estimate_clipped"].notna()
            )
            augmented_target.loc[fill_mask] = plot_data.loc[
                fill_mask, "duration_estimate_clipped"
            ]
        plot_data["_plot_target"] = augmented_target
    else:
        plot_data["_plot_target"] = pd.to_numeric(
            plot_data[target_col], errors="coerce"
        )
    return plot_data


def dense_violin_style(
    n_levels: int,
    *,
    min_levels: int = 8,
    dense_width: float = 0.78,
    force_dense: bool = False,
) -> tuple[float, str | None]:
    """Return violin width and optional side for dense category panels."""
    if force_dense or n_levels >= min_levels:
        return dense_width, "positive"
    return 0.34, None


def dense_box_style(
    n_levels: int,
    *,
    min_levels: int = 8,
    dense_width: float = 0.18,
) -> float:
    """Return box width for dense category panels."""
    if n_levels >= min_levels:
        return dense_width
    return 0.08


def build_violin_trace(
    fig: go.Figure,
    values: pd.Series,
    x_position: float,
    name: str,
    color: str,
    width: float,
    opacity: float,
    side: str | None = None,
    hover_label: str | None = None,
) -> None:
    """Add a violin trace to a figure."""
    clean = pd.to_numeric(values, errors="coerce").dropna()
    hover_label = hover_label or name
    violin_kwargs = dict(
        x=[x_position] * len(clean),
        y=clean,
        name=name,
        line=dict(color=color, width=2),
        fillcolor=color,
        opacity=opacity,
        width=width,
        box_visible=False,
        meanline_visible=False,
        points=False,
        spanmode="hard",
        scalemode="width",
        hoveron="kde",
        customdata=[hover_label] * len(clean),
        hovertemplate="%{customdata}<br>Wert: %{y:.2f}<extra></extra>",
        yhoverformat=".2f",
    )
    if side is not None:
        violin_kwargs["side"] = side

    fig.add_trace(go.Violin(**violin_kwargs))


def build_box_trace(
    fig: go.Figure,
    values: pd.Series,
    x_position: float,
    name: str,
    color: str,
    width: float,
    line_width: float,
) -> None:
    """Add a box trace to a figure when data is available."""
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return

    fig.add_trace(
        go.Box(
            x=[x_position] * len(clean),
            y=clean,
            name=name,
            width=width,
            fillcolor="rgba(255,255,255,0.0)",
            line=dict(color=color, width=line_width),
            marker=dict(opacity=0),
            showlegend=False,
            boxpoints=False,
            boxmean=True,
            hovertemplate="%{name}<br>Wert: %{y:.2f}<extra></extra>",
        )
    )


def plot_category_contrast_violins(
    data,
    target_col,
    target_transform="log",
    min_count=10,
    merge_cols=(),
    cov_type="HC3",
    output_dir=None,
    plot_output_dir=None,
):
    """Create violin plots for non-significant category levels from contrast tables.

    The function first recomputes the contrast tables via
    :func:`category_contrast_analysis`, then loads the saved CSV and builds one
    violin plot per categorical feature and model kind.

    The plot data are always shown on the original target scale. If the plotting
    config enables augmentation and ``target_col`` is ``duration_computed``, we
    fill missing target values only for plotting with ``duration_estimate_clipped``
    from ``duration_predictions.csv``.
    """

    result = category_contrast_analysis(
        data,
        target_col=target_col,
        target_transform=target_transform,
        min_count=min_count,
        merge_cols=merge_cols,
        cov_type=cov_type,
        output_dir=output_dir,
    )

    output_path = result["output_dir"]
    contrast_path = output_path / f"{target_col}__category_contrasts.csv"
    contrasts = pd.read_csv(contrast_path)

    plot_params, params_global, _ = load_plot_config("catcontrast_violins")
    feature_label_names = plot_params.get("feature_label_names", {})
    omitted_features = set(plot_params.get("omitted_features", []))
    augment_for_plotting = bool(plot_params.get("augment_for_plotting", False))
    significance_col = plot_params.get("significance_column", "p_adj")
    significance_level = float(plot_params.get("significance_level", 0.05))
    cutoff_years = float(plot_params.get("cutoff_years", 12))
    dense_violin_min_levels = int(plot_params.get("dense_violin_min_levels", 8))
    dense_violin_width = float(plot_params.get("dense_violin_width", 0.78))
    dense_box_width = float(plot_params.get("dense_box_width", 0.18))
    always_dense_violin_style = bool(
        plot_params.get("always_dense_violin_style", False)
    )
    show_forest_overlay = bool(plot_params.get("show_forest_overlay", True))
    forest_dense_offset = float(plot_params.get("forest_dense_offset", 0.24))
    category_replacements = plot_params.get("category_replacements", {})
    color_order_by_feature = plot_params.get("color_order", {})

    plot_data = prefer_duration_computed(data, target_col)
    plot_data = engineer_features(plot_data, target_col)
    plot_data = prepare_plot_target(plot_data, target_col, augment_for_plotting)

    specs = [
        {
            "feature": "institute_name",
            "term": categorical_term("institute_name", coding="Sum"),
            "kind": "cat",
            "reference": "Alle Institute",
            "coding": "Sum",
        },
        {
            "feature": "pba_category",
            "term": categorical_term(
                "pba_category", coding="Treatment", reference="GER"
            ),
            "kind": "cat",
            "reference": "GER",
            "coding": "Treatment",
        },
        {
            "feature": "is_male",
            "term": categorical_term("is_male", coding="Treatment", reference=True),
            "kind": "cat",
            "reference": "True",
            "coding": "Treatment",
        },
        # {
        #     "feature": "pba_grade",
        #     "term": "pba_grade",
        #     "kind": "num",
        #     "reference": None,
        #     "coding": None,
        # },
        # {
        #     "feature": "pba_grade_missing",
        #     "term": categorical_term(
        #         "pba_grade_missing", coding="Treatment", reference=False
        #     ),
        #     "kind": "cat",
        #     "reference": "False",
        #     "coding": "Treatment",
        # },
        # {
        #     "feature": "pba_type",
        #     "term": categorical_term(
        #         "pba_type",
        #         coding="Treatment",
        #         reference="Konsekutives Masterstudium",
        #     ),
        #     "kind": "cat",
        #     "reference": "Konsekutives Masterstudium",
        #     "coding": "Treatment",
        # },
    ]

    plotted_features = [
        spec
        for spec in specs
        if spec["kind"] == "cat" and spec["feature"] not in omitted_features
    ]

    out_dir = (
        Path(plot_output_dir)
        if plot_output_dir is not None
        else Path(__file__).resolve().parents[2] / "int_plots"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    def plot_feature_kind(feature: str, model_kind: str) -> None:
        spec = next(item for item in plotted_features if item["feature"] == feature)
        feature_contrasts = contrasts[
            (contrasts["feature"] == feature) & (contrasts["model_kind"] == model_kind)
        ].copy()
        if feature_contrasts.empty:
            return

        if significance_col not in feature_contrasts.columns:
            raise KeyError(f"Missing significance column {significance_col!r}")

        sig = feature_contrasts.loc[
            feature_contrasts[significance_col].le(significance_level)
        ].copy()

        feature_series = plot_data[["_plot_target", feature]].dropna().copy()
        if feature in merge_cols:
            feature_series[feature] = merge_rare_categories(
                feature_series[feature], min_count=min_count, other_label="Other"
            )
        replacements = category_replacements.get(feature, {})
        feature_series[feature] = feature_series[feature].astype("object")
        feature_series["_level_key"] = feature_series[feature].astype(str)

        if spec["reference"] == "Alle Institute":
            ref_values = feature_series["_plot_target"]
            ref_label = "Alle Institute"
            ref_mean = float(pd.to_numeric(ref_values, errors="coerce").dropna().mean())
        else:
            ref_values = feature_series.loc[
                feature_series["_level_key"] == str(spec["reference"]), "_plot_target"
            ]
            ref_label = str(spec["reference"])
            ref_mean = float(pd.to_numeric(ref_values, errors="coerce").dropna().mean())
        ref_display_label = display_category_label_with_replacements(
            feature, ref_label, replacements
        )
        ref_line_label = reference_line_label(feature, ref_display_label)

        level_rows = []
        for _, row in sig.iterrows():
            level = contrast_level_from_label(str(row["contrast"]))
            if level is None:
                continue
            if str(level) == "Other":
                continue
            level_values = feature_series.loc[
                feature_series["_level_key"] == str(level), "_plot_target"
            ]
            if level_values.empty:
                continue
            level_mean = float(
                pd.to_numeric(level_values, errors="coerce").dropna().mean()
            )
            pct = percent_vs_reference(level_mean, ref_mean)
            model_pct = model_percent_from_effect(
                float(row["effect"]), target_transform
            )
            estimate, ci_low, ci_high = model_estimate_and_ci_from_effect(
                float(row["effect"]),
                float(row["se"]),
                ref_mean,
                target_transform,
            )
            level_rows.append(
                {
                    "level": str(level),
                    "values": level_values,
                    "mean": level_mean,
                    "pct": pct,
                    "model_pct": model_pct,
                    "estimate": estimate,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "p_adj": float(row[significance_col]),
                }
            )

        if not level_rows:
            return

        left = sorted(
            [row for row in level_rows if np.isfinite(row["pct"]) and row["pct"] < 0],
            key=lambda row: row["pct"],
        )
        right = sorted(
            [row for row in level_rows if np.isfinite(row["pct"]) and row["pct"] >= 0],
            key=lambda row: row["pct"],
        )
        ordered = (
            left
            + [
                {
                    "level": ref_display_label,
                    "values": ref_values,
                    "mean": ref_mean,
                    "pct": 0.0,
                    "model_pct": 0.0,
                    "estimate": ref_mean,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "p_adj": np.nan,
                    "is_reference": True,
                }
            ]
            + right
        )

        x_positions = list(range(-len(left), 0)) + [0] + list(range(1, len(right) + 1))
        feature_color_order = color_order_by_feature.get(feature, [])
        non_reference_rows = [row for row in ordered if not row.get("is_reference")]
        ordered_color_levels = ordered_with_preference(
            [row["level"] for row in non_reference_rows],
            feature_color_order,
            rest_label="__no_rest__",
        )
        color_by_level = {
            level: params_global["color_cycle"][idx % len(params_global["color_cycle"])]
            for idx, level in enumerate(ordered_color_levels)
        }
        all_values = pd.concat(
            [pd.to_numeric(row["values"], errors="coerce") for row in ordered],
            ignore_index=True,
        ).dropna()
        y_max = float(all_values.max()) if not all_values.empty else np.nan
        y_min = float(all_values.min()) if not all_values.empty else np.nan
        y_range = None
        if np.isfinite(y_max) and y_max > cutoff_years:
            lower = 0.0 if not np.isfinite(y_min) else min(0.0, y_min)
            y_range = [lower, cutoff_years]

        fig = go.Figure()
        forest_rows = [row for row in ordered if not row.get("is_reference")]
        for idx, (x_pos, row) in enumerate(zip(x_positions, ordered, strict=True)):
            label_pct = row["pct"]
            label_model_pct = row.get("model_pct", np.nan)
            pct_txt = "0.0%" if not np.isfinite(label_pct) else f"{label_pct:+.1f}%"
            model_pct_txt = (
                ""
                if not np.isfinite(label_model_pct)
                else f" [{label_model_pct:+.1f}%]"
            )
            display_level = display_category_label_with_replacements(
                feature, str(row["level"]), replacements
            )
            stars = (
                ""
                if row.get("is_reference")
                else sig_stars(float(row.get("p_adj", np.nan)))
            )
            display_label = f"{display_level}{(' ' + stars) if stars else ''}<br>{pct_txt}{model_pct_txt}"
            hover_label = f"{display_level}{(' ' + stars) if stars else ''}"
            if row.get("is_reference"):
                color = "#9a9a9a"
            elif feature in color_order_by_feature:
                color = color_by_level[row["level"]]
            else:
                color = params_global["color_cycle"][idx % len(params_global["color_cycle"])]
            opacity = 0.36 if row.get("is_reference") else 0.58
            width, side = dense_violin_style(
                len(ordered),
                min_levels=dense_violin_min_levels,
                dense_width=dense_violin_width,
                force_dense=always_dense_violin_style,
            )

            build_violin_trace(
                fig,
                row["values"],
                x_pos,
                display_label,
                color,
                width,
                opacity,
                side=side,
                hover_label=hover_label,
            )
            build_box_trace(
                fig,
                row["values"],
                x_pos,
                display_label,
                color,
                dense_box_style(
                    len(ordered),
                    min_levels=dense_violin_min_levels,
                    dense_width=dense_box_width,
                ),
                2.5 if row.get("is_reference") else 2.0,
            )

        forest_offset = (
            forest_dense_offset if len(ordered) >= dense_violin_min_levels else 0.0
        )
        forest_x = [
            x_pos + forest_offset
            for x_pos, row in zip(x_positions, ordered, strict=True)
            if not row.get("is_reference")
        ]
        forest_y = [float(row["estimate"]) for row in forest_rows]
        forest_err_minus = [
            float(row["estimate"] - row["ci_low"]) for row in forest_rows
        ]
        forest_err_plus = [
            float(row["ci_high"] - row["estimate"]) for row in forest_rows
        ]
        forest_customdata = np.column_stack(
            [
                [row["ci_low"] for row in forest_rows],
                [row["ci_high"] for row in forest_rows],
                [row["p_adj"] for row in forest_rows],
            ]
        )
        forest_colors = [
            "#E15759" if float(row["p_adj"]) < significance_level else "#9a9a9a"
            for row in forest_rows
        ]
        if show_forest_overlay:
            fig.add_trace(
                go.Scatter(
                    x=forest_x,
                    y=forest_y,
                    mode="markers",
                    marker=dict(
                        size=7,
                        color=forest_colors,
                        line=dict(color="#FFFFFF", width=0.8),
                    ),
                    error_y=dict(
                        type="data",
                        array=forest_err_plus,
                        arrayminus=forest_err_minus,
                        visible=True,
                        color="#333333",
                        thickness=1.4,
                    ),
                    customdata=forest_customdata,
                    hovertemplate=(
                        "Modellschätzer: %{y:.2f}<br>"
                        "95% CI: [%{customdata[0]:.2f}, %{customdata[1]:.2f}]<br>"
                        "p_adj: %{customdata[2]:.4g}<extra></extra>"
                    ),
                    showlegend=False,
                    cliponaxis=False,
                )
            )

        fig.add_trace(
            go.Scatter(
                x=np.linspace(x_positions[0] - 0.6, x_positions[-1] + 0.6, 80),
                y=[ref_mean] * 80,
                mode="lines+markers",
                line=dict(color="black", width=1, dash="dash"),
                marker=dict(size=10, color="rgba(0,0,0,0.001)"),
                opacity=0.75,
                hovertemplate=f"{ref_line_label}: {ref_mean:.2f}<extra></extra>",
                showlegend=False,
                name=ref_line_label,
                cliponaxis=False,
            )
        )

        ticktext = [
            f"{display_category_label_with_replacements(feature, str(row['level']), replacements)}"
            f"{'' if row.get('is_reference') else (' ' + sig_stars(float(row.get('p_adj', np.nan))))}"
            f"<br>{'0.0%' if not np.isfinite(row['pct']) else f'{row['pct']:+.1f}%'}"
            f"{'' if not np.isfinite(row.get('model_pct', np.nan)) else f' [{row['model_pct']:+.1f}%]'}"
            for row in ordered
        ]

        fig.update_layout(
            violinmode="group",
            showlegend=False,
            hovermode="closest",
            violingap=0.02,
            violingroupgap=0.02,
            margin=dict(l=60, r=30, t=35, b=90),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(
                title="",
                tickmode="array",
                tickvals=x_positions,
                ticktext=ticktext,
                tickangle=45 if feature == "institute_name" else 0,
                automargin=True,
            ),
            yaxis=dict(
                title=plot_params.get("y_label", "Dauer"),
                showgrid=True,
                gridcolor="rgba(0,0,0,0.15)",
                zeroline=False,
                range=y_range,
                automargin=True,
            ),
        )

        feature_display_name = str(feature_label_names.get(feature, feature))
        out_name = (
            f"{target_col}__{model_kind}__{feature}__"
            f"{slugify_for_filename(feature_display_name)}.html"
        )
        write_html(
            fig,
            out_dir / model_kind / out_name,
            trace_map=None,
            plot_type="violin_contrast",
        )

    for spec in plotted_features:
        for model_kind in ("full", "solo"):
            plot_feature_kind(spec["feature"], model_kind)

    return {
        "analysis": result,
        "contrasts": contrasts,
        "plot_dir": out_dir,
    }


def visualize_feature_effects(
    *,
    target_cols,
    comparison_blocks,
    output_dir=None,
):
    """Load previously computed feature-comparison tables from CSV.

    The signature mirrors :func:`compare_feature_blocks_grid` so callers can
    use the same configuration object for either recomputation or plotting.

    Returns
    -------
    dict
        Nested mapping ``results[target_col][table_name] = DataFrame``.
    """

    output_path = (
        Path(output_dir)
        if output_dir is not None
        else DATA_DIR / "regression_feature_selection"
    )

    tables = {}
    for target_col in target_cols:
        target_tables = {}
        for block in comparison_blocks:
            if len(block) == 2:
                table_name, _ = block
            elif len(block) == 3:
                table_name, _, _ = block
            else:
                raise ValueError("comparison_blocks entries must have 2 or 3 items")
            table_path = output_path / f"{target_col}__{table_name}.csv"
            if not table_path.exists():
                raise FileNotFoundError(f"Missing comparison table: {table_path}")
            target_tables[table_name] = pd.read_csv(table_path)
        tables[target_col] = target_tables

    target_keys = list(tables.keys())
    for key in target_keys:
        tbs = tables[key]

    return tables
