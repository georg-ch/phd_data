"""Duration estimation and evaluation helpers.

Active workflow
---------------
1. ``engineer_features`` derives the feature set used by the current model.
2. ``find_parameters`` tunes CatBoost for ``RMSEWithUncertainty``.
3. ``predict_durations`` fits that model on all labeled rows with core timing
   features present, predicts the mean and variance for every row, and writes a
   post-processed output file.
4. ``cross_validated_model_estimates`` repeats the same prediction and
   post-processing logic out of fold for evaluation.

Modeling choice
---------------
The current pipeline uses only ``RMSEWithUncertainty``. Its mean prediction is
treated as the canonical duration estimate. The estimated variance is used to
construct raw confidence intervals and to derive an uncertainty score for
filtering.

Post-processing summary
-----------------------
``postprocess_predictions`` applies the domain-specific rules that make the raw
model output usable:

- convert model variance into a raw Gaussian-style interval
- clip the estimate and interval to the feasible duration range implied by
  ``d_acceptance`` and ``d_pba`` with a one-month buffer
- handle impossible clipped intervals by either:
  - collapsing narrow-window rows to the midpoint between acceptance and PBA
  - or falling back to the raw interval when constraints still conflict
- optionally replace estimates in small-gap rows with the naive midpoint
  baseline, because that rule is empirically very strong in that regime

The result is that the exported ``duration_estimate`` is not just the raw model
mean. It is the uncertainty-model mean after the project-specific feasibility
and hybridization rules have been applied.
"""

import pandas as pd
import json
import matplotlib.pyplot as plt

from catboost import CatBoostRegressor, Pool, cv
from sklearn.model_selection import KFold, train_test_split
from hyperopt import fmin, tpe, hp, STATUS_OK
from hyperopt.pyll import scope
import numpy as np
from pathlib import Path


def target_split(labeled_data, target_col_name):
    """
    Split the labeled data into features (X) and target (y) based on the target column name.
    """
    return (
        labeled_data.drop(columns=[target_col_name]),
        labeled_data[target_col_name],
    )


def prepare_data_raw(
    data, target_col_name, feature_col_info, dropna=("d_pba", "d_acceptance")
):
    """
    Prepare the labeled training table for CatBoost.

    This is the active training-data path. It keeps categorical columns in raw
    form, converts them to object dtype, and replaces missing categories with a
    dedicated ``"__MISSING__"`` level so CatBoost can handle them natively.

    Rows with missing target values are removed. Rows missing any columns named
    in ``dropna`` are also removed, because those features are treated as core
    inputs for the current model.
    """
    feature_col_names = [name for name, _is_cat in feature_col_info]
    labeled = data[feature_col_names + [target_col_name]].copy()
    labeled = labeled[labeled[target_col_name].notna()].copy()

    na_mask = pd.Series(False, index=labeled.index)
    for col in dropna:
        if col in labeled.columns:
            na_mask = na_mask | labeled[col].isna()
    labeled = labeled[~na_mask]

    # Ensure categoricals are objects and have no pd.NA
    cat_cols = [name for name, is_cat in feature_col_info if is_cat]
    for c in cat_cols:
        labeled[c] = labeled[c].astype("object")
        labeled[c] = labeled[c].where(labeled[c].notna(), "__MISSING__")

    return labeled, cat_cols


def catboost_paramsearch(
    labeled_data: pd.DataFrame,
    cat_features: list[str] | None,
    N=6000,
    ES=200,
    target_col_name="duration_computed",
    loss_function="RMSE",
):
    """Tune CatBoost hyperparameters with CatBoost CV and Hyperopt.

    Only the active CatBoost path is kept here. The search objective is the
    fold-mean validation loss reported by CatBoost CV. For the current
    production workflow, this is used with ``RMSEWithUncertainty``.
    """

    SPACE = {
        "depth": scope.int(hp.quniform("depth", 2, 8, 1)),
        "learning_rate": hp.loguniform("learning_rate", np.log(0.005), np.log(0.2)),
        "subsample": hp.uniform("subsample", 0.5, 1.0),
        "rsm": hp.uniform("rsm", 0.5, 1.0),
        "l2_leaf_reg": hp.loguniform("l2_leaf_reg", np.log(1e-2), np.log(1e2)),
    }
    FIXED = {
        "task_type": "CPU",
        # "bootstrap_type": "Bernoulli",
        # "boosting_type": "Ordered",
        "random_strength": 1.0,
        "one_hot_max_size": 8,
    }
    if loss_function == "RMSE":
        FIXED["loss_function"] = "RMSE"
        test_result_col = "test-RMSE-mean"
    elif loss_function == "RMSEWithUncertainty":
        FIXED["loss_function"] = "RMSEWithUncertainty"
        # FIXED["boosting_type"] = "Plain"
        # FIXED["bootstrap_type"] = "Bernoulli"
        FIXED["one_hot_max_size"] = 8
        test_result_col = "test-RMSEWithUncertainty-mean"
    else:
        raise ValueError(f"Unknown loss_function: {loss_function}")

    X, y = target_split(labeled_data, target_col_name)
    base_pool = Pool(X, y, cat_features=cat_features)

    def objective(params):
        params = {**FIXED, **params}
        cv_res = cv(
            pool=base_pool,
            params=params,
            fold_count=5,
            iterations=N,
            early_stopping_rounds=ES,
            verbose=False,
            partition_random_seed=42,
        )
        best = float(cv_res[test_result_col].iloc[-1])
        return {"loss": best, "status": STATUS_OK}

    best_params = fmin(
        fn=objective,
        space=SPACE,
        algo=tpe.suggest,
        max_evals=80,
        rstate=np.random.default_rng(42),
    )
    if not best_params:
        raise ValueError("Hyperopt did not return any best parameters.")
    final_params = {
        **FIXED,
        "depth": int(best_params["depth"]),
        "learning_rate": float(best_params["learning_rate"]),
        "subsample": float(best_params["subsample"]),
        "rsm": float(best_params["rsm"]),
        "l2_leaf_reg": float(best_params["l2_leaf_reg"]),
    }
    print("Final CatBoost params:", final_params)
    return final_params


def catboost_cv(
    labeled_data,
    target_col_name,
    cat_features=None,
    params={},
    N=20000,
    ES=200,
    random_seed=1,
    fold_count=10,
    feature_importance=True,
    loss_function="RMSE",
    test_size=0.25,
):
    """Estimate the best iteration count for a fitted parameter set.

    The main output is the CV-best iteration index, which is then stored in the
    final CatBoost parameter dictionary. The optional feature-importance branch
    performs a separate train/validation fit only for diagnostics.
    """

    X, y = target_split(labeled_data, target_col_name)
    pool = Pool(X, y, cat_features=cat_features)

    cv_data = cv(
        pool,
        params,
        fold_count=fold_count,
        iterations=N,
        early_stopping_rounds=ES,
        verbose=False,
        partition_random_seed=random_seed,
    )
    best_iter = cv_data[f"test-{loss_function}-mean"].idxmin()
    mean_rmse = cv_data.loc[best_iter, f"test-{loss_function}-mean"]
    std_rmse = cv_data.loc[best_iter, f"test-{loss_function}-std"]

    print(f"_Best iteration_: {best_iter}")
    print(f"CV {loss_function}: {mean_rmse:.3f} ± {std_rmse:.3f}")

    if feature_importance:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_seed
        )
        train_pool = Pool(X_train, y_train, cat_features=cat_features)
        valid_pool = Pool(X_test, y_test, cat_features=cat_features)

        cb_final = CatBoostRegressor(**params)
        cb_final.fit(
            train_pool,
            eval_set=valid_pool,
            use_best_model=True,
            early_stopping_rounds=ES,
            verbose=False,
        )
        print("Best iter:", cb_final.get_best_iteration())
        print(
            f"Val {loss_function}:",
            cb_final.get_best_score()["validation"][loss_function],
        )
        fi = cb_final.get_feature_importance(data=train_pool, type="LossFunctionChange")
        fi_df = pd.DataFrame({"feature": X.columns, "importance": fi}).sort_values(
            "importance", ascending=False
        )
        fi_df.to_csv("data/_feature_importance.csv", index=False)
        print(fi_df.head(20))

        return best_iter


def train_catboost(
    labeled_data, target_col_name, cat_features, params, loss_function="RMSE"
):
    """Fit a CatBoost regressor using the provided parameter dictionary."""
    assert params.get("loss_function") == loss_function, (
        "Params loss_function must match the one specified in the argument."
    )
    X, y = target_split(labeled_data, target_col_name)
    train_pool = Pool(X, y, cat_features=cat_features)

    model = CatBoostRegressor(**params)
    model.fit(
        train_pool,
        verbose=False,
    )

    print("Best iteration:", model.get_best_iteration())
    print("Best training RMSE:", model.get_best_score()["learn"][loss_function])
    return model


def sanitize_cat_cols(df, feature_col_info):
    """Normalize categorical columns for CatBoost inference/training."""
    df = df.copy()
    cat_cols = [name for name, is_cat in feature_col_info if is_cat]
    for c in cat_cols:
        df[c] = df[c].astype("object")
        df[c] = df[c].where(df[c].notna(), "__MISSING__")
    return df


def prepare_data_for_prediction(data, feature_col_info, target_col_name):
    """
    Prepare the feature matrix used at prediction time.

    Unlike ``prepare_data_raw``, this keeps all rows, including rows without a
    known target. Categorical columns are sanitized in the same way as in the
    training data so that CatBoost sees a consistent schema.
    """
    feature_col_names = [item[0] for item in feature_col_info]

    pred_data = data[feature_col_names + [target_col_name]].copy()

    # cat_cols = [name for name, is_cat in feature_col_info if is_cat]
    # for c in cat_cols:
    #     pred_data[c] = pred_data[c].astype("object")
    #     pred_data[c] = pred_data[c].where(pred_data[c].notna(), "__MISSING__")
    pred_data = sanitize_cat_cols(pred_data, feature_col_info)

    # Apply the same categorical → aggregated mean transformation
    # pred_data = cat_to_aggmean(pred_data, feature_col_info)
    pred_data = pred_data.drop(columns=[target_col_name])

    return pred_data


def engineer_features(data):
    """Derive the current model feature set from the cleaned input table.

    The current production model is intentionally small. It uses:

    - ``d_pba`` and ``d_acceptance`` as the two main timing anchors
    - ``d_gap`` as their difference, which helps the uncertainty model react to
      narrow acceptance/PBA windows
    - a small number of demographic and institutional covariates

    Returns
    -------
    tuple[pd.DataFrame, list[tuple[str, bool]]]
        The augmented dataframe and the feature specification consumed by the
        training/prediction helpers.
    """
    data = data.copy()
    data["age_at_defense"] = data["defense_year"] - data["birth_year"]
    data["d_gap"] = data["d_pba"] - data["d_acceptance"]

    feature_col_info = [
        ("d_pba", False),
        ("d_acceptance", False),
        ("d_gap", False),  # important for uncertainty model
        ("age_at_defense", False),
        ("institute_name", True),
        ("has_german_cship", True),
        ("pba_state", True),
    ]
    return data, feature_col_info


def predict_durations(
    data,
    target_col_name,
    feature_col_info,
    params_unc,
    carry_cols,
    out_path,
    out_fname,
    override_impossible=True,
    naive_estimator_for_small_gap=(),
):
    """Train the uncertainty model, predict durations for all rows, and write output.

    This is the production prediction entrypoint.

    The fitted model is ``RMSEWithUncertainty``. Its mean prediction is passed
    into ``postprocess_predictions``, which applies feasibility clipping and the
    optional small-gap midpoint override before the final CSV is written.
    """
    labeled_data, cat_features = prepare_data_raw(
        data, target_col_name, feature_col_info, dropna=("d_pba", "d_acceptance")
    )

    model_rmseunc = train_catboost(
        labeled_data,
        target_col_name,
        cat_features,
        params_unc,
        loss_function="RMSEWithUncertainty",
    )

    data_for_pred = prepare_data_for_prediction(data, feature_col_info, target_col_name)
    pred_mean_var = model_rmseunc.predict(
        data_for_pred, prediction_type="RMSEWithUncertainty"
    )
    predicted_duration_unc = pred_mean_var[:, 0]  # mean
    predicted_duration_var = pred_mean_var[:, 1]  # variance
    out = postprocess_predictions(
        data,
        predicted_duration_unc,
        predicted_duration_var,
        override_impossible=override_impossible,
        carry_cols=carry_cols,
        naive_estimator_for_small_gap=naive_estimator_for_small_gap,
    )

    out_cols = carry_cols + [
        "duration_estimate",
        "duration_estimate_clipped",
        "lower_bound_raw",
        "upper_bound_raw",
        "ci_size_raw",
        "lower_bound_adjusted",
        "upper_bound_adjusted",
        "ci_size_adjusted",
    ]
    out = out[out_cols]

    # estimates do not make sense for rows without a defense date, so set to NA to be sure
    out.loc[
        out["defense_date"].isna(),
        [
            "duration_estimate",
            "duration_estimate_clipped",
            "lower_bound_raw",
            "upper_bound_raw",
            "ci_size_raw",
            "lower_bound_adjusted",
            "upper_bound_adjusted",
            "ci_size_adjusted",
        ],
    ] = pd.NA
    # out.to_csv("data/duration_predictions.csv", index=False)
    # out.to_csv(out_path / "duration_predictions.csv", index=False)
    out_path = Path(out_path)
    out.to_csv(out_path / out_fname, index=False)


# check duplication issue!
def find_parameters(
    data,
    target_col_name,
    feature_col_info,
    loss_function="RMSE",
    rseed=67,
    N=10000,
    ES=200,
    dump_folder=None,
):
    """Tune hyperparameters and choose the final iteration count."""
    labeled_data, cat_features = prepare_data_raw(
        data,
        target_col_name,
        feature_col_info,
    )
    params = catboost_paramsearch(
        labeled_data,
        cat_features,
        N=N,
        ES=ES,
        loss_function=loss_function,
    )
    iterations = catboost_cv(
        labeled_data,
        target_col_name,
        cat_features,
        params=params,
        loss_function=loss_function,
        random_seed=rseed,
    )
    params["iterations"] = iterations

    if dump_folder:
        with open(f"{dump_folder}/best_params_{loss_function}.json", "w") as f:
            json.dump(params, f, indent=4)
    return params


def postprocess_predictions(
    df,
    predicted_duration_unc,
    predicted_duration_var,
    override_impossible=True,
    carry_cols=None,
    naive_estimator_for_small_gap=(),
):
    """Convert raw uncertainty-model outputs into final estimates and CI metrics.

    Parameters
    ----------
    df:
        Input dataframe containing at least ``d_pba`` and ``d_acceptance``.
    predicted_duration_unc:
        Mean prediction returned by CatBoost ``RMSEWithUncertainty``.
    predicted_duration_var:
        Variance-like second output returned by CatBoost
        ``RMSEWithUncertainty``.
    override_impossible:
        Whether to repair cases where the raw model interval and the feasible
        duration range do not overlap.
    carry_cols:
        Original columns to copy into the output frame.
    naive_estimator_for_small_gap:
        Iterable of estimate columns that should be replaced by the midpoint
        baseline when the acceptance/PBA gap is small and internally
        consistent.

    What happens here
    -----------------
    1. Build a raw interval ``mean ± 1.96 * std`` from the model variance.
    2. Define a feasible duration range from:
       - ``d_pba + 1/12`` as an upper duration bound
       - ``d_acceptance - 1/12`` as a lower duration bound
       The one-month buffer makes the rule less brittle around date granularity.
    3. Clip the raw interval to those feasible bounds when the relevant anchor
       exists.
    4. If the clipped interval becomes impossible:
       - for narrow acceptance/PBA windows, set the estimate to the midpoint and
         collapse the adjusted interval onto the feasible range
       - otherwise mark the adjusted bounds missing and finally fall back to the
         raw interval if needed
    5. Produce:
       - ``duration_estimate``
       - ``duration_estimate_clipped``
       - raw/adjusted bounds
       - raw/adjusted CI widths

    The exported estimate is therefore deliberately more structured than the raw
    model mean. It is the model prediction after project-specific feasibility
    and small-gap corrections.
    """
    predicted_duration_std = np.sqrt(predicted_duration_var)
    lower_bound_raw = predicted_duration_unc - 1.96 * predicted_duration_std
    upper_bound_raw = predicted_duration_unc + 1.96 * predicted_duration_std

    max_duration = df["d_pba"] + 1 / 12
    min_duration = df["d_acceptance"] - 1 / 12
    inconsistency_mask = max_duration < min_duration
    max_duration = max_duration.copy()
    min_duration = min_duration.copy()
    max_duration[inconsistency_mask] = 100
    min_duration[inconsistency_mask] = 0

    has_acceptance = df["d_acceptance"].notna()
    has_pba = df["d_pba"].notna()

    lower_bound_adjusted = pd.Series(lower_bound_raw, index=df.index).copy()
    upper_bound_adjusted = pd.Series(upper_bound_raw, index=df.index).copy()

    lower_bound_adjusted[has_acceptance] = np.maximum(
        lower_bound_adjusted[has_acceptance],
        min_duration[has_acceptance],
    )
    upper_bound_adjusted[has_pba] = np.minimum(
        upper_bound_adjusted[has_pba],
        max_duration[has_pba],
    )

    duration_estimate = pd.Series(predicted_duration_unc, index=df.index).clip(lower=0)

    if override_impossible:
        impossible_mask = upper_bound_adjusted < lower_bound_adjusted
        small_diff_mask = (df["d_pba"] - df["d_acceptance"]) <= 0.5
        override_mask = impossible_mask & small_diff_mask

        midpoint = (max_duration + min_duration) / 2
        duration_estimate.loc[override_mask] = midpoint[override_mask]
        lower_bound_adjusted.loc[override_mask] = min_duration[override_mask]
        upper_bound_adjusted.loc[override_mask] = max_duration[override_mask]

        lower_bound_adjusted.loc[impossible_mask & ~small_diff_mask] = pd.NA
        upper_bound_adjusted.loc[impossible_mask & ~small_diff_mask] = pd.NA

        impossible_mask = upper_bound_adjusted < lower_bound_adjusted
        lower_bound_adjusted.loc[impossible_mask] = lower_bound_raw[impossible_mask]
        upper_bound_adjusted.loc[impossible_mask] = upper_bound_raw[impossible_mask]

    if not carry_cols:
        out = pd.DataFrame(index=df.index)
    else:
        out = df.loc[:, carry_cols].copy()
    out["duration_estimate"] = duration_estimate
    out["duration_estimate_clipped"] = duration_estimate.clip(
        lower=min_duration, upper=max_duration
    )
    out["lower_bound_raw"] = lower_bound_raw
    out["upper_bound_raw"] = upper_bound_raw
    out["ci_size_raw"] = out["upper_bound_raw"] - out["lower_bound_raw"]
    out["lower_bound_adjusted"] = lower_bound_adjusted
    out["upper_bound_adjusted"] = upper_bound_adjusted
    out["ci_size_adjusted"] = out["upper_bound_adjusted"] - out["lower_bound_adjusted"]

    if naive_estimator_for_small_gap:
        small_diff_mask = (df["d_pba"] - df["d_acceptance"]) <= 0.5
        small_diff_mask &= (
            ~inconsistency_mask
        )  # no touching the inconsistent intervals.
        for col_name in naive_estimator_for_small_gap:
            out.loc[small_diff_mask, col_name] = (max_duration + min_duration) / 2
    return out


def acc_estimator(df):
    """Midpoint baseline in duration space using ``d_pba`` and ``d_acceptance``."""
    d_acceptance = df["d_acceptance"]
    return d_acceptance


def naive_estimator(df):
    """Midpoint baseline in duration space using ``d_pba`` and ``d_acceptance``."""
    d_acceptance = df["d_acceptance"]
    d_pba = df["d_pba"]
    naive_duration = (d_pba + d_acceptance) / 2
    valid_mask = d_acceptance.notna() & d_pba.notna()
    return naive_duration, valid_mask


def compute_double_long_side(df, *, estimate_col="duration_estimate_clipped"):
    """Return twice the longer adjusted-CI side around ``estimate_col``."""
    left_side = df[estimate_col] - df["lower_bound_adjusted"]
    right_side = df["upper_bound_adjusted"] - df[estimate_col]
    return 2 * pd.concat([left_side, right_side], axis=1).max(axis=1)


def cross_validated_model_estimates(
    data,
    target_col_name,
    params_unc,
    feature_col_info,
    n_splits=5,
    random_state=42,
    override_impossible=True,
    naive_estimator_for_small_gap=("duration_estimate", "duration_estimate_clipped"),
):
    """Generate out-of-fold predictions using the same post-processing as production.

    This function exists so that evaluation is based on held-out predictions,
    not on the full-fit output file. Each fold:

    - trains the uncertainty model on the training folds
    - predicts on the held-out fold
    - applies ``postprocess_predictions`` to the held-out rows
    - stores the naive midpoint baseline for comparison

    The returned dataframe is the basis for the plotting script and for the
    CI-threshold summaries.
    """
    labeled_data, cat_features = prepare_data_raw(
        data, target_col_name, feature_col_info, dropna=("d_pba", "d_acceptance")
    )

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_outputs = []

    for train_idx, test_idx in kf.split(labeled_data):
        train_df = labeled_data.iloc[train_idx].copy()
        test_df = labeled_data.iloc[test_idx].copy()

        model_rmseunc = train_catboost(
            train_df,
            target_col_name,
            cat_features,
            params_unc,
            loss_function="RMSEWithUncertainty",
        )

        X_test = prepare_data_for_prediction(test_df, feature_col_info, target_col_name)
        pred_mean_var = model_rmseunc.predict(
            X_test, prediction_type="RMSEWithUncertainty"
        )

        fold_out = postprocess_predictions(
            test_df,
            predicted_duration_unc=pred_mean_var[:, 0],
            predicted_duration_var=pred_mean_var[:, 1],
            override_impossible=override_impossible,
            naive_estimator_for_small_gap=naive_estimator_for_small_gap,
        )
        fold_out["duration_computed"] = test_df[target_col_name]
        fold_out["gap_size"] = test_df["d_pba"] - test_df["d_acceptance"]
        fold_out["double_long_side"] = compute_double_long_side(fold_out)

        naive_duration, naive_valid_mask = naive_estimator(test_df)
        acc_duration = acc_estimator(test_df)
        fold_out["naive_duration"] = naive_duration
        fold_out["naive_valid"] = naive_valid_mask
        fold_out["acc_duration"] = acc_duration
        fold_outputs.append(fold_out)

    return pd.concat(fold_outputs).sort_index()


def calculate_errors(cv_result):
    """Add squared-error columns for the model and naive baseline."""
    cv_result = cv_result.copy()
    cv_result["estimate_error"] = (
        cv_result["duration_estimate"] - cv_result["duration_computed"]
    ) ** 2
    cv_result["estimate_error_clipped"] = (
        cv_result["duration_estimate_clipped"] - cv_result["duration_computed"]
    ) ** 2
    cv_result["naive_error"] = (
        cv_result["naive_duration"] - cv_result["duration_computed"]
    ) ** 2
    cv_result["acc_error"] = (
        cv_result["acc_duration"] - cv_result["duration_computed"]
    ) ** 2
    return cv_result


def get_binned_errors(df, error_col, bin_col, bin_edges):
    """Compute RMSE and counts within half-open bins ``[low, high)``."""
    bins = [[] for _ in range(len(bin_edges) - 1)]
    frequencies = [0] * (len(bin_edges) - 1)
    for row in df.itertuples():
        for i, (low, high) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
            if low <= getattr(row, bin_col) < high:
                bins[i].append(getattr(row, error_col))
                frequencies[i] += 1
                break
    error_means = []
    for bin_values in bins:
        bin_array = np.array(bin_values)
        error_means.append(np.sqrt(np.mean(bin_array)))
    return error_means, frequencies


def summarize_ci_thresholds(
    cv_result,
    thresholds=(1.0, 1.5, 2.0, 2.5, 3.0),
    ci_col="double_long_side",
    estimate_col="duration_estimate_clipped",
    target_col="duration_computed",
    out_path=None,
):
    """Summarize retention and accuracy at several CI thresholds.

    This is meant for selecting a practical ``ci_size_adjusted`` cutoff. The
    summary is computed on cross-validated predictions so that the reported
    errors are out-of-fold rather than full-fit.
    """
    valid_mask = (
        cv_result[ci_col].notna()
        & cv_result[estimate_col].notna()
        & cv_result[target_col].notna()
    )
    valid = cv_result.loc[valid_mask].copy()

    rows = []
    text_lines = []
    for threshold in thresholds:
        mask = valid[ci_col] <= threshold
        abs_error = (valid.loc[mask, estimate_col] - valid.loc[mask, target_col]).abs()
        kept = int(mask.sum())
        keep_pct = 100 * kept / len(valid)
        row = {
            "ci_threshold": threshold,
            "kept": kept,
            "keep_pct": keep_pct,
            "mae": float(abs_error.mean()),
            "p75_abs_error": float(abs_error.quantile(0.75)),
            "p95_abs_error": float(abs_error.quantile(0.95)),
            "rmse": float(np.sqrt(np.mean(abs_error**2))),
        }
        rows.append(row)
        text_lines.extend(
            [
                f"CI <= {threshold:.1f}",
                f"  - kept: {kept} ({keep_pct:.1f}%)",
                f"  - average absolute error: {row['mae']:.3f} years",
                f"  - 75th percentile absolute error: {row['p75_abs_error']:.3f} years",
                f"  - 95th percentile absolute error: {row['p95_abs_error']:.3f} years",
                f"  - RMSE: {row['rmse']:.3f} years",
            ]
        )

    summary = pd.DataFrame(rows)
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(out_path, index=False)
        out_txt = out_path.with_suffix(".txt")
        out_txt.write_text("\n".join(text_lines) + "\n")
    return summary


def find_parameters_and_predict(
    data,
    target_col_name,
    feature_col_info,
    orig_cols,
    out_path,
    out_fname,
    N=10000,
    ES=200,
    dump_folder=None,
    naive_estimator_for_small_gap=(),
):
    """Tune the uncertainty model and write full-data duration predictions.

    This is the entrypoint used by ``scripts.data_processing.estimate_duration``.
    It tunes CatBoost for ``RMSEWithUncertainty`` and then runs the production
    prediction path on the full dataset.
    """
    params_unc = find_parameters(
        data,
        target_col_name,
        feature_col_info,
        loss_function="RMSEWithUncertainty",
        N=N,
        ES=ES,
        dump_folder=dump_folder,
    )

    predict_durations(
        data,
        target_col_name,
        feature_col_info,
        params_unc,
        orig_cols,
        out_path,
        out_fname,
        naive_estimator_for_small_gap=naive_estimator_for_small_gap,
    )


def visualize_binned_errors(cv_result, bin_edges, bin_labels, out_dir, theme, bin_var):
    """Plot RMSE by gap size or CI size and save the figure to disk."""
    est = get_binned_errors(cv_result, "estimate_error", bin_var, bin_edges)
    est_c = get_binned_errors(cv_result, "estimate_error_clipped", bin_var, bin_edges)
    est_n = get_binned_errors(cv_result, "naive_error", bin_var, bin_edges)
    est_a = get_binned_errors(cv_result, "acc_error", bin_var, bin_edges)

    mean_est = np.sqrt(np.mean(cv_result["estimate_error"]))
    mean_est_c = np.sqrt(np.mean(cv_result["estimate_error_clipped"]))
    mean_est_n = np.sqrt(np.mean(cv_result["naive_error"]))
    mean_est_a = np.sqrt(np.mean(cv_result["acc_error"]))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        bin_edges[:-1],
        est[0],
        label=f"Model Estimate, mean: {mean_est:.4f}",
        marker="o",
    )
    ax.plot(
        bin_edges[:-1],
        est_c[0],
        label=f"Clipped Estimate, mean: {mean_est_c:.4f}",
        marker="o",
    )
    ax.plot(
        bin_edges[:-1],
        est_n[0],
        label=f"Naive Estimate, mean: {mean_est_n:.4f}",
        marker="o",
    )
    ax.plot(
        bin_edges[:-1],
        est_a[0],
        label=f"Acceptance-based Estimate, mean: {mean_est_a:.4f}",
        marker="o",
    )

    frequencies = est[1]
    frequencies_frac = [f / sum(frequencies) for f in frequencies]

    ax2 = ax.twinx()
    ax2.plot(
        bin_edges[:-1],
        frequencies_frac,
        label="Frequency",
        marker="x",
        linestyle="--",
        color="gray",
    )
    ax2.plot(
        bin_edges[:-1],
        np.cumsum(frequencies_frac),
        label="Cumulative Frequency",
        marker="x",
        linestyle="-",
        color="lightgray",
    )
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    if bin_var == "gap_size":
        xlabel = "Gap Size (Years)"
    elif bin_var == "ci_size_adjusted":
        xlabel = "CI Size (Years)"
    elif bin_var == "double_long_side":
        xlabel = "2 x Long CI Side (Years)"
    else:
        raise ValueError("Invalid bin_var")

    ax.set_xlabel(xlabel)
    ax.set_xticks(bin_edges[:-1])
    ax.set_xticklabels(bin_labels, rotation=45)
    ax.set_ylabel("RMSE")
    ax2.set_ylabel("Frequency")
    theme.apply_transforms()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"duration_eval_{bin_var}.png"
    fig.tight_layout()
    fig.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close(fig)


def scatter_gap_ci(cv_result, plot_dir, theme, ci_col="ci_size_adjusted"):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(cv_result[ci_col], cv_result["gap_size"], alpha=0.5, s=1)
    ax.set_ylabel("Gap Size")
    if ci_col == "ci_size_adjusted":
        ax.set_xlabel("CI Size (Adjusted)")
    elif ci_col == "ci_size_raw":
        ax.set_xlabel("CI Size (Raw)")
    theme.apply_transforms()
    plt.savefig(plot_dir / f"gap_size_vs_{ci_col}.png", dpi=200, bbox_inches="tight")
    plt.close()


def get_combined_duration(duration_data, ci_thereshold, mode="adjusted_ci"):
    duration_combined = duration_data["duration_estimate_clipped"].copy()
    if mode == "adjusted_ci":
        ci_mask = (duration_data["ci_size_adjusted"] > ci_thereshold) | duration_data[
            "ci_size_adjusted"
        ].isna()
    elif mode == "double_long_side":
        double_long_side = compute_double_long_side(duration_data)
        ci_mask = (double_long_side > ci_thereshold) | double_long_side.isna()
    else:
        raise ValueError(f"Unknown mode: {mode}")

    duration_combined[ci_mask] = pd.NA
    duration_combined[duration_data["duration_computed"].notna()] = duration_data[
        "duration_computed"
    ]
    return duration_combined
