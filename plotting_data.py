import numpy as np
import pandas as pd
from typing import Callable

PBA_CATEGORY_LABELS = {
    "TUB": "TU Berlin",
    "GER": "in Deutschland",
    "EU": "in EU",
    "NEU": "außerhalb EU",
}


def get_singular_count_name(count_name: str) -> str:
    """Return the singular form for known plot count labels."""
    singular_map = {
        "Anmeldungen": "Anmeldung",
        "Abschluesse": "Abschluss",
        "Abschlüsse": "Abschluss",
    }
    return singular_map.get(count_name, count_name)


def prepare_category_frame(
    df: pd.DataFrame, category_col: str, category_label_map: dict
) -> pd.DataFrame:
    """Return a dataframe copy with category_col normalized into the shared category_name column."""
    df = df.copy()
    df[category_col] = df[category_col].fillna("Unklare Zuordnung")
    df["category_name"] = (
        df[category_col].map(category_label_map).fillna(df[category_col])
    )
    df["category_name"] = df["category_name"].fillna("Unklare Zuordnung").astype(str)
    return df


def build_faculties(faculty_institute_mappings: dict) -> dict[str, list[str]]:
    """Return a sorted faculty-to-institute-name mapping derived from the color config."""
    faculties = {}

    for faculty in sorted(faculty_institute_mappings):
        inst_colors = faculty_institute_mappings[faculty].get("institute_colors", {})
        faculties[faculty] = sorted(inst_colors)

    return faculties


def remap_institutes(df: pd.DataFrame, params_global: dict):
    """Return a dataframe copy with institute names remapped via params_global."""
    df = df.copy()
    institute_remap = params_global["institute_remap"]
    df["institute_name"] = df["institute_name"].replace(institute_remap)
    return df


def normalize_institute_names(
    df: pd.DataFrame, faculties: dict[str, list[str]]
) -> pd.DataFrame:
    """Return a dataframe where invalid faculty/institute pairs are relabeled as 'Unklare Zuordnung'."""
    faculty_sets = {k: set(v) for k, v in faculties.items()}

    invalid = ~df.apply(
        lambda r: r["institute_name"] in faculty_sets.get(r["faculty"], set()),
        axis=1,
    )

    df.loc[invalid, "institute_name"] = "Unklare Zuordnung"
    return df


def remove_dubious_institute_assignments(
    df: pd.DataFrame, faculties: dict[str, list[str]]
) -> pd.DataFrame:
    """Return a dataframe where invalid faculty/institute pairs have institute_name set to NA."""
    faculty_sets = {k: set(v) for k, v in faculties.items()}

    invalid = ~df.apply(
        lambda r: r["institute_name"] in faculty_sets.get(r["faculty"], set()),
        axis=1,
    )

    df.loc[invalid, "institute_name"] = pd.NA
    return df


def limit_year_range(
    df: pd.DataFrame,
    *,
    year_col: str,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    """Return only rows whose year_col lies within the inclusive [start_year, end_year] interval."""
    return df[df[year_col].between(start_year, end_year)]


def prepare_yearly_category_counts(
    df: pd.DataFrame,
    *,
    year_col: str,
    faculty_col: str,
    category_col: str,
    start_year: int,
    end_year: int,
    category_label_map: dict | None = None,
) -> pd.DataFrame:
    """Return grouped yearly counts by faculty/category after optional label remapping."""
    df = df[[year_col, faculty_col, category_col]].copy()
    if category_label_map:
        df.replace({category_col: category_label_map}, inplace=True)

    df = df.groupby([year_col, faculty_col, category_col], as_index=False).size()
    df = df.rename(columns={year_col: "year", "size": "count"})
    return limit_year_range(
        df,
        year_col="year",
        start_year=start_year,
        end_year=end_year,
    )


def year_total_agg(
    df: pd.DataFrame,
    *,
    year_col: str,
    value_col: str,
    total_col: str = "year_total",
    agg: str = "sum",
) -> pd.DataFrame:
    """Return one row per year with value_col aggregated into total_col."""
    return (
        df.groupby(year_col, as_index=False)[value_col]
        .agg(agg)
        .rename(columns={value_col: total_col})
    )


def fac_by_year_agg(
    df: pd.DataFrame,
    year_total: pd.DataFrame,
    *,
    year_col: str,
    faculty_col: str,
    value_col: str,
    total_col: str = "year_total",
    out_value_col: str | None = None,
    pct_col: str | None = None,
    agg: str = "sum",
) -> pd.DataFrame:
    """Return one row per (year, faculty) with aggregated values and optional percentage columns."""
    fac = (
        df.groupby([year_col, faculty_col], as_index=False)[value_col]
        .agg(agg)
        .merge(year_total, on=year_col, how="left")
    )
    if out_value_col is not None and out_value_col != value_col:
        fac = fac.rename(columns={value_col: out_value_col})
        value_name = out_value_col
    else:
        value_name = value_col
    if pct_col is not None:
        fac[pct_col] = np.where(
            fac[total_col] > 0,
            fac[value_name] / fac[total_col],
            0.0,
        )
    return fac


def inst_by_year_agg(
    df: pd.DataFrame,
    year_total: pd.DataFrame,
    *,
    year_col: str,
    faculty_col: str,
    institute_col: str,
    value_col: str,
    total_col: str = "year_total",
    faculty_total_col: str = "faculty_year_total",
    out_value_col: str | None = None,
    pct_of_faculty_col: str | None = None,
    pct_of_year_total_col: str | None = None,
    agg: str = "sum",
) -> pd.DataFrame:
    """Return one row per (year, faculty, institute) with aggregated values and optional share columns."""
    inst = (
        df.groupby([year_col, faculty_col, institute_col], as_index=False)[value_col]
        .agg(agg)
        .merge(year_total, on=year_col, how="left")
    )
    fac_year_total = (
        df.groupby([year_col, faculty_col], as_index=False)[value_col]
        .agg(agg)
        .rename(columns={value_col: faculty_total_col})
    )
    inst = inst.merge(fac_year_total, on=[year_col, faculty_col], how="left")
    if out_value_col is not None and out_value_col != value_col:
        inst = inst.rename(columns={value_col: out_value_col})
        value_name = out_value_col
    else:
        value_name = value_col
    if pct_of_faculty_col is not None:
        inst[pct_of_faculty_col] = np.where(
            inst[faculty_total_col] > 0,
            inst[value_name] / inst[faculty_total_col],
            0.0,
        )
    if pct_of_year_total_col is not None:
        inst[pct_of_year_total_col] = np.where(
            inst[total_col] > 0,
            inst[value_name] / inst[total_col],
            0.0,
        )
    return inst


def prune_groups_for_plot(
    df: pd.DataFrame,
    *,
    year_col: str,
    group_cols: list[str],
    years_sorted,
    n_min_year_prune: int,
    n_min_abs: int,
    n_min_mean: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return survivor keys, remainder keys, and per-key stats after applying prune thresholds over the selected years."""
    years_prune = [y for y in years_sorted if y >= n_min_year_prune]

    counts = df.groupby([year_col] + group_cols).size().rename("n").reset_index()
    counts = counts[counts[year_col].isin(years_prune)]

    all_keys = df[group_cols].drop_duplicates()

    grid = (
        all_keys.assign(_tmp=1)
        .merge(pd.DataFrame({year_col: years_prune, "_tmp": 1}), on="_tmp")
        .drop(columns="_tmp")
    )

    counts_full = grid.merge(counts, on=[year_col] + group_cols, how="left")
    counts_full["n"] = counts_full["n"].fillna(0).astype(int)

    inst_stats = counts_full.groupby(group_cols, as_index=False).agg(
        n_min=("n", "min"), n_mean=("n", "mean")
    )

    survivors_keys = inst_stats[
        (inst_stats["n_min"] >= n_min_abs) & (inst_stats["n_mean"] >= n_min_mean)
    ][group_cols].copy()

    remainder_keys = (
        all_keys.merge(survivors_keys, on=group_cols, how="left", indicator=True)
        .query('_merge == "left_only"')
        .drop(columns="_merge")
    )

    return survivors_keys, remainder_keys, inst_stats


def series_with_rest_from_keys(
    df_full: pd.DataFrame,
    year_total: pd.DataFrame,
    fac_year_total: pd.DataFrame,
    survivors_keys: pd.DataFrame,
    remainder_keys: pd.DataFrame,
    *,
    year_col: str,
    faculty_col: str,
    label_col: str,
    value_col: str,
    rest_label: str = "Rest",
) -> pd.DataFrame:
    """Return a yearly series frame where survivor groups stay separate and remainder groups are merged into Rest."""
    keep_df = df_full.merge(survivors_keys, on=[faculty_col, label_col], how="inner")

    keep = keep_df.groupby([year_col, faculty_col, label_col], as_index=False).agg(
        female_count=(value_col, "sum"), total_count=(value_col, "count")
    )
    keep["female_percentage"] = np.where(
        keep["total_count"] > 0,
        keep["female_count"] / keep["total_count"],
        np.nan,
    )
    keep["rest_hover"] = ""

    rest_df = df_full.merge(remainder_keys, on=[faculty_col, label_col], how="inner")

    rest = rest_df.groupby([year_col, faculty_col], as_index=False).agg(
        female_count=(value_col, "sum"), total_count=(value_col, "count")
    )
    rest[label_col] = rest_label
    rest["female_percentage"] = np.where(
        rest["total_count"] > 0,
        rest["female_count"] / rest["total_count"],
        np.nan,
    )

    rest_details = (
        rest_df.groupby([year_col, faculty_col, label_col], as_index=False)
        .agg(n=(value_col, "count"), n_female=(value_col, "sum"))
        .sort_values([year_col, faculty_col, "n"], ascending=[True, True, False])
    )

    rest_hover = (
        rest_details.groupby([year_col, faculty_col], sort=False)
        .agg(_names=(label_col, list), _n=("n", list), _nf=("n_female", list))
        .reset_index()
    )
    rest_hover["rest_hover"] = [
        "<br>".join(f"{name}: n={n} (w/d={nf})" for name, n, nf in zip(names, ns, nfs))
        for names, ns, nfs in zip(
            rest_hover["_names"], rest_hover["_n"], rest_hover["_nf"]
        )
    ]
    rest_hover = rest_hover[[year_col, faculty_col, "rest_hover"]]

    rest = rest.merge(rest_hover, on=[year_col, faculty_col], how="left")
    rest["rest_hover"] = rest["rest_hover"].fillna("")

    return (
        pd.concat([keep, rest], ignore_index=True)
        .merge(year_total, on=year_col, how="left")
        .merge(fac_year_total, on=[year_col, faculty_col], how="left")
    )


def aggregate_small_groups(
    df: pd.DataFrame,
    *,
    group_cols: list[str],
    label_col: str,
    count_col: str,
    total_cols: list[str],
    pct_specs: list[tuple[str, str]],
    n_min: int,
    invalid_label: str = "Unklare Zuordnung",
    rest_label: str = "Rest",
    singular_name: str,
    plural_name: str,
) -> pd.DataFrame:
    """Return a grouped frame where small labels are folded into Rest, with percentages and Rest hover text recomputed."""
    df = df.copy()
    grouping_cols = list(group_cols)
    has_groups = bool(grouping_cols)

    invalid_mask = df[label_col].eq(invalid_label)
    small_mask = (df[count_col] < n_min) | invalid_mask

    if has_groups:
        n_small = (
            df[small_mask]
            .groupby(grouping_cols)[label_col]
            .nunique()
            .rename("n_small")
            .reset_index()
        )
        df = df.merge(n_small, on=grouping_cols, how="left")
        df["n_small"] = df["n_small"].fillna(0).astype(int)
    else:
        df["n_small"] = int(df.loc[small_mask, label_col].nunique())

    df["group_label"] = np.where(
        invalid_mask,
        rest_label,
        np.where(
            ~small_mask,
            df[label_col],
            np.where(df["n_small"] == 1, df[label_col], rest_label),
        ),
    )

    rest_details = (
        df[df["group_label"] == rest_label]
        .groupby(grouping_cols + [label_col], as_index=False)
        .agg(**{count_col: (count_col, "sum")})
        .sort_values(
            grouping_cols + [count_col],
            ascending=[True] * len(grouping_cols) + [False],
        )
    )

    if has_groups:
        rest_hover = (
            rest_details.groupby(grouping_cols, sort=False)
            .agg(_names=(label_col, list), _counts=(count_col, list))
            .reset_index()
        )
        rest_hover["rest_hover"] = [
            "<br>".join(
                f"{name}: {count:,} {singular_name if count == 1 else plural_name}"
                for name, count in zip(names, counts)
            )
            for names, counts in zip(rest_hover["_names"], rest_hover["_counts"])
        ]
        rest_hover = rest_hover[grouping_cols + ["rest_hover"]]
    else:
        rest_hover = "<br>".join(
            f"{name}: {count:,} {singular_name if count == 1 else plural_name}"
            for name, count in zip(rest_details[label_col], rest_details[count_col])
        )

    agg_spec = {count_col: (count_col, "sum")}
    for total_col in total_cols:
        agg_spec[total_col] = (total_col, "first")

    out = (
        df.groupby(grouping_cols + ["group_label"], as_index=False)
        .agg(**agg_spec)
        .rename(columns={"group_label": label_col})
    )

    for pct_col, total_col in pct_specs:
        out[pct_col] = np.where(
            out[total_col] > 0, out[count_col] / out[total_col], 0.0
        )

    if has_groups:
        out = out.merge(rest_hover, on=grouping_cols, how="left")
        out["rest_hover"] = np.where(
            out[label_col] == rest_label,
            out["rest_hover"].fillna(""),
            "",
        )
    else:
        out["rest_hover"] = np.where(out[label_col] == rest_label, rest_hover, "")

    return out


def aggregate_top_groups(
    df: pd.DataFrame,
    *,
    group_cols: list[str],
    label_col: str,
    count_col: str,
    total_cols: list[str],
    pct_specs: list[tuple[str, str]],
    top_n: int,
    invalid_label: str = "Unklare Zuordnung",
    rest_label: str = "Rest",
    singular_name: str,
    plural_name: str,
) -> pd.DataFrame:
    """Return a grouped frame where only the top_n labels remain separate and the rest are folded into Rest."""
    if top_n < 1:
        raise ValueError("top_n must be at least 1.")

    df = df.copy()
    grouping_cols = list(group_cols)
    has_groups = bool(grouping_cols)

    invalid_mask = df[label_col].eq(invalid_label)

    rank_source = df.loc[~invalid_mask].sort_values(
        grouping_cols + [count_col, label_col],
        ascending=[True] * len(grouping_cols) + [False, True],
    )
    rank_source["label_rank"] = (
        rank_source.groupby(grouping_cols).cumcount() + 1
        if has_groups
        else np.arange(1, len(rank_source) + 1)
    )

    rank_cols = grouping_cols + [label_col, "label_rank"]
    df = df.merge(rank_source[rank_cols], on=grouping_cols + [label_col], how="left")

    keep_mask = (~invalid_mask) & df["label_rank"].le(top_n).fillna(False)
    df["group_label"] = np.where(keep_mask, df[label_col], rest_label)

    rest_details = (
        df[df["group_label"] == rest_label]
        .groupby(grouping_cols + [label_col], as_index=False)
        .agg(**{count_col: (count_col, "sum")})
        .sort_values(
            grouping_cols + [count_col, label_col],
            ascending=[True] * len(grouping_cols) + [False, True],
        )
    )

    if has_groups:
        rest_hover = (
            rest_details.groupby(grouping_cols, sort=False)
            .agg(_names=(label_col, list), _counts=(count_col, list))
            .reset_index()
        )
        rest_hover["rest_hover"] = [
            "<br>".join(
                f"{name}: {count:,} {singular_name if count == 1 else plural_name}"
                for name, count in zip(names, counts)
            )
            for names, counts in zip(rest_hover["_names"], rest_hover["_counts"])
        ]
        rest_hover = rest_hover[grouping_cols + ["rest_hover"]]
    else:
        rest_hover = "<br>".join(
            f"{name}: {count:,} {singular_name if count == 1 else plural_name}"
            for name, count in zip(rest_details[label_col], rest_details[count_col])
        )

    agg_spec = {count_col: (count_col, "sum")}
    for total_col in total_cols:
        agg_spec[total_col] = (total_col, "first")

    out = (
        df.groupby(grouping_cols + ["group_label"], as_index=False)
        .agg(**agg_spec)
        .rename(columns={"group_label": label_col})
    )

    for pct_col, total_col in pct_specs:
        out[pct_col] = np.where(
            out[total_col] > 0, out[count_col] / out[total_col], 0.0
        )

    if has_groups:
        out = out.merge(rest_hover, on=grouping_cols, how="left")
        out["rest_hover"] = np.where(
            out[label_col] == rest_label,
            out["rest_hover"].fillna(""),
            "",
        )
    else:
        out["rest_hover"] = np.where(out[label_col] == rest_label, rest_hover, "")

    return out


def finalize_group_percentages(
    df: pd.DataFrame,
    *,
    count_col: str,
    pct_specs: list[tuple[str, str]],
) -> pd.DataFrame:
    """Return a copy with pct_specs recomputed from count_col and an empty rest_hover column added."""
    df = df.copy()
    for pct_col, total_col in pct_specs:
        df[pct_col] = np.where(df[total_col] > 0, df[count_col] / df[total_col], 0.0)
    df["rest_hover"] = ""
    return df


def prep_cship_and_pba_cats(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["citizenship_category"] = pd.Series("", index=df.index)

    df.loc[~df["has_eu_cship"], "citizenship_category"] = "NEU"
    df.loc[df["has_eu_cship"] & ~df["has_german_cship"], "citizenship_category"] = "EU"
    df.loc[df["has_german_cship"], "citizenship_category"] = "GER"

    df["pba_category"] = df["pba_category"].replace({"FH": "GER"})
    return df


def prepare_country_cat_matrix(
    df: pd.DataFrame,
    *,
    start_year: int,
    end_year: int,
    year_col: str,
    fac_col: str,
    category_1: str,
    category_2: str,
    label_map_1: dict | None,
    label_map_2: dict | None,
    df_transform_func: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
) -> pd.DataFrame:
    if df_transform_func:
        df = df_transform_func(df)

    df = df[[year_col, category_1, category_2, fac_col]].copy()
    if label_map_1:
        df.replace({category_1: label_map_1}, inplace=True)
    if label_map_2:
        df.replace({category_2: label_map_2}, inplace=True)
    df = limit_year_range(
        df,
        year_col=year_col,
        start_year=start_year,
        end_year=end_year,
    )
    df = df.groupby([category_1, category_2, fac_col], as_index=False).size()
    df = df.rename(columns={"size": "count"})

    return df


def get_total_catmatrix_counts(
    df: pd.DataFrame, category_1: str, category_2: str
) -> pd.DataFrame:
    total_count = int(df["count"].sum())
    category_total = (
        df.groupby([category_1, category_2], as_index=False)["count"]
        .sum()
        .assign(total_count=total_count)
    )
    return category_total
