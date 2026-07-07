from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import patsy
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

# Make the repository root importable when the script is executed directly.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.paths import CONFIG_DIR, DATA_DIR
from src.data_loading.load_data import load_data
from src.data_processing.regression import engineer_features
from src.plotting.plot_config import load_plot_config


def load_analysis_frame() -> pd.DataFrame:
    params, _, _ = load_plot_config("gender_regression")
    start_year = int(params["start_year"])
    end_year = int(params["end_year"])

    df = load_data(DATA_DIR / "clean_data.csv", DATA_DIR / "cleaned_data_dtypes.json")
    df = df[df["defense_year"].between(start_year, end_year, inclusive="both")].copy()
    df = df[df["duration_computed"].notna()].copy()

    # The requested contrasts are male-vs-female, so keep only those rows.
    df = df[df["gender"].isin(["m", "w"])].copy()
    df = engineer_features(df, "duration_computed")
    df["log_duration"] = np.log(df["duration_computed"].where(df["duration_computed"] > 0))
    df = df.dropna(
        subset=[
            "log_duration",
            "institute_name",
            "is_male",
            "cship_category",
            "age_at_start",
            "pba_grade",
            "pba_grade_missing",
            "pba_type",
            "covid_overlap",
        ]
    ).copy()
    df["is_male"] = df["is_male"].astype("object")
    df["pba_grade_missing"] = df["pba_grade_missing"].astype("object")
    df["institute_name"] = df["institute_name"].astype("object")
    df["cship_category"] = df["cship_category"].astype("object")
    df["pba_type"] = df["pba_type"].astype("object")
    return df


def contrast_table(model, contrast_map: dict[str, np.ndarray]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for label, contrast in contrast_map.items():
        test = model.t_test(contrast)
        effect = float(np.asarray(test.effect).ravel()[0])
        se = float(np.asarray(test.sd).ravel()[0])
        t_value = float(np.asarray(test.tvalue).ravel()[0])
        p_value = float(np.asarray(test.pvalue).ravel()[0])
        rows.append(
            {
                "contrast": label,
                "effect_log": effect,
                "effect_pct": float(np.expm1(effect)),
                "se": se,
                "t": t_value,
                "p": p_value,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(
            columns=["contrast", "effect_log", "effect_pct", "se", "t", "p", "p_adj"]
        )

    out["p_adj"] = multipletests(out["p"], method="fdr_bh")[1]
    return out.sort_values(["p_adj", "p", "contrast"]).reset_index(drop=True)


def build_institute_contrasts(model, data: pd.DataFrame) -> pd.DataFrame:
    design_info = model.model.data.design_info
    covariate_values = {
        "is_male": False,
        "cship_category": data["cship_category"].mode(dropna=True).iloc[0],
        "age_at_start": float(data["age_at_start"].median()),
        "pba_grade": float(data["pba_grade"].median()),
        "pba_grade_missing": False,
        "pba_type": data["pba_type"].mode(dropna=True).iloc[0],
        "covid_overlap": float(data["covid_overlap"].median()),
    }

    insts = sorted(data["institute_name"].dropna().astype(str).unique())
    design_rows = []
    for inst in insts:
        row = {
            **covariate_values,
            "institute_name": inst,
        }
        x = patsy.build_design_matrices([design_info], pd.DataFrame([row]))[0]
        design_rows.append(np.asarray(x))

    x_mean = np.mean(np.vstack(design_rows), axis=0, keepdims=True)

    contrast_map = {
        inst: np.asarray(x - x_mean)
        for inst, x in zip(insts, design_rows, strict=False)
    }
    out = contrast_table(model, contrast_map)
    out["contrast"] = out["contrast"].astype(str)
    out["effect_log"] = out["effect_log"].astype(float)
    out["effect_pct"] = out["effect_pct"].astype(float)
    return out


def build_gender_contrasts(model, data: pd.DataFrame) -> pd.DataFrame:
    design_info = model.model.data.design_info
    covariate_values = {
        "cship_category": data["cship_category"].mode(dropna=True).iloc[0],
        "age_at_start": float(data["age_at_start"].median()),
        "pba_grade": float(data["pba_grade"].median()),
        "pba_grade_missing": False,
        "pba_type": data["pba_type"].mode(dropna=True).iloc[0],
        "covid_overlap": float(data["covid_overlap"].median()),
    }

    contrast_map: dict[str, np.ndarray] = {}
    for inst in sorted(data["institute_name"].dropna().astype(str).unique()):
        female = {
            **covariate_values,
            "institute_name": inst,
            "is_male": False,
        }
        male = {
            **covariate_values,
            "institute_name": inst,
            "is_male": True,
        }
        x_female = patsy.build_design_matrices([design_info], pd.DataFrame([female]))[0]
        x_male = patsy.build_design_matrices([design_info], pd.DataFrame([male]))[0]
        contrast_map[inst] = np.asarray(x_male - x_female)

    return contrast_table(model, contrast_map)


def select_top_institutes_by_squared_mean(
    institute_ct: pd.DataFrame, top_k: int
) -> pd.Index:
    ranked = institute_ct.copy()
    ranked["score"] = ranked["effect_log"] ** 2
    ranked = ranked.sort_values(["score", "p_adj", "contrast"], ascending=[False, True, True])
    return pd.Index(ranked.head(top_k)["contrast"].tolist())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--top-k",
        type=int,
        default=16,
        help="Keep only the institutes with the largest squared deviation from the grand mean.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("plots/gender_selection_compare/institute_gender_contrasts_top16.csv"),
    )
    args = parser.parse_args()

    df = load_analysis_frame()

    institute_formula = (
        "log_duration ~ C(institute_name, Sum) + C(is_male) + "
        "C(cship_category) + age_at_start + pba_grade + "
        "C(pba_grade_missing) + C(pba_type) + covid_overlap"
    )
    gender_formula = (
        "log_duration ~ C(institute_name, Sum) * C(is_male) + "
        "C(cship_category) + age_at_start + pba_grade + "
        "C(pba_grade_missing) + C(pba_type) + covid_overlap"
    )

    institute_model = smf.ols(institute_formula, data=df).fit(cov_type="HC3")
    gender_model = smf.ols(gender_formula, data=df).fit(cov_type="HC3")

    inst_ct = build_institute_contrasts(institute_model, df)
    top_institutes = select_top_institutes_by_squared_mean(inst_ct, args.top_k)

    df_top = df[df["institute_name"].isin(top_institutes)].copy()
    institute_model_top = smf.ols(institute_formula, data=df_top).fit(cov_type="HC3")
    gender_model_top = smf.ols(gender_formula, data=df_top).fit(cov_type="HC3")

    inst_ct_top = build_institute_contrasts(institute_model_top, df_top)
    gender_ct_top = build_gender_contrasts(gender_model_top, df_top)

    print(f"Rows used: {len(df)}")
    print(f"Gender rows: {df['is_male'].value_counts(dropna=False).to_dict()}")
    print(f"Top {args.top_k} institutes by squared mean deviation:")
    print(", ".join(top_institutes.tolist()))

    print("\nInstitute contrasts vs grand mean (top-k restricted)")
    print(inst_ct_top.to_string(index=False))

    print("\nWithin-institute gender contrasts (top-k restricted)")
    print(gender_ct_top.to_string(index=False))

    out = {
        "institute": inst_ct_top.assign(table="institute_vs_grand_mean"),
        "gender": gender_ct_top.assign(table="within_institute_gender"),
    }
    out_df = pd.concat(out.values(), ignore_index=True, sort=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
