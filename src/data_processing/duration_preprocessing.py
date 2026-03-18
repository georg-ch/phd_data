import pandas as pd

from datetime import datetime
import numpy as np


def impute_start_month_year(filtered_data, slack_months=2):
    no_start_month_mask = (filtered_data["start_month"] == 0) | (
        filtered_data["start_month"].isna()
    )

    in_ival_mask_acc = check_col_in_month_ival(
        filtered_data,
        "start_year",
        "acceptance_year",
        "acceptance_month",
        slack_months=slack_months,
    )
    # start_year in pba +- 3
    in_ival_mask_pba = check_col_in_month_ival(
        filtered_data,
        "start_year",
        "pba_year",
        "pba_month",
        slack_months=slack_months,
    )
    in_both_ival_mask = in_ival_mask_acc & in_ival_mask_pba
    in_none_mask = ~in_ival_mask_acc & ~in_ival_mask_pba

    in_none_mask_relevant = in_none_mask & no_start_month_mask
    in_ival_mask_acc_relevant = in_ival_mask_acc & no_start_month_mask
    in_ival_mask_pba_relevant = in_ival_mask_pba & no_start_month_mask

    in_both_ival_mask_relevant = in_both_ival_mask & no_start_month_mask

    ay = filtered_data.loc[in_both_ival_mask_relevant, "acceptance_year"]
    am = filtered_data.loc[in_both_ival_mask_relevant, "acceptance_month"]
    py = filtered_data.loc[in_both_ival_mask_relevant, "pba_year"]
    pm = filtered_data.loc[in_both_ival_mask_relevant, "pba_month"]

    # Absolute month indices (0-based within a year)
    abs_a = ay * 12 + (am - 1)
    abs_b = py * 12 + (pm - 1)

    # Midpoint in absolute-month space (floor midpoint like your // 2)
    mid_abs = (abs_a + abs_b) // 2

    start_year = (mid_abs // 12).astype("Int64")
    start_month = (mid_abs % 12 + 1).astype("Int64")

    filtered_data.loc[in_both_ival_mask_relevant, "start_year"] = start_year
    filtered_data.loc[in_both_ival_mask_relevant, "start_month"] = start_month

    filtered_data.loc[
        in_ival_mask_acc_relevant & ~in_ival_mask_pba_relevant, "start_month"
    ] = filtered_data.loc[
        in_ival_mask_acc_relevant & ~in_ival_mask_pba_relevant, "acceptance_month"
    ]
    filtered_data.loc[
        in_ival_mask_acc_relevant & ~in_ival_mask_pba_relevant, "start_year"
    ] = filtered_data.loc[
        in_ival_mask_acc_relevant & ~in_ival_mask_pba_relevant, "acceptance_year"
    ]

    filtered_data.loc[
        in_ival_mask_pba_relevant & ~in_ival_mask_acc_relevant, "start_month"
    ] = filtered_data.loc[
        in_ival_mask_pba_relevant & ~in_ival_mask_acc_relevant, "pba_month"
    ]
    filtered_data.loc[
        in_ival_mask_pba_relevant & ~in_ival_mask_acc_relevant, "start_year"
    ] = filtered_data.loc[
        in_ival_mask_pba_relevant & ~in_ival_mask_acc_relevant, "pba_year"
    ]

    filtered_data.loc[in_none_mask_relevant, "start_month"] = 6
    filtered_data["start_month_guessed"] = in_none_mask_relevant
    return filtered_data


def impute_end_month_year(filtered_data, slack_months=2):
    in_ival_mask = check_col_in_month_ival(
        filtered_data,
        "end_year",
        "defense_year",
        "defense_month",
        slack_months=slack_months,
    )
    no_end_month_mask = (filtered_data["end_month"] == 0) | (
        filtered_data["end_month"].isna()
    )

    out_ival_mask = ~in_ival_mask
    out_ival_mask_relevant = out_ival_mask & no_end_month_mask
    in_ival_mask_relevant = in_ival_mask & no_end_month_mask

    filtered_data.loc[out_ival_mask_relevant, "end_month"] = 6
    filtered_data.loc[in_ival_mask_relevant, "end_month"] = filtered_data.loc[
        in_ival_mask_relevant, "defense_month"
    ]

    filtered_data.loc[in_ival_mask_relevant, "end_year"] = filtered_data.loc[
        in_ival_mask_relevant, "defense_year"
    ]
    return filtered_data


def check_col_in_month_ival(
    data,
    checked_year_col_name,
    checker_year_col_name,
    checker_month_col_name,
    slack_months=3,
):
    checked_year = data[checked_year_col_name]
    checker_year = data[checker_year_col_name]
    checker_month = data[checker_month_col_name]

    same_year_mask = checked_year == checker_year

    month_over_mask = checker_month + slack_months > 12
    month_under_mask = checker_month - slack_months < 1

    next_year_mask = checker_year + 1 == checked_year
    previous_year_mask = checker_year - 1 == checked_year

    yr_in_upper_ival_mask = next_year_mask & month_over_mask
    yr_in_lower_ival_mask = previous_year_mask & month_under_mask

    return (same_year_mask | yr_in_upper_ival_mask | yr_in_lower_ival_mask).fillna(
        False
    )


def extract_month(date_str):
    if pd.isna(date_str):
        return pd.NA
    try:
        return datetime.strptime(date_str, "%d-%m-%Y").month
    except ValueError:
        return pd.NA


def start_and_acceptance(joined_data, mode="official", months_soft=12):
    if mode == "official":
        late_start_mask_over12 = within_months_of_signed(
            joined_data,
            "start_year",
            "start_month",
            "acceptance_year",
            "acceptance_month",
            months=months_soft,
        )
        late_start_mask_under12 = within_months_of_signed(
            joined_data,
            "start_year",
            "start_month",
            "acceptance_year",
            "acceptance_month",
            months=1,
        )
        mask_correct = late_start_mask_under12 & ~late_start_mask_over12
        joined_data.loc[mask_correct, ["start_year", "start_month"]] = joined_data.loc[
            mask_correct, ["acceptance_year", "acceptance_month"]
        ].to_numpy()
        joined_data.loc[late_start_mask_over12, ["start_year", "start_month"]] = pd.NA
    return joined_data


def start_and_pba(joined_data, months=12):
    mask_more_than12_before_pba = within_months_of_signed(
        joined_data, "pba_year", "pba_month", "start_year", "start_month", months=months
    )
    joined_data.loc[mask_more_than12_before_pba, ["start_year", "start_month"]] = pd.NA
    return joined_data


def end_after_defense(joined_data):
    end_after_defense_mask = within_months_of_signed(
        joined_data, "end_year", "end_month", "defense_year", "defense_month", months=1
    )
    joined_data.loc[end_after_defense_mask, ["end_year", "end_month"]] = (
        joined_data.loc[
            end_after_defense_mask, ["defense_year", "defense_month"]
        ].to_numpy()
    )
    return joined_data


def end_before_defense(joined_data, months=12, mode="official"):
    if mode == "official":
        mask_end_before_defense_over12 = within_months_of_signed(
            joined_data,
            "defense_year",
            "defense_month",
            "end_year",
            "end_month",
            months=months,
        )
        mask_end_before_defense = within_months_of_signed(
            joined_data,
            "defense_year",
            "defense_month",
            "end_year",
            "end_month",
            months=1,
        )
        mask_end_before_defense_under12 = (
            mask_end_before_defense & ~mask_end_before_defense_over12
        )
        joined_data.loc[mask_end_before_defense_over12, ["end_year", "end_month"]] = (
            pd.NA
        )

        joined_data.loc[mask_end_before_defense_under12, ["end_year", "end_month"]] = (
            joined_data.loc[
                mask_end_before_defense_under12, ["defense_year", "defense_month"]
            ].to_numpy()
        )
    return joined_data


def within_months_of_signed(
    data,
    colname_yr_checked,
    colname_mo_checked,
    colname_yr_ref,
    colname_mo_ref,
    months=3,
):
    # return true if checked < / > ref + months, depending on larger=True
    mask = data[colname_yr_checked].notna() & data[colname_mo_checked].notna()
    mask &= data[colname_yr_ref].notna() & data[colname_mo_ref].notna()

    checked_years = data[colname_yr_checked]
    checked_months = data[colname_mo_checked]
    ref_months = data[colname_mo_ref]
    ref_years = data[colname_yr_ref]

    checked_total_months = checked_years * 12 + checked_months - 1
    ref_total_months = ref_years * 12 + ref_months - 1

    return (checked_total_months >= ref_total_months + months).fillna(False)


def check_start_after_acceptance(
    accept_year, accept_month, start_year, start_month, slack_months=3
):
    # return True if start date is after acceptance date plus slack_months
    accept_month_adjusted = accept_month + slack_months
    if accept_month_adjusted > 12:
        accept_year += 1
        accept_month_adjusted -= 12
    if start_year > accept_year:
        return True
    if start_year == accept_year and start_month > accept_month_adjusted:
        return True
    return False


def check_startafter_all(filtered_data, slack_months=3):
    mask = filtered_data["start_year"].notna() & filtered_data["end_year"].notna()
    filtered_data = filtered_data[mask]

    counter = 0
    nmask = (
        filtered_data["acceptance_year"].notna()
        & filtered_data["acceptance_month"].notna()
    )
    for row in filtered_data[nmask].itertuples():
        if check_start_after_acceptance(
            row.acceptance_year,
            row.acceptance_month,
            row.start_year,
            row.start_month,
            slack_months=slack_months,
        ):
            counter += 1
    return counter / len(filtered_data[nmask])


def check_defense_before_end(def_year, def_month, end_year, end_month, slack_months=3):
    # return True if defense date is before end date minus slack_months
    end_month_adjusted = end_month - slack_months
    if end_month_adjusted <= 0:
        end_year -= 1
        end_month_adjusted += 12
    if def_year < end_year:
        return True
    if def_year == end_year and def_month < end_month_adjusted:
        return True
    return False


def check_defbefore_all(filtered_data, slack_months=3):
    mask = filtered_data["start_year"].notna() & filtered_data["end_year"].notna()
    filtered_data = filtered_data[mask]

    counter = 0
    for row in filtered_data.itertuples():
        if check_defense_before_end(
            row.defense_year,
            row.defense_month,
            row.end_year,
            row.end_month,
            slack_months=slack_months,
        ):
            counter += 1
    return counter / len(filtered_data)


def deviation_distribution(
    data,
    year_checked_colname,
    month_checked_colname,
    year_ref_colname,
    month_ref_colname,
    slack_months=2,
):
    rel_data = data[data[year_ref_colname].notna() & data[month_ref_colname].notna()]

    delta_months = (
        rel_data[year_checked_colname] - rel_data[year_ref_colname]
    ) * 12 + (rel_data[month_checked_colname] - rel_data[month_ref_colname])
    return np.where(delta_months <= slack_months, 0, delta_months)


def get_large_deviations(
    data,
    year_checked_colname,
    month_checked_colname,
    year_ref_colname,
    month_ref_colname,
    threshold=10,
):
    rel_data = data[data[year_ref_colname].notna() & data[month_ref_colname].notna()]

    delta_months = (
        rel_data[year_checked_colname] - rel_data[year_ref_colname]
    ) * 12 + (rel_data[month_checked_colname] - rel_data[month_ref_colname])

    large_deviations = rel_data[delta_months > threshold]
    return large_deviations, delta_months[delta_months > threshold]


# data = load_data("data/clean_data.csv", "data/cleaned_data_dtypes.json")
# durations = pd.read_csv("data/durations.csv")


def add_merge_linkedin_data(data, fname_durations):
    durations = pd.read_csv(fname_durations)
    joined_data = pd.merge(
        data, durations.drop(columns=["full_name"]), on="pagination_nr", how="left"
    )

    # mask where no valid duration/years was input
    start_null_mask = joined_data["start_year"] == 0.0
    end_null_mask = joined_data["end_year"] == 0.0
    nulldur_mask = start_null_mask | end_null_mask

    # setting no NA for clarity
    joined_data.loc[nulldur_mask, "duration"] = pd.NA
    joined_data.loc[nulldur_mask, "start_year"] = pd.NA
    joined_data.loc[nulldur_mask, "end_year"] = pd.NA
    joined_data.loc[nulldur_mask, "start_month"] = pd.NA
    joined_data.loc[nulldur_mask, "end_month"] = pd.NA

    joined_data = joined_data.rename(columns={"duration": "duration_linkedin"})

    # rename duration to duration_linkedin
    int_cols = [
        "start_year",
        "end_year",
        "start_month",
        "end_month",
        "acceptance_year",
        "pba_year",
        "defense_year",
        "acceptance_month",
        "pba_month",
        "defense_month",
    ]

    for c in int_cols:
        if c in joined_data.columns:
            joined_data[c] = joined_data[c].astype("Int64")

    acceptance_date = joined_data["acceptance_date"]
    pba_date = joined_data["pba_date"]
    defense_date = joined_data["defense_date"]

    acceptance_month = acceptance_date.apply(extract_month)
    joined_data["acceptance_month"] = acceptance_month

    pba_month = pba_date.apply(extract_month)
    joined_data["pba_month"] = pba_month

    defense_month = defense_date.apply(extract_month)
    joined_data["defense_month"] = defense_month

    # keep raw start and end dates with suffix _raw
    joined_data["start_year_raw"] = joined_data["start_year"]
    joined_data["end_year_raw"] = joined_data["end_year"]
    joined_data["start_month_raw"] = joined_data["start_month"]
    joined_data["end_month_raw"] = joined_data["end_month"]

    joined_data = impute_start_month_year(joined_data, slack_months=2)
    joined_data = impute_end_month_year(joined_data, slack_months=2)

    joined_data = start_and_acceptance(joined_data, mode="official", months_soft=12)
    joined_data = start_and_pba(joined_data, months=12)
    joined_data = end_after_defense(joined_data)
    joined_data = end_before_defense(joined_data, months=12, mode="official")

    computed_duration = get_duration(
        joined_data["start_year"],
        joined_data["end_year"],
        joined_data["start_month"],
        joined_data["end_month"],
    )
    joined_data["duration_computed"] = computed_duration

    return joined_data


def get_duration(start_year, end_year, start_month, end_month):
    duration = (end_year - start_year) + (end_month - start_month) / 12
    return duration


# joined_data = add_merge_linkedin_data(data, "data/durations.csv")
#
# slacks = np.arange(0, 13)
# defbefore_ratios = []
# acceptance_ratios = []
#
# delta_end = []
# delta_start = []
#
# for slack in slacks:
#     defbefore_ratio = check_defbefore_all(joined_data, slack_months=slack)
#     acceptance_ratio = check_startafter_all(joined_data, slack_months=slack)
#     defbefore_ratios.append(defbefore_ratio)
#     acceptance_ratios.append(acceptance_ratio)
#
#     # starts AFTER acceptance
#     d_start = deviation_distribution(
#         joined_data,
#         "start_year",
#         "start_month",
#         "acceptance_year",
#         "acceptance_month",
#         slack_months=slack,
#     )
#     # ends BEFORE defense
#     d_end = deviation_distribution(
#         joined_data,
#         "end_year",
#         "end_month",
#         "defense_year",
#         "defense_month",
#         slack_months=-slack,
#     )
#     delta_start.append(d_start)
#     delta_end.append(d_end)
#
# if False:
#     plt.plot(slacks, defbefore_ratios, label="Defense before end")
#     plt.plot(slacks, acceptance_ratios, label="Start after acceptance")
#     plt.xlabel("Slack months")
#     plt.ylabel("Ratio")
#     plt.title("Ratio of consistent dates vs. slack months")
#     plt.legend()
#     plt.grid()
#     plt.show()
# if False:
#     plt.boxplot(delta_start[0], positions=[0], widths=0.4)
#     plt.boxplot(delta_end[0], positions=[1], widths=0.4)
#     plt.axhline(2, color="red", linestyle="--", label="2 months")
#     plt.xticks([0, 1], ["Start - Acceptance", "End - Defense"])
#     plt.ylabel("Months deviation")
#     plt.title("Deviation distributions vs. slack months")
#     plt.legend()
#     plt.grid()
#     plt.show()
#


def print_large_deviations(
    filtered_data,
    col_name_checked_yr,
    col_name_checked_mo,
    col_name_ref_yr,
    col_name_ref_mo,
    threshold=10,
):
    large_deviations, d_months = get_large_deviations(
        joined_data,
        col_name_checked_yr,
        col_name_checked_mo,
        col_name_ref_yr,
        col_name_ref_mo,
        threshold=threshold,
    )
    large_deviations["dev"] = d_months
    print(
        large_deviations[
            [
                "pagination_nr",
                col_name_checked_yr,
                col_name_checked_mo,
                col_name_ref_yr,
                col_name_ref_mo,
                "dev",
            ]
        ].sort_values(by="dev", ascending=False)
    )


# print_large_deviations(
#     joined_data, "pba_year", "pba_month", "start_year", "start_month", threshold=12
# )

# print_large_deviations(
#     filtered_data,
#     "start_year",
#     "start_month",
#     "acceptance_year",
#     "acceptance_month",
#     threshold=10,
# )

# print_large_deviations(
#     filtered_data,
#     "end_year",
#     "end_month",
#     "defense_year",
#     "defense_month",
#     threshold=10,
# )
# print_large_deviations(
#     filtered_data,
#     "defense_year",
#     "defense_month",
#     "end_year",
#     "end_month",
#     threshold=10,
# )
