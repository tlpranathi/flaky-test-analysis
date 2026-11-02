import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.config import (
    OUTPUT_DIR,
    TARGET_COL
)

from src.feature_engineering import FINAL_DATASET

# OUTPUT FOLDERS

EDA_DIR = os.path.join(OUTPUT_DIR, "eda")

TABLES_DIR = os.path.join(EDA_DIR, "tables")

FIGURES_DIR = os.path.join(EDA_DIR, "figures")

# PLOT COLOURS

NON_FLAKY_COLOR = "#4C72B0"

FLAKY_COLOR = "#C44E52"

# FAILURE FREQUENCY DEFINITIONS

# Outcome of the reruns for each test, in display order

OUTCOME_ORDER = [
    "Always passed",
    "Failed intermittently",
    "Always failed",
    "No rerun data"
]

# fail_rate_pct buckets for flaky tests (right edge included)

FAIL_RATE_BINS = [0, 0.1, 1, 10, 50, 100]

FAIL_RATE_LABELS = ["<=0.1%", "0.1-1%", "1-10%", "10-50%", ">50%"]

# 1. LOAD DATASET

def load_dataset(path=FINAL_DATASET):
    """Load the final dataset produced by feature_engineering.py."""

    print("\nLoading final dataset...")

    df = pd.read_csv(path)

    print("Rows:", len(df))
    print("Columns:", len(df.columns))

    return df

# 2. SAVE HELPERS

def save_table(df, name, index=False):
    """Save a summary table to outputs/eda/tables."""

    os.makedirs(TABLES_DIR, exist_ok=True)

    path = os.path.join(TABLES_DIR, name)

    df.to_csv(path, index=index)

    print("Table saved to:", path)


def save_figure(fig, name):
    """Save a figure to outputs/eda/figures and close it."""

    os.makedirs(FIGURES_DIR, exist_ok=True)

    path = os.path.join(FIGURES_DIR, name)

    fig.savefig(path, dpi=150, bbox_inches="tight")

    plt.close(fig)

    print("Figure saved to:", path)

# 3. CLASS DISTRIBUTION

def class_distribution(df):
    """Count flaky vs non-flaky tests and measure the class imbalance."""

    print("\n========== CLASS DISTRIBUTION ==========")

    counts = df[TARGET_COL].value_counts()

    non_flaky = int(counts.get(0, 0))
    flaky = int(counts.get(1, 0))

    table = pd.DataFrame({
        "label": ["Non-Flaky", "Flaky"],
        "tests": [non_flaky, flaky]
    })

    table["percentage"] = (table["tests"] / len(df) * 100).round(2)

    imbalance_ratio = round(non_flaky / flaky, 2)

    print(table.to_string(index=False))
    print(f"Imbalance ratio: {imbalance_ratio} non-flaky tests per flaky test")

    save_table(table, "class_distribution.csv")

    fig, ax = plt.subplots(figsize=(6, 4.5))

    bars = ax.bar(
        table["label"],
        table["tests"],
        color=[NON_FLAKY_COLOR, FLAKY_COLOR]
    )

    for bar, pct in zip(bars, table["percentage"]):

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{int(bar.get_height()):,}\n({pct}%)",
            ha="center",
            va="bottom"
        )

    ax.set_ylim(0, table["tests"].max() * 1.15)
    ax.set_ylabel("Number of tests")
    ax.set_title("Class Distribution: Flaky vs Non-Flaky Tests")

    save_figure(fig, "class_distribution.png")

    return table, imbalance_ratio

# 4. FLAKY TESTS PER PROJECT

def flaky_per_project(df):
    """Count tests, flaky tests and flaky rate for every project."""

    print("\n========== FLAKY TESTS PER PROJECT ==========")

    table = df.groupby("project").agg(
        tests=(TARGET_COL, "size"),
        flaky_tests=(TARGET_COL, "sum")
    )

    table["non_flaky_tests"] = table["tests"] - table["flaky_tests"]

    table["flaky_pct"] = (
        table["flaky_tests"] / table["tests"] * 100
    ).round(2)

    table["share_of_all_flaky_pct"] = (
        table["flaky_tests"] / table["flaky_tests"].sum() * 100
    ).round(2)

    table = table.sort_values(
        ["flaky_tests", "flaky_pct"],
        ascending=False
    )

    table = table[[
        "tests",
        "flaky_tests",
        "non_flaky_tests",
        "flaky_pct",
        "share_of_all_flaky_pct"
    ]]

    print(table.to_string())

    no_flaky = list(table.index[table["flaky_tests"] == 0])

    print("\nProjects with no flaky tests:", no_flaky if no_flaky else "none")

    top3_share = table["share_of_all_flaky_pct"].head(3).sum().round(2)

    print(
        f"Top 3 projects ({', '.join(table.index[:3])}) "
        f"hold {top3_share}% of all flaky tests"
    )

    save_table(table.reset_index(), "flaky_per_project.csv")

    # Projects are ordered by flaky count in both panels

    fig, (ax_count, ax_rate) = plt.subplots(
        1, 2,
        figsize=(14, 8),
        sharey=True
    )

    projects = table.index

    ax_count.barh(projects, table["flaky_tests"], color=FLAKY_COLOR)
    ax_rate.barh(projects, table["flaky_pct"], color=NON_FLAKY_COLOR)

    for ax, values, fmt in [
        (ax_count, table["flaky_tests"], "{:,.0f}"),
        (ax_rate, table["flaky_pct"], "{:.1f}%")
    ]:

        for y, value in enumerate(values):

            ax.text(
                value,
                y,
                " " + fmt.format(value),
                va="center",
                fontsize=8
            )

        ax.set_xlim(0, values.max() * 1.15)

    ax_count.invert_yaxis()

    ax_count.set_xlabel("Flaky tests")
    ax_rate.set_xlabel("Flaky rate (% of project's tests)")

    ax_count.set_title("Flaky Tests per Project (count)")
    ax_rate.set_title("Flaky Tests per Project (rate)")

    save_figure(fig, "flaky_per_project.png")

    return table

# 5. FAILURE FREQUENCY

def classify_outcomes(df):
    """Label each test by what happened across its reruns."""

    return pd.Series(
        np.select(
            [
                df["NumFailingRuns"].isna(),
                df["NumFailingRuns"] == 0,
                df["NumPassingRuns"] == 0
            ],
            [
                "No rerun data",
                "Always passed",
                "Always failed"
            ],
            default="Failed intermittently"
        ),
        index=df.index
    )


def failure_frequency(df):
    """Analyse how often tests fail across reruns (EDA only, uses rerun columns)."""

    print("\n========== FAILURE FREQUENCY ==========")

    # A. Rerun outcome of every test, split by flaky label

    outcome = classify_outcomes(df)

    outcomes = pd.DataFrame({
        "outcome": OUTCOME_ORDER,
        "flaky_tests": [
            int(((outcome == name) & (df[TARGET_COL] == 1)).sum())
            for name in OUTCOME_ORDER
        ],
        "non_flaky_tests": [
            int(((outcome == name) & (df[TARGET_COL] == 0)).sum())
            for name in OUTCOME_ORDER
        ]
    })

    outcomes["total_tests"] = (
        outcomes["flaky_tests"] + outcomes["non_flaky_tests"]
    )

    outcomes["percentage"] = (
        outcomes["total_tests"] / len(df) * 100
    ).round(2)

    print("\nRerun outcome per test:")
    print(outcomes.to_string(index=False))

    save_table(outcomes, "failure_outcomes.csv")

    # B. Fail-rate buckets for flaky tests

    flaky = df[df[TARGET_COL] == 1]

    fail_rate = flaky["fail_rate_pct"].dropna()

    buckets = pd.cut(
        fail_rate,
        bins=FAIL_RATE_BINS,
        labels=FAIL_RATE_LABELS,
        right=True
    )

    rate_table = (
        buckets.value_counts()
        .reindex(FAIL_RATE_LABELS)
        .rename_axis("fail_rate_bucket")
        .reset_index(name="flaky_tests")
    )

    rate_table["percentage"] = (
        rate_table["flaky_tests"] / len(fail_rate) * 100
    ).round(2)

    print("\nFail rate of flaky tests:")
    print(rate_table.to_string(index=False))

    print(
        f"Median fail rate: {fail_rate.median():.2f}%, "
        f"mean: {fail_rate.mean():.2f}%"
    )

    save_table(rate_table, "flaky_fail_rate_buckets.csv")

    # C. Distinct exception types seen in failing runs of flaky tests

    exceptions = (
        flaky["UniqueFailingExceptionTypes"]
        .dropna()
        .astype(int)
        .value_counts()
        .sort_index()
        .rename_axis("unique_exception_types")
        .reset_index(name="flaky_tests")
    )

    exceptions["percentage"] = (
        exceptions["flaky_tests"] / exceptions["flaky_tests"].sum() * 100
    ).round(2)

    print("\nUnique exception types per flaky test:")
    print(exceptions.to_string(index=False))

    save_table(exceptions, "flaky_exception_types.csv")

    # Tests that always fail are broken, not flaky

    always_failed = int(outcomes.loc[
        outcomes["outcome"] == "Always failed", "non_flaky_tests"
    ].iloc[0])

    print(
        f"\n{always_failed} non-flaky tests failed in every run "
        "(consistently broken, not flaky)."
    )

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))

    panels = [
        (
            axes[0],
            outcomes["outcome"],
            outcomes["total_tests"],
            "Rerun Outcome per Test (log scale)",
            True
        ),
        (
            axes[1],
            rate_table["fail_rate_bucket"],
            rate_table["flaky_tests"],
            "Fail Rate of Flaky Tests",
            False
        ),
        (
            axes[2],
            exceptions["unique_exception_types"].astype(str),
            exceptions["flaky_tests"],
            "Unique Exception Types per Flaky Test",
            False
        )
    ]

    for ax, labels, values, title, log in panels:

        ax.bar(labels, values, color=FLAKY_COLOR)

        if log:
            ax.set_yscale("log")
            ax.set_ylim(top=values.max() * 3)

        for x, value in enumerate(values):

            ax.text(
                x,
                value,
                f"{int(value):,}",
                ha="center",
                va="bottom",
                fontsize=9
            )

        ax.set_title(title)
        ax.set_ylabel("Number of tests")

    axes[0].tick_params(axis="x", rotation=15)

    axes[1].set_xlabel("Failing runs as % of total runs")
    axes[2].set_xlabel("Distinct exception types")

    fig.tight_layout()

    save_figure(fig, "failure_frequency.png")

    return outcomes, rate_table, exceptions

# 6. SUMMARY TABLE

def build_summary(df, project_table, imbalance_ratio):
    """Headline numbers for the presentation, saved as one small table."""

    print("\n========== EDA SUMMARY ==========")

    flaky = df[df[TARGET_COL] == 1]

    fail_rate = flaky["fail_rate_pct"].dropna()

    by_rate = project_table.sort_values("flaky_pct", ascending=False)

    summary = pd.DataFrame([
        ("total_tests", len(df)),
        ("projects", df["project"].nunique()),
        ("flaky_tests", len(flaky)),
        ("non_flaky_tests", len(df) - len(flaky)),
        ("flaky_pct", round(len(flaky) / len(df) * 100, 2)),
        ("imbalance_ratio_non_flaky_per_flaky", imbalance_ratio),
        ("projects_with_flaky_tests", int((project_table["flaky_tests"] > 0).sum())),
        (
            "project_most_flaky_tests",
            f"{project_table.index[0]} ({project_table['flaky_tests'].iloc[0]})"
        ),
        (
            "project_highest_flaky_pct",
            f"{by_rate.index[0]} ({by_rate['flaky_pct'].iloc[0]}%)"
        ),
        ("tests_without_rerun_data", int(df["NumFailingRuns"].isna().sum())),
        (
            "non_flaky_tests_always_failing",
            int(((df["NumPassingRuns"] == 0) & (df["NumFailingRuns"] > 0)).sum())
        ),
        ("flaky_median_fail_rate_pct", round(fail_rate.median(), 2)),
        ("flaky_mean_fail_rate_pct", round(fail_rate.mean(), 2))
    ], columns=["metric", "value"])

    print(summary.to_string(index=False))

    save_table(summary, "eda_summary.csv")

    return summary

# 7. MAIN EXECUTION

def main():

    df = load_dataset()

    _, imbalance_ratio = class_distribution(df)

    project_table = flaky_per_project(df)

    failure_frequency(df)

    build_summary(df, project_table, imbalance_ratio)


if __name__ == "__main__":
    main()
