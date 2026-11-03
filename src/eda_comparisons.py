import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from src.config import (
    COVERAGE_COLS,
    DEPENDENCY_COLS,
    EXECUTION_COLS,
    SMELL_COLS,
    TARGET_COL,
    TEST_CHARACTERISTIC_COLS,
    WINDOWS,
    RENAME,
)
from src.eda_distribution import save_figure
from src.feature_engineering import (
    EXEC_TIME_LABELS,
    LIBS_LABELS,
)


NON_FLAKY_COLOR = "#4C72B0"
FLAKY_COLOR = "#C44E52"
CLASS_GROUPS = [
    (0, "Non-Flaky", NON_FLAKY_COLOR),
    (1, "Flaky", FLAKY_COLOR),
]

NUMERICAL_COMPARISON_COLS = [
    RENAME[column]
    for column in (
        EXECUTION_COLS
        + COVERAGE_COLS
        + TEST_CHARACTERISTIC_COLS
        + DEPENDENCY_COLS
    )
]
CHURN_COLS = [
    RENAME[f"hIndexModificationsPerCoveredLine_window{window}"]
    for window in WINDOWS
]
RENAMED_SMELL_COLS = [RENAME[column] for column in SMELL_COLS]


def _require_columns(df, columns):
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"DataFrame is missing required EDA columns: {missing}")


def _validate_binary_values(df, column):
    values = df[column].dropna()
    if not values.isin([0, 1]).all():
        raise ValueError(f"{column} must contain only binary 0/1 values.")


def compare_numerical_feature_distributions(df):
    """Compare numerical feature distributions for flaky and non-flaky tests."""
    _require_columns(df, [TARGET_COL] + NUMERICAL_COMPARISON_COLS)
    _validate_binary_values(df, TARGET_COL)

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.flatten()

    for ax, column in zip(axes, NUMERICAL_COMPARISON_COLS):
        samples = []
        labels = []
        colors = []

        for value, label, color in CLASS_GROUPS:
            sample = df.loc[df[TARGET_COL] == value, column].dropna()
            if not sample.empty:
                samples.append(sample.to_numpy())
                labels.append(f"{label}\n(n={len(sample):,})")
                colors.append(color)

        if samples:
            boxes = ax.boxplot(
                samples,
                patch_artist=True,
                showfliers=False,
                medianprops={"color": "black"},
            )
            for box, color in zip(boxes["boxes"], colors):
                box.set_facecolor(color)
                box.set_alpha(0.75)
            ax.set_xticks(range(1, len(labels) + 1), labels)
        else:
            ax.text(0.5, 0.5, "No observations", ha="center", va="center")

        ax.set_title(column.replace("_", " ").title())
        ax.set_xlabel("Test class")
        ax.set_ylabel("Feature value")
        ax.grid(axis="y", alpha=0.25)

    for ax in axes[len(NUMERICAL_COMPARISON_COLS):]:
        ax.set_visible(False)

    fig.suptitle("Numerical Feature Distributions by Flaky Label", fontsize=15)
    fig.legend(
        handles=[
            Patch(facecolor=color, label=label, alpha=0.75)
            for _, label, color in CLASS_GROUPS
        ],
        loc="lower center",
        ncol=2,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    save_figure(fig, "numerical_feature_distributions_by_label.png")


def _bucket_flaky_rate(df, bucket_column, categories, title, filename):
    _require_columns(df, [TARGET_COL, bucket_column])
    _validate_binary_values(df, TARGET_COL)

    analysis = df.loc[
        df[TARGET_COL].isin([0, 1]), [bucket_column, TARGET_COL]
    ].copy()
    analysis["_bucket"] = (
        analysis[bucket_column]
        .astype(object)
        .astype("string")
        .fillna("Missing")
    )

    bucket_order = [str(category) for category in categories]
    if analysis["_bucket"].eq("Missing").any():
        bucket_order.append("Missing")

    grouped = analysis.groupby("_bucket", sort=False)[TARGET_COL].agg(
        tests="count",
        flaky_rate="mean",
    )
    summary = grouped.reindex(bucket_order)
    summary["tests"] = summary["tests"].fillna(0).astype(int)
    summary["flaky_rate"] = summary["flaky_rate"] * 100

    fig, ax = plt.subplots(figsize=(9, 5))
    positions = np.arange(len(bucket_order))
    bars = ax.bar(
        positions,
        summary["flaky_rate"].fillna(0),
        color=FLAKY_COLOR,
        width=0.7,
    )

    for bar, tests, rate in zip(
        bars, summary["tests"], summary["flaky_rate"]
    ):
        if tests:
            ax.annotate(
                f"{rate:.1f}%\nn={tests:,}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    if not summary["tests"].sum():
        ax.text(0.5, 0.5, "No observations", ha="center", va="center",
                transform=ax.transAxes)

    ax.set_xticks(positions, bucket_order)
    ax.set_ylim(0, 100)
    ax.set_xlabel(bucket_column.replace("_", " ").title())
    ax.set_ylabel("Flaky tests (% of tests in bucket)")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save_figure(fig, filename)
    return summary


def flaky_rate_by_execution_time(df):
    """Plot flaky percentage by the existing execution-time buckets."""
    return _bucket_flaky_rate(
        df,
        "execution_time_bucket",
        EXEC_TIME_LABELS,
        "Flaky-Test Rate by Execution-Time Bucket",
        "flaky_rate_by_execution_time_bucket.png",
    )


def flaky_rate_by_third_party_libraries(df):
    """Plot flaky percentage by the existing third-party-library buckets."""
    return _bucket_flaky_rate(
        df,
        "third_party_libs_bucket",
        LIBS_LABELS,
        "Flaky-Test Rate by Third-Party-Library Bucket",
        "flaky_rate_by_third_party_lib_bucket.png",
    )


def compare_changed_line_churn(df):
    """Compare changed-line churn across the available commit windows."""
    _require_columns(df, [TARGET_COL] + CHURN_COLS)
    _validate_binary_values(df, TARGET_COL)

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.flatten()

    for ax, column, window in zip(axes, CHURN_COLS, WINDOWS):
        samples = []
        labels = []
        colors = []

        for value, label, color in CLASS_GROUPS:
            sample = df.loc[df[TARGET_COL] == value, column].dropna()
            if not sample.empty:
                samples.append(sample.to_numpy())
                labels.append(f"{label}\n(n={len(sample):,})")
                colors.append(color)

        if samples:
            boxes = ax.boxplot(
                samples,
                patch_artist=True,
                showfliers=False,
                medianprops={"color": "black"},
            )
            for box, color in zip(boxes["boxes"], colors):
                box.set_facecolor(color)
                box.set_alpha(0.75)
            ax.set_xticks(range(1, len(labels) + 1), labels)
        else:
            ax.text(0.5, 0.5, "No observations", ha="center", va="center")

        ax.set_title(f"Last {window} commits")
        ax.set_xlabel("Test class")
        ax.set_ylabel("Changed-line churn")
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle("Changed-Line Churn by Flaky Label and Commit Window",
                 fontsize=15)
    fig.legend(
        handles=[
            Patch(facecolor=color, label=label, alpha=0.75)
            for _, label, color in CLASS_GROUPS
        ],
        loc="lower center",
        ncol=2,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    save_figure(fig, "changed_line_churn_by_label.png")


def compare_test_smell_prevalence(df):
    """Compare within-class smell prevalence to account for label imbalance."""
    _require_columns(df, [TARGET_COL] + RENAMED_SMELL_COLS)
    _validate_binary_values(df, TARGET_COL)
    for column in RENAMED_SMELL_COLS:
        _validate_binary_values(df, column)

    class_sizes = {
        value: int((df[TARGET_COL] == value).sum())
        for value, _, _ in CLASS_GROUPS
    }
    positions = np.arange(len(RENAMED_SMELL_COLS))
    width = 0.36
    fig, ax = plt.subplots(figsize=(14, 6))

    for offset, (value, label, color) in zip(
        [-width / 2, width / 2], CLASS_GROUPS
    ):
        percentages = []
        annotations = []
        for column in RENAMED_SMELL_COLS:
            sample = df.loc[df[TARGET_COL] == value, column].dropna()
            percentages.append(sample.mean() * 100 if len(sample) else 0)
            annotations.append(
                f"{int(sample.sum())}/{len(sample):,}" if len(sample) else "0/0"
            )

        bars = ax.bar(
            positions + offset,
            percentages,
            width,
            color=color,
            label=f"{label} (n={class_sizes[value]:,})",
        )
        for bar, annotation in zip(bars, annotations):
            ax.annotate(
                annotation,
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
            )

    ax.set_xticks(
        positions,
        [column[len("smell_"):].replace("_", "\n")
         for column in RENAMED_SMELL_COLS],
    )
    ax.set_ylabel("Tests with smell (% within test class)")
    ax.set_xlabel("Test smell")
    ax.set_title("Test-Smell Prevalence Within Flaky and Non-Flaky Tests")
    ax.set_ylim(0, 100)
    ax.legend(title="Class sample size")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save_figure(fig, "test_smell_prevalence_by_label.png")


def main(df):
    """Generate all comparison plots from the prepared dataset."""
    compare_numerical_feature_distributions(df)
    flaky_rate_by_execution_time(df)
    flaky_rate_by_third_party_libraries(df)
    compare_changed_line_churn(df)
    compare_test_smell_prevalence(df)
