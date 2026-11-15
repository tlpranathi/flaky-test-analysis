import os
import numpy as np
import pandas as pd

from src.train_models import require_inputs

import matplotlib.pyplot as plt

from src.config import OUTPUT_DIR, TARGET_COL, RENAME, SMELL_COLS
from src.feature_engineering import (
    FINAL_DATASET,
    EXEC_TIME_LABELS,
    LIBS_LABELS,
    COVERAGE_LABELS
)
from src.plot_style import (
    apply_style,
    save_figure,
    BLUE,
    AXIS,
    TEXT_MUTED,
    TEXT_SECONDARY
)

# SETTINGS

MIN_GROUP_SIZE = 50

TOP_PROJECTS = 10

CHURN_COL = RENAME["hIndexModificationsPerCoveredLine_window100"]

CHURN_BINS = [-1, 0, 1, 2, 5, np.inf]

CHURN_LABELS = ["0", "1", "2", "3-5", ">5"]

SMELL_FEATURES = [RENAME[col] for col in SMELL_COLS]

# OUTPUT FILES

PATTERN_DIR = os.path.join(OUTPUT_DIR, "patterns")

PATTERNS_FILE = os.path.join(PATTERN_DIR, "patterns.csv")

FIGS = {
    "execution_time": os.path.join(PATTERN_DIR, "p1_execution_time.png"),
    "coverage_libraries": os.path.join(PATTERN_DIR, "p2_coverage_and_libraries.png"),
    "churn": os.path.join(PATTERN_DIR, "p3_churn.png"),
    "smells": os.path.join(PATTERN_DIR, "p4_smells.png"),
    "projects": os.path.join(PATTERN_DIR, "p5_projects.png")
}

# 1. LOAD DATA

def load_data():
    """Load the final dataset (rerun-outcome leakage columns are not used)."""

    require_inputs([FINAL_DATASET])

    df = pd.read_csv(FINAL_DATASET)

    print("\nTests:", len(df))
    print(f"Overall flaky share: {df[TARGET_COL].mean() * 100:.2f}%")

    return df

# 2. FLAKY SHARE PER GROUP

def flaky_share(df, col, order=None):
    """Tests, flaky tests and flaky % for each value of col."""

    table = df.groupby(col, observed=False)[TARGET_COL].agg(
        tests="size",
        flaky="sum"
    )

    table["flaky_pct"] = (table["flaky"] / table["tests"] * 100).round(2)

    if order is not None:
        table = table.reindex(order)

    return table


def highest_vs_lowest(table):
    """Highest and lowest flaky-% groups, ignoring groups under 50 tests."""

    big = table[table["tests"] >= MIN_GROUP_SIZE]

    return big["flaky_pct"].idxmax(), big["flaky_pct"].idxmin(), big

# 3. CHART HELPERS

def bar_panel(ax, table, overall, xlabel):
    """Flaky % per bucket, labelled with % and test count, plus the overall line."""

    x = np.arange(len(table))

    ax.bar(x, table["flaky_pct"], color=BLUE, width=0.6)

    for i, pct in enumerate(table["flaky_pct"]):
        ax.text(i, pct, f"{pct:.1f}%\n", ha="center", va="bottom",
                fontsize=9, color=TEXT_SECONDARY)

    ax.axhline(overall, color=TEXT_MUTED, linestyle="--", linewidth=1)
    ax.text(len(table) - 0.45, overall, f" all tests\n {overall:.1f}%",
            ha="left", va="center", fontsize=8, color=TEXT_SECONDARY, clip_on=False)

    ticks = [f"{label}\nn={n:,}" for label, n in zip(table.index.astype(str), table["tests"])]

    ax.set_xticks(x, ticks)
    ax.grid(axis="x", visible=False)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("% of tests that are flaky")
    ax.set_ylim(0, table["flaky_pct"].max() * 1.25)

# 4. PATTERNS

def pattern_execution_time(df, overall):
    """P1: slow tests are far more often flaky."""

    table = flaky_share(df, "execution_time_bucket", EXEC_TIME_LABELS)

    hi, lo, _ = highest_vs_lowest(table)

    statement = (
        f"Slow tests are much more likely to be flaky: {table.loc[hi, 'flaky_pct']:.1f}% of tests "
        f"taking {hi} are flaky, vs {table.loc[lo, 'flaky_pct']:.1f}% of tests taking {lo} "
        f"({table.loc[hi, 'flaky_pct'] / table.loc[lo, 'flaky_pct']:.0f}x)."
    )

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bar_panel(ax, table, overall, "Execution time")
    ax.set_title("P1. Flaky share rises with execution time")

    save_figure(fig, FIGS["execution_time"])

    return statement, table


def pattern_coverage_libraries(df, overall):
    """P2: tests covering more code and using many libraries are riskier."""

    coverage = flaky_share(df, "source_lines_covered_bucket", COVERAGE_LABELS)
    libs = flaky_share(df, "third_party_libs_bucket", LIBS_LABELS)

    c_hi, c_lo, _ = highest_vs_lowest(coverage)
    l_hi, l_lo, _ = highest_vs_lowest(libs)

    statement = (
        f"Tests that touch more of the project and more libraries are riskier: "
        f"{coverage.loc[c_hi, 'flaky_pct']:.1f}% of tests covering {c_hi} source lines are flaky "
        f"vs {coverage.loc[c_lo, 'flaky_pct']:.1f}% for {c_lo} lines, and "
        f"{libs.loc[l_hi, 'flaky_pct']:.1f}% of tests using {l_hi} third-party libraries are flaky "
        f"vs {libs.loc[l_lo, 'flaky_pct']:.1f}% for {l_lo}."
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)

    bar_panel(axes[0], coverage, overall, "Project source lines covered")
    bar_panel(axes[1], libs, overall, "Third-party libraries used")

    axes[1].set_ylabel("")
    axes[0].set_ylim(0, max(coverage["flaky_pct"].max(), libs["flaky_pct"].max()) * 1.25)

    axes[0].set_title("P2. Flaky share by coverage and by libraries")

    small = coverage[coverage["tests"] < MIN_GROUP_SIZE]

    if len(small):
        axes[0].text(0, -0.2, f"Bucket '{small.index[0]}' has only {int(small['tests'].iloc[0])} tests",
                     transform=axes[0].transAxes, fontsize=7, color=TEXT_MUTED)

    save_figure(fig, FIGS["coverage_libraries"])

    return statement, pd.concat({"coverage": coverage, "libraries": libs})


def pattern_churn(df, overall):
    """P3: tests covering recently changed code (moderate churn) are riskier."""

    df = df.assign(churn_bucket=pd.cut(df[CHURN_COL], bins=CHURN_BINS, labels=CHURN_LABELS))

    table = flaky_share(df, "churn_bucket", CHURN_LABELS)

    hi, lo, _ = highest_vs_lowest(table)

    statement = (
        f"Tests covering frequently changed code are riskier: with a churn score "
        f"({CHURN_COL}) of {hi}, {table.loc[hi, 'flaky_pct']:.1f}% of tests are flaky, vs "
        f"{table.loc[lo, 'flaky_pct']:.1f}% at {lo}. The very highest churn group (>5) is small "
        f"({int(table.loc['>5', 'tests'])} tests) and drops back to "
        f"{table.loc['>5', 'flaky_pct']:.1f}%, so the effect is not a straight line."
    )

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bar_panel(ax, table, overall, f"Changed-line churn score ({CHURN_COL})")
    ax.set_title("P3. Flaky share by code churn (last 100 commits)")

    save_figure(fig, FIGS["churn"])

    return statement, table


def pattern_smells(df, overall):
    """P4: some smells (resource optimism, fire-and-forget, mystery guest) are red flags."""

    rows = []

    for smell in SMELL_FEATURES:

        present = df[df[smell] == 1][TARGET_COL]
        absent = df[df[smell] == 0][TARGET_COL]

        rows.append({
            "smell": smell.replace("smell_", "").replace("_", " "),
            "tests_with_smell": len(present),
            "flaky_pct_with": round(present.mean() * 100, 2),
            "flaky_pct_without": round(absent.mean() * 100, 2)
        })

    table = pd.DataFrame(rows).sort_values("flaky_pct_with").set_index("smell")

    table["ratio"] = (table["flaky_pct_with"] / table["flaky_pct_without"]).round(2)

    top3 = table.sort_values("flaky_pct_with", ascending=False).head(3)

    count = df[SMELL_FEATURES].sum(axis=1).clip(upper=4)
    by_count = df.groupby(count)[TARGET_COL].mean() * 100

    lowers = table[table["ratio"] < 1]

    exception = (
        " Exception: " + ", ".join(
            f"{s} ({r['flaky_pct_with']:.1f}% with vs {r['flaky_pct_without']:.1f}% without)"
            for s, r in lowers.iterrows()
        ) + " goes the other way."
        if len(lowers) else ""
    )

    statement = (
        "A few test smells are red flags: tests with "
        + ", ".join(f"{s} ({r['flaky_pct_with']:.1f}% flaky)" for s, r in top3.iterrows())
        + f" are far above the {overall:.1f}% average. Smells add up: "
        f"{by_count.loc[4]:.1f}% of tests with 4+ smells are flaky vs {by_count.loc[1]:.1f}% "
        f"with exactly one." + exception
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    y = np.arange(len(table))
    h = 0.36

    ax.barh(y + h / 2, table["flaky_pct_with"], height=h, color=BLUE, label="Smell present")
    ax.barh(y - h / 2, table["flaky_pct_without"], height=h, color=AXIS, label="Smell absent")

    for i, (pct, n) in enumerate(zip(table["flaky_pct_with"], table["tests_with_smell"])):
        ax.text(pct, i + h / 2, f" {pct:.1f}% (n={n:,})", va="center",
                fontsize=8, color=TEXT_SECONDARY)

    ax.axvline(overall, color=TEXT_MUTED, linestyle="--", linewidth=1)

    ax.set_yticks(y, table.index)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, table["flaky_pct_with"].max() * 1.35)
    ax.set_xlabel("% of tests that are flaky")
    ax.set_title(f"P4. Flaky share with vs without each test smell (dashed = all tests {overall:.1f}%)",
                 fontsize=11)
    ax.legend(loc="lower right")

    save_figure(fig, FIGS["smells"])

    return statement, table


def pattern_projects(df, overall):
    """P5: flakiness is concentrated in a few projects."""

    table = flaky_share(df, "project")

    big = table[table["tests"] >= MIN_GROUP_SIZE].sort_values("flaky_pct", ascending=False)

    top = big.head(TOP_PROJECTS)

    top3 = big.head(3)

    share_top3 = top3["flaky"].sum() / table["flaky"].sum() * 100

    statement = (
        "Flakiness is concentrated in a few projects: "
        + ", ".join(f"{p} ({r['flaky_pct']:.1f}%)" for p, r in top3.iterrows())
        + f" have the highest flaky share, and these 3 projects alone hold "
        f"{share_top3:.0f}% of all flaky tests. "
        f"{int((table['flaky'] <= 1).sum())} of 24 projects have at most one flaky test."
    )

    plot = top.iloc[::-1]

    fig, ax = plt.subplots(figsize=(7.5, 5))

    ax.barh(plot.index, plot["flaky_pct"], color=BLUE, height=0.6)

    for i, (pct, f, n) in enumerate(zip(plot["flaky_pct"], plot["flaky"], plot["tests"])):
        ax.text(pct, i, f" {pct:.1f}% ({int(f)}/{n:,})", va="center",
                fontsize=8, color=TEXT_SECONDARY)

    ax.axvline(overall, color=TEXT_MUTED, linestyle="--", linewidth=1)

    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, plot["flaky_pct"].max() * 1.3)
    ax.set_xlabel("% of the project's tests that are flaky")
    ax.set_title(f"P5. Top {TOP_PROJECTS} projects by flaky share (projects with 50+ tests)")

    save_figure(fig, FIGS["projects"])

    return statement, table

# 5. MAIN EXECUTION

def main():

    df = load_data()

    os.makedirs(PATTERN_DIR, exist_ok=True)

    apply_style()

    overall = df[TARGET_COL].mean() * 100

    patterns = [
        ("P1", "Execution time", "execution_time", pattern_execution_time),
        ("P2", "Coverage and libraries", "coverage_libraries", pattern_coverage_libraries),
        ("P3", "Code churn", "churn", pattern_churn),
        ("P4", "Test smells", "smells", pattern_smells),
        ("P5", "Projects", "projects", pattern_projects)
    ]

    rows = []

    print("\n========== FLAKY TEST PATTERNS ==========")

    for pid, topic, key, func in patterns:

        statement, table = func(df, overall)

        rows.append({
            "pattern": pid,
            "topic": topic,
            "statement": statement,
            "chart": FIGS[key].replace(os.sep, "/")
        })

        print(f"\n{pid} ({topic}): {statement}")
        print(table.to_string())

    pd.DataFrame(rows).to_csv(PATTERNS_FILE, index=False)

    print("\nPatterns saved to:", PATTERNS_FILE)


if __name__ == "__main__":
    main()
