import os
import numpy as np
import pandas as pd

from src.config import (
    RENAME,
    ID_COLS,
    TARGET_COL,
    FEATURE_COLS,
    OUTPUT_DIR
)

from src.data_cleaning import CLEANED_DATA, RERUN_COLS

# OUTPUT FILES

FINAL_DATASET = os.path.join(
    OUTPUT_DIR,
    "cleaned_dataset.csv"
)

COLUMN_ROLES = os.path.join(
    OUTPUT_DIR,
    "column_roles.csv"
)

# BUCKET DEFINITIONS

EXEC_TIME_COL = RENAME["ExecutionTime"]

EXEC_TIME_BINS = [0, 0.1, 1, 10, np.inf]

EXEC_TIME_LABELS = ["<0.1s", "0.1-1s", "1-10s", ">10s"]


LIBS_COL = RENAME["num_third_party_libs"]

LIBS_BINS = [-1, 0, 2, 5, 10, np.inf]

LIBS_LABELS = ["0", "1-2", "3-5", "6-10", ">10"]


COVERAGE_COL = RENAME["projectSourceLinesCovered"]

COVERAGE_BINS = [-1, 0, 25, 100, 500, np.inf]

COVERAGE_LABELS = ["0", "1-25", "26-100", "101-500", ">500"]

# COLUMN ROLES

BUCKET_COLS = [
    "execution_time_bucket",
    "third_party_libs_bucket",
    "source_lines_covered_bucket"
]

# Model inputs: the original FlakeFlagger features (renamed)

MODEL_FEATURE_COLS = [RENAME.get(col, col) for col in FEATURE_COLS]

# Rerun outcomes and anything derived from them leak the label.
# NOT model features: use them for EDA / Power BI only.

LEAKAGE_COLS = RERUN_COLS + ["fail_rate_pct"]

# Text version of the target, for Power BI Key Influencers

LABEL_TEXT_COL = "flaky_label"

# 1. LOAD CLEANED DATA

def load_cleaned_data(path=CLEANED_DATA):
    """Load the output of data_cleaning.py."""

    print("\nLoading cleaned data...")

    df = pd.read_csv(path)

    print("Rows:", len(df))
    print("Columns:", len(df.columns))

    return df

# 2. EXECUTION-TIME BUCKETS

def add_execution_time_buckets(df):
    """Bucket execution time into <0.1s, 0.1-1s, 1-10s, >10s (ordered)."""

    # right=False: 0.1 falls in "0.1-1s", 10 falls in ">10s"

    df["execution_time_bucket"] = pd.cut(
        df[EXEC_TIME_COL],
        bins=EXEC_TIME_BINS,
        labels=EXEC_TIME_LABELS,
        right=False,
        ordered=True
    )

    return df

# 3. THIRD-PARTY LIBRARY BUCKETS

def add_third_party_lib_buckets(df):
    """Bucket third-party library count into 0, 1-2, 3-5, 6-10, >10 (ordered)."""

    df["third_party_libs_bucket"] = pd.cut(
        df[LIBS_COL],
        bins=LIBS_BINS,
        labels=LIBS_LABELS,
        ordered=True
    )

    return df

# 4. SOURCE-LINES-COVERED BUCKETS

def add_coverage_buckets(df):
    """Bucket project source lines covered into 0, 1-25, 26-100, 101-500, >500."""

    df["source_lines_covered_bucket"] = pd.cut(
        df[COVERAGE_COL],
        bins=COVERAGE_BINS,
        labels=COVERAGE_LABELS,
        ordered=True
    )

    return df

# 5. TEXT LABEL

def add_flaky_label(df):
    """Add a "Flaky"/"Non-Flaky" text label (target, not a feature)."""

    df[LABEL_TEXT_COL] = df[TARGET_COL].map({
        1: "Flaky",
        0: "Non-Flaky"
    })

    return df

# 6. FAIL RATE

def add_fail_rate(df):
    """Add fail_rate_pct from rerun counts (leakage column, not a feature)."""

    total_runs = df["NumFailingRuns"] + df["NumPassingRuns"]

    df["fail_rate_pct"] = (
        df["NumFailingRuns"] / total_runs * 100
    ).astype("Float64").round(4)

    missing = int(df["fail_rate_pct"].isna().sum())

    print("\nfail_rate_pct missing (no rerun data):", missing)

    return df

# 7. COLUMN ROLES

def build_column_roles(df):
    """Label every column as identifier, target, feature, bucket or leakage."""

    roles = {}

    for col in df.columns:

        if col in ID_COLS:
            roles[col] = "identifier"
        elif col in (TARGET_COL, LABEL_TEXT_COL):
            roles[col] = "target"
        elif col in MODEL_FEATURE_COLS:
            roles[col] = "model_feature"
        elif col in BUCKET_COLS:
            roles[col] = "bucket (EDA / Power BI)"
        elif col in LEAKAGE_COLS:
            roles[col] = "leakage - NOT a model feature"
        else:
            roles[col] = "unassigned"

    roles = pd.DataFrame(
        list(roles.items()),
        columns=["column", "role"]
    )

    unassigned = roles[roles["role"] == "unassigned"]

    if len(unassigned):
        print("WARNING: Columns without a role:", list(unassigned["column"]))

    return roles

# 8. SAVE FINAL DATASET

def save_final_dataset(df, roles):
    """Save the final dataset and its column roles to the outputs folder."""

    # Identifiers, target, features, buckets, then leakage columns last

    ordered = list(roles.sort_values(
        "role",
        key=lambda r: r.map({
            "identifier": 0,
            "target": 1,
            "model_feature": 2,
            "bucket (EDA / Power BI)": 3,
            "leakage - NOT a model feature": 4
        }).fillna(5),
        kind="stable"
    )["column"])

    df = df[ordered]

    roles = roles.set_index("column").loc[ordered].reset_index()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df.to_csv(FINAL_DATASET, index=False)
    roles.to_csv(COLUMN_ROLES, index=False)

    print("\nFinal dataset saved to:", FINAL_DATASET)
    print("Column roles saved to:", COLUMN_ROLES)

# 9. PRINT BUCKET COUNTS

def print_bucket_counts(df, col):
    """Print row and flaky counts for each bucket."""

    print(f"\n{col}:")

    counts = df.groupby(col, observed=False).agg(
        tests=("flaky", "size"),
        flaky=("flaky", "sum")
    )

    counts["flaky_pct"] = (counts["flaky"] / counts["tests"] * 100).round(2)

    print(counts)

    unassigned = int(df[col].isna().sum())

    if unassigned:
        print(f"WARNING: {unassigned} rows not assigned to a bucket.")

# 10. MAIN EXECUTION

def main():

    df = load_cleaned_data()

    cols_before = len(df.columns)

    df = add_execution_time_buckets(df)
    df = add_third_party_lib_buckets(df)
    df = add_coverage_buckets(df)
    df = add_flaky_label(df)
    df = add_fail_rate(df)

    print("\n========== BUCKETS ==========")

    for col in BUCKET_COLS:
        print_bucket_counts(df, col)

    roles = build_column_roles(df)

    print("\n========== COLUMN ROLES ==========")

    print(roles["role"].value_counts().to_string())

    print("\nNOT model features (leak the label):", LEAKAGE_COLS)

    print("\n========== FEATURE ENGINEERING SUMMARY ==========")

    print("Rows:", len(df))
    print(f"Columns: {cols_before} -> {len(df.columns)}")
    print("Added:", BUCKET_COLS + [LABEL_TEXT_COL, "fail_rate_pct"])

    save_final_dataset(df, roles)


if __name__ == "__main__":
    main()
