import numpy as np
import pandas as pd

from src.config import RENAME

from src.data_cleaning import CLEANED_DATA

# BUCKET DEFINITIONS

EXEC_TIME_COL = RENAME["ExecutionTime"]

EXEC_TIME_BINS = [0, 0.1, 1, 10, np.inf]

EXEC_TIME_LABELS = ["<0.1s", "0.1-1s", "1-10s", ">10s"]


LIBS_COL = RENAME["num_third_party_libs"]

LIBS_BINS = [-1, 0, 2, 5, 10, np.inf]

LIBS_LABELS = ["0", "1-2", "3-5", "6-10", ">10"]

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

# 4. PRINT BUCKET COUNTS

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

# 5. MAIN EXECUTION

def main():

    df = load_cleaned_data()

    cols_before = len(df.columns)

    df = add_execution_time_buckets(df)
    df = add_third_party_lib_buckets(df)

    print("\n========== BUCKETS ==========")

    print_bucket_counts(df, "execution_time_bucket")
    print_bucket_counts(df, "third_party_libs_bucket")

    print("\n========== FEATURE ENGINEERING SUMMARY ==========")

    print("Rows:", len(df))
    print(f"Columns: {cols_before} -> {len(df.columns)}")


if __name__ == "__main__":
    main()
