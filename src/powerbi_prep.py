import os
import pandas as pd

from src.train_models import require_inputs, METRICS_FILE, SAMPLINGS

from src.config import OUTPUT_DIR, TARGET_COL, RENAME, SMELL_COLS
from src.feature_engineering import (
    FINAL_DATASET,
    COLUMN_ROLES,
    EXEC_TIME_LABELS,
    LIBS_LABELS,
    COVERAGE_LABELS
)

# OUTPUT FILES

POWERBI_DIR = os.path.join(OUTPUT_DIR, "powerbi")

POWERBI_TABLE = os.path.join(POWERBI_DIR, "flaky_tests_powerbi.csv")

MODEL_COMPARISON = os.path.join(POWERBI_DIR, "model_comparison.csv")

MODEL_COMPARISON_LONG = os.path.join(POWERBI_DIR, "model_comparison_long.csv")

# BUCKET SORT ORDER (Power BI sorts text alphabetically otherwise)

BUCKET_ORDERS = {
    "execution_time_bucket": EXEC_TIME_LABELS,
    "third_party_libs_bucket": LIBS_LABELS,
    "source_lines_covered_bucket": COVERAGE_LABELS
}

SMELL_FEATURES = [RENAME[col] for col in SMELL_COLS]

METRIC_COLS = ["precision", "recall", "f1", "roc_auc", "pr_auc"]

# 1. LOAD DATA

def load_data():
    """Load the final dataset and its column roles."""

    require_inputs([FINAL_DATASET, COLUMN_ROLES])

    df = pd.read_csv(FINAL_DATASET)
    roles = pd.read_csv(COLUMN_ROLES)

    print("\nRows:", len(df))
    print("Columns:", len(df.columns))

    return df, roles

# 2. REMOVE LEAKAGE COLUMNS

def remove_leakage(df, roles):
    """Drop the 6 rerun-outcome columns that leak the label."""

    print("\n========== REMOVE LEAKAGE COLUMNS ==========")

    leakage = list(roles.loc[roles["role"].str.startswith("leakage"), "column"])

    assert len(leakage) == 6, f"Expected 6 leakage columns, found {leakage}"

    df = df.drop(columns=leakage)

    assert not set(leakage) & set(df.columns)

    print("Removed:", leakage)

    return df

# 3. ADD POWER BI HELPER COLUMNS

def add_helper_columns(df):
    """Add smell count and bucket sort-order columns for Power BI."""

    print("\n========== HELPER COLUMNS ==========")

    df["smell_count"] = df[SMELL_FEATURES].sum(axis=1).astype(int)

    df["has_any_smell"] = df["smell_count"].gt(0).map({True: "Yes", False: "No"})

    added = ["smell_count", "has_any_smell"]

    for col, labels in BUCKET_ORDERS.items():

        order_col = f"{col}_order"

        df[order_col] = df[col].map({label: i for i, label in enumerate(labels)})

        added.append(order_col)

    print("Added:", added)

    return df

# 4. MODEL COMPARISON TABLE

def build_model_comparison():
    """Model x balancing table from metrics.csv (wide and long)."""

    print("\n========== MODEL COMPARISON ==========")

    if not os.path.exists(METRICS_FILE):
        print(f"WARNING: {METRICS_FILE} not found, model comparison skipped.")
        print("Run python -m src.train_models, then rerun this script.")
        return None, None

    metrics = pd.read_csv(METRICS_FILE)

    order = {name: i for i, name in enumerate(SAMPLINGS)}

    wide = metrics[["model_label", "sampling", "sampling_label", "is_best"]].copy()

    wide["sampling_order"] = wide["sampling"].map(order)

    for metric in METRIC_COLS:
        wide[metric] = metrics[f"{metric}_mean"].round(4)

    wide = wide.drop(columns="sampling")

    long = wide.melt(
        id_vars=["model_label", "sampling_label", "sampling_order", "is_best"],
        value_vars=METRIC_COLS,
        var_name="metric",
        value_name="value"
    )

    print(wide.drop(columns=["sampling_order"]).to_string(index=False))

    return wide, long

# 5. SAVE OUTPUTS

def save_outputs(df, wide, long):
    """Write all Power BI CSV files."""

    os.makedirs(POWERBI_DIR, exist_ok=True)

    df.to_csv(POWERBI_TABLE, index=False)

    print("\nPower BI table saved to:", POWERBI_TABLE)

    if wide is not None:
        wide.to_csv(MODEL_COMPARISON, index=False)
        long.to_csv(MODEL_COMPARISON_LONG, index=False)
        print("Model comparison saved to:", MODEL_COMPARISON)
        print("Model comparison (long) saved to:", MODEL_COMPARISON_LONG)

# 6. MAIN EXECUTION

def main():

    df, roles = load_data()

    cols_before = len(df.columns)

    df = remove_leakage(df, roles)
    df = add_helper_columns(df)

    wide, long = build_model_comparison()

    print("\n========== POWER BI PREP SUMMARY ==========")

    print("Rows:", len(df))
    print(f"Columns: {cols_before} -> {len(df.columns)}")
    print(f"Flaky tests: {int(df[TARGET_COL].sum())} "
          f"({df[TARGET_COL].mean() * 100:.2f}%)")

    save_outputs(df, wide, long)


if __name__ == "__main__":
    main()
