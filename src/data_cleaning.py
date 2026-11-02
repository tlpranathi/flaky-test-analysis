import os
import pandas as pd

from src.config import (
    FEATURES_PATH,
    RESULTS_PATH,
    RENAME,
    SMELL_COLS,
    TARGET_COL,
    TEST_CHARACTERISTIC_COLS,
    OUTPUT_DIR
)

# OUTPUT FILE

CLEANED_DATA = os.path.join(
    OUTPUT_DIR,
    "cleaned_data.csv"
)

# Columns that carry no information (leftover CSV index)

UNUSED_COLS = [
    "Unnamed: 0"
]

# Columns with missing values (renamed names): test_length,
# num_asserts and covered_lines

FILL_COLS = [
    RENAME[col] for col in TEST_CHARACTERISTIC_COLS
] + [RENAME["numCoveredLines"]]

# 1. LOAD DATASETS

def load_datasets():
    """Load the raw features and results datasets."""

    print("\nLoading datasets...")

    features = pd.read_csv(FEATURES_PATH)
    results = pd.read_csv(RESULTS_PATH)

    print("Features:", features.shape)
    print("Results:", results.shape)

    return features, results

# 2. REMOVE UNUSED COLUMNS

def remove_unused_columns(df):
    """Drop leftover index columns that are not part of the data."""

    print("\n========== REMOVE UNUSED COLUMNS ==========")

    to_drop = [
        col for col in df.columns
        if col in UNUSED_COLS
        or col.startswith("Unnamed:")
    ]

    df = df.drop(columns=to_drop)

    print("Dropped columns:", to_drop if to_drop else "none")

    return df

# 3. RENAME COLUMNS

def rename_columns(df):
    """Rename feature columns using the team mapping in config.RENAME."""

    print("\n========== RENAME COLUMNS ==========")

    mapping = {
        old: new for old, new in RENAME.items()
        if old in df.columns
    }

    missing = [
        old for old in RENAME
        if old not in df.columns
    ]

    df = df.rename(columns=mapping)

    print("Renamed columns:", len(mapping))

    if missing:
        print("WARNING: Mapped columns not found:", missing)

    return df

# 4. FILL MISSING VALUES

def fill_missing_values(df):
    """Fill missing values with the project median, then the overall median."""

    print("\n========== FILL MISSING VALUES ==========")

    filled = {}

    for col in FILL_COLS:

        missing_before = int(df[col].isna().sum())

        project_median = df.groupby("project")[col].transform("median")

        df[col] = df[col].fillna(project_median)

        from_project = missing_before - int(df[col].isna().sum())

        df[col] = df[col].fillna(df[col].median())

        filled[col] = missing_before

        print(
            f"{col}: {missing_before} filled "
            f"({from_project} project median, "
            f"{missing_before - from_project} overall median)"
        )

    print("Remaining missing values:", int(df.isna().sum().sum()))

    return df, filled

# 5. FIX INVALID VALUES

def fix_invalid_values(results):
    """Keep only the leading number of malformed exception-type counts."""

    print("\n========== FIX INVALID VALUES ==========")

    col = "UniqueFailingExceptionTypes"

    raw = results[col].astype(str).str.strip()

    invalid = ~raw.str.fullmatch(r"\d+")

    print(f"Malformed values in {col}:", int(invalid.sum()))

    leading_number = raw.str.extract(r"^(\d+)", expand=False)

    for value, fixed in zip(raw[invalid], leading_number[invalid]):
        print(f"  '{value}' -> {fixed}")

    results[col] = pd.to_numeric(leading_number, errors="coerce")

    unparsable = int(results[col].isna().sum())

    if unparsable:
        print(f"WARNING: {unparsable} values had no leading number.")
    else:
        results[col] = results[col].astype(int)

    return results, int(invalid.sum())

# 6. CHECK NEGATIVE / IMPOSSIBLE VALUES

def check_impossible_values(features, results):
    """Report negative or impossible values (report only, nothing changed)."""

    print("\n========== IMPOSSIBLE VALUE CHECK ==========")

    feature_numeric = features.select_dtypes("number")

    negatives = (feature_numeric < 0).sum()
    negatives = negatives[negatives > 0]

    print("Negative values in features:", dict(negatives) if len(negatives) else "none")

    binary_cols = [
        RENAME.get(col, col) for col in SMELL_COLS
    ] + [TARGET_COL]

    for col in binary_cols:

        bad = int((~features[col].isin([0, 1])).sum())

        if bad:
            print(f"WARNING: {col} has {bad} non-binary values.")

    for col in ["NumFailingRuns", "NumPassingRuns"]:

        bad = int((results[col] < 0).sum())

        if bad:
            print(f"WARNING: {col} has {bad} negative values.")

    no_runs = int(
        (results["NumFailingRuns"] + results["NumPassingRuns"] == 0).sum()
    )

    print("Results with zero total runs:", no_runs)

    # -1 in the run-ID columns means "never happened", not an error

    for col in ["FirstFailingRunID", "FirstPassingRunID"]:

        sentinel = int((results[col] == -1).sum())
        other = int((results[col] < -1).sum())

        print(f"{col}: {sentinel} rows use -1 (never occurred), {other} invalid")

    never_passed = int(
        ((results["NumPassingRuns"] == 0) & (results["NumFailingRuns"] > 0)).sum()
    )

    print("Tests that failed in every run (never passed):", never_passed)

# 7. SAVE CLEANED DATA

def save_cleaned_data(df, path=CLEANED_DATA):
    """Save the cleaned dataset to the outputs folder."""

    os.makedirs(os.path.dirname(path), exist_ok=True)

    df.to_csv(path, index=False)

    print("\nCleaned data saved to:", path)

# 8. MAIN EXECUTION

def main():

    features, results = load_datasets()

    rows_before, cols_before = features.shape

    features = remove_unused_columns(features)
    features = rename_columns(features)

    features, filled = fill_missing_values(features)
    results, fixed = fix_invalid_values(results)

    check_impossible_values(features, results)

    print("\n========== CLEANING SUMMARY ==========")

    print(f"Rows: {rows_before} -> {len(features)}")
    print(f"Columns: {cols_before} -> {len(features.columns)}")
    print("Values filled:", sum(filled.values()), filled)
    print("Malformed values fixed:", fixed)

    save_cleaned_data(features)


if __name__ == "__main__":
    main()
