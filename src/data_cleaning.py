import os
import pandas as pd

from src.config import (
    FEATURES_PATH,
    RESULTS_PATH,
    RENAME,
    SMELL_COLS,
    TARGET_COL,
    TEST_CHARACTERISTIC_COLS,
    RESULTS_TARGET_COL,
    OUTPUT_DIR
)

from src.dataset_matching import normalize_project_name

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

# Rerun outcome columns from test_results.csv. They describe how the
# test behaved when rerun, so they leak the label (NOT model features).

RERUN_COLS = [
    "NumFailingRuns",
    "NumPassingRuns",
    "FirstFailingRunID",
    "FirstPassingRunID",
    "UniqueFailingExceptionTypes"
]

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

# 7. CHECK DUPLICATE RECORDS

def check_duplicates(df, key_cols, dataset_name):
    """Report duplicate test names / full rows and drop full duplicates."""

    print(f"\n========== DUPLICATE CHECK: {dataset_name} ==========")

    full_duplicates = int(df.duplicated().sum())

    key_duplicates = int(df.duplicated(subset=key_cols).sum())

    print("Full duplicate rows:", full_duplicates)
    print(f"Duplicate {key_cols}:", key_duplicates)

    df = df.drop_duplicates().reset_index(drop=True)

    # Same test name but different values: not a true duplicate,
    # so it is reported but kept for manual review

    conflicting = int(df.duplicated(subset=key_cols).sum())

    if conflicting:
        print(
            f"WARNING: {conflicting} records share a test name "
            "but differ in values. Kept for review."
        )

    print("Rows dropped:", full_duplicates)

    return df, full_duplicates

# 8. MERGE RERUN RESULTS

def merge_results(features, results):
    """Left-join rerun results onto features using normalised test keys."""

    print("\n========== MERGE RERUN RESULTS ==========")

    features = features.copy()
    results = results.copy()

    # Features store test_name as lowercase "class.method", results
    # store "Class#method". Lowercasing would merge tests that differ
    # only in case, so the features key is rebuilt case-preserving
    # from testClassName and testMethodName.

    features["project_key"] = features["project"].apply(normalize_project_name)
    results["project_key"] = results["Project"].apply(normalize_project_name)

    features["test_key"] = (
        features["testClassName"].astype(str).str.strip()
        + "#"
        + features["testMethodName"].astype(str).str.strip()
    )

    results["test_key"] = results["Test"].astype(str).str.strip()

    consistent_names = (
        features["test_key"].str.replace("#", ".", regex=False).str.lower()
        == features["test_name"]
    ).mean()

    print(
        "test_name matches lowercase Class.method:",
        f"{consistent_names * 100:.2f}%"
    )

    case_collisions = int(
        results.assign(
            lower_key=results["test_key"].str.lower()
        ).duplicated(subset=["project_key", "lower_key"]).sum()
    )

    print("Result tests differing only by case:", case_collisions)

    results = results.drop(columns=["Project", "Test"])

    merged = pd.merge(
        features,
        results,
        on=["project_key", "test_key"],
        how="left",
        indicator=True,
        validate="one_to_one"
    )

    matched = int((merged["_merge"] == "both").sum())
    unmatched = len(merged) - matched

    print("Feature rows:", len(features))
    print("Matched with results:", matched)
    print(f"Match rate: {matched / len(features) * 100:.2f}%")
    print("Without rerun data (empty rerun columns):", unmatched)

    # Compare the two flaky labels on matched rows

    both = merged["_merge"] == "both"

    mismatches = int(
        (merged.loc[both, TARGET_COL] != merged.loc[both, RESULTS_TARGET_COL]).sum()
    )

    print("Label mismatches (flaky vs IsFlaky):", mismatches)

    # flaky from test_features is the label; IsFlaky is redundant

    merged = merged.drop(
        columns=["project_key", "test_key", "_merge", RESULTS_TARGET_COL]
    )

    for col in RERUN_COLS:
        merged[col] = merged[col].astype("Int64")

    return merged, matched, unmatched

# 9. SAVE CLEANED DATA

def save_cleaned_data(df, path=CLEANED_DATA):
    """Save the cleaned dataset to the outputs folder."""

    os.makedirs(os.path.dirname(path), exist_ok=True)

    df.to_csv(path, index=False)

    print("\nCleaned data saved to:", path)

# 10. MAIN EXECUTION

def main():

    features, results = load_datasets()

    rows_before, cols_before = features.shape

    features = remove_unused_columns(features)
    features = rename_columns(features)

    features, filled = fill_missing_values(features)
    results, fixed = fix_invalid_values(results)

    check_impossible_values(features, results)

    features, feature_dups = check_duplicates(
        features, ["project", "test_name"], "test_features"
    )

    results, result_dups = check_duplicates(
        results, ["Project", "Test"], "test_results"
    )

    cleaned, matched, unmatched = merge_results(features, results)

    print("\n========== CLEANING SUMMARY ==========")

    print(f"Rows: {rows_before} -> {len(cleaned)}")
    print(f"Columns: {cols_before} -> {len(cleaned.columns)}")
    print("Values filled:", sum(filled.values()), filled)
    print("Malformed values fixed:", fixed)
    print("Duplicates removed:", feature_dups + result_dups)
    print(f"Rerun data attached: {matched} rows ({unmatched} without)")

    save_cleaned_data(cleaned)


if __name__ == "__main__":
    main()
