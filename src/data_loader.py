import os
import numpy as np
import pandas as pd

from src.config import (
    FEATURES_PATH,
    RESULTS_PATH,
    RENAME,
    ID_COLS,
    TARGET_COL,
    FEATURE_COLS,
    RESULTS_COLS,
    EXPECTED_PROJECTS,
    VALIDATION_REPORT,
    SMELL_COLS,
    WINDOWS
)

# 1. LOAD FEATURES DATASET

def load_features(path=FEATURES_PATH):

    print("\nLoading test_features.csv...")

    df = pd.read_csv(path)

    print("Features dataset loaded successfully!")
    print("Rows:", len(df))
    print("Columns:", len(df.columns))

    return df

# 2. LOAD RESULTS DATASET

def load_results(path=RESULTS_PATH):

    print("\nLoading test_results.csv...")

    df = pd.read_csv(path)

    print("Results dataset loaded successfully!")
    print("Rows:", len(df))
    print("Columns:", len(df.columns))

    return df


# 3. CHECK REQUIRED COLUMNS

def check_columns(df, required_columns, dataset_name):

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        print(f"\nERROR: Missing columns in {dataset_name}:")
        print(missing)

        return False

    print(f"\nAll required columns present in {dataset_name}.")

    return True

# 4. VALIDATE FEATURES DATASET

def validate_features(df):

    print("\n========== FEATURES VALIDATION ==========")

    # Check required columns

    required = (
        ID_COLS
        + [TARGET_COL]
        + FEATURE_COLS
    )

    columns_valid = check_columns(
        df,
        required,
        "test_features.csv"
    )

    if not columns_valid:
        return False

    valid = True

    # Check target values

    target_values = set(df[TARGET_COL].dropna().unique())

    if not target_values.issubset({0, 1}):
        print("ERROR: flaky contains values other than 0 and 1.")
        valid = False
    else:
        print("Target labels are valid.")

    if df[TARGET_COL].isna().any():
        print("ERROR: Missing values in flaky column.")
        valid = False

    # Check numerical feature data types

    for col in FEATURE_COLS:

        if not pd.api.types.is_numeric_dtype(df[col]):
            print(f"ERROR: {col} is not numerical.")
            valid = False

    # Check missing values

    print("\nMissing values:")

    missing = df.isnull().sum()

    print(missing[missing > 0])

    # Check infinite values

    numeric_df = df[FEATURE_COLS].select_dtypes(
        include=np.number
    )

    infinite = np.isinf(numeric_df).sum()

    print("\nInfinite values:")

    print(infinite[infinite > 0])

    # Check duplicate records

    duplicates = df.duplicated(
        subset=["project", "test_name"]
    ).sum()

    print("\nDuplicate test records:", duplicates)

    # Check number of projects

    projects = df["project"].nunique()

    print("\nNumber of projects:", projects)

    if projects != EXPECTED_PROJECTS:
        print(
            f"WARNING: Expected {EXPECTED_PROJECTS} projects, "
            f"found {projects}."
        )
    else:
        print("Project count is as expected.")

    # Check negative values in non-negative features

    non_negative_cols = [
        "ExecutionTime",
        "numCoveredLines",
        "projectSourceLinesCovered",
        "projectSourceClassesCovered",
        "num_third_party_libs",
        "testLength",
        "numAsserts"
    ] + [
        col for col in FEATURE_COLS
        if col.startswith("hIndexModifications")
    ]

    for col in non_negative_cols:

        negative_count = (df[col] < 0).sum()

        if negative_count > 0:
            print(
                f"WARNING: {col} has "
                f"{negative_count} negative values."
            )

    print("\nFeatures validation completed.")

    return valid

# 5. VALIDATE RESULTS DATASET

def validate_results(df):

    print("\n========== RESULTS VALIDATION ==========")

    columns_valid = check_columns(
        df,
        RESULTS_COLS,
        "test_results.csv"
    )

    if not columns_valid:
        return False

    valid = True

    # Check target values

    target_values = set(
        df["IsFlaky"].dropna().unique()
    )

    if not target_values.issubset({0, 1}):
        print("ERROR: IsFlaky contains invalid labels.")
        valid = False
    else:
        print("Results labels are valid.")

    # Check missing values

    print("\nMissing values:")

    missing = df.isnull().sum()

    print(missing[missing > 0])

    # Check execution counts

    count_cols = [
        "NumFailingRuns",
        "NumPassingRuns"
    ]

    for col in count_cols:

        if not pd.api.types.is_numeric_dtype(df[col]):
            print(f"ERROR: {col} is not numerical.")
            valid = False

        elif (df[col].dropna() < 0).any():
            print(f"ERROR: Negative values in {col}.")
            valid = False

    # Check duplicate records

    duplicates = df.duplicated(
        subset=["Project", "Test"]
    ).sum()

    print("\nDuplicate result records:", duplicates)

    print("\nResults validation completed.")

    return valid

# 6. DATASET SUMMARY

def dataset_summary(df):

    print("\n========== DATASET SUMMARY ==========")

    print("\nNumber of rows:", len(df))
    print("Number of columns:", len(df.columns))

    print("Number of projects:", df["project"].nunique())

    flaky = (df["flaky"] == 1).sum()
    non_flaky = (df["flaky"] == 0).sum()

    print("\nFlaky tests:", flaky)
    print("Non-flaky tests:", non_flaky)

    print(
        "Flaky percentage:",
        round(flaky / len(df) * 100, 2),
        "%"
    )

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isnull().sum())

# 7. SAVE VALIDATION REPORT

def save_validation_report(features_valid, results_valid,
                           features_df, results_df):

    os.makedirs(os.path.dirname(VALIDATION_REPORT), exist_ok=True)

    with open(VALIDATION_REPORT, "w") as file:

        file.write("FLAKEFLAGGER DATASET VALIDATION REPORT\n")
        file.write("=" * 45 + "\n\n")

        file.write("FEATURES DATASET\n")
        file.write("-" * 25 + "\n")

        file.write(f"Rows: {len(features_df)}\n")
        file.write(f"Columns: {len(features_df.columns)}\n")
        file.write(f"Projects: {features_df['project'].nunique()}\n")

        file.write(
            f"Flaky tests: {(features_df['flaky'] == 1).sum()}\n"
        )

        file.write(
            f"Missing values: {features_df.isnull().sum().sum()}\n"
        )

        file.write(
            f"Validation successful: {features_valid}\n\n"
        )

        file.write("RESULTS DATASET\n")
        file.write("-" * 25 + "\n")

        file.write(f"Rows: {len(results_df)}\n")
        file.write(f"Columns: {len(results_df.columns)}\n")

        file.write(
            f"Missing values: {results_df.isnull().sum().sum()}\n"
        )

        file.write(
            f"Validation successful: {results_valid}\n"
        )

    print("\nValidation report saved to:", VALIDATION_REPORT)

def create_feature_dictionary():

    descriptions = {
        "ExecutionTime": "Time taken to execute the test.",
        "numCoveredLines": "Number of source code lines covered by the test.",
        "projectSourceLinesCovered": "Number of project source lines covered.",
        "projectSourceClassesCovered": "Number of project source classes covered.",
        "num_third_party_libs": "Number of third-party libraries used.",
        "testLength": "Length of the test in lines.",
        "numAsserts": "Number of assertions in the test.",

        "assertion-roulette": "Test smell related to ambiguous assertions.",
        "conditional-test-logic": "Test smell involving conditional logic inside tests.",
        "eager-test": "Test smell involving testing too much functionality in one test.",
        "fire-and-forget": "Test smell involving asynchronous operations without proper verification.",
        "indirect-testing": "Test smell where functionality is tested indirectly.",
        "mystery-guest": "Test smell involving external resources or dependencies.",
        "resource-optimism": "Test smell involving assumptions about resource availability.",
        "test-run-war": "Test smell involving execution-order or shared-state dependencies.",

        "flaky": "Target label: 1 means flaky, 0 means non-flaky.",

        "test_name": "Unique test identifier.",
        "project": "Name of the Java project.",
        "testClassName": "Name of the test class.",
        "testMethodName": "Name of the test method."
    }

    rows = []

    for original, description in descriptions.items():

        rows.append({
            "feature": original,
            "renamed_feature": RENAME.get(original, original),
            "description": description,
            "type": (
                "Binary" if original == "flaky"
                or original in SMELL_COLS
                else "Identifier" if original in ID_COLS
                or original == "test_name"
                else "Numerical"
            )
        })

    for window in WINDOWS:

        original = (
            f"hIndexModificationsPerCoveredLine_window{window}"
        )

        rows.append({
            "feature": original,
            "renamed_feature": RENAME[original],
            "description": (
                f"Modification h-index for covered lines "
                f"over the last {window} commits."
            ),
            "type": "Numerical"
        })

    dictionary = pd.DataFrame(rows)

    os.makedirs("documentation", exist_ok=True)

    dictionary.to_csv(
        "documentation/feature_dictionary.csv",
        index=False
    )

    print("\nFeature dictionary created successfully!")

# 8. MAIN EXECUTION

def main():

    features_df = load_features()
    results_df = load_results()

    features_valid = validate_features(features_df)
    results_valid = validate_results(results_df)

    dataset_summary(features_df)

    save_validation_report(
        features_valid,
        results_valid,
        features_df,
        results_df
    )

    print("\n========== FINAL STATUS ==========")

    if features_valid and results_valid:
        print("Both datasets passed the validation checks.")
    else:
        print("Some validation checks failed. Review the output.")

    create_feature_dictionary()


if __name__ == "__main__":
    main()
