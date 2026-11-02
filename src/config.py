import os

FEATURES_PATH = "data/test_features.csv"
RESULTS_PATH = "data/test_results.csv"

# COLUMN RENAMING

RENAME = {
    "ExecutionTime": "execution_time_s",
    "numCoveredLines": "covered_lines",
    "projectSourceLinesCovered": "source_lines_covered",
    "projectSourceClassesCovered": "source_classes_covered",
    "num_third_party_libs": "third_party_libs",
    "testLength": "test_length",
    "numAsserts": "num_asserts",

    "assertion-roulette": "smell_assertion_roulette",
    "conditional-test-logic": "smell_conditional_logic",
    "eager-test": "smell_eager_test",
    "fire-and-forget": "smell_fire_and_forget",
    "indirect-testing": "smell_indirect_testing",
    "mystery-guest": "smell_mystery_guest",
    "resource-optimism": "smell_resource_optimism",
    "test-run-war": "smell_test_run_war",
}

WINDOWS = (5, 10, 25, 50, 75, 100, 500, 10000)

for window in WINDOWS:
    RENAME[
        f"hIndexModificationsPerCoveredLine_window{window}"
    ] = f"changed_line_churn_w{window}"

# COLUMN GROUPS

ID_COLS = [
    "test_name",
    "project",
    "testClassName",
    "testMethodName"
]

TARGET_COL = "flaky"

SMELL_COLS = [
    "assertion-roulette",
    "conditional-test-logic",
    "eager-test",
    "fire-and-forget",
    "indirect-testing",
    "mystery-guest",
    "resource-optimism",
    "test-run-war"
]

EXECUTION_COLS = [
    "ExecutionTime"
]

COVERAGE_COLS = [
    "numCoveredLines",
    "projectSourceLinesCovered",
    "projectSourceClassesCovered"
]

TEST_CHARACTERISTIC_COLS = [
    "testLength",
    "numAsserts"
]

DEPENDENCY_COLS = [
    "num_third_party_libs"
]

CHURN_COLS = [
    f"hIndexModificationsPerCoveredLine_window{w}"
    for w in WINDOWS
]

FEATURE_COLS = (
    EXECUTION_COLS
    + COVERAGE_COLS
    + TEST_CHARACTERISTIC_COLS
    + DEPENDENCY_COLS
    + SMELL_COLS
    + CHURN_COLS
)

# RESULTS DATASET COLUMNS

RESULTS_COLS = [
    "Project",
    "Test",
    "IsFlaky",
    "NumFailingRuns",
    "NumPassingRuns",
    "FirstFailingRunID",
    "FirstPassingRunID",
    "UniqueFailingExceptionTypes"
]

RESULTS_TARGET_COL = "IsFlaky"

# EXPECTED DATASET INFORMATION

EXPECTED_PROJECTS = 24

OUTPUT_DIR = "outputs"

REPORT_DIR = "documentation"

VALIDATION_REPORT = os.path.join(
    REPORT_DIR,
    "dataset_validation_report.txt"
)

FEATURE_DICTIONARY = os.path.join(
    REPORT_DIR,
    "feature_dictionary.csv"
)
