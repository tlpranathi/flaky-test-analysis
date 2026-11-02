
import os
import pandas as pd

from src.config import (
    FEATURES_PATH,
    RESULTS_PATH,
    PROJECT_NAME_MAP,
    PROJECT_NAME_REPORT,
    MATCHING_REPORT,
    UNMATCHED_FEATURES,
    UNMATCHED_RESULTS,
    LABEL_MISMATCHES,
)


# 1. LOAD DATASETS

def load_datasets():
    print("\nLoading datasets...")

    features = pd.read_csv(FEATURES_PATH)
    results = pd.read_csv(RESULTS_PATH)

    print("Features rows:", len(features))
    print("Results rows:", len(results))

    return features, results


# 2. NORMALIZE PROJECT NAMES

def normalize_project_name(name):
    name = str(name).strip().lower()
    return PROJECT_NAME_MAP.get(name, name)


# 3. PREPARE MATCHING KEYS

def prepare_matching_keys(features, results):

    features = features.copy()
    results = results.copy()

    # Normalize project names
    features["project_key"] = (
        features["project"].apply(normalize_project_name)
    )

    results["project_key"] = (
        results["Project"].apply(normalize_project_name)
    )

    # Check for missing test identifiers
    required_feature_cols = [
        "testClassName",
        "testMethodName"
    ]

    if features[required_feature_cols].isnull().any().any():
        raise ValueError("Missing class or method names in features dataset.")

    if results["Test"].isnull().any():
        raise ValueError("Missing test identifiers in results dataset.")

    # Construct the identifier in the format:
    # ClassName#MethodName
    features["test_key"] = (
        features["testClassName"].astype(str).str.strip()
        + "#"
        + features["testMethodName"].astype(str).str.strip()
    )

    # Results already use ClassName#MethodName
    results["test_key"] = (
        results["Test"].astype(str).str.strip()
    )

    return features, results


# 4. INSPECT PROJECT NAMES

def inspect_project_names(features, results):

    feature_projects = sorted(features["project_key"].unique())
    result_projects = sorted(results["project_key"].unique())

    common_projects = sorted(
        set(feature_projects) & set(result_projects)
    )

    print("\n========== PROJECT NAME INSPECTION ==========")

    print("Projects in features:", len(feature_projects))
    print("Projects in results:", len(result_projects))
    print("Common projects:", len(common_projects))

    print("\nCommon project names:")
    for project in common_projects:
        print(project)

    # Save project comparison
    report = pd.DataFrame({
        "results_project": results["Project"].unique()
    })

    report["normalized_project"] = (
        report["results_project"].apply(normalize_project_name)
    )

    report["exists_in_features"] = (
        report["normalized_project"].isin(feature_projects)
    )

    os.makedirs(os.path.dirname(PROJECT_NAME_REPORT), exist_ok=True)
    report.to_csv(PROJECT_NAME_REPORT, index=False)


# 5. MATCH THE DATASETS

def match_datasets(features, results):

    features, results = prepare_matching_keys(features, results)

    # Check duplicate matching keys
    feature_duplicates = features.duplicated(
        subset=["project_key", "test_key"]
    ).sum()

    result_duplicates = results.duplicated(
        subset=["project_key", "test_key"]
    ).sum()

    print("\n========== DUPLICATE KEY CHECK ==========")

    print("Feature duplicates:", feature_duplicates)
    print("Result duplicates:", result_duplicates)

    if feature_duplicates > 0 or result_duplicates > 0:
        raise ValueError(
            "Duplicate matching keys found. "
            "Investigate them before merging."
        )

    # Outer join keeps matched and unmatched records
    merged = pd.merge(
        features,
        results,
        on=["project_key", "test_key"],
        how="outer",
        suffixes=("_features", "_results"),
        indicator=True,
        validate="one_to_one"
    )

    matched = merged[merged["_merge"] == "both"].copy()

    only_features = merged[
        merged["_merge"] == "left_only"
    ].copy()

    only_results = merged[
        merged["_merge"] == "right_only"
    ].copy()

    print("\n========== MATCHING SUMMARY ==========")

    print("Total merged records:", len(merged))
    print("Matched records:", len(matched))
    print("Only in features:", len(only_features))
    print("Only in results:", len(only_results))

    return merged, matched, only_features, only_results


# 6. COMPARE FLAKY LABELS

def compare_labels(matched):

    matched = matched.copy()

    feature_labels = pd.to_numeric(
        matched["flaky"], errors="coerce"
    )

    result_labels = pd.to_numeric(
        matched["IsFlaky"], errors="coerce"
    )

    matched["label_match"] = (
        feature_labels == result_labels
    )

    consistent = matched["label_match"].sum()
    inconsistent = (~matched["label_match"]).sum()

    print("\n========== LABEL COMPARISON ==========")

    print("Consistent labels:", consistent)
    print("Inconsistent labels:", inconsistent)

    return matched


# 7. SAVE REPORTS

def save_reports(
    features,
    results,
    merged,
    matched,
    only_features,
    only_results
):

    os.makedirs("documentation", exist_ok=True)

    label_mismatches = matched[
        matched["label_match"] == False
    ].copy()

    summary = pd.DataFrame([{
        "total_feature_records": len(features),
        "total_result_records": len(results),
        "matched_records": len(matched),
        "only_in_features": len(only_features),
        "only_in_results": len(only_results),
        "consistent_labels": int(matched["label_match"].sum()),
        "inconsistent_labels": len(label_mismatches),
    }])

    summary.to_csv(MATCHING_REPORT, index=False)

    only_features.to_csv(UNMATCHED_FEATURES, index=False)

    only_results.to_csv(UNMATCHED_RESULTS, index=False)

    label_mismatches.to_csv(LABEL_MISMATCHES, index=False)

    print("\nMatching reports saved successfully!")


# 8. MAIN FUNCTION

def main():

    features, results = load_datasets()

    # Prepare normalized project and test identifiers
    features, results = prepare_matching_keys(
        features, results
    )

    inspect_project_names(features, results)

    merged, matched, only_features, only_results = match_datasets(
        features, results
    )

    matched = compare_labels(matched)

    save_reports(
        features,
        results,
        merged,
        matched,
        only_features,
        only_results
    )

    print("\n========== FINAL STATUS ==========")
    print("Dataset matching analysis completed.")


if __name__ == "__main__":
    main()
