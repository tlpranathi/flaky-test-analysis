from src import data_cleaning, data_loader, eda, feature_engineering


def main():
    print("========== FLAKY TEST ANALYSIS PIPELINE ==========")

    print("\n[1/6] Loading original datasets...")
    features = data_loader.load_features()
    results = data_loader.load_results()

    print("\n[2/6] Validating original datasets...")
    features_valid = data_loader.validate_features(features)
    results_valid = data_loader.validate_results(results)
    data_loader.dataset_summary(features)
    data_loader.save_validation_report(
        features_valid,
        results_valid,
        features,
        results,
    )
    if not features_valid or not results_valid:
        raise ValueError(
            "Input dataset validation failed; review "
            "documentation/dataset_validation_report.txt."
        )

    print("\n[3/6] Cleaning and preparing datasets...")
    rows_before, columns_before = features.shape
    features = data_cleaning.remove_unused_columns(features)
    features = data_cleaning.rename_columns(features)
    features, filled_values = data_cleaning.fill_missing_values(features)
    results, fixed_values = data_cleaning.fix_invalid_values(results)
    data_cleaning.check_impossible_values(features, results)
    features, feature_duplicates = data_cleaning.check_duplicates(
        features, ["project", "test_name"], "test_features"
    )
    results, result_duplicates = data_cleaning.check_duplicates(
        results, ["Project", "Test"], "test_results"
    )
    cleaned, matched, unmatched = data_cleaning.merge_results(
        features, results
    )
    print("\nCleaning summary:")
    print(f"Rows: {rows_before} -> {len(cleaned)}")
    print(f"Columns: {columns_before} -> {len(cleaned.columns)}")
    print("Values filled:", sum(filled_values.values()))
    print("Malformed result values fixed:", fixed_values)
    print("Duplicate rows removed:", feature_duplicates + result_duplicates)
    print(f"Rerun results matched: {matched}; without rerun data: {unmatched}")
    data_cleaning.save_cleaned_data(cleaned)

    print("\n[4/6] Engineering features...")
    engineered = feature_engineering.add_execution_time_buckets(cleaned)
    engineered = feature_engineering.add_third_party_lib_buckets(engineered)
    engineered = feature_engineering.add_coverage_buckets(engineered)
    engineered = feature_engineering.add_flaky_label(engineered)
    engineered = feature_engineering.add_fail_rate(engineered)
    roles = feature_engineering.build_column_roles(engineered)
    feature_engineering.save_final_dataset(engineered, roles)

    print("\n[5/6] Running Person 3 statistical EDA...")
    eda.run_statistical_eda(engineered)

    print("\n[6/6] Running Person 4 comparison visualizations...")
    eda.run_comparison_eda(engineered)

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()
