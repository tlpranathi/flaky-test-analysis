from src import eda_correlations, eda_distribution, eda_comparisons


def run_statistical_eda(df):
    """Run Person 3's existing distribution and correlation analyses."""
    distribution_df = df.copy()
    for column in ("NumFailingRuns", "NumPassingRuns"):
        distribution_df[column] = distribution_df[column].astype("float64")

    class_table, imbalance_ratio = eda_distribution.class_distribution(
        distribution_df
    )
    project_table = eda_distribution.flaky_per_project(distribution_df)
    failure_tables = eda_distribution.failure_frequency(distribution_df)
    summary = eda_distribution.build_summary(
        distribution_df, project_table, imbalance_ratio
    )

    correlation = eda_correlations.correlation_matrix(df)
    eda_correlations.correlation_heatmap(correlation)
    target_correlations = eda_correlations.target_correlations(correlation)
    high_correlation_pairs = eda_correlations.high_correlation_pairs(
        correlation
    )
    feature_statistics = eda_correlations.feature_statistics(df)

    return {
        "class_distribution": class_table,
        "project_distribution": project_table,
        "failure_frequency": failure_tables,
        "summary": summary,
        "correlation_matrix": correlation,
        "target_correlations": target_correlations,
        "high_correlation_pairs": high_correlation_pairs,
        "feature_statistics": feature_statistics,
    }


def run_comparison_eda(df):
    """Run the Person 4 comparisons on the same prepared DataFrame."""
    return eda_comparisons.main(df)


def run_eda(df):
    """Run statistical EDA followed by flaky/non-flaky comparisons."""
    statistical_results = run_statistical_eda(df)
    run_comparison_eda(df)
    return statistical_results
