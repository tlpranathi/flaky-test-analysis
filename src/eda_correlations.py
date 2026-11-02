import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

from src.config import TARGET_COL

from src.feature_engineering import MODEL_FEATURE_COLS

from src.eda_distribution import (
    load_dataset,
    save_table,
    save_figure
)

# CORRELATION SETTINGS

# Spearman is used because most features are heavily right-skewed
# and the smells and target are binary

CORR_METHOD = "spearman"

# Feature pairs at or above this absolute correlation are reported

HIGH_CORR_THRESHOLD = 0.8

# 1. CORRELATION MATRIX

def correlation_matrix(df):
    """Correlation of every model feature with every other feature and the target."""

    print("\n========== CORRELATION MATRIX ==========")

    columns = MODEL_FEATURE_COLS + [TARGET_COL]

    corr = df[columns].corr(method=CORR_METHOD)

    print("Method:", CORR_METHOD)
    print("Matrix size:", corr.shape)

    save_table(
        corr.round(4).rename_axis("feature").reset_index(),
        "correlation_matrix.csv"
    )

    return corr

# 2. CORRELATION HEATMAP

def correlation_heatmap(corr):
    """Lower-triangle heatmap of the correlation matrix."""

    print("\n========== CORRELATION HEATMAP ==========")

    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    fig, ax = plt.subplots(figsize=(18, 15))

    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 7},
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.7, "label": f"{CORR_METHOD.title()} correlation"},
        ax=ax
    )

    ax.set_title(
        f"Feature Correlation Heatmap ({CORR_METHOD.title()})",
        fontsize=14
    )

    save_figure(fig, "correlation_heatmap.png")

# 3. CORRELATION WITH THE TARGET

def target_correlations(corr):
    """Rank features by their correlation with the flaky label."""

    print("\n========== CORRELATION WITH FLAKY LABEL ==========")

    with_target = corr[TARGET_COL].drop(TARGET_COL)

    table = pd.DataFrame({
        "feature": with_target.index,
        "correlation": with_target.values.round(4),
        "abs_correlation": with_target.abs().values.round(4)
    })

    table = table.sort_values(
        "abs_correlation",
        ascending=False
    ).reset_index(drop=True)

    print(table.to_string(index=False))

    save_table(table, "feature_target_correlation.csv")

    return table

# 4. HIGHLY CORRELATED FEATURE PAIRS

def high_correlation_pairs(corr):
    """List feature pairs that are almost redundant (multicollinearity)."""

    print("\n========== HIGHLY CORRELATED FEATURE PAIRS ==========")

    features = corr.loc[MODEL_FEATURE_COLS, MODEL_FEATURE_COLS]

    upper = features.where(
        np.triu(np.ones(features.shape, dtype=bool), k=1)
    )

    pairs = upper.stack().rename("correlation").reset_index()

    pairs.columns = ["feature_1", "feature_2", "correlation"]

    pairs = pairs[pairs["correlation"].abs() >= HIGH_CORR_THRESHOLD]

    pairs = pairs.sort_values(
        "correlation",
        key=abs,
        ascending=False
    ).reset_index(drop=True)

    pairs["correlation"] = pairs["correlation"].round(4)

    print(f"Pairs with |correlation| >= {HIGH_CORR_THRESHOLD}:", len(pairs))

    if len(pairs):
        print(pairs.to_string(index=False))

    save_table(pairs, "high_correlation_pairs.csv")

    return pairs

# 5. FEATURE STATISTICS

def feature_statistics(df):
    """Descriptive statistics per feature, overall and for flaky vs non-flaky."""

    print("\n========== FEATURE STATISTICS ==========")

    features = df[MODEL_FEATURE_COLS]

    flaky = df.loc[df[TARGET_COL] == 1, MODEL_FEATURE_COLS]
    non_flaky = df.loc[df[TARGET_COL] == 0, MODEL_FEATURE_COLS]

    stats = features.describe().T.rename(columns={"50%": "median"})

    stats["skew"] = features.skew()

    stats["mean_flaky"] = flaky.mean()
    stats["mean_non_flaky"] = non_flaky.mean()

    stats["median_flaky"] = flaky.median()
    stats["median_non_flaky"] = non_flaky.median()

    stats = stats.round(4)

    # Mann-Whitney U: do flaky and non-flaky tests differ in this feature?
    # Added after rounding so small p-values are not rounded to 0.

    stats["mannwhitney_p"] = [
        mannwhitneyu(flaky[col], non_flaky[col]).pvalue
        for col in MODEL_FEATURE_COLS
    ]

    stats = stats.rename_axis("feature").reset_index()

    stats["count"] = stats["count"].astype(int)

    print(
        stats[[
            "feature", "mean", "median", "max",
            "median_flaky", "median_non_flaky", "mannwhitney_p"
        ]].to_string(index=False)
    )

    save_table(stats, "feature_statistics.csv")

    return stats

# 6. MAIN EXECUTION

def main():

    df = load_dataset()

    corr = correlation_matrix(df)

    correlation_heatmap(corr)

    target_correlations(corr)

    high_correlation_pairs(corr)

    feature_statistics(df)


if __name__ == "__main__":
    main()
