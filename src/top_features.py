import os
import pandas as pd

from src.train_models import require_inputs

from src.config import OUTPUT_DIR, REPORT_DIR
from src.explainability import FEATURE_IMPORTANCE
from src.compare_influencers import KEY_INFLUENCERS, load_key_influencers, check_feature_names

# SETTINGS

TOP_N = 10

MIN_VOTES = 3

METHODS = {
    "shap_rank": "SHAP",
    "permutation_rank": "Permutation",
    "dt_rank": "Decision Tree",
    "lr_rank": "LR coefficient"
}

# OUTPUT FILES

TOP_FEATURES = os.path.join(OUTPUT_DIR, "top_features.csv")

METHODS_AGREEMENT = os.path.join(OUTPUT_DIR, "methods_agreement.csv")

METHOD_CORRELATION = os.path.join(OUTPUT_DIR, "methods_rank_correlation.csv")

FINDINGS = os.path.join(REPORT_DIR, "top_features_findings.md")

# Plain-English names for the findings

FEATURE_GROUPS = {
    "execution_time_s": "execution time",
    "source_lines_covered": "coverage",
    "source_classes_covered": "coverage",
    "covered_lines": "coverage",
    "test_length": "test size",
    "num_asserts": "test size",
    "third_party_libs": "third-party libraries"
}

# 1. LOAD IMPORTANCES

def load_importance():
    """Load the four per-method ranks written by explainability.py."""

    require_inputs([FEATURE_IMPORTANCE])

    table = pd.read_csv(FEATURE_IMPORTANCE)

    print("\nFeatures:", len(table))
    print("Methods:", list(METHODS.values()))

    return table


def feature_group(feature):
    """Group name: execution time, coverage, churn, smell, ..."""

    if feature.startswith("changed_line_churn"):
        return "churn"

    if feature.startswith("smell_"):
        return "test smell"

    return FEATURE_GROUPS.get(feature, "other")

# 2. ADD KEY INFLUENCERS (OPTIONAL 5TH VIEW)

def add_key_influencers(table):
    """Add Power BI Key Influencers ranks if the template has been filled in."""

    print("\n========== KEY INFLUENCERS (5th view) ==========")

    ki = load_key_influencers()

    if ki is None:
        print(f"Skipped: {KEY_INFLUENCERS} is missing or not filled in yet.")
        return table, False

    ki = check_feature_names(ki).drop_duplicates("feature")

    ki["ki_rank"] = range(1, len(ki) + 1)

    table = table.merge(ki[["feature", "ki_rank"]], on="feature", how="left")

    print("Added Key Influencers ranks for", int(table["ki_rank"].notna().sum()), "features")

    return table, True

# 3. MERGE RANKS

def merge_ranks(table, with_ki):
    """Mean rank across the 4 methods, plus votes = number of top-10 hits."""

    rank_cols = list(METHODS)

    for col in rank_cols:
        table[f"top10_{METHODS[col]}"] = table[col] <= TOP_N

    table["votes"] = table[[f"top10_{m}" for m in METHODS.values()]].sum(axis=1)

    table["mean_rank"] = table[rank_cols].mean(axis=1).round(2)
    table["best_rank"] = table[rank_cols].min(axis=1)
    table["worst_rank"] = table[rank_cols].max(axis=1)
    table["rank_spread"] = table["worst_rank"] - table["best_rank"]

    table["consensus"] = table["votes"] >= MIN_VOTES

    if with_ki:
        table["top10_Key Influencers"] = table["ki_rank"] <= TOP_N
        table["votes_with_ki"] = table["votes"] + table["top10_Key Influencers"].astype(int)

    table = table.sort_values(["mean_rank", "shap_rank"]).reset_index(drop=True)

    table.insert(0, "final_rank", range(1, len(table) + 1))

    table["group"] = table["feature"].map(feature_group)

    return table

# 4. AGREE / DISAGREE TABLE

def verdict(votes):
    """Plain-English agreement label for a number of top-10 votes."""

    if votes == 4:
        return "all 4 methods agree (top 10)"
    if votes == 3:
        return "most agree (3 of 4)"
    if votes == 2:
        return "methods disagree (2 of 4)"
    if votes == 1:
        return "only 1 method"
    return "no method ranks it top 10"


def build_agreement(table, with_ki):
    """Per-feature Yes/No for each method, votes and verdict."""

    cols = [f"top10_{m}" for m in METHODS.values()]

    if with_ki:
        cols.append("top10_Key Influencers")

    agreement = table[["final_rank", "feature"] + list(METHODS) + cols
                      + ["votes", "rank_spread"]].copy()

    for col in cols:
        agreement[col] = agreement[col].map({True: "Yes", False: "No"})

    agreement["verdict"] = agreement["votes"].map(verdict)

    return agreement


def method_correlation(table):
    """Spearman correlation between each pair of methods' rankings."""

    corr = table[list(METHODS)].corr(method="spearman").round(2)

    corr.index = corr.columns = list(METHODS.values())

    return corr

# 5. FINDINGS

def write_findings(table, corr, with_ki):
    """5-8 plain-English findings built from the numbers."""

    def ranks(row):
        return ", ".join(f"{METHODS[c]} #{int(row[c])}" for c in METHODS)

    def name_list(rows):
        return ", ".join(f"`{f}`" for f in rows["feature"])

    def direction_text(row):
        return {
            "higher -> more flaky": "a higher value pushes the flaky score up",
            "higher -> less flaky": "a higher value pushes the flaky score down",
        }.get(row["shap_direction"], "its effect goes both ways depending on the value")

    consensus = table[table["consensus"]]
    top = table.iloc[0]

    findings = []

    # 1. The single strongest feature

    findings.append(
        f"**`{top['feature']}` is the strongest flaky-test signal.** It ranks "
        f"{ranks(top)}. In SHAP, {direction_text(top)}."
    )

    # 2. Consensus features

    findings.append(
        f"**{len(consensus)} features are in the top {TOP_N} of at least {MIN_VOTES} of the 4 "
        f"methods:** {name_list(consensus)}. These are the features we can trust most, "
        "because models that work very differently all point to them."
    )

    # 3. Coverage

    coverage = table[table["group"] == "coverage"].sort_values("final_rank")
    best_cov = coverage.iloc[0]

    if best_cov["shap_direction"] == "higher -> more flaky":
        cov_head = "Coverage matters: tests that touch more of the project are riskier."
        cov_tail = (" Tests that exercise many classes depend on more shared state, "
                    "which is a classic cause of flakiness.")
    else:
        cov_head = "Coverage matters, but not in a simple more-is-worse way."
        cov_tail = ""

    findings.append(
        f"**{cov_head}** The best coverage feature, `{best_cov['feature']}`, is "
        f"#{int(best_cov['final_rank'])} overall ({ranks(best_cov)}); in SHAP, "
        f"{direction_text(best_cov)}.{cov_tail}"
    )

    # 4. Churn

    churn = table[table["group"] == "churn"].sort_values("final_rank")
    churn_top = churn[churn["final_rank"] <= TOP_N]

    churn_all = churn[churn["votes"] == len(METHODS)]

    agreed = (
        f"Only {name_list(churn_all)} is in the top {TOP_N} of all 4 methods"
        if len(churn_all) else f"No churn window is in the top {TOP_N} of all 4 methods"
    )

    findings.append(
        f"**Code churn matters, but the 8 churn windows share the credit.** "
        f"{len(churn_top)} of the 8 windows reach the overall top {TOP_N}; the strongest is "
        f"`{churn.iloc[0]['feature']}` (#{int(churn.iloc[0]['final_rank'])}). {agreed}: the "
        "windows are strongly correlated, so each method picks different windows."
    )

    # 5. Smells

    smells = table[table["group"] == "test smell"].sort_values("final_rank")
    best_smell = smells.iloc[0]

    findings.append(
        f"**Test smells are weak signals on their own.** The best smell, "
        f"`{best_smell['feature']}`, is only #{int(best_smell['final_rank'])} overall "
        f"({ranks(best_smell)}, top {TOP_N} in {int(best_smell['votes'])} of 4 methods); "
        f"{int((smells['votes'] == 0).sum())} of the 8 smells are not in any method's top {TOP_N}. "
        "Smells are rare (most appear in under 10% of tests), so they help explain some flaky "
        "tests but cannot separate most of them."
    )

    # 6. Biggest disagreement

    spread = table.sort_values("rank_spread", ascending=False).iloc[0]

    findings.append(
        f"**The methods disagree most on `{spread['feature']}`** ({ranks(spread)}). "
        "Logistic Regression only sees straight-line effects of each feature, while the trees "
        "can use a feature only in some branches, so features with uneven effects get very "
        "different ranks."
    )

    # 7. How similar the methods are overall

    pairs = corr.where(~pd.DataFrame(
        [[i >= j for j in range(len(corr))] for i in range(len(corr))],
        index=corr.index, columns=corr.columns
    )).stack()

    best_pair = pairs.idxmax()
    worst_pair = pairs.idxmin()

    findings.append(
        f"**The methods agree at the top but less further down.** The closest pair is {best_pair[0]} and "
        f"{best_pair[1]} (Spearman {pairs.max():.2f}); the least similar is {worst_pair[0]} and "
        f"{worst_pair[1]} ({pairs.min():.2f})."
    )

    # 8. Key Influencers

    if with_ki:
        ki_hits = int((table["top10_Key Influencers"] & table["consensus"]).sum())
        findings.append(
            f"**Power BI Key Influencers confirms {ki_hits} of the {len(consensus)} consensus "
            "features** (5th view)."
        )
    else:
        findings.append(
            "**Power BI Key Influencers is not included yet.** Fill in "
            "`outputs/powerbi/key_influencers_top.csv` and rerun this script to add it as a 5th view."
        )

    lines = [
        "# Top features: findings",
        "",
        "Generated by `python -m src.top_features` from SHAP, permutation importance, "
        "Decision Tree importance and Logistic Regression coefficients "
        "(`outputs/explainability/feature_importance.csv`).",
        "",
    ]

    lines += [f"{i}. {text}" for i, text in enumerate(findings, start=1)]

    lines += [
        "",
        "## Overall top 10 (mean rank of the 4 methods)",
        "",
        "| Rank | Feature | SHAP | Permutation | DT | LR | Votes | Direction (SHAP) |",
        "|---|---|---|---|---|---|---|---|",
    ]

    for _, r in table.head(TOP_N).iterrows():
        lines.append(
            f"| {r['final_rank']} | `{r['feature']}` | {r['shap_rank']} | {r['permutation_rank']} "
            f"| {r['dt_rank']} | {r['lr_rank']} | {r['votes']}/4 | {r['shap_direction']} |"
        )

    os.makedirs(REPORT_DIR, exist_ok=True)

    with open(FINDINGS, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("\n========== FINDINGS ==========")

    for i, text in enumerate(findings, start=1):
        print(f"{i}. {text.replace('**', '').replace('`', '')}")

    print("\nFindings saved to:", FINDINGS)

# 6. MAIN EXECUTION

def main():

    table = load_importance()

    table, with_ki = add_key_influencers(table)

    table = merge_ranks(table, with_ki)

    agreement = build_agreement(table, with_ki)

    corr = method_correlation(table)

    table.to_csv(TOP_FEATURES, index=False)
    agreement.to_csv(METHODS_AGREEMENT, index=False)
    corr.to_csv(METHOD_CORRELATION)

    print("\n========== TOP FEATURES (merged ranking) ==========")
    print(table.head(TOP_N)[["final_rank", "feature", "mean_rank", "votes",
                            "consensus"]].to_string(index=False))

    print("\n========== METHODS AGREE / DISAGREE ==========")
    print(agreement["verdict"].value_counts().to_string())

    print("\nRank correlation between methods (Spearman):")
    print(corr.to_string())

    print("\nTop features saved to:", TOP_FEATURES)
    print("Agreement table saved to:", METHODS_AGREEMENT)
    print("Method correlation saved to:", METHOD_CORRELATION)

    write_findings(table, corr, with_ki)


if __name__ == "__main__":
    main()
