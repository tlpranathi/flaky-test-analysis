import os
import sys
import pandas as pd

from src.train_models import require_inputs

from src.config import OUTPUT_DIR
from src.feature_engineering import COLUMN_ROLES

# SETTINGS

TOP_N = 10

TEMPLATE_ROWS = 10

TEMPLATE_COLS = ["rank", "feature", "condition", "likelihood_multiplier", "notes"]

# FILES

POWERBI_DIR = os.path.join(OUTPUT_DIR, "powerbi")

KEY_INFLUENCERS = os.path.join(POWERBI_DIR, "key_influencers_top.csv")

FEATURE_IMPORTANCE = os.path.join(OUTPUT_DIR, "explainability", "feature_importance.csv")

COMPARISON_FILE = os.path.join(POWERBI_DIR, "influencers_vs_shap.csv")

NOTES_FILE = os.path.join(POWERBI_DIR, "influencers_agreement_notes.md")

# Words in the Power BI condition text that say which way the effect goes

UP_WORDS = [">", "more than", "higher", "high", "is 1", "= 1", "increases"]
DOWN_WORDS = ["<", "less than", "lower", "low", "is 0", "= 0", "decreases"]

# 1. CREATE TEMPLATE

def create_template(path=KEY_INFLUENCERS):
    """Write an empty template for the Power BI Key Influencers results."""

    os.makedirs(os.path.dirname(path), exist_ok=True)

    template = pd.DataFrame({
        "rank": range(1, TEMPLATE_ROWS + 1),
        "feature": "",
        "condition": "",
        "likelihood_multiplier": "",
        "notes": ""
    })[TEMPLATE_COLS]

    template.to_csv(path, index=False)

    print("Template created:", path)

# 2. LOAD KEY INFLUENCERS

def load_key_influencers(path=KEY_INFLUENCERS):
    """Return the filled-in Key Influencers rows, or None if not filled in yet."""

    if not os.path.exists(path):
        return None

    ki = pd.read_csv(path, dtype=str).fillna("")

    missing = [col for col in ["rank", "feature"] if col not in ki.columns]

    if missing:
        raise ValueError(f"{path} is missing columns {missing}. "
                         f"Expected: {TEMPLATE_COLS}")

    ki["feature"] = ki["feature"].str.strip()

    ki = ki[ki["feature"] != ""].copy()

    if ki.empty:
        return None

    ki["rank"] = pd.to_numeric(ki["rank"], errors="coerce")

    return ki.sort_values("rank").reset_index(drop=True)


def check_feature_names(ki):
    """Match names case-insensitively to the 23 model features."""

    require_inputs([COLUMN_ROLES])

    roles = pd.read_csv(COLUMN_ROLES)

    features = list(roles.loc[roles["role"] == "model_feature", "column"])

    lookup = {f.lower(): f for f in features}

    ki["feature"] = ki["feature"].map(lambda f: lookup.get(f.lower(), f))

    unknown = sorted(set(ki["feature"]) - set(features))

    if unknown:
        print("WARNING: Not one of the 23 model features (ignored):", unknown)
        print("Use the exact column names, e.g.", features[:3])

    return ki[ki["feature"].isin(features)].copy()

# 3. COMPARE WITH SHAP

def condition_direction(condition):
    """Read 'higher -> more flaky' / 'lower -> more flaky' from the condition text."""

    text = str(condition).lower()

    if any(word in text for word in DOWN_WORDS):
        return "lower -> more flaky"

    if any(word in text for word in UP_WORDS):
        return "higher -> more flaky"

    return "unknown"


def compare_with_shap(ki, importance):
    """Per-feature table: Key Influencers rank vs SHAP rank and direction."""

    # A feature can appear more than once (different conditions): keep the first

    ki_first = ki.drop_duplicates("feature").reset_index(drop=True)
    ki_first["ki_rank"] = range(1, len(ki_first) + 1)
    ki_first["ki_direction"] = ki_first["condition"].map(condition_direction) \
        if "condition" in ki_first else "unknown"

    shap = importance[["feature", "shap_rank", "shap_direction"]]

    shap_top = set(shap.loc[shap["shap_rank"] <= TOP_N, "feature"])
    ki_top = set(ki_first.loc[ki_first["ki_rank"] <= TOP_N, "feature"])

    table = shap.merge(
        ki_first[["feature", "ki_rank", "condition", "ki_direction"]],
        on="feature",
        how="left"
    )

    table["in_shap_top10"] = table["feature"].isin(shap_top)
    table["in_ki_top10"] = table["feature"].isin(ki_top)

    table["agreement"] = "neither"
    table.loc[table["in_shap_top10"] & table["in_ki_top10"], "agreement"] = "both"
    table.loc[table["in_shap_top10"] & ~table["in_ki_top10"], "agreement"] = "SHAP only"
    table.loc[~table["in_shap_top10"] & table["in_ki_top10"], "agreement"] = "Key Influencers only"

    table["rank_difference"] = (table["ki_rank"] - table["shap_rank"]).abs()

    def same_direction(row):
        if pd.isna(row["ki_rank"]) or row["ki_direction"] == "unknown" \
                or row["shap_direction"] == "mixed":
            return ""
        return "yes" if row["ki_direction"] == row["shap_direction"] else "no"

    table["direction_agrees"] = table.apply(same_direction, axis=1)

    table = table[(table["agreement"] != "neither")].sort_values(
        ["agreement", "shap_rank"]
    )

    return table, shap_top, ki_top

# 4. WRITE NOTES

def write_notes(table, shap_top, ki_top):
    """Plain-English agreement notes."""

    both = table[table["agreement"] == "both"]
    shap_only = table[table["agreement"] == "SHAP only"]
    ki_only = table[table["agreement"] == "Key Influencers only"]

    overlap = len(shap_top & ki_top)
    union = len(shap_top | ki_top)

    rank_pairs = both.dropna(subset=["ki_rank"])

    spearman = (
        rank_pairs["ki_rank"].corr(rank_pairs["shap_rank"], method="spearman")
        if len(rank_pairs) >= 3 else float("nan")
    )

    direction_checked = table[table["direction_agrees"] != ""]
    direction_ok = int((direction_checked["direction_agrees"] == "yes").sum())

    def names(rows):
        return ", ".join(f"`{f}`" for f in rows["feature"]) or "none"

    lines = [
        "# Key Influencers vs SHAP: agreement notes",
        "",
        f"- **Overlap:** {overlap} of the Key Influencers top {len(ki_top)} are also in the "
        f"SHAP top {TOP_N} (Jaccard {overlap / union:.2f})." if union else "- No features to compare.",
        f"- **Both methods:** {names(both)}.",
        f"- **SHAP only:** {names(shap_only)}.",
        f"- **Key Influencers only:** {names(ki_only)}.",
    ]

    if pd.notna(spearman):
        lines.append(f"- **Rank agreement** on shared features (Spearman): {spearman:.2f} "
                     "(1 = same order, 0 = unrelated).")

    if len(direction_checked):
        lines.append(f"- **Direction:** for {direction_ok} of {len(direction_checked)} features "
                     "the Power BI condition points the same way as SHAP "
                     "(e.g. higher value -> more likely flaky).")

    lines += [
        "",
        "## Why they can differ",
        "",
        "- Key Influencers fits a simple logistic regression per feature (one feature at a time),",
        "  so correlated features such as the 8 churn windows can all look important.",
        "- SHAP explains the trained Random Forest, which uses all features together and",
        "  splits the credit among correlated features.",
        "- Key Influencers reports conditions on a bucket (\"execution_time_s > 1.2\"), while",
        "  SHAP ranks the whole feature, so a feature that only matters in a narrow range",
        "  may rank lower in SHAP.",
    ]

    with open(NOTES_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("\n".join(lines[2:8]))
    print("\nNotes saved to:", NOTES_FILE)

# 5. MAIN EXECUTION

def main():

    # Optional: python -m src.compare_influencers <path to a filled csv>

    path = sys.argv[1] if len(sys.argv) > 1 else KEY_INFLUENCERS

    print("\n========== KEY INFLUENCERS vs SHAP ==========")

    if not os.path.exists(path):
        print(f"{path} not found, so a blank template was created.")
        create_template(path)

    ki = load_key_influencers(path)

    if ki is None:
        print(f"\nThe template {path} is not filled in yet.")
        print("Fill it in from Power BI (see documentation/key_influencers_guide.md),")
        print("one row per influencer: rank, feature, condition, likelihood_multiplier.")
        print("Then rerun: python -m src.compare_influencers")
        return

    require_inputs([FEATURE_IMPORTANCE])

    ki = check_feature_names(ki)

    print("Key Influencers rows:", len(ki))

    importance = pd.read_csv(FEATURE_IMPORTANCE)

    table, shap_top, ki_top = compare_with_shap(ki, importance)

    table.to_csv(COMPARISON_FILE, index=False)

    print(table[["feature", "agreement", "shap_rank", "ki_rank",
                 "direction_agrees"]].to_string(index=False))

    print("\nComparison saved to:", COMPARISON_FILE)

    write_notes(table, shap_top, ki_top)


if __name__ == "__main__":
    main()
