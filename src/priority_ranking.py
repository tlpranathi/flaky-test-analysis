import os
import json
import numpy as np
import pandas as pd

from src.train_models import PREDICTIONS_FILE, BEST_MODEL_INFO, require_inputs

import matplotlib.pyplot as plt

from src.config import OUTPUT_DIR, TARGET_COL, RENAME
from src.feature_engineering import FINAL_DATASET
from src.plot_style import (
    apply_style,
    save_figure,
    BLUE,
    ORANGE,
    TEXT_MUTED,
    TEXT_SECONDARY
)

# SETTINGS

K_SHARES = [0.05, 0.10, 0.20]

TOP_LIST = 20

EXEC_TIME_COL = RENAME["ExecutionTime"]

# OUTPUT FILES

PRIORITY_RANKING = os.path.join(OUTPUT_DIR, "priority_ranking.csv")

RECALL_AT_K = os.path.join(OUTPUT_DIR, "recall_at_k.csv")

TOP20_FILE = os.path.join(OUTPUT_DIR, "top20_priority.csv")

GAINS_FIG = os.path.join(OUTPUT_DIR, "priority_gains_curve.png")

# 1. LOAD OUT-OF-FOLD PREDICTIONS

def load_predictions():
    """Load out-of-fold probabilities and attach execution time."""

    require_inputs([PREDICTIONS_FILE, FINAL_DATASET])

    preds = pd.read_csv(PREDICTIONS_FILE)
    data = pd.read_csv(FINAL_DATASET, usecols=["test_name", "project", EXEC_TIME_COL])

    # Both files are written in the same row order by the pipeline

    same_rows = (
        len(preds) == len(data)
        and (preds["test_name"].values == data["test_name"].values).all()
        and (preds["project"].values == data["project"].values).all()
    )

    if not same_rows:
        raise ValueError("predictions.csv and cleaned_dataset.csv rows do not line up. "
                         "Rerun python -m src.train_models.")

    preds[EXEC_TIME_COL] = data[EXEC_TIME_COL].values

    model = "best model"

    if os.path.exists(BEST_MODEL_INFO):
        with open(BEST_MODEL_INFO) as f:
            info = json.load(f)
        model = f"{info['model_label']} ({info['sampling_label']})"

    print("\nTests:", len(preds))
    print(f"Flaky tests: {int(preds[TARGET_COL].sum())} ({preds[TARGET_COL].mean() * 100:.2f}%)")
    print("Scores: out-of-fold prob_best from", model)
    print("Every test is scored by a model that never saw it during training.")

    return preds

# 2. RANK TESTS

def position_rank(df, sort_cols):
    """Rank 1..N by sorting on sort_cols (all descending, stable)."""

    order = df.sort_values(sort_cols, ascending=False, kind="stable").index

    rank = pd.Series(np.arange(1, len(df) + 1), index=order)

    return rank.reindex(df.index).astype(int)


def rank_tests(preds):
    """Rank 1 = most likely flaky. Second rank: probability x execution time."""

    preds = preds.rename(columns={"prob_best": "flaky_probability"})

    # Expected seconds lost to flakiness: likely-flaky AND slow tests first

    preds["expected_flaky_seconds"] = (
        preds["flaky_probability"] * preds[EXEC_TIME_COL]
    ).round(6)

    # Many tests tie at the same probability: the slower one goes first

    preds["rank"] = position_rank(preds, ["flaky_probability", "expected_flaky_seconds"])

    preds["cost_rank"] = position_rank(preds, ["expected_flaky_seconds", "flaky_probability"])

    ranking = preds[[
        "rank", "test_name", "project", "flaky_probability",
        TARGET_COL, EXEC_TIME_COL, "expected_flaky_seconds", "cost_rank"
    ]].sort_values("rank")

    return ranking

# 3. RECALL@K AND LIFT

def recall_at_k(ranking, rank_col):
    """Recall@k and lift over random for the top 5%, 10% and 20%."""

    ordered = ranking.sort_values(rank_col)

    total_flaky = ordered[TARGET_COL].sum()
    total_time = ordered.loc[ordered[TARGET_COL] == 1, EXEC_TIME_COL].sum()

    rows = []

    for share in K_SHARES:

        n = int(np.ceil(share * len(ordered)))

        top = ordered.head(n)

        found = int(top[TARGET_COL].sum())

        recall = found / total_flaky

        rows.append({
            "ranking": "probability" if rank_col == "rank" else "probability x execution time",
            "top_share_pct": int(share * 100),
            "tests_reviewed": n,
            "flaky_found": found,
            "total_flaky": int(total_flaky),
            "recall_at_k": round(recall, 4),
            "precision_at_k": round(found / n, 4),
            "random_recall": share,
            "lift_over_random": round(recall / share, 2),
            "flaky_exec_time_captured_pct": round(
                top.loc[top[TARGET_COL] == 1, EXEC_TIME_COL].sum() / total_time * 100, 2
            )
        })

    return pd.DataFrame(rows)

# 4. GAINS CURVE

def plot_gains(ranking):
    """Cumulative share of flaky tests found vs share of tests reviewed."""

    apply_style()

    fig, ax = plt.subplots(figsize=(7, 5))

    for rank_col, label, color in [
        ("rank", "Ranked by flaky probability", BLUE),
        ("cost_rank", "Ranked by probability x execution time", ORANGE)
    ]:
        ordered = ranking.sort_values(rank_col)
        found = ordered[TARGET_COL].cumsum() / ordered[TARGET_COL].sum()
        reviewed = np.arange(1, len(ordered) + 1) / len(ordered)
        ax.plot(reviewed * 100, found * 100, color=color, label=label)

    ax.plot([0, 100], [0, 100], color=TEXT_MUTED, linestyle="--",
            linewidth=1, label="Random order")

    ordered = ranking.sort_values("rank")
    n10 = int(np.ceil(0.10 * len(ordered)))
    r10 = ordered.head(n10)[TARGET_COL].sum() / ordered[TARGET_COL].sum() * 100

    ax.scatter([10], [r10], color=BLUE, s=40, zorder=3,
               edgecolor="#fcfcfb", linewidth=2)
    ax.annotate(f"Top 10% -> {r10:.0f}% of flaky tests", (10, r10),
                xytext=(28, 68), fontsize=9, color=TEXT_SECONDARY,
                arrowprops={"arrowstyle": "-", "color": TEXT_MUTED, "linewidth": 1})

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 102)
    ax.set_xlabel("% of tests reviewed (highest score first)")
    ax.set_ylabel("% of flaky tests found")
    ax.set_title("How many flaky tests does a ranked review find?")
    ax.legend(loc="lower right")

    save_figure(fig, GAINS_FIG)

# 5. SAVE OUTPUTS

def save_outputs(ranking, recall_table):
    """Write the ranking, the recall@k table and the top-20 list."""

    ranking.to_csv(PRIORITY_RANKING, index=False)
    recall_table.to_csv(RECALL_AT_K, index=False)

    top = ranking.head(TOP_LIST)
    top.to_csv(TOP20_FILE, index=False)

    print("\nPriority ranking saved to:", PRIORITY_RANKING)
    print("Recall@k table saved to:", RECALL_AT_K)
    print("Top-20 list saved to:", TOP20_FILE)

# 6. MAIN EXECUTION

def main():

    preds = load_predictions()

    ranking = rank_tests(preds)

    recall_table = pd.concat([
        recall_at_k(ranking, "rank"),
        recall_at_k(ranking, "cost_rank")
    ], ignore_index=True)

    print("\n========== RECALL@K AND LIFT ==========")
    print(recall_table[[
        "ranking", "top_share_pct", "tests_reviewed", "flaky_found",
        "recall_at_k", "precision_at_k", "lift_over_random",
        "flaky_exec_time_captured_pct"
    ]].to_string(index=False))

    print(f"\n========== TOP {TOP_LIST} TESTS TO REVIEW ==========")
    top = ranking.head(TOP_LIST)
    print(top[["rank", "project", "flaky_probability", TARGET_COL, "test_name"]]
          .to_string(index=False))
    print(f"Actually flaky in the top {TOP_LIST}: {int(top[TARGET_COL].sum())}")

    save_outputs(ranking, recall_table)

    plot_gains(ranking)

    top10 = recall_table[(recall_table["ranking"] == "probability")
                         & (recall_table["top_share_pct"] == 10)].iloc[0]

    print("\n========== HEADLINE ==========")
    print(f"Reviewing the top 10% of tests finds {top10['recall_at_k'] * 100:.1f}% "
          f"of flaky tests ({top10['lift_over_random']:.1f}x better than random).")


if __name__ == "__main__":
    main()
