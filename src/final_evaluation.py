import os
import json
import warnings
import numpy as np
import pandas as pd

from src.train_models import (
    METRICS_FILE,
    BEST_MODEL_INFO,
    SAMPLINGS,
    SAMPLING_LABELS,
    MODEL_LABELS,
    THRESHOLD,
    build_pipeline,
    load_model_data,
    require_inputs,
    score_predictions
)

import matplotlib.pyplot as plt

from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from src.config import OUTPUT_DIR
from src.plot_style import apply_style, save_figure, BLUE, TEXT_SECONDARY, TEXT_MUTED

# OUTPUT FILES

FINAL_METRICS = os.path.join(OUTPUT_DIR, "final_metrics.csv")

LOPO_RESULTS = os.path.join(OUTPUT_DIR, "lopo_results.csv")

FINAL_READING = os.path.join(OUTPUT_DIR, "final_reading.md")

LOPO_FIG = os.path.join(OUTPUT_DIR, "lopo_f1_by_project.png")

METRIC_COLS = ["precision", "recall", "f1", "roc_auc", "pr_auc"]

UNDERSAMPLING = ["none", "under_10", "under_25", "under_50"]

# 1. LOAD INPUTS

def load_inputs():
    """Load the CV metrics and the best model settings from train_models."""

    require_inputs([METRICS_FILE, BEST_MODEL_INFO])

    metrics = pd.read_csv(METRICS_FILE)

    with open(BEST_MODEL_INFO) as f:
        best = json.load(f)

    print("\nBest model from Stage 3:", best["model_label"], "-", best["sampling_label"])

    return metrics, best

# 2. FINAL METRICS TABLE

def build_final_metrics(metrics):
    """No balancing vs 10/25/50% undersampling vs SMOTE, for every model."""

    print("\n========== FINAL METRICS (stratified 5-fold CV, mean) ==========")

    final = metrics[["model_label", "sampling", "sampling_label",
                     "train_flaky_share_pct", "test_flaky_share_pct"]].copy()

    for metric in METRIC_COLS:
        final[metric] = metrics[f"{metric}_mean"].round(4)

    final["f1_std"] = metrics["f1_std"].round(4)
    final["is_best_overall"] = metrics["is_best"]

    final["sampling_order"] = final["sampling"].map(
        {name: i for i, name in enumerate(SAMPLINGS)}
    )

    final = final.sort_values(["model_label", "sampling_order"]).drop(columns="sampling_order")

    print(final[["model_label", "sampling_label", "precision", "recall",
                 "f1", "roc_auc"]].to_string(index=False))

    return final

# 3. LEAVE-ONE-PROJECT-OUT

def run_lopo(df, X, y, best):
    """Train on 23 projects, test on the held-out one, for all 24 projects."""

    print("\n========== LEAVE-ONE-PROJECT-OUT ==========")
    print("Model:", best["model_label"], "-", best["sampling_label"])

    projects = sorted(df["project"].unique())

    lopo_prob = np.zeros(len(y))
    rows = []

    for project in projects:

        test_mask = (df["project"] == project).values

        pipe = build_pipeline(best["model"], best["sampling"])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe.fit(X[~test_mask], y[~test_mask])

        prob = pipe.predict_proba(X[test_mask])[:, 1]

        lopo_prob[test_mask] = prob

        y_test = y[test_mask]
        pred = (prob >= THRESHOLD).astype(int)

        flaky = int(y_test.sum())

        scores = score_predictions(y_test, prob)

        # Recall and F1 have no meaning when a project has no flaky tests

        if flaky == 0:
            scores["recall"] = np.nan
            scores["f1"] = np.nan

        rows.append({
            "project": project,
            "tests": int(test_mask.sum()),
            "flaky": flaky,
            "flaky_share_pct": round(y_test.mean() * 100, 2),
            "predicted_flaky": int(pred.sum()),
            "true_positives": int(((pred == 1) & (y_test == 1)).sum()),
            "false_positives": int(((pred == 1) & (y_test == 0)).sum()),
            "false_negatives": int(((pred == 0) & (y_test == 1)).sum()),
            **{m: scores[m] for m in METRIC_COLS}
        })

        f1_text = "n/a (no flaky tests)" if flaky == 0 else f"{scores['f1']:.3f}"

        print(f"{project:<18} tests={int(test_mask.sum()):>5} flaky={flaky:>4} "
              f"F1={f1_text}")

    lopo = pd.DataFrame(rows).sort_values("flaky", ascending=False)

    return lopo, lopo_prob

# 4. SUMMARISE LOPO

def summarise_lopo(lopo, lopo_prob, y):
    """Unseen-project F1 (all held-out predictions pooled) and per-project mean."""

    pred = (lopo_prob >= THRESHOLD).astype(int)

    summary = {
        "unseen_f1": f1_score(y, pred, zero_division=0),
        "unseen_precision": precision_score(y, pred, zero_division=0),
        "unseen_recall": recall_score(y, pred, zero_division=0),
        "unseen_roc_auc": roc_auc_score(y, lopo_prob),
        "median_project_roc_auc": lopo["roc_auc"].median(skipna=True),
        "mean_project_f1": lopo["f1"].mean(skipna=True),
        "median_project_f1": lopo["f1"].median(skipna=True),
        "projects_scored": int(lopo["f1"].notna().sum()),
        "projects_f1_zero": int((lopo["f1"] == 0).sum())
    }

    print("\n========== LOPO SUMMARY ==========")
    print(f"Unseen-project F1 (all held-out predictions pooled): {summary['unseen_f1']:.3f}")
    print(f"  precision {summary['unseen_precision']:.3f}, recall {summary['unseen_recall']:.3f}")
    print(f"Mean per-project F1 ({summary['projects_scored']} projects with flaky tests): "
          f"{summary['mean_project_f1']:.3f}")
    print("Projects where no flaky test was found (F1 = 0):", summary["projects_f1_zero"])
    print(f"ROC-AUC on unseen projects: pooled {summary['unseen_roc_auc']:.3f}, "
          f"median per project {summary['median_project_roc_auc']:.3f}")

    return summary

# 5. LOPO CHART

def plot_lopo(lopo, pooled_f1):
    """Per-project F1 on unseen projects, with the pooled CV F1 for reference."""

    data = lopo.dropna(subset=["f1"]).sort_values("f1")

    apply_style()

    fig, ax = plt.subplots(figsize=(7.5, 7))

    ax.barh(data["project"], data["f1"], color=BLUE, height=0.6)
    ax.grid(axis="y", visible=False)

    for i, (f1, flaky) in enumerate(zip(data["f1"], data["flaky"])):
        ax.text(f1, i, f" {f1:.2f}  (n flaky={flaky})", va="center",
                fontsize=8, color=TEXT_SECONDARY)

    ax.axvline(pooled_f1, color=TEXT_MUTED, linestyle="--", linewidth=1)
    ax.text(pooled_f1, len(data) - 0.4, f" pooled CV F1 = {pooled_f1:.2f}",
            fontsize=8, color=TEXT_SECONDARY, va="bottom")

    ax.set_xlim(0, 1.15)
    ax.set_xlabel("F1 on the held-out project")
    ax.set_title("Leave-one-project-out: F1 on projects the model never saw\n"
                 "(jimfs omitted: it has no flaky tests)", fontsize=11)

    save_figure(fig, LOPO_FIG)

# 6. WRITE THE READING

def best_undersampling(final, model_label):
    """Best of the 10 / 25 / 50% undersampling ratios for one model, by F1."""

    rows = final[(final["model_label"] == model_label)
                 & final["sampling"].isin(UNDERSAMPLING[1:])]

    return rows.sort_values("f1", ascending=False).iloc[0]


def write_reading(final, best, summary):
    """One-paragraph plain-English reading of the final results."""

    best_row = final[final["is_best_overall"]].iloc[0]

    model_rows = final[final["model_label"] == best["model_label"]].set_index("sampling")

    pooled_f1 = best_row["f1"]

    under = best_undersampling(final, best["model_label"])

    none_f1 = model_rows.loc["none", "f1"]
    none_recall = model_rows.loc["none", "recall"]
    u50 = model_rows.loc["under_50"]
    smote = model_rows.loc["smote_50"]

    lr_rows = final[final["model_label"] == MODEL_LABELS["logistic_regression"]].set_index("sampling")
    lr_under = best_undersampling(final, MODEL_LABELS["logistic_regression"])

    ratio_text = SAMPLING_LABELS[under["sampling"]].lower().replace("undersample to ", "")

    if best["sampling"] == "smote_50":
        smote_text = (f"SMOTE to 50% gives the best F1 overall ({smote['f1']:.3f}), "
                      f"{smote['f1'] - none_f1:+.3f} vs no balancing.")
    else:
        smote_text = f"SMOTE to 50% gives F1 {smote['f1']:.3f}."

    if abs(under["f1"] - none_f1) < 0.01:
        versus = "essentially tied with"
    elif under["f1"] > none_f1:
        versus = "better than"
    else:
        versus = "worse than"

    paragraph = (
        f"**Pooled F1 = {pooled_f1:.2f}, unseen-project F1 = {summary['unseen_f1']:.2f}.** "
        f"The best model is a {best['model_label']} trained with {best['sampling_label']}. "
        f"With stratified 5-fold cross-validation (projects mixed across folds) it reaches "
        f"F1 {pooled_f1:.2f} (precision {best_row['precision']:.2f}, recall {best_row['recall']:.2f}, "
        f"ROC-AUC {best_row['roc_auc']:.2f}). When a whole project is held out and the model "
        f"has never seen it (leave-one-project-out), F1 falls to {summary['unseen_f1']:.2f} "
        f"(mean per-project F1 {summary['mean_project_f1']:.2f}, and in "
        f"{summary['projects_f1_zero']} of {summary['projects_scored']} projects no flaky test was found), "
        f"so the model relies heavily on project-specific patterns. It still ranks tests better than "
        f"random inside a new project (median per-project ROC-AUC {summary['median_project_roc_auc']:.2f}, "
        f"vs {best_row['roc_auc']:.2f} pooled), so mainly the fixed 0.5 cut-off breaks on unseen projects. "
        f"The best undersampling ratio for the {best['model_label']} is **{ratio_text} flaky** "
        f"(F1 {under['f1']:.3f}, {versus} no balancing at {none_f1:.3f}): heavier undersampling to 50% raises "
        f"recall to {u50['recall']:.2f} but cuts precision to {u50['precision']:.2f}, so F1 drops to "
        f"{u50['f1']:.2f}. {smote_text} Balancing matters far more for Logistic Regression, "
        f"whose F1 rises from {lr_rows.loc['none', 'f1']:.2f} with no balancing to "
        f"{lr_under['f1']:.2f} at {SAMPLING_LABELS[lr_under['sampling']].lower()} flaky."
    )

    notes = f"""
## How to read this

- **The "no balancing" baseline uses no class weights.** The model sees the real
  3.65% flaky share and nothing tells it that missing a flaky test is costly, so
  low recall there is expected (here {none_recall:.2f} for the {best['model_label']}).
  That baseline is the point of the comparison: it shows what balancing adds.
- **Balancing is applied only to the training folds.** Undersampling removes
  non-flaky tests and SMOTE creates synthetic flaky tests, but only in the data the
  model learns from. The test folds always keep the real 3.65% flaky share.
- **Why:** in real use the model will see the natural mix of tests, where only about
  1 in 27 is flaky. If the test folds were balanced too, precision would look much
  better than it really is, and SMOTE's synthetic tests would be graded as if they
  were real tests. Keeping test folds untouched makes the scores honest.
- Thresholds are fixed at 0.5. "Pooled F1" is the mean F1 over the 5 stratified folds.
  "Unseen-project F1" is the F1 over all leave-one-project-out predictions combined.

## Files

- `outputs/final_metrics.csv`: every model x balancing setting
- `outputs/lopo_results.csv`: per-project leave-one-project-out results
- `outputs/lopo_f1_by_project.png`: per-project F1 chart
"""

    with open(FINAL_READING, "w", encoding="utf-8") as f:
        f.write("# Final evaluation: reading\n\n" + paragraph + "\n" + notes)

    print("\n========== READING ==========")
    print(paragraph.replace("**", ""))
    print("\nReading saved to:", FINAL_READING)

    return pooled_f1

# 7. MAIN EXECUTION

def main():

    metrics, best = load_inputs()

    df, X, y, features = load_model_data()

    if features != best["features"]:
        raise ValueError("Feature list differs from the one used in train_models.")

    final = build_final_metrics(metrics)

    final.to_csv(FINAL_METRICS, index=False)

    print("\nFinal metrics saved to:", FINAL_METRICS)

    lopo, lopo_prob = run_lopo(df, X.values, y.values, best)

    lopo.round(4).to_csv(LOPO_RESULTS, index=False)

    print("\nLOPO results saved to:", LOPO_RESULTS)

    summary = summarise_lopo(lopo, lopo_prob, y.values)

    pooled_f1 = write_reading(final, best, summary)

    plot_lopo(lopo, pooled_f1)


if __name__ == "__main__":
    main()
