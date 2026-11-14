import os

# Stops a harmless joblib "could not find the number of physical
# cores" traceback on Windows. joblib only skips that check when the
# limit is below the logical core count, hence "- 1".

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(max(1, (os.cpu_count() or 2) - 1)))

import sys
import json
import warnings
import joblib
import numpy as np
import pandas as pd

from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

from src.config import TARGET_COL, OUTPUT_DIR
from src.feature_engineering import FINAL_DATASET, COLUMN_ROLES, LEAKAGE_COLS
from src.plot_style import apply_style, save_figure, BLUE_RAMP, TEXT_PRIMARY

import matplotlib.pyplot as plt

# SETTINGS

RANDOM_STATE = 42

N_FOLDS = 5

THRESHOLD = 0.5

# OUTPUT FILES

MODEL_DIR = os.path.join(OUTPUT_DIR, "model")

METRICS_FILE = os.path.join(MODEL_DIR, "metrics.csv")

PREDICTIONS_FILE = os.path.join(MODEL_DIR, "predictions.csv")

BEST_MODEL_FILE = os.path.join(MODEL_DIR, "best_model.joblib")

BEST_MODEL_INFO = os.path.join(MODEL_DIR, "best_model_info.json")

CONFUSION_FIG = os.path.join(MODEL_DIR, "confusion_matrix.png")

# MODELS

MODEL_LABELS = {
    "logistic_regression": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest"
}

# BALANCING SETTINGS
# share = flaky share of the TRAINING folds after balancing.
# Undersampling removes non-flaky rows; SMOTE creates synthetic
# flaky rows. Both run inside the pipeline, so they only touch the
# training folds. Test folds always keep the real ~3.65% flaky share.

SAMPLINGS = {
    "none": {"method": None, "share": None},
    "under_10": {"method": "undersample", "share": 0.10},
    "under_25": {"method": "undersample", "share": 0.25},
    "under_50": {"method": "undersample", "share": 0.50},
    "smote_50": {"method": "smote", "share": 0.50},
}

SAMPLING_LABELS = {
    "none": "No balancing",
    "under_10": "Undersample to 10%",
    "under_25": "Undersample to 25%",
    "under_50": "Undersample to 50%",
    "smote_50": "SMOTE to 50%"
}

# 1. CHECK INPUTS

def require_inputs(paths):
    """Stop with a clear message if any input file is missing."""

    missing = [p for p in paths if not os.path.exists(p)]

    if missing:

        for path in missing:
            print(f"ERROR: Required input not found: {path}")

        print("Run the earlier pipeline steps first (see documentation/TEAM_SPLIT.md).")

        sys.exit(1)

# 2. LOAD MODEL DATA

def load_model_data():
    """Load the final dataset and return it with X (23 model features) and y."""

    require_inputs([FINAL_DATASET, COLUMN_ROLES])

    df = pd.read_csv(FINAL_DATASET)
    roles = pd.read_csv(COLUMN_ROLES)

    features = list(roles.loc[roles["role"] == "model_feature", "column"])

    leaked = [col for col in features if col in LEAKAGE_COLS]

    assert not leaked, f"Leakage columns in model features: {leaked}"
    assert len(features) == 23, f"Expected 23 model features, got {len(features)}"

    X = df[features].astype(float)
    y = df[TARGET_COL].astype(int)

    return df, X, y, features

# 3. BUILD PIPELINE

def sampling_strategy(share):
    """Convert a target flaky share into imblearn's minority/majority ratio."""

    return share / (1 - share)


def build_classifier(model_name):
    """Return an untrained classifier with a fixed random seed."""

    if model_name == "logistic_regression":
        return LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)

    if model_name == "decision_tree":
        return DecisionTreeClassifier(
            max_depth=10,
            min_samples_leaf=5,
            random_state=RANDOM_STATE
        )

    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=1,
            n_jobs=-1,
            random_state=RANDOM_STATE
        )

    raise ValueError(f"Unknown model: {model_name}")


def build_pipeline(model_name, sampling):
    """Pipeline: [scaler for LR] -> [sampler] -> classifier.

    imblearn applies the sampler only in fit(), never in predict(),
    so test data is never resampled.
    """

    steps = []

    if model_name == "logistic_regression":
        steps.append(("scaler", StandardScaler()))

    method = SAMPLINGS[sampling]["method"]
    share = SAMPLINGS[sampling]["share"]

    if method == "undersample":
        steps.append(("sampler", RandomUnderSampler(
            sampling_strategy=sampling_strategy(share),
            random_state=RANDOM_STATE
        )))

    elif method == "smote":
        steps.append(("sampler", SMOTE(
            sampling_strategy=sampling_strategy(share),
            random_state=RANDOM_STATE
        )))

    steps.append(("clf", build_classifier(model_name)))

    return Pipeline(steps)

# 4. FOLD METRICS

def score_predictions(y_true, prob, threshold=THRESHOLD):
    """Precision, recall, F1, ROC-AUC and PR-AUC for one set of predictions."""

    pred = (prob >= threshold).astype(int)

    has_both = len(np.unique(y_true)) == 2

    return {
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, prob) if has_both else np.nan,
        "pr_auc": average_precision_score(y_true, prob) if has_both else np.nan
    }

# 5. CROSS-VALIDATION

def run_cross_validation(X, y):
    """Stratified 5-fold CV for every model x balancing setting."""

    print("\n========== STRATIFIED 5-FOLD CROSS-VALIDATION ==========")

    skf = StratifiedKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    folds = list(skf.split(X, y))

    for i, (train_idx, test_idx) in enumerate(folds, start=1):
        print(
            f"Fold {i}: train {len(train_idx)} rows, test {len(test_idx)} rows, "
            f"test flaky share {y.iloc[test_idx].mean() * 100:.2f}%"
        )

    rows = []
    oof = {}

    for model_name in MODEL_LABELS:

        for sampling in SAMPLINGS:

            key = f"{model_name}__{sampling}"

            oof_prob = np.zeros(len(y))
            fold_scores = []
            train_shares = []

            for train_idx, test_idx in folds:

                pipe = build_pipeline(model_name, sampling)

                X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pipe.fit(X_train, y_train)

                train_shares.append(train_share_after_sampling(
                    pipe, X_train, y_train
                ))

                prob = pipe.predict_proba(X.iloc[test_idx])[:, 1]

                oof_prob[test_idx] = prob

                fold_scores.append(score_predictions(y.iloc[test_idx], prob))

            oof[key] = oof_prob

            fold_scores = pd.DataFrame(fold_scores)

            row = {
                "model": model_name,
                "model_label": MODEL_LABELS[model_name],
                "sampling": sampling,
                "sampling_label": SAMPLING_LABELS[sampling],
                "train_flaky_share_pct": round(np.mean(train_shares) * 100, 2),
                "test_flaky_share_pct": round(y.mean() * 100, 2)
            }

            for metric in fold_scores.columns:
                row[f"{metric}_mean"] = fold_scores[metric].mean()
                row[f"{metric}_std"] = fold_scores[metric].std()

            rows.append(row)

            print(
                f"{MODEL_LABELS[model_name]:<20} {SAMPLING_LABELS[sampling]:<20} "
                f"P={row['precision_mean']:.3f} R={row['recall_mean']:.3f} "
                f"F1={row['f1_mean']:.3f} ROC-AUC={row['roc_auc_mean']:.3f} "
                f"PR-AUC={row['pr_auc_mean']:.3f}"
            )

    metrics = pd.DataFrame(rows)

    return metrics, oof


def train_share_after_sampling(pipe, X_train, y_train):
    """Flaky share the classifier actually saw in this training fold."""

    if "sampler" not in pipe.named_steps:
        return y_train.mean()

    data = X_train

    if "scaler" in pipe.named_steps:
        data = pipe.named_steps["scaler"].transform(X_train)

    _, y_res = pipe.named_steps["sampler"].fit_resample(data, y_train)

    return float(np.mean(y_res))

# 6. PICK BEST MODEL

def pick_best(metrics):
    """Best model = highest mean F1; PR-AUC breaks ties."""

    ranked = metrics.sort_values(
        ["f1_mean", "pr_auc_mean"],
        ascending=False
    )

    best = ranked.iloc[0]

    metrics["is_best"] = (
        (metrics["model"] == best["model"])
        & (metrics["sampling"] == best["sampling"])
    )

    print("\n========== BEST MODEL ==========")
    print("Model:", best["model_label"])
    print("Balancing:", best["sampling_label"])
    print(f"Mean F1: {best['f1_mean']:.3f}  PR-AUC: {best['pr_auc_mean']:.3f}  "
          f"ROC-AUC: {best['roc_auc_mean']:.3f}")

    return metrics, best

# 7. SAVE METRICS AND OUT-OF-FOLD PREDICTIONS

def save_metrics_and_predictions(df, y, metrics, oof, best):
    """Save metrics.csv and predictions.csv (out-of-fold probabilities)."""

    os.makedirs(MODEL_DIR, exist_ok=True)

    metrics.round(4).to_csv(METRICS_FILE, index=False)

    preds = pd.DataFrame({
        "test_name": df["test_name"],
        "project": df["project"],
        TARGET_COL: y
    })

    for key, prob in oof.items():
        preds[f"prob_{key}"] = prob.round(6)

    best_key = f"{best['model']}__{best['sampling']}"

    preds["prob_best"] = oof[best_key].round(6)

    preds.to_csv(PREDICTIONS_FILE, index=False)

    print("\nMetrics saved to:", METRICS_FILE)
    print("Out-of-fold predictions saved to:", PREDICTIONS_FILE)
    print("prob_best column =", f"prob_{best_key}")

# 8. REFIT AND SAVE BEST MODEL

def save_best_model(X, y, features, best):
    """Refit the best pipeline on all data and save it with its settings."""

    pipe = build_pipeline(best["model"], best["sampling"])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe.fit(X, y)

    settings = SAMPLINGS[best["sampling"]]

    info = {
        "model": best["model"],
        "model_label": best["model_label"],
        "sampling": best["sampling"],
        "sampling_label": best["sampling_label"],
        "sampling_method": settings["method"],
        "train_flaky_share": settings["share"],
        "sampling_strategy": (
            sampling_strategy(settings["share"]) if settings["share"] else None
        ),
        "threshold": THRESHOLD,
        "random_state": RANDOM_STATE,
        "features": features,
        "cv_f1_mean": round(float(best["f1_mean"]), 4),
        "cv_pr_auc_mean": round(float(best["pr_auc_mean"]), 4),
        "cv_roc_auc_mean": round(float(best["roc_auc_mean"]), 4)
    }

    joblib.dump({"pipeline": pipe, **info}, BEST_MODEL_FILE, compress=3)

    with open(BEST_MODEL_INFO, "w") as f:
        json.dump(info, f, indent=2)

    print("\nBest model saved to:", BEST_MODEL_FILE)
    print("Best model settings saved to:", BEST_MODEL_INFO)

# 9. CONFUSION MATRIX

def plot_confusion_matrix(y, prob, best):
    """Confusion matrix of the best model's out-of-fold predictions."""

    pred = (prob >= THRESHOLD).astype(int)

    cm = confusion_matrix(y, pred)

    print("\n========== CONFUSION MATRIX (out-of-fold, best model) ==========")
    print(pd.DataFrame(
        cm,
        index=["actual Non-Flaky", "actual Flaky"],
        columns=["pred Non-Flaky", "pred Flaky"]
    ))

    apply_style()

    fig, ax = plt.subplots(figsize=(5.2, 4.4))

    row_pct = cm / cm.sum(axis=1, keepdims=True) * 100

    cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list(
        "blues", ["#f0efec", BLUE_RAMP[0], BLUE_RAMP[2], BLUE_RAMP[4]]
    )

    ax.imshow(row_pct, cmap=cmap, vmin=0, vmax=100)
    ax.grid(False)

    for i in range(2):
        for j in range(2):
            color = "#ffffff" if row_pct[i, j] > 55 else TEXT_PRIMARY
            ax.text(
                j, i, f"{cm[i, j]:,}\n({row_pct[i, j]:.1f}% of row)",
                ha="center", va="center", color=color, fontsize=10
            )

    ax.set_xticks([0, 1], ["Non-Flaky", "Flaky"])
    ax.set_yticks([0, 1], ["Non-Flaky", "Flaky"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title(
        f"{best['model_label']} ({best['sampling_label']})\n"
        "Out-of-fold confusion matrix, threshold 0.5",
        fontsize=11
    )

    save_figure(fig, CONFUSION_FIG)

# 10. MAIN EXECUTION

def main():

    df, X, y, features = load_model_data()

    print("\nRows:", len(df))
    print("Model features:", len(features))
    print(f"Flaky share: {y.mean() * 100:.2f}% ({int(y.sum())} flaky)")

    metrics, oof = run_cross_validation(X, y)

    metrics, best = pick_best(metrics)

    save_metrics_and_predictions(df, y, metrics, oof, best)

    save_best_model(X, y, features, best)

    plot_confusion_matrix(y, oof[f"{best['model']}__{best['sampling']}"], best)

    print("\n========== METRICS SUMMARY (mean of 5 folds) ==========")

    summary = metrics.pivot(
        index="sampling_label",
        columns="model_label",
        values="f1_mean"
    ).reindex([SAMPLING_LABELS[s] for s in SAMPLINGS])

    print("F1 by model and balancing:")
    print(summary.round(3).to_string())


if __name__ == "__main__":
    main()
