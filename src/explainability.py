import os
import warnings
import joblib
import numpy as np
import pandas as pd

from src.train_models import (
    RANDOM_STATE,
    BEST_MODEL_FILE,
    SAMPLINGS,
    build_pipeline,
    load_model_data
)

import shap
import matplotlib.pyplot as plt

from joblib import Parallel, delayed

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.tree import export_text

from src.config import OUTPUT_DIR
from src.plot_style import (
    apply_style,
    save_figure,
    BLUE,
    FLAKY_COLOR,
    NON_FLAKY_COLOR,
    TEXT_SECONDARY
)

# SETTINGS

SHAP_SAMPLE_SIZE = 800

N_JOBS = int(os.environ["LOKY_MAX_CPU_COUNT"])

PERMUTATION_REPEATS = 5

TOP_N = 10

DT_RULES_DEPTH = 4

# OUTPUT FILES

EXPLAIN_DIR = os.path.join(OUTPUT_DIR, "explainability")

FEATURE_IMPORTANCE = os.path.join(EXPLAIN_DIR, "feature_importance.csv")

TOP_FEATURES_FILE = os.path.join(EXPLAIN_DIR, "top10_features.csv")

LOCAL_EXPLANATION = os.path.join(EXPLAIN_DIR, "local_explanation.csv")

DT_RULES = os.path.join(EXPLAIN_DIR, "dt_rules.txt")

SHAP_BAR_FIG = os.path.join(EXPLAIN_DIR, "shap_global_importance.png")
SHAP_BEESWARM_FIG = os.path.join(EXPLAIN_DIR, "shap_beeswarm.png")
SHAP_WATERFALL_FIG = os.path.join(EXPLAIN_DIR, "shap_waterfall_flagged_test.png")
PERMUTATION_FIG = os.path.join(EXPLAIN_DIR, "permutation_importance.png")
LR_COEF_FIG = os.path.join(EXPLAIN_DIR, "lr_coefficients.png")

# 1. LOAD BEST MODEL

def load_best_model(X, y, features):
    """Load best_model.joblib, or fit a placeholder Random Forest if missing."""

    print("\n========== LOAD BEST MODEL ==========")

    if os.path.exists(BEST_MODEL_FILE):

        bundle = joblib.load(BEST_MODEL_FILE)

        if list(bundle["features"]) != list(features):
            raise ValueError("best_model.joblib was trained on a different feature list.")

        print("Loaded:", BEST_MODEL_FILE)

    else:

        print(f"WARNING: {BEST_MODEL_FILE} not found. "
              "Fitting a PLACEHOLDER Random Forest (no balancing).")
        print("Run python -m src.train_models for the real best model.")

        pipe = build_pipeline("random_forest", "none")
        pipe.fit(X, y)

        bundle = {
            "pipeline": pipe,
            "model": "random_forest",
            "model_label": "Random Forest (placeholder)",
            "sampling": "none",
            "sampling_label": "No balancing",
            "features": features
        }

    print("Model:", bundle["model_label"])
    print("Balancing:", bundle["sampling_label"])

    return bundle

# 2. SPLIT TRAIN / HOLDOUT

def split_holdout(X, y):
    """Stratified 80/20 split, used only for permutation importance."""

    return train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE
    )

# 3. SHAP VALUES

def tree_shap_chunk(clf, X_chunk):
    """TreeSHAP values for one chunk of rows (runs in a worker process)."""

    explainer = shap.TreeExplainer(clf)

    values = explainer.shap_values(X_chunk, check_additivity=False)

    return values, explainer.expected_value


def model_input(pipe, X):
    """Transform X the way the classifier sees it (scaler for LR, else raw)."""

    if "scaler" in pipe.named_steps:
        return pd.DataFrame(
            pipe.named_steps["scaler"].transform(X),
            columns=X.columns,
            index=X.index
        )

    return X


def compute_shap(bundle, X, y):
    """SHAP values for the flaky class on a stratified sample of tests."""

    print("\n========== SHAP ==========")

    pipe = bundle["pipeline"]
    clf = pipe.named_steps["clf"]

    sample_idx, _ = train_test_split(
        X.index,
        train_size=SHAP_SAMPLE_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )

    X_sample = X.loc[sample_idx]
    X_input = model_input(pipe, X_sample)

    if bundle["model"] == "logistic_regression":

        explainer = shap.LinearExplainer(clf, X_input)
        values = explainer.shap_values(X_input)
        base_value = explainer.expected_value

    else:

        # TreeSHAP on a full-depth forest is slow (~0.3 s per test), so a
        # stratified sample of 800 tests is explained and
        # split across CPU cores

        chunks = np.array_split(X_input.values, N_JOBS)

        results = Parallel(n_jobs=N_JOBS)(
            delayed(tree_shap_chunk)(clf, chunk) for chunk in chunks
        )

        values = np.vstack([r[0] for r in results])
        base_value = results[0][1]

    # Tree models return one set of values per class: keep class 1 (flaky)

    values = np.asarray(values)

    if values.ndim == 3:
        values = values[:, :, 1]

    base_value = np.ravel(base_value)[-1]

    # Raw feature values are shown on the plots, even for the scaled LR

    explanation = shap.Explanation(
        values=values,
        base_values=np.full(len(values), base_value),
        data=X_sample.values,
        feature_names=list(X.columns)
    )

    print("Explained tests:", len(X_sample),
          f"({int(y.loc[sample_idx].sum())} flaky)")

    return explanation, X_sample

# 4. SHAP CHARTS

def plot_shap_bar(shap_importance):
    """Global importance: mean |SHAP| per feature (top 15)."""

    top = shap_importance.head(15).iloc[::-1]

    fig, ax = plt.subplots(figsize=(7, 5.5))

    ax.barh(top.index, top.values, color=BLUE, height=0.6)
    ax.grid(axis="y", visible=False)

    for i, value in enumerate(top.values):
        ax.text(value, i, f" {value:.3f}", va="center",
                fontsize=8, color=TEXT_SECONDARY)

    ax.set_xlabel("Mean |SHAP value| (impact on flaky probability)")
    ax.set_title("Which features drive the flaky prediction? (SHAP, top 15)")

    save_figure(fig, SHAP_BAR_FIG)


def plot_shap_beeswarm(explanation):
    """Beeswarm: direction and spread of each feature's effect."""

    plt.figure()

    shap.plots.beeswarm(explanation, max_display=15, show=False)

    fig = plt.gcf()
    fig.set_size_inches(8, 6)

    plt.title("SHAP beeswarm: red = high feature value, blue = low",
              loc="left", fontsize=11)

    save_figure(fig, SHAP_BEESWARM_FIG)


def plot_shap_waterfall(bundle, explanation, X_sample, df, y):
    """Local example: why the highest-scored flaky test was flagged."""

    prob = bundle["pipeline"].predict_proba(X_sample)[:, 1]

    flaky_pos = np.where(y.loc[X_sample.index].values == 1)[0]

    pos = flaky_pos[np.argmax(prob[flaky_pos])]

    row_index = X_sample.index[pos]

    test_name = df.loc[row_index, "test_name"]
    project = df.loc[row_index, "project"]

    print("\nLocal example (highest-scored flaky test in the sample):")
    print("  Test:", test_name)
    print("  Project:", project)
    print(f"  Predicted flaky probability: {prob[pos]:.3f}")

    local = pd.DataFrame({
        "feature": explanation.feature_names,
        "value": X_sample.iloc[pos].values,
        "shap_value": explanation.values[pos]
    }).sort_values("shap_value", key=abs, ascending=False)

    local.insert(0, "test_name", test_name)
    local.insert(1, "project", project)
    local.insert(2, "predicted_probability", round(float(prob[pos]), 4))

    local.round(4).to_csv(LOCAL_EXPLANATION, index=False)

    print("  Top reasons:")
    for _, r in local.head(5).iterrows():
        direction = "raises" if r["shap_value"] > 0 else "lowers"
        print(f"    {r['feature']} = {r['value']:g} {direction} the score "
              f"by {abs(r['shap_value']):.3f}")

    plt.figure()

    shap.plots.waterfall(explanation[pos], max_display=12, show=False)

    fig = plt.gcf()
    fig.set_size_inches(8, 6)

    short_name = test_name if len(test_name) < 70 else "..." + test_name[-67:]

    plt.title(f"Why was this test flagged? ({project})\n{short_name}",
              loc="left", fontsize=9)

    save_figure(fig, SHAP_WATERFALL_FIG)

    print("Local explanation saved to:", LOCAL_EXPLANATION)

# 5. PERMUTATION IMPORTANCE

def compute_permutation_importance(bundle, X_train, X_hold, y_train, y_hold):
    """Fit a fresh copy of the best pipeline on 80%, permute on the 20% holdout.

    A separate copy is used because the saved model was refit on ALL
    data, so the holdout would be part of its training data.
    """

    print("\n========== PERMUTATION IMPORTANCE ==========")

    pipe = clone(bundle["pipeline"])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe.fit(X_train, y_train)

    result = permutation_importance(
        pipe,
        X_hold,
        y_hold,
        scoring="average_precision",
        n_repeats=PERMUTATION_REPEATS,
        random_state=RANDOM_STATE,
        n_jobs=1
    )

    perm = pd.DataFrame({
        "permutation_importance": result.importances_mean,
        "permutation_std": result.importances_std
    }, index=X_hold.columns)

    print("Fitted on", len(X_train), "rows; scored on", len(X_hold),
          "held-out rows (PR-AUC drop when a feature is shuffled)")

    return perm


def plot_permutation(perm):
    """Bar chart of the PR-AUC drop per feature (top 15)."""

    top = perm.sort_values("permutation_importance").tail(15)

    fig, ax = plt.subplots(figsize=(7, 5.5))

    ax.barh(top.index, top["permutation_importance"],
            xerr=top["permutation_std"], color=BLUE, height=0.6,
            error_kw={"ecolor": TEXT_SECONDARY, "elinewidth": 1})
    ax.grid(axis="y", visible=False)

    ax.set_xlabel("Drop in PR-AUC when the feature is shuffled (20% holdout)")
    ax.set_title("Permutation importance (top 15)")

    save_figure(fig, PERMUTATION_FIG)

# 6. DECISION TREE AND LOGISTIC REGRESSION

def fit_dt_and_lr(bundle, X, y):
    """Refit DT and LR on all data with the best model's balancing setting."""

    print("\n========== DECISION TREE + LOGISTIC REGRESSION ==========")

    sampling = bundle["sampling"] if bundle["sampling"] in SAMPLINGS else "none"

    print("Balancing used for both:", sampling)

    dt = build_pipeline("decision_tree", sampling)
    lr = build_pipeline("logistic_regression", sampling)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dt.fit(X, y)
        lr.fit(X, y)

    return dt, lr


def save_dt_rules(dt, features):
    """Write the top levels of the Decision Tree as readable if/else rules."""

    rules = export_text(
        dt.named_steps["clf"],
        feature_names=list(features),
        max_depth=DT_RULES_DEPTH,
        show_weights=True,
        decimals=3
    )

    header = (
        f"Decision Tree rules (first {DT_RULES_DEPTH} levels of a depth-10 tree).\n"
        "class: 1 = flaky, 0 = non-flaky. weights = training rows per class\n"
        "after balancing. '...' marks deeper branches that were cut off.\n\n"
    )

    with open(DT_RULES, "w") as f:
        f.write(header + rules)

    print("Decision Tree rules saved to:", DT_RULES)
    print("\nFirst rules:")
    print("\n".join(rules.splitlines()[:12]))


def lr_coefficients(lr, features):
    """Standardised LR coefficients (per 1 standard deviation of the feature)."""

    coef = lr.named_steps["clf"].coef_[0]

    return pd.Series(coef, index=features)


def plot_lr_coefficients(coef):
    """Diverging bars: positive = higher flaky risk, negative = lower."""

    top = coef.reindex(coef.abs().sort_values().index).tail(15)

    colors = [FLAKY_COLOR if c > 0 else NON_FLAKY_COLOR for c in top]

    fig, ax = plt.subplots(figsize=(7, 5.5))

    ax.barh(top.index, top.values, color=colors, height=0.6)
    ax.axvline(0, color=TEXT_SECONDARY, linewidth=1)
    ax.grid(axis="y", visible=False)

    ax.set_xlabel("Standardised coefficient (log-odds per 1 SD)")
    ax.set_title("Logistic Regression coefficients (top 15 by size)\n"
                 "orange = raises flaky odds, blue = lowers them", fontsize=11)

    save_figure(fig, LR_COEF_FIG)

# 7. COMBINE IMPORTANCES

def shap_direction(explanation, features):
    """Does a HIGHER feature value push the flaky score up or down?

    Spearman correlation between each feature's value and its SHAP value
    across the explained tests.
    """

    values = pd.DataFrame(explanation.values, columns=features)
    data = pd.DataFrame(explanation.data, columns=features)

    corr = pd.Series({
        col: values[col].corr(data[col], method="spearman")
        for col in features
    }).fillna(0)

    direction = np.select(
        [corr > 0.1, corr < -0.1],
        ["higher -> more flaky", "higher -> less flaky"],
        default="mixed"
    )

    return corr, pd.Series(direction, index=features)


def build_importance_table(shap_importance, direction, perm, dt, coef):
    """One table with each method's score and rank for all 23 features."""

    corr, label = direction

    table = pd.DataFrame({
        "shap_mean_abs": shap_importance,
        "shap_direction": label,
        "shap_value_corr": corr,
        "permutation_importance": perm["permutation_importance"],
        "permutation_std": perm["permutation_std"],
        "dt_importance": pd.Series(
            dt.named_steps["clf"].feature_importances_,
            index=coef.index
        ),
        "lr_coef": coef,
        "lr_abs_coef": coef.abs()
    })

    for score, rank in [
        ("shap_mean_abs", "shap_rank"),
        ("permutation_importance", "permutation_rank"),
        ("dt_importance", "dt_rank"),
        ("lr_abs_coef", "lr_rank")
    ]:
        table[rank] = table[score].rank(ascending=False, method="min").astype(int)

    table = table.sort_values("shap_rank")
    table.index.name = "feature"

    return table.reset_index()

# 8. SAVE OUTPUTS

def save_importance(table):
    """Save feature_importance.csv and the SHAP top-10 list."""

    table.round(5).to_csv(FEATURE_IMPORTANCE, index=False)

    top = table.head(TOP_N)[["shap_rank", "feature", "shap_mean_abs"]]
    top.round(5).to_csv(TOP_FEATURES_FILE, index=False)

    print("\nFeature importance saved to:", FEATURE_IMPORTANCE)
    print("Top 10 saved to:", TOP_FEATURES_FILE)

# 9. MAIN EXECUTION

def main():

    df, X, y, features = load_model_data()

    os.makedirs(EXPLAIN_DIR, exist_ok=True)

    apply_style()

    bundle = load_best_model(X, y, features)

    explanation, X_sample = compute_shap(bundle, X, y)

    shap_importance = pd.Series(
        np.abs(explanation.values).mean(axis=0),
        index=features
    ).sort_values(ascending=False)

    plot_shap_bar(shap_importance)
    plot_shap_beeswarm(explanation)
    plot_shap_waterfall(bundle, explanation, X_sample, df, y)

    X_train, X_hold, y_train, y_hold = split_holdout(X, y)

    perm = compute_permutation_importance(bundle, X_train, X_hold, y_train, y_hold)
    plot_permutation(perm)

    dt, lr = fit_dt_and_lr(bundle, X, y)

    save_dt_rules(dt, features)

    coef = lr_coefficients(lr, features)
    plot_lr_coefficients(coef)

    direction = shap_direction(explanation, features)

    table = build_importance_table(shap_importance, direction, perm, dt, coef)

    save_importance(table)

    print("\n========== TOP 10 FEATURES (ranked by SHAP) ==========")

    print(table.head(TOP_N)[[
        "feature", "shap_rank", "permutation_rank", "dt_rank", "lr_rank",
        "shap_direction"
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
