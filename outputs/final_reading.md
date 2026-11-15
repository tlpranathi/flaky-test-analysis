# Final evaluation: reading

**Pooled F1 = 0.72, unseen-project F1 = 0.12.** The best model is a Random Forest trained with SMOTE to 50%. With stratified 5-fold cross-validation (projects mixed across folds) it reaches F1 0.72 (precision 0.78, recall 0.67, ROC-AUC 0.97). When a whole project is held out and the model has never seen it (leave-one-project-out), F1 falls to 0.12 (mean per-project F1 0.05, and in 17 of 23 projects no flaky test was found), so the model relies heavily on project-specific patterns. It still ranks tests better than random inside a new project (median per-project ROC-AUC 0.74, vs 0.97 pooled), so mainly the fixed 0.5 cut-off breaks on unseen projects. The best undersampling ratio for the Random Forest is **10% flaky** (F1 0.718, essentially tied with no balancing at 0.718): heavier undersampling to 50% raises recall to 0.90 but cuts precision to 0.25, so F1 drops to 0.39. SMOTE to 50% gives the best F1 overall (0.722), +0.004 vs no balancing. Balancing matters far more for Logistic Regression, whose F1 rises from 0.12 with no balancing to 0.29 at undersample to 25% flaky.

## How to read this

- **The "no balancing" baseline uses no class weights.** The model sees the real
  3.65% flaky share and nothing tells it that missing a flaky test is costly, so
  low recall there is expected (here 0.61 for the Random Forest).
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
