# Stage 3 slides: ML classification and visualization (outline)

**Owner:** Person 4 (assembles the deck)  ·  12 slides, about 12-15 minutes
Numbers below come from the current runs. If anything is rerun, refresh them from the files
named on each slide.

Each person presents their own part: P1 slides 3-6, P2 slides 7-9, P3 slide 10,
P4 slides 11-12.

---

### 1. Title
- "Predicting flaky tests from test features: FlakeFlagger dataset"
- Team names and roles (P1 models · P2 explainability · P3 Power BI · P4 Key Influencers and ranking)

### 2. Recap: data and the problem
- 22,236 tests from 24 Java projects, **811 flaky (3.65%)**, so very imbalanced
- 23 model features: execution time, coverage (3), test size (2), third-party libraries,
  8 test smells, 8 code-churn windows
- **6 leakage columns excluded** (rerun outcomes such as NumFailingRuns, fail_rate_pct):
  they record what happened on reruns, so using them would be cheating
- Visual: a bar chart of flaky vs non-flaky counts, or a single "3.65%" figure

### 3. Method (P1)
- 3 models: Logistic Regression, Decision Tree, Random Forest
- Stratified 5-fold CV: every test fold keeps the real 3.65% flaky share
- Balancing applied **only to the training folds** (inside an imblearn pipeline):
  none, undersample to 10% / 25% / 50% flaky, and SMOTE to 50%
- Metrics: precision, recall, F1, ROC-AUC, PR-AUC. **Not accuracy**: predicting "never flaky"
  already scores 96.35% accuracy
- Diagram: train fold → balance → fit → test on an untouched fold

### 4. Results table (P1)
- Source: `outputs/model/metrics.csv` (or the Power BI Model comparison page)
- Random Forest is clearly best: F1 0.72, ROC-AUC 0.97, PR-AUC 0.77
- Decision Tree F1 0.59; Logistic Regression only 0.12-0.29
- Visual: clustered columns of F1 by balancing setting, one colour per model

### 5. What balancing does (P1)
- "No balancing" uses **no class weights**, so its low recall is expected (LR recall 0.07)
- Undersampling to 50%: RF recall 0.61 → 0.90, but precision 0.88 → 0.25, so F1 falls to 0.39
- For RF: 10% undersampling (F1 0.718) ≈ no balancing (0.718); SMOTE 50% is best (0.722)
- Balancing helps the weak LR far more: F1 0.12 → 0.29 at 25%
- Message: balancing trades precision for recall; the best setting depends on the model

### 6. Best model: confusion matrix (P1)
- `outputs/model/confusion_matrix.png`, Random Forest + SMOTE 50%, out-of-fold
- 543 of 811 flaky tests caught (67%), with 150 false alarms among 21,425 non-flaky tests (0.7%)

### 7. Which features matter? SHAP (P2)
- `outputs/explainability/shap_global_importance.png` and `shap_beeswarm.png`
- Top 4: execution_time_s, source_classes_covered, source_lines_covered, changed_line_churn_w100
- Beeswarm: high values of these features (red dots) push the flaky score up

### 8. Why was this test flagged? (P2)
- `outputs/explainability/shap_waterfall_flagged_test.png`
- Example: alluxio `RemoteBlockInStreamTest.readTest5`, predicted probability 1.00
- The biggest pushes come from classes covered (156), churn w100 (3) and lines covered (382)
- Message: the model's decision can be explained for each individual test

### 9. Do other methods agree? (P2)
- Permutation importance, Decision Tree rules (`dt_rules.txt`), LR coefficients
- execution_time_s is **#1 in all four methods**
- SHAP vs permutation rank correlation 0.79; DT vs LR only 0.32
- Show the first 3-4 lines of the tree rules: the first split is on execution time

### 10. Power BI dashboard (P3)
- Screenshots: Overview (cards), Flaky vs non-flaky page, Model comparison page
- Leakage columns removed from the Power BI table
- Flaky % by execution time bucket: 1.5% (<0.1s) → 21.0% (>10s)

### 11. Key Influencers (P4)
- Screenshot of the visual: Analyze = flaky_label "Flaky", Explain by = the 23 features
- Top influencers and their "likelihood increases by Nx" values (from `key_influencers_top.csv`)
- Agreement with SHAP: overlap of the two top-10 lists (from `influencers_agreement_notes.md`)
- Why they can differ: one-feature-at-a-time vs the whole model; correlated churn windows

### 12. Takeaways and next stage
- A Random Forest on 23 static features detects flaky tests well (F1 0.72) within known projects
- Slow tests with broad coverage and recent churn are the riskiest
- Stage 4 preview: leave-one-project-out testing, a merged feature ranking, flaky patterns,
  and a review priority list
