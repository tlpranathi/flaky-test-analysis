# Key Influencers vs SHAP: agreement notes

- **Overlap:** 7 of the Key Influencers top 8 are also in the SHAP top 10 (Jaccard 0.64).
- **Both methods:** `execution_time_s`, `source_classes_covered`, `changed_line_churn_w100`, `covered_lines`, `changed_line_churn_w50`, `test_length`, `changed_line_churn_w10`.
- **SHAP only:** `source_lines_covered`, `changed_line_churn_w75`, `smell_fire_and_forget`.
- **Key Influencers only:** `changed_line_churn_w500`.
- **Rank agreement** on shared features (Spearman): -0.04 (1 = same order, 0 = unrelated).
- **Direction:** for 2 of 4 features the Power BI condition points the same way as SHAP (e.g. higher value -> more likely flaky).

## Why they can differ

- Key Influencers fits a simple logistic regression per feature (one feature at a time),
  so correlated features such as the 8 churn windows can all look important.
- SHAP explains the trained Random Forest, which uses all features together and
  splits the credit among correlated features.
- Key Influencers reports conditions on a bucket ("execution_time_s > 1.2"), while
  SHAP ranks the whole feature, so a feature that only matters in a narrow range
  may rank lower in SHAP.
