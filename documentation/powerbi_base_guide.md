# Power BI base report: click-by-click guide

**Owner:** Person 3  ·  **Input files:** made by `python -m src.powerbi_prep`  ·  **Result:** `flaky_tests.pbix`

This guide builds the base `.pbix`: data import, Power Query clean-up, DAX measures, and
the first visuals comparing flaky and non-flaky tests, plus a model comparison page.

> **No leakage columns.** `flaky_tests_powerbi.csv` already has the 6 rerun-outcome columns
> removed (`NumFailingRuns`, `NumPassingRuns`, `FirstFailingRunID`, `FirstPassingRunID`,
> `UniqueFailingExceptionTypes`, `fail_rate_pct`). They describe what happened when the test
> was rerun, so they give away the answer. Do **not** import `cleaned_dataset.csv` instead.

---

## 0. Before you start

1. From the repo root, run:
   ```
   python -m src.train_models      (Person 1, only if outputs/model/metrics.csv is missing)
   python -m src.powerbi_prep
   ```
2. Check that `outputs/powerbi/` contains:

| File | What it is | Rows |
|---|---|---|
| `flaky_tests_powerbi.csv` | one row per test: identifiers, label, 23 features, buckets, helper columns | 22,236 |
| `model_comparison.csv` | one row per model x balancing setting (metrics as columns) | 15 |
| `model_comparison_long.csv` | the same metrics, one row per metric (for a metric slicer) | 75 |

3. Open **Power BI Desktop**, then **File > Options and settings > Options > Regional Settings**.
   If your locale uses a comma as the decimal separator, set *Locale for import* to
   **English (United States)**. The CSVs use `.` for decimals.

---

## 1. Import the data

1. **Home > Get data > Text/CSV**, then pick `outputs/powerbi/flaky_tests_powerbi.csv` > **Open**.
2. In the preview window, check *Delimiter = Comma* and *Data type detection = Based on entire dataset*.
3. Click **Transform Data** (not Load) to open Power Query.
4. Repeat steps 1-3 for `model_comparison.csv` and `model_comparison_long.csv`.

## 2. Power Query steps

Do these in the Power Query Editor. Every step appears under *Applied Steps* on the right.

### 2.1 Query `flaky_tests_powerbi` → rename to **Tests**
1. Right-click the query on the left, choose **Rename**, and type `Tests`.
2. **Check data types** (click the icon left of each column header):
   - `flaky`, `smell_*`, `smell_count`, `*_order` → **Whole Number**
   - `execution_time_s`, `changed_line_churn_*`, `test_length`, `num_asserts`, `covered_lines`,
     `source_lines_covered`, `source_classes_covered`, `third_party_libs` → **Decimal Number**
   - `test_name`, `project`, `testClassName`, `testMethodName`, `flaky_label`, `*_bucket`,
     `has_any_smell` → **Text**
3. **Add a churn group** (used in visual 4):
   **Add Column > Conditional Column**, name it `churn_w100_group`:
   - if `changed_line_churn_w100` *equals* 0 → `0`
   - else if `changed_line_churn_w100` *equals* 1 → `1`
   - else if `changed_line_churn_w100` *equals* 2 → `2`
   - else if `changed_line_churn_w100` *is less than or equal to* 5 → `3-5`
   - else → `>5`

   Then add a second conditional column, `churn_w100_group_order`, with the outputs 0-4 in the
   same order, and set its type to Whole Number.
4. Check the row count in the bottom-left status bar: it should say **22,236 rows**.

### 2.2 New query **Smells** (unpivot the 8 smell columns)
1. Right-click **Tests** and choose **Reference**. Rename the new query to `Smells`.
2. Ctrl+click `test_name`, `project`, `flaky`, `flaky_label` **and** the 8 `smell_*` columns
   (12 columns in total), then right-click a selected header and choose **Remove Other Columns**.
3. Select the 8 `smell_*` columns, then **Transform > Unpivot Columns**. Rename `Attribute` to
   `smell` and `Value` to `present`.
4. Select `smell`, then **Transform > Replace Values**: replace `smell_` with nothing.
   Replace `_` with a space too.
5. The result is one row per test x smell (177,888 rows). Power Query builds it, so the repo
   doesn't need a separate 23 MB file.

### 2.3 Queries `model_comparison` and `model_comparison_long`
1. Rename them to **ModelComparison** and **ModelMetricsLong**.
2. Set the types:
   - `precision`, `recall`, `f1`, `roc_auc`, `pr_auc` and `value` → **Decimal Number**
   - `sampling_order` → **Whole Number**
   - `is_best` → **True/False**
3. **Home > Close & Apply.**

## 3. Model view and sort orders

1. Open the **Model** view (third icon on the left). **Don't create relationships.** Every table
   carries its own label columns, and `test_name` is not guaranteed unique across projects.
2. Fix the text sort order (otherwise `>10s` sorts before `0.1-1s`). In the **Data** view, for
   each column below, select it and choose **Column tools > Sort by column**:

| Table | Column | Sort by column |
|---|---|---|
| Tests | `execution_time_bucket` | `execution_time_bucket_order` |
| Tests | `third_party_libs_bucket` | `third_party_libs_bucket_order` |
| Tests | `source_lines_covered_bucket` | `source_lines_covered_bucket_order` |
| Tests | `churn_w100_group` | `churn_w100_group_order` |
| ModelComparison | `sampling_label` | `sampling_order` |
| ModelMetricsLong | `sampling_label` | `sampling_order` |

3. Select the `*_order` columns and choose **Hide in report view** to keep the field list tidy.

## 4. DAX measures

Select the **Tests** table, then choose **Table tools > New measure** for each measure.
Set the format in **Measure tools** (% for the percentage measures).

```DAX
Total Tests = COUNTROWS ( Tests )

Flaky Tests = CALCULATE ( COUNTROWS ( Tests ), Tests[flaky] = 1 )

Non-Flaky Tests = CALCULATE ( COUNTROWS ( Tests ), Tests[flaky] = 0 )

Flaky % = DIVIDE ( [Flaky Tests], [Total Tests] )

Overall Flaky % = CALCULATE ( [Flaky %], ALL ( Tests ) )

Flaky % vs Overall = DIVIDE ( [Flaky %], [Overall Flaky %] )            -- format: 0.0"x"

Share of All Flaky Tests = DIVIDE ( [Flaky Tests], CALCULATE ( [Flaky Tests], ALL ( Tests ) ) )

Avg Execution Time (s) = AVERAGE ( Tests[execution_time_s] )

Median Execution Time (s) = MEDIAN ( Tests[execution_time_s] )

Avg Third-Party Libs = AVERAGE ( Tests[third_party_libs] )

Avg Source Lines Covered = AVERAGE ( Tests[source_lines_covered] )

Avg Source Classes Covered = AVERAGE ( Tests[source_classes_covered] )

Avg Churn (w100) = AVERAGE ( Tests[changed_line_churn_w100] )

Avg Smell Count = AVERAGE ( Tests[smell_count] )

Projects = DISTINCTCOUNT ( Tests[project] )
```

On the **Smells** table:

```DAX
Smell Flaky % = DIVIDE ( CALCULATE ( COUNTROWS ( Smells ), Smells[flaky] = 1 ), COUNTROWS ( Smells ) )
```

On the **ModelComparison** table:

```DAX
Best F1 = MAX ( ModelComparison[f1] )

Best Model = CALCULATE ( SELECTEDVALUE ( ModelComparison[model_label] ), ModelComparison[is_best] = TRUE () )
```

## 5. Theme colours (use the same everywhere)

**View > Themes > Customize current theme > Name and colors**: set *Color 1* = `#2A78D6` (blue)
and *Color 2* = `#EB6834` (orange). In every visual split by `flaky_label`, use
**Format > Columns/Bars > Colors**:
- **Flaky = orange `#EB6834`**
- **Non-Flaky = blue `#2A78D6`**

The Python charts use the same colours.

## 6. Visuals

### Page 1: "Overview"
1. **Four cards** (Visualizations > Card): `Total Tests`, `Flaky Tests`, `Flaky %`, `Projects`.
   Expected values: 22,236 · 811 · 3.65% · 24.
2. **Slicer**: field `project` (style: Dropdown). It lets the viewer filter the whole page.
3. **Clustered bar chart, "Flaky % by project"**:
   Y-axis = `project`, X-axis = `Flaky %`. Sort by `Flaky %` descending (... > Sort axis).

### Page 2: "Flaky vs non-flaky"
Each visual compares the two groups.

1. **Execution time**: *Clustered column chart*. X-axis = `execution_time_bucket`,
   Y-axis = `Flaky %`. Add a constant line under **Analytics > Constant line** at 0.0365, labelled
   "All tests".
2. **Third-party libraries**: same as 1, but X-axis = `third_party_libs_bucket`.
3. **Coverage**: same as 1, but X-axis = `source_lines_covered_bucket`.
   Note: the `0` bucket has only 22 tests. Mention this if you show it.
4. **Churn**: same as 1, but X-axis = `churn_w100_group`.
5. **Smells**: *Clustered bar chart* on the **Smells** table. Y-axis = `smell`,
   Legend = `present`, X-axis = `Smell Flaky %`.
6. **Averages table**: *Matrix*. Rows = `flaky_label`. Values = `Avg Execution Time (s)`,
   `Median Execution Time (s)`, `Avg Third-Party Libs`, `Avg Source Lines Covered`,
   `Avg Churn (w100)`, `Avg Smell Count`. This is the side-by-side "what does a flaky test look
   like" summary.

### Page 3: "Model comparison"
1. **Clustered column chart**: X-axis = `sampling_label`, Legend = `model_label`, Y-axis = `f1`
   (set to *Don't summarize* or *Max*). It shows how each balancing setting changes F1.
2. **Slicer** on `ModelMetricsLong[metric]` (single select) plus a second clustered column chart
   from **ModelMetricsLong**: X-axis = `sampling_label`, Legend = `model_label`,
   Y-axis = `value`. The viewer can then switch between precision, recall, F1, ROC-AUC and PR-AUC.
3. **Card**: `Best Model`, and a card `Best F1`.
4. **Text box** (Insert > Text box), pasted from `outputs/final_reading.md`: "The no-balancing
   baseline uses no class weights, so its low recall is expected. Balancing is applied only to
   training folds; test folds keep the real 3.65% flaky share."

## 7. Save

**File > Save as** → `powerbi/flaky_tests.pbix` (or wherever the team keeps it). Person 4 adds
the Key Influencers page (`documentation/key_influencers_guide.md`). Person 3 finishes the report
with `documentation/powerbi_final_guide.md`.

## Checklist

- [ ] Row count 22,236, Flaky % = 3.65%
- [ ] None of the 6 leakage columns appear in the field list
- [ ] Buckets are in the right order (`<0.1s, 0.1-1s, 1-10s, >10s`)
- [ ] Flaky = orange and Non-Flaky = blue on every visual
- [ ] The model comparison shows 3 models x 5 balancing settings
