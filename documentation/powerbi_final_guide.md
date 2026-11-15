# Power BI final report: click-by-click guide

**Owner:** Person 3  ·  **Needs:** the base `.pbix` from `documentation/powerbi_base_guide.md`,
plus `python -m src.flaky_patterns` (writes `outputs/patterns/patterns.csv` and 5 charts)
·  **Result:** the finished `flaky_tests.pbix`

This adds a **"What flaky tests look like"** page with **5 visuals, one per pattern** from
`outputs/patterns/patterns.csv`, so the Power BI report and the Python charts tell the same story.

> **Leakage check:** build everything from the **Tests** and **Smells** tables (from
> `flaky_tests_powerbi.csv`). The 6 rerun-outcome columns (`NumFailingRuns`, `NumPassingRuns`,
> `FirstFailingRunID`, `FirstPassingRunID`, `UniqueFailingExceptionTypes`, `fail_rate_pct`) are
> not in that file and must not be added back.

---

## 0. Refresh the data

1. Rerun `python -m src.powerbi_prep`, then in Power BI click **Home > Refresh**.
2. Open `outputs/patterns/patterns.csv` and keep it next to you. Each visual's title and
   subtitle text come from it. The numbers below are the current values, so update them if
   the data changes.

## 1. Create the page

1. Add a new page and rename it **What flaky tests look like**.
2. **View > Page view > Fit to page**. Use a 3-over-2 grid layout.
3. Add a **text box** across the top: "Flaky tests are slow, broad, churn-heavy and
   concentrated in a few projects. Overall flaky share: 3.65% (811 of 22,236 tests)."

For every visual below:
- **Format > General > Title**: on, using the title text given below.
- **Analytics > Constant line** (on column/bar charts): value `0.0365`, dashed, grey `#898781`,
  data label "All tests 3.65%". This line is the reference the eye compares against.
- **Format > Data labels**: on, display units **None**, 1 decimal.
- Colour: one series in blue `#2A78D6`. Where a visual splits by Flaky / Non-Flaky,
  Flaky = orange `#EB6834`.

## 2. The 5 visuals

### P1. Execution time
**Pattern:** slow tests are much more likely to be flaky: 21.0% of tests taking >10s are flaky,
vs 1.5% of tests taking <0.1s (14x).

1. **Clustered column chart**.
2. X-axis = `Tests[execution_time_bucket]` (sorted by `execution_time_bucket_order`).
3. Y-axis = `[Flaky %]`.
4. Tooltips: add `[Total Tests]` and `[Flaky Tests]` so hovering shows the counts.
5. Title: **"Flaky share rises with execution time"**.

### P2. Coverage and libraries
**Pattern:** tests that touch more of the project and use more libraries are riskier: 13.7% of
tests covering 101-500 source lines are flaky (vs 2.1% for 26-100), and 12.6% of tests using
>10 third-party libraries are flaky (vs 2.2% for 1-2).

1. **Two clustered column charts side by side**, both Y-axis = `[Flaky %]`.
   - Left: X-axis = `source_lines_covered_bucket`.
   - Right: X-axis = `third_party_libs_bucket`.
2. Select both and set the same Y-axis range: **Format > Y-axis > Range > Maximum = 0.16**,
   so the bars can be compared directly.
3. Add a footnote text box: "Coverage bucket '0' has only 22 tests."
4. Title: **"More code covered and more libraries → more flaky"**.

### P3. Code churn
**Pattern:** tests covering frequently changed code are riskier: with a churn score of 3-5,
21.4% of tests are flaky, vs 1.1% at 1. The >5 group is small (209 tests) and drops back,
so the effect is not a straight line.

1. **Clustered column chart**.
2. X-axis = `churn_w100_group` (the Power Query column from the base guide, sorted by
   `churn_w100_group_order`).
3. Y-axis = `[Flaky %]`, with tooltips `[Total Tests]`.
4. Title: **"Flaky share by code churn (last 100 commits)"**.

### P4. Test smells
**Pattern:** a few smells are red flags: resource optimism (23.3% flaky), fire-and-forget
(12.1%) and mystery guest (12.0%), against a 3.6% average. Tests with 4+ smells are 14.3%
flaky vs 1.9% with exactly one. Indirect testing goes the other way.

1. **Clustered bar chart** on the **Smells** table.
2. Y-axis = `smell`. Legend = `present` (rename the legend items to "Smell present" /
   "Smell absent" under **Format > Legend**). X-axis = `[Smell Flaky %]`.
3. **Sort** by `Smell Flaky %` descending.
4. Colours: present = blue `#2A78D6`, absent = light grey `#C3C2B7`.
5. Title: **"Flaky share with vs without each test smell"**.

### P5. Projects
**Pattern:** flakiness is concentrated in a few projects: alluxio (62.0%), hbase (33.6%) and
hector (23.2%) have the highest flaky share; these 3 hold 36% of all flaky tests. 5 of 24
projects have at most one flaky test.

1. **Clustered bar chart**.
2. Y-axis = `project`, X-axis = `[Flaky %]`. Tooltips = `[Flaky Tests]`, `[Total Tests]`,
   `[Share of All Flaky Tests]`.
3. **Filters pane > Filters on this visual > project > Filter type: Top N**:
   Show items **Top 10** by value `[Flaky %]`.
4. Add a second visual-level filter: `[Total Tests]` **is greater than or equal to 50**,
   so tiny projects don't dominate (same rule as the Python chart).
5. Title: **"Top 10 projects by flaky share (50+ tests)"**.

## 3. Finishing touches

1. **Slicer**: add a `project` dropdown slicer at the top right. **View > Sync slicers**:
   sync it to the "Flaky vs non-flaky" page too, but **not** to P5 (that visual compares
   projects). Select P5, choose **Format > Edit interactions**, and set the slicer to *None*
   on P5.
2. **Page navigation**: **Insert > Buttons > Navigator > Page navigator**.
3. Check the **Key influencers** page (Person 4) is still there and refreshes.
4. **Model comparison** page: confirm the 15 rows (3 models x 5 balancing settings) appear
   and the text box says that balancing is applied only to training folds.
5. **File > Save**. Then **File > Export > Export to PDF** for the submission/slides.

## Checklist

- [ ] 5 visuals, each titled with its pattern, each matching a chart in `outputs/patterns/`
- [ ] Constant line at 3.65% on every column/bar chart
- [ ] Numbers in the visuals match `patterns.csv` (spot-check P1 and P5)
- [ ] No leakage column in the field list of any table
- [ ] Flaky = orange, Non-Flaky = blue, single-series bars blue
- [ ] Report pages: Overview · Flaky vs non-flaky · Model comparison · Key influencers ·
      What flaky tests look like
