# Key Influencers visual: click-by-click guide

**Owner:** Person 4  ·  **Needs:** the base `.pbix` from `documentation/powerbi_base_guide.md`
(the **Tests** table)  ·  **Result:** a "Key influencers" page plus a filled-in
`outputs/powerbi/key_influencers_top.csv`

Key Influencers is Power BI's built-in AI visual. For each feature it asks: "when this feature
has value X, how much more likely is the test to be Flaky?" We then compare its answers with
the SHAP ranking from Python (`python -m src.compare_influencers`).

---

## 1. Add the page and the visual

1. Open the `.pbix`. Click **+** at the bottom to add a page, and rename it **Key influencers**.
2. In **Visualizations**, click the **Key influencers** icon (a light-bulb chart).
   Stretch it across the whole page; it needs room.

## 2. Analyze = `flaky_label`

1. Drag **Tests > `flaky_label`** into the **Analyze** well.
2. At the top of the visual, set *"What influences flaky_label to be"* to **Flaky**.

> Use `flaky_label` (text), not `flaky` (0/1). A text field makes Power BI treat the target as
> a category, which is what we want.

## 3. Explain by = the 23 model features (and nothing else)

Drag **exactly these 23 fields** from the **Tests** table into **Explain by**:

| Group | Fields |
|---|---|
| Execution | `execution_time_s` |
| Coverage | `covered_lines`, `source_lines_covered`, `source_classes_covered` |
| Test size | `test_length`, `num_asserts` |
| Dependencies | `third_party_libs` |
| Test smells (8) | `smell_assertion_roulette`, `smell_conditional_logic`, `smell_eager_test`, `smell_fire_and_forget`, `smell_indirect_testing`, `smell_mystery_guest`, `smell_resource_optimism`, `smell_test_run_war` |
| Churn (8) | `changed_line_churn_w5`, `changed_line_churn_w10`, `changed_line_churn_w25`, `changed_line_churn_w50`, `changed_line_churn_w75`, `changed_line_churn_w100`, `changed_line_churn_w500`, `changed_line_churn_w10000` |

**Do NOT add:**
- `project`, `test_name`, or the class/method names. They identify a test; they don't describe it.
- The `*_bucket`, `*_order`, `smell_count` or `has_any_smell` columns. They repeat the features
  above, and the comparison script only accepts the 23 names.
- `flaky`, which is the target itself.
- Any rerun-outcome column. These are already removed from `flaky_tests_powerbi.csv`.

**Numeric fields:** in each field's dropdown inside the *Explain by* well, make sure
**Don't summarize** is selected (it is the default for columns). Power BI then finds thresholds
such as "is more than 1.6" by itself.

## 4. Settings

In **Format visual > Visual > Analysis**:
- **Analysis type: Categorical**. This compares the Flaky share in each group of a feature.
  Our numeric features are then split into ranges automatically.
- Leave *Enable counts* **off** for a cleaner list. You can turn it on to see how many tests
  each influencer covers.

Wait for the visual to finish ("Analyzing...").

## 5. Read the "Key influencers" tab

The left list shows influencers in order, for example (illustrative numbers; yours will differ):

> *When `execution_time_s` is more than 1.62, the likelihood of flaky_label being **Flaky**
> increases by **6.85x***

Click an influencer to see the bar chart on the right (Flaky % for that group vs the rest).

The **Top segments** tab shows combinations of conditions (for example: high execution time AND
high churn). Take a screenshot for the slides, but it is not needed for the template.

## 6. Fill in the template

Open `outputs/powerbi/key_influencers_top.csv` in Excel or a text editor. Fill in one row per
influencer, top to bottom, for the **first 10**:

| Column | What to write | Example |
|---|---|---|
| `rank` | position in the list (1 = top) | `1` |
| `feature` | the **exact** field name | `execution_time_s` |
| `condition` | the condition text Power BI shows | `is more than 1.62` |
| `likelihood_multiplier` | the "increases by" number, without the x | `6.85` |
| `notes` | optional | `bar chart: 21% vs 3%` |

Rules:
- If a feature appears twice (two different ranges), give it two rows. The script keeps the
  first, highest-ranked one.
- Keep the header row exactly as it is. Save as **CSV**, not .xlsx.
- For condition text, copy Power BI's wording ("is more than", "is less than", "is 1"). The
  script uses those words to check whether the direction agrees with SHAP.

## 7. Run the comparison

From the repo root:

```
python -m src.compare_influencers
```

It checks your feature names against the 23 features and compares the Key Influencers top 10
with the SHAP top 10. It writes:
- `outputs/powerbi/influencers_vs_shap.csv`: per feature, "both", "SHAP only" or
  "Key Influencers only", the two ranks, and whether the direction agrees
- `outputs/powerbi/influencers_agreement_notes.md`: plain-English notes for the slides

Then Person 2 reruns `python -m src.top_features`. It adds Key Influencers as the 5th view
automatically.

If the template is still empty, the script just prints *"not filled in yet"* and stops. That
is expected.

## Why Key Influencers and SHAP can disagree (for the viva)

- Key Influencers looks at **one feature at a time** (a simple logistic regression per
  feature). SHAP explains the **Random Forest**, which uses all features together.
- Correlated features (the 8 churn windows; lines vs classes covered) can **all** look strong
  in Key Influencers. In SHAP they split the credit between them.
- Key Influencers reports a **range** ("more than 1.62"). SHAP ranks the **whole feature**.
