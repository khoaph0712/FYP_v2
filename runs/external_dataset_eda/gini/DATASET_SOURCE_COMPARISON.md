# Dataset Source Analysis

## Why This Exists
The lecturer feedback is that the merged dataset is heavily imbalanced and hides source-level bias. This report keeps the original filename source prefix and analyses each dataset/source separately before any final merge.

- Data YAML: `C:\FYP_v2\prepared_datasets\gini\data.yaml`
- Classes after previous merge: garbage

## Source Summary
| Source | Images | Objects | Classes Present | Imbalance Ratio | Main Risk |
|---|---:|---:|---:|---:|---|
| garbage | 333 | 333 | 1 | 1.0 | OK |
| background | 484 | 0 | 0 | 0.0 | missing classes |

## Files Generated
- `source_summary.csv`: image/object counts, class distribution, imbalance ratio.
- `source_feature_summary.csv`: sampled handcrafted feature summary by source.
- `features_by_source/*.csv`: mean/std of all 637 features for each source.
- `chart_source_class_distribution.png`: stacked class distribution by source.

## Interpretation For Report
- Do not claim the merged dataset is balanced.
- Present each source separately first.
- Train ANN/CNN by source, then compare with the merged result.
- If merging later, cap per class and per source so one source cannot dominate the final dataset.