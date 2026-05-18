# Dataset Source Analysis

## Why This Exists
The lecturer feedback is that the merged dataset is heavily imbalanced and hides source-level bias. This report keeps the original filename source prefix and analyses each dataset/source separately before any final merge.

- Data YAML: `C:\FYP_v2\prepared_datasets\trashnet\data.yaml`
- Classes after previous merge: cardboard, glass, metal, paper, plastic, trash

## Source Summary
| Source | Images | Objects | Classes Present | Imbalance Ratio | Main Risk |
|---|---:|---:|---:|---:|---|
| paper | 594 | 594 | 1 | 1.0 | missing classes |
| glass | 501 | 501 | 1 | 1.0 | missing classes |
| plastic | 482 | 482 | 1 | 1.0 | missing classes |
| metal | 410 | 410 | 1 | 1.0 | missing classes |
| cardboard | 403 | 403 | 1 | 1.0 | missing classes |
| trash | 137 | 137 | 1 | 1.0 | missing classes |

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