# Dataset Source Analysis

## Why This Exists
The lecturer feedback is that the merged dataset is heavily imbalanced and hides source-level bias. This report keeps the original filename source prefix and analyses each dataset/source separately before any final merge.

- Data YAML: `C:\FYP_v2\prepared_datasets\taco_official_partial\data.yaml`
- Classes after previous merge: Aluminium foil, Battery, Aluminium blister pack, Carded blister pack, Other plastic bottle, Clear plastic bottle, Glass bottle, Plastic bottle cap, Metal bottle cap, Broken glass, Food Can, Aerosol, Drink can, Toilet tube, Other carton, Egg carton, Drink carton, Corrugated carton, Meal carton, Pizza box, Paper cup, Disposable plastic cup, Foam cup, Glass cup, Other plastic cup, Food waste, Glass jar, Plastic lid, Metal lid, Other plastic, Magazine paper, Tissues, Wrapping paper, Normal paper, Paper bag, Plastified paper bag, Plastic film, Six pack rings, Garbage bag, Other plastic wrapper, Single-use carrier bag, Polypropylene bag, Crisp packet, Spread tub, Tupperware, Disposable food container, Foam food container, Other plastic container, Plastic glooves, Plastic utensils, Pop tab, Rope & strings, Scrap metal, Shoe, Squeezable tube, Plastic straw, Paper straw, Styrofoam piece, Unlabeled litter, Cigarette

## Source Summary
| Source | Images | Objects | Classes Present | Imbalance Ratio | Main Risk |
|---|---:|---:|---:|---:|---|
| batch_12 | 100 | 505 | 37 | 81.0 | severe imbalance |
| batch_8 | 100 | 433 | 39 | 81.0 | severe imbalance |
| batch_6 | 96 | 402 | 33 | 72.0 | severe imbalance |
| batch_10 | 100 | 377 | 39 | 86.0 | severe imbalance |
| batch_7 | 107 | 358 | 39 | 37.0 | severe imbalance |
| batch_15 | 85 | 317 | 32 | 47.0 | severe imbalance |
| batch_9 | 100 | 311 | 30 | 59.0 | severe imbalance |
| batch_1 | 101 | 309 | 48 | 32.0 | severe imbalance |
| batch_5 | 107 | 290 | 37 | 45.0 | severe imbalance |
| batch_2 | 92 | 288 | 33 | 95.0 | severe imbalance |
| batch_13 | 100 | 274 | 35 | 43.0 | severe imbalance |
| batch_11 | 100 | 235 | 34 | 37.0 | severe imbalance |
| batch_14 | 100 | 180 | 30 | 27.0 | severe imbalance |
| batch_4 | 89 | 164 | 28 | 43.0 | severe imbalance |

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