# Source-Level Dataset Analysis And ANN/CNN Plan

This document records the new lecturer-directed direction: stop treating the merged dataset as the only experiment, pause YOLO as the main focus, and analyse/train each dataset source separately.

## Lecturer Feedback Converted To Tasks

- The merged dataset is too imbalanced.
- The original dataset/source identity must be preserved.
- TACO-style datasets with many annotated classes should be analysed before being mapped into 6/7 final classes.
- Extract characteristic features per dataset/source, not only after merging.
- Train ANN/CNN per dataset/source, with saved models, saved logs, accuracy, confusion matrix, and classification report.
- YOLO is paused for this stage.

## What Was Implemented

### 1. Source-Level EDA And Feature Extraction

Command:

```powershell
.\.venv311\Scripts\python.exe scripts\analyze_dataset_sources.py `
  --data merged_dataset_v3\data.yaml `
  --out runs\source_analysis\merged_dataset_v3 `
  --max-feature-objects-per-source 1500
```

Outputs:

- `runs\source_analysis\merged_dataset_v3\DATASET_SOURCE_COMPARISON.md`
- `runs\source_analysis\merged_dataset_v3\source_summary.csv`
- `runs\source_analysis\merged_dataset_v3\source_feature_summary.csv`
- `runs\source_analysis\merged_dataset_v3\features_by_source\*.csv`
- `runs\source_analysis\merged_dataset_v3\chart_source_class_distribution.png`

## Source-Level Findings

| Source | Images | Objects | Classes Present | Imbalance Ratio | Decision |
|---|---:|---:|---:|---:|---|
| `rf_garbage_cls` | 8,387 | 28,683 | 5 | 1.779 | train ANN/CNN |
| `rf_food_waste` | 6,734 | 28,446 | 1 | 1.000 | EDA only; not useful as standalone classifier |
| `rf_waste_sorting` | 15,630 | 16,027 | 5 | 356.125 | train ANN/CNN, but mark severe imbalance |
| `rf_taco_trash` | 9,305 | 16,004 | 6 | 2.720 | train ANN/CNN; note original TACO-style labels were already mapped in this merged copy |
| `rf_uca_recyclable` | 8,365 | 14,529 | 3 | 2.595 | train ANN/CNN |
| `rf_cigarettes` | 2,427 | 3,077 | 1 | 1.000 | EDA only; not useful as standalone classifier |

Important interpretation:

- The previous merged dataset is not balanced at source level.
- Some sources are single-class only; training ANN/CNN on those alone would produce misleading results.
- `rf_waste_sorting` is the clearest imbalance problem: `glass` has only 16 objects while `plastic` has 5,698 and `metal` has 5,546.
- The current `rf_taco_trash` copy has already been mapped into the project class scheme. If the lecturer wants the original TACO 60-class analysis, the next step is to add the original TACO annotation file separately instead of relying only on the merged copy.

## Source-Level ANN/CNN Training

Command pattern:

```powershell
.\.venv311\Scripts\python.exe scripts\deep_learning_baseline.py `
  --data merged_dataset_v3\data.yaml `
  --out runs\dl\source_ann_cnn\<source_name> `
  --model both `
  --source-filter <source_name> `
  --max-train-objects 3000 `
  --max-val-objects 800 `
  --max-test-objects 800 `
  --epochs 5 `
  --batch-size 64
```

Sources trained:

- `rf_garbage_cls`
- `rf_waste_sorting`
- `rf_taco_trash`
- `rf_uca_recyclable`

Sources not trained as standalone classifiers:

- `rf_food_waste`: only organic class.
- `rf_cigarettes`: only other class.

## ANN/CNN Results By Source

| Source | Model | Test Accuracy | Test F1-macro | Best Val Accuracy |
|---|---|---:|---:|---:|
| `rf_garbage_cls` | ANN | 0.4362 | 0.3843 | 0.4450 |
| `rf_garbage_cls` | CNN | 0.3175 | 0.2836 | 0.5100 |
| `rf_taco_trash` | ANN | 0.4444 | 0.2988 | 0.4850 |
| `rf_taco_trash` | CNN | 0.5247 | 0.4333 | 0.4688 |
| `rf_uca_recyclable` | ANN | 0.6938 | 0.6982 | 0.7175 |
| `rf_uca_recyclable` | CNN | 0.7038 | 0.7003 | 0.7212 |
| `rf_waste_sorting` | ANN | 0.5716 | 0.4297 | 0.5525 |
| `rf_waste_sorting` | CNN | 0.6672 | 0.5015 | 0.6725 |

Each source output folder contains:

- `metrics_summary.json`
- `REPORT.md`
- `ann\ann.pt`
- `ann\ann_full.pt`
- `ann\training_log.csv`
- `ann\training_curves_ann.png`
- `ann\confusion_ann.png`
- `ann\classification_report.json`
- `cnn\cnn.pt`
- `cnn\cnn_full.pt`
- `cnn\training_log.csv`
- `cnn\training_curves_cnn.png`
- `cnn\confusion_cnn.png`
- `cnn\classification_report.json`

## How To Explain This To Lecturer

The earlier merged dataset gave one combined result, but it hid source-level imbalance. The new experiment separates the sources using the original filename prefix. For each source, the project now reports class distribution, imbalance ratio, object statistics, sampled 637-feature summaries, and ANN/CNN training results where the source has enough classes.

Current conclusion:

- Do not use the merged dataset alone as the main proof.
- Use source-level EDA/features to justify which datasets are useful.
- Use ANN/CNN source experiments as the new deep-learning baseline.
- Keep YOLO results as previous work, but not the current focus.

## Raw Dataset Download And Per-Dataset Training

After the lecturer note about dataset imbalance, the raw source folders were downloaded again instead of only relying on `merged_dataset_v3`.

Downloaded raw Roboflow folders:

| Folder | Raw classes | Use |
|---|---:|---|
| `rf_taco_trash` | 12 | Train ANN/CNN and raw feature analysis |
| `rf_garbage_cls` | 6 | Train ANN/CNN and raw feature analysis |
| `rf_food_waste` | 32 | Train ANN/CNN and raw feature analysis; maps to organic in project-level classes |
| `rf_trash_detection` | 1 | EDA/features only; not useful as standalone classifier |
| `rf_cigarettes` | 1 | EDA/features only; not useful as standalone classifier |

Download commands used:

```powershell
.\.venv311\Scripts\python.exe scripts\download_datasets.py
.\.venv311\Scripts\python.exe scripts\download_extra.py
```

Note: `rf_waste_saah1` was attempted by `download_extra.py`, but no public version was available from Roboflow during this run.

Raw feature analysis commands:

```powershell
.\.venv311\Scripts\python.exe scripts\analyze_dataset_sources.py --data rf_taco_trash\data.yaml --out runs\raw_source_analysis\rf_taco_trash --max-feature-objects-per-source 1500
.\.venv311\Scripts\python.exe scripts\analyze_dataset_sources.py --data rf_garbage_cls\data.yaml --out runs\raw_source_analysis\rf_garbage_cls --max-feature-objects-per-source 1500
.\.venv311\Scripts\python.exe scripts\analyze_dataset_sources.py --data rf_food_waste\data.yaml --out runs\raw_source_analysis\rf_food_waste --max-feature-objects-per-source 1500
.\.venv311\Scripts\python.exe scripts\analyze_dataset_sources.py --data rf_trash_detection\data.yaml --out runs\raw_source_analysis\rf_trash_detection --max-feature-objects-per-source 1500
.\.venv311\Scripts\python.exe scripts\analyze_dataset_sources.py --data rf_cigarettes\data.yaml --out runs\raw_source_analysis\rf_cigarettes --max-feature-objects-per-source 1500
```

Raw ANN/CNN commands:

```powershell
.\.venv311\Scripts\python.exe scripts\deep_learning_baseline.py --data rf_taco_trash\data.yaml --out runs\dl\raw_ann_cnn\rf_taco_trash --model both --max-train-objects 3000 --max-val-objects 800 --max-test-objects 800 --epochs 5 --batch-size 64
.\.venv311\Scripts\python.exe scripts\deep_learning_baseline.py --data rf_garbage_cls\data.yaml --out runs\dl\raw_ann_cnn\rf_garbage_cls --model both --max-train-objects 3000 --max-val-objects 800 --max-test-objects 800 --epochs 5 --batch-size 64
.\.venv311\Scripts\python.exe scripts\deep_learning_baseline.py --data rf_food_waste\data.yaml --out runs\dl\raw_ann_cnn\rf_food_waste --model both --max-train-objects 3000 --max-val-objects 800 --max-test-objects 800 --epochs 5 --batch-size 64
```

Raw ANN/CNN results:

| Raw dataset | Model | Test Accuracy | Test F1-macro | Best Val Accuracy | Note |
|---|---|---:|---:|---:|---|
| `rf_taco_trash` | ANN | 0.0000 | 0.0000 | 0.2963 | test split has only 10 objects from one class; use val as fairer signal |
| `rf_taco_trash` | CNN | 0.0000 | 0.0000 | 0.3137 | test split has only 10 objects from one class; use val as fairer signal |
| `rf_garbage_cls` | ANN | 0.3450 | 0.2128 | 0.4537 | raw split still imbalanced |
| `rf_garbage_cls` | CNN | 0.4200 | 0.2980 | 0.4213 | raw split still imbalanced |
| `rf_food_waste` | ANN | 0.2588 | 0.0885 | 0.2650 | 32 fine-grained food classes; severe imbalance |
| `rf_food_waste` | CNN | 0.1725 | 0.0561 | 0.2037 | 32 fine-grained food classes; severe imbalance |

Important TACO clarification:

- The downloaded Roboflow TACO-style dataset used here has **12 classes**, not the official full TACO 60-class taxonomy.
- If the lecturer specifically requires the original TACO 60-class annotation, the next task is to add the official TACO COCO annotation file and convert it to YOLO/crop-classification format.
- The current result is still useful because it proves the new workflow: raw dataset first, feature analysis per dataset, ANN/CNN per dataset, then controlled mapping/merge later.
