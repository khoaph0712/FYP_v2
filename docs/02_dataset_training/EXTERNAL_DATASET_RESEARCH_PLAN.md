# External Dataset Research Plan

This document turns the lecturer feedback into a dataset-first research workflow. The goal is no longer to merge every source immediately. The goal is to prove which datasets are useful, what each dataset contributes, and how each one should be converted before any controlled merge.

## Lecturer Problem

The previous `merged_dataset_v3` result hides severe imbalance and source bias. A model trained on the merge may look like one experiment, but it does not answer:

- which source contributed which class,
- whether one dataset dominates the distribution,
- whether TACO-like real-world annotations behave differently from clean classification datasets,
- whether handcrafted features are dataset-specific or class-specific.

## Candidate Dataset Registry

The machine-readable registry is stored in:

`docs\02_dataset_training\external_dataset_registry.json`

| Dataset | Format | Classes | Similarity To TACO | Use Decision |
|---|---|---:|---|---|
| TACO official | COCO instance segmentation | 60 | reference | Convert first; use as main real-world litter source |
| MJU-Waste | PASCAL VOC / COCO segmentation | 1 | medium | EDA/features; useful for segmentation/localization, not sorting alone |
| ZeroWaste | detection/segmentation | unknown until download | medium-high | External-domain robustness/stress test |
| TrashCan | instance segmentation masks | 22 | medium | Marine debris comparison; do not merge into household sorting without justification |
| TrashNet | classification folders | 6 | low-medium | Clean-background classification baseline only |
| GINI | one-class bounding boxes | 1 | medium | EDA/features or binary trash-vs-background only |

## Sources Checked

- TACO official/reference: `https://github.com/pedropro/TACO`, `https://datasetninja.com/taco`
- MJU-Waste official: `https://github.com/realwecan/mju-waste`
- ZeroWaste paper/project: `https://arxiv.org/abs/2106.02740`
- TrashCan data repository: `https://conservancy.umn.edu/items/6dd6a960-c44a-4510-a679-efb8c82ebfb7`
- TrashNet official repository: `https://github.com/garythung/trashnet`
- GINI reference listing: `https://github.com/AgaMiko/waste-datasets-review`

## Correct Pipeline Per Dataset

### Step 1: Keep Raw Dataset Separate

Every dataset must live under a source folder:

```text
external_datasets/
  taco_official/
  mju_waste/
  zerowaste/
  trashcan/
  trashnet/
  gini/
```

Do not merge during this step.

### Step 2: Convert To One Internal Format

Preferred internal format for this project:

```text
prepared_datasets/<dataset_id>/
  data.yaml
  train/images
  train/labels
  valid/images
  valid/labels
  test/images
  test/labels
```

Conversion rules:

| Original Format | Required Adapter |
|---|---|
| COCO boxes/masks | COCO to YOLO labels, then crop extraction |
| PASCAL VOC XML | VOC to YOLO labels |
| segmentation masks | mask-to-box conversion or segmentation-only EDA |
| classification folders | classification feature pipeline; optional full-image pseudo boxes only if clearly documented |

### Step 3: EDA Per Dataset

For each dataset:

- number of images,
- number of objects,
- number of classes,
- class distribution,
- imbalance ratio,
- object size distribution,
- boxes per image,
- image resolution distribution,
- sample images,
- missing/rare classes.

Expected output:

```text
runs/external_dataset_eda/<dataset_id>/
  DATASET_SOURCE_COMPARISON.md
  source_summary.csv
  chart_source_class_distribution.png
```

### Step 4: Feature Extraction Per Dataset

For each dataset, extract the same 637-feature vector:

- 8 spatial features,
- 9 frequency/FFT features,
- 44 color features,
- 576 HOG features.

Expected output:

```text
runs/ml/features_by_dataset/<dataset_id>/
  feature_summary.csv
  spatial_summary.csv
  frequency_summary.csv
  color_summary.csv
  hog_summary.csv
```

### Step 5: Classical ML Per Dataset

Train only datasets that have enough classes for a fair classifier:

- Decision Tree,
- SVM,
- Random Forest,
- XGBoost.

Single-class datasets such as GINI or cigarette-only datasets should be EDA/features only, or treated as binary trash-vs-background if negative/background images are added.

Expected output:

```text
runs/ml/by_dataset/<dataset_id>/
  metrics_summary.json
  classification_reports.json
  confusion_decision_tree.png
  confusion_linear_svm.png
  confusion_rf.png
  confusion_xgboost.png
```

### Step 6: Controlled Merge Only After Evidence

Merge only after source-level analysis is finished.

Merge rules:

- map dataset-specific labels into the project taxonomy,
- cap per class and per source,
- do not let one source dominate one class,
- keep source contribution CSV,
- document every dropped class.

Example mapping:

| External Label Type | Project Class |
|---|---|
| bottle, plastic film, plastic wrapper, plastic bag | plastic |
| can, aluminium foil, metal cap | metal |
| glass bottle, glass cup | glass |
| paper cup, normal paper, tissues | paper |
| carton, cardboard box | cardboard |
| food waste, fruit, vegetable | organic |
| cigarette, unknown litter, styrofoam if not recyclable | other |

Expected controlled merge output:

```text
controlled_dataset_v1/
  data.yaml
  source_contribution.csv
  class_mapping.json
  dropped_labels.json
  balance_report.md
```

## Current Project Status

Already done locally:

- raw Roboflow source folders downloaded again,
- raw source EDA/features generated,
- ANN/CNN trained separately for multi-class raw datasets,
- merged-source EDA/features generated using filename prefixes,
- source-level ANN/CNN runs generated for the merged source prefixes.
- external dataset repos downloaded where public GitHub access worked:
  - official TACO repository,
  - TrashNet repository,
  - GINI repository,
  - MJU-Waste metadata repository.
- prepared internal YOLO-style datasets:
  - `prepared_datasets\trashnet`
  - `prepared_datasets\gini`
  - `prepared_datasets\taco_official_partial`
- external EDA/features generated:
  - `runs\external_dataset_eda\trashnet`
  - `runs\external_dataset_eda\gini`
  - `runs\external_dataset_eda\taco_official_partial`
- external per-dataset classical ML generated:
  - `runs\ml\by_dataset\trashnet`
  - `runs\ml\by_dataset\taco_official_partial`

Still needed before final controlled merge:

- complete official TACO image download if full 1500 images are required; current run prepared a partial official TACO dataset from available downloaded images,
- decide whether MJU-Waste / TrashCan / ZeroWaste are only comparison datasets or also training data,
- add TrashNet only as a clean-background classification baseline, not as detection evidence,
- create `controlled_dataset_v1` with per-source caps.

## External Dataset Run Results

### Prepared Datasets

| Prepared dataset | Source | Classes | Use | Notes |
|---|---|---:|---|---|
| `prepared_datasets\trashnet` | TrashNet GitHub | 6 | ML classification/features | Full-image pseudo boxes because TrashNet has no object boxes |
| `prepared_datasets\gini` | GINI GitHub | 1 | EDA/features only | Positive garbage boxes plus empty-label background images; not material sorting |
| `prepared_datasets\taco_official_partial` | official TACO GitHub | 60 | ML classification/features | TACO downloader produced 732 images during this run; conversion skipped missing images |

TACO prepare summary:

| Split | Images | Objects |
|---|---:|---:|
| train | 1095 | 3469 |
| valid | 141 | 488 |
| test | 141 | 486 |

The TACO numbers above are from annotations whose images were available locally after the downloader run. Some image filenames had multiple annotations, so converted split image counts can exceed the number of unique downloaded files.

### Classical ML By External Dataset

| Dataset | Best model | Accuracy | F1-macro | Interpretation |
|---|---|---:|---:|---|
| TrashNet | XGBoost | 0.8103 | 0.7841 | Clean-background classification features are separable; useful baseline but not TACO-like detection |
| TACO official partial | XGBoost | 0.1399 | 0.0675 | Real-world 60-class litter is much harder and highly sparse/imbalanced |

TrashNet full ranking:

| Model | Accuracy | F1-macro |
|---|---:|---:|
| XGBoost | 0.8103 | 0.7841 |
| Random Forest | 0.7391 | 0.6688 |
| Extra Trees | 0.7115 | 0.6597 |
| Logistic Regression | 0.6166 | 0.5622 |
| Linear SVM | 0.5613 | 0.5155 |
| Decision Tree | 0.5296 | 0.5073 |

TACO official partial full ranking:

| Model | Accuracy | F1-macro |
|---|---:|---:|
| Random Forest | 0.1552 | 0.0411 |
| XGBoost | 0.1399 | 0.0675 |
| Extra Trees | 0.1349 | 0.0457 |
| Logistic Regression | 0.1145 | 0.0479 |
| Linear SVM | 0.0840 | 0.0368 |
| Decision Tree | 0.0611 | 0.0278 |

Important TACO limitation:

- 18 rare official TACO classes had no usable test support after the partial image download / split and were dropped from the ML run.
- This is not a failure of the project; it is evidence for the lecturer's point that raw TACO-style data is highly sparse and needs class mapping before final training.
- The next correct step is to map 60 TACO classes into a smaller project taxonomy or TACO-10 style grouping, then cap per mapped class/source.

## Commands Actually Run For External Datasets

```powershell
git clone --depth 1 https://github.com/pedropro/TACO.git external_datasets\taco_official
git clone --depth 1 https://github.com/garythung/trashnet.git external_datasets\trashnet
git clone --depth 1 https://github.com/spotgarbage/spotgarbage-GINI.git external_datasets\gini
git clone --depth 1 https://github.com/realwecan/mju-waste.git external_datasets\mju_waste_repo

.\.venv311\Scripts\python.exe scripts\prepare_classification_dataset.py --source external_datasets\trashnet\data\dataset-resized --out prepared_datasets\trashnet --overwrite
.\.venv311\Scripts\python.exe scripts\prepare_gini_dataset.py --source external_datasets\gini --out prepared_datasets\gini --overwrite --max-negative-images 500
C:\FYP_v2\.venv311\Scripts\python.exe external_datasets\taco_official\download.py
.\.venv311\Scripts\python.exe scripts\prepare_coco_dataset.py --annotations external_datasets\taco_official\data\annotations.json --images-root external_datasets\taco_official\data --out prepared_datasets\taco_official_partial --overwrite

.\.venv311\Scripts\python.exe scripts\analyze_dataset_sources.py --data prepared_datasets\trashnet\data.yaml --out runs\external_dataset_eda\trashnet --max-feature-objects-per-source 1500
.\.venv311\Scripts\python.exe scripts\analyze_dataset_sources.py --data prepared_datasets\gini\data.yaml --out runs\external_dataset_eda\gini --max-feature-objects-per-source 1500
.\.venv311\Scripts\python.exe scripts\analyze_dataset_sources.py --data prepared_datasets\taco_official_partial\data.yaml --out runs\external_dataset_eda\taco_official_partial --max-feature-objects-per-source 1500

.\.venv311\Scripts\python.exe scripts\feature_ml_analysis.py --data prepared_datasets\trashnet\data.yaml --out runs\ml\by_dataset\trashnet --exclude-classes= --max-per-class-train 400 --max-per-class-test 120 --domain-out runs\ml\features_by_dataset\trashnet
.\.venv311\Scripts\python.exe scripts\feature_ml_analysis.py --data prepared_datasets\taco_official_partial\data.yaml --out runs\ml\by_dataset\taco_official_partial --exclude-classes= --max-per-class-train 120 --max-per-class-test 40 --domain-out runs\ml\features_by_dataset\taco_official_partial
```

## What To Tell Lecturer

The project no longer treats the merged dataset as the only experiment. The next dataset section will show a source-level comparison first. Each dataset is inspected separately for class imbalance, annotation format, object size, and feature distribution. Only after that will compatible datasets be mapped into a controlled final taxonomy with caps per class and per source.
