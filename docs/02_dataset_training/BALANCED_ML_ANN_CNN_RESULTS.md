# Balanced ML And ANN/CNN Results

YOLOv8 is intentionally paused for this stage. This report only covers:

- classical ML on handcrafted features,
- ANN/CNN crop classifiers,
- balanced sampling by per-class caps.

## Why Balance

The lecturer feedback was that the datasets are too imbalanced. The raw datasets are still kept for EDA evidence, but model comparison now uses capped per-class sampling where possible.

Balancing method:

- keep the original raw/prepared dataset unchanged,
- sample object crops per class,
- cap each class independently,
- drop classes without enough train/validation/test support for ANN/CNN,
- report support in `class_support.json`.

## Classical ML Results

All classical ML models use the same 637 handcrafted features:

- 8 spatial,
- 9 frequency/FFT,
- 44 color,
- 576 HOG.

| Dataset | Balanced cap | Best model | Accuracy | F1-macro |
|---|---|---|---:|---:|
| `merged_dataset_v3` 6-class | train 1000/class, test 250/class | XGBoost | 0.6497 | 0.6422 |
| `prepared_datasets\trashnet` | train 400/class, test 120/class | XGBoost | 0.8103 | 0.7841 |
| `prepared_datasets\taco_official_partial` | train 120/class, test 40/class | XGBoost | 0.1399 | 0.0675 |

Important interpretation:

- TrashNet scores highest because it is a clean-background classification dataset.
- The project 6-class dataset is more realistic and therefore lower.
- TACO official partial is much harder because it has many fine-grained classes and rare labels; it should be mapped into fewer project classes before final training.

Output folders:

- `runs\ml\balanced_by_dataset\merged_6class`
- `runs\ml\by_dataset\trashnet`
- `runs\ml\by_dataset\taco_official_partial`

## ANN/CNN Results

ANN/CNN were trained with balanced per-class caps. These are crop classifiers, not YOLO detectors.

| Dataset | Model | Test Accuracy | Test F1-macro | Best Val Accuracy |
|---|---|---:|---:|---:|
| `merged_dataset_v3` 6-class | ANN | 0.4083 | 0.3836 | 0.4072 |
| `merged_dataset_v3` 6-class | CNN | 0.4792 | 0.4676 | 0.4688 |
| `prepared_datasets\trashnet` | ANN | 0.3458 | 0.2792 | 0.3655 |
| `prepared_datasets\trashnet` | CNN | 0.4167 | 0.4048 | 0.4118 |
| `prepared_datasets\taco_official_partial` | ANN | 0.0573 | 0.0028 | 0.0532 |
| `prepared_datasets\taco_official_partial` | CNN | 0.0573 | 0.0028 | 0.0532 |

Important interpretation:

- CNN is better than ANN on the balanced project dataset because CNN keeps spatial structure.
- TrashNet remains easier than TACO because TrashNet images are cleaner and class count is smaller.
- TACO 60-class partial is not suitable for direct ANN/CNN classification without class mapping or more complete/balanced data.

Output folders:

- `runs\dl\balanced_ann_cnn\merged_6class`
- `runs\dl\balanced_ann_cnn\trashnet`
- `runs\dl\balanced_ann_cnn\taco_official_partial`

Each folder contains:

- saved model files,
- `training_log.csv`,
- train/validation loss and accuracy curves,
- confusion matrix,
- classification report,
- `class_support.json`.

## Commands

Classical ML:

```powershell
.\.venv311\Scripts\python.exe scripts\feature_ml_analysis.py --data merged_dataset_v3\data.yaml --out runs\ml\balanced_by_dataset\merged_6class --exclude-classes other --max-per-class-train 1000 --max-per-class-test 250 --domain-out runs\ml\features_by_dataset\merged_6class

.\.venv311\Scripts\python.exe scripts\feature_ml_analysis.py --data prepared_datasets\trashnet\data.yaml --out runs\ml\by_dataset\trashnet --exclude-classes= --max-per-class-train 400 --max-per-class-test 120 --domain-out runs\ml\features_by_dataset\trashnet

.\.venv311\Scripts\python.exe scripts\feature_ml_analysis.py --data prepared_datasets\taco_official_partial\data.yaml --out runs\ml\by_dataset\taco_official_partial --exclude-classes= --max-per-class-train 120 --max-per-class-test 40 --domain-out runs\ml\features_by_dataset\taco_official_partial
```

ANN/CNN:

```powershell
.\.venv311\Scripts\python.exe scripts\deep_learning_baseline.py --data merged_dataset_v3\data.yaml --out runs\dl\balanced_ann_cnn\merged_6class --model both --source-filter rf_taco_trash,rf_garbage_cls,rf_waste_sorting,rf_uca_recyclable --max-per-class-train 1000 --max-per-class-val 250 --max-per-class-test 250 --max-train-objects 6000 --max-val-objects 1500 --max-test-objects 1500 --epochs 5 --batch-size 64 --min-per-class-train 50 --min-per-class-val 10 --min-per-class-test 10

.\.venv311\Scripts\python.exe scripts\deep_learning_baseline.py --data prepared_datasets\trashnet\data.yaml --out runs\dl\balanced_ann_cnn\trashnet --model both --max-per-class-train 350 --max-per-class-val 60 --max-per-class-test 60 --max-train-objects 2100 --max-val-objects 360 --max-test-objects 360 --epochs 5 --batch-size 64 --min-per-class-train 50 --min-per-class-val 20 --min-per-class-test 20

.\.venv311\Scripts\python.exe scripts\deep_learning_baseline.py --data prepared_datasets\taco_official_partial\data.yaml --out runs\dl\balanced_ann_cnn\taco_official_partial --model both --max-per-class-train 50 --max-per-class-val 15 --max-per-class-test 15 --max-train-objects 2500 --max-val-objects 600 --max-test-objects 600 --epochs 5 --batch-size 64 --min-per-class-train 5 --min-per-class-val 1 --min-per-class-test 1
```

## Conclusion

Balanced sampling improves fairness of comparison, but it does not magically solve dataset difficulty. Clean datasets such as TrashNet are easier; real-world TACO-style 60-class data needs class mapping and controlled merging before final training.
