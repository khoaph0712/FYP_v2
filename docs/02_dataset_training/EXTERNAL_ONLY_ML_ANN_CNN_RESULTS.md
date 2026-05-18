# External-Only Dataset Results

This run ignores the old merged/project datasets and uses only datasets downloaded in the external dataset stage.

YOLOv8 was not run.

## Datasets Used

| Dataset | Prepared folder | Task used here | Note |
|---|---|---|---|
| TrashNet | `prepared_datasets\trashnet` | 6-class clean-background classification | Full-image pseudo boxes |
| GINI binary | `prepared_datasets\gini_binary` | garbage vs background | Binary classification, not material sorting |
| TACO official partial | `prepared_datasets\taco_official_partial` | 60-class real-world litter | Partial official image download; sparse/rare classes |

MJU-Waste was downloaded as a metadata repo only. It does not include the image files in the cloned repository, so it is not trained in this run.

## Classical ML Results

All ML models use the same handcrafted 637-feature vector: spatial + frequency/FFT + color + HOG.

| Dataset | Best model | Accuracy | F1-macro | Output |
|---|---|---:|---:|---|
| TrashNet | XGBoost | 0.8103 | 0.7841 | `runs\ml\external_only\trashnet` |
| GINI binary | Random Forest | 0.8837 | 0.8516 | `runs\ml\external_only\gini_binary` |
| TACO official partial | Extra Trees | 0.0705 | 0.0461 | `runs\ml\external_only\taco_official_partial_fast` |

## ANN/CNN Results

| Dataset | Model | Test Accuracy | Test F1-macro | Best Val Accuracy | Output |
|---|---|---:|---:|---:|---|
| TrashNet | ANN | 0.3458 | 0.2792 | 0.3655 | `runs\dl\external_only_ann_cnn\trashnet` |
| TrashNet | CNN | 0.4292 | 0.4172 | 0.4202 | `runs\dl\external_only_ann_cnn\trashnet` |
| GINI binary | ANN | 0.6211 | 0.3831 | 0.6452 | `runs\dl\external_only_ann_cnn\gini_binary` |
| GINI binary | CNN | 0.7895 | 0.7883 | 0.7742 | `runs\dl\external_only_ann_cnn\gini_binary` |
| TACO official partial | ANN | 0.0225 | 0.0020 | 0.0402 | `runs\dl\external_only_ann_cnn\taco_official_partial` |
| TACO official partial | CNN | 0.0586 | 0.0124 | 0.0580 | `runs\dl\external_only_ann_cnn\taco_official_partial` |

## Interpretation

- TrashNet is much easier because the images are clean-background and only 6 classes.
- GINI binary performs well because it is only `garbage` vs `background`, not material sorting.
- TACO official partial performs very poorly because it is real-world, fine-grained, 60-class, sparse, and partially downloaded. This supports the lecturer feedback: raw TACO-like datasets need class mapping and balancing before final training.

## Commands Used

```powershell
.\.venv311\Scripts\python.exe scripts\prepare_gini_binary_classification.py --source external_datasets\gini --out prepared_datasets\gini_binary --max-per-class 1000 --overwrite

.\.venv311\Scripts\python.exe scripts\feature_ml_analysis.py --data prepared_datasets\trashnet\data.yaml --out runs\ml\external_only\trashnet --exclude-classes= --max-per-class-train 400 --max-per-class-test 120 --domain-out runs\ml\external_only_features\trashnet
.\.venv311\Scripts\python.exe scripts\feature_ml_analysis.py --data prepared_datasets\gini_binary\data.yaml --out runs\ml\external_only\gini_binary --exclude-classes= --max-per-class-train 400 --max-per-class-test 120 --domain-out runs\ml\external_only_features\gini_binary
.\.venv311\Scripts\python.exe scripts\feature_ml_analysis.py --data prepared_datasets\taco_official_partial\data.yaml --out runs\ml\external_only\taco_official_partial_fast --exclude-classes= --max-per-class-train 30 --max-per-class-test 10 --domain-out runs\ml\external_only_features\taco_official_partial_fast

.\.venv311\Scripts\python.exe scripts\deep_learning_baseline.py --data prepared_datasets\trashnet\data.yaml --out runs\dl\external_only_ann_cnn\trashnet --model both --max-per-class-train 350 --max-per-class-val 60 --max-per-class-test 60 --max-train-objects 2100 --max-val-objects 360 --max-test-objects 360 --epochs 5 --batch-size 64 --min-per-class-train 50 --min-per-class-val 20 --min-per-class-test 20
.\.venv311\Scripts\python.exe scripts\deep_learning_baseline.py --data prepared_datasets\gini_binary\data.yaml --out runs\dl\external_only_ann_cnn\gini_binary --model both --max-per-class-train 350 --max-per-class-val 60 --max-per-class-test 60 --max-train-objects 700 --max-val-objects 120 --max-test-objects 120 --epochs 5 --batch-size 64 --min-per-class-train 50 --min-per-class-val 20 --min-per-class-test 20
.\.venv311\Scripts\python.exe scripts\deep_learning_baseline.py --data prepared_datasets\taco_official_partial\data.yaml --out runs\dl\external_only_ann_cnn\taco_official_partial --model both --max-per-class-train 30 --max-per-class-val 10 --max-per-class-test 10 --max-train-objects 1500 --max-val-objects 500 --max-test-objects 500 --epochs 5 --batch-size 64 --min-per-class-train 5 --min-per-class-val 1 --min-per-class-test 1
```

## Lecturer Talking Point

The downloaded external datasets do not behave equally. Clean classification data gives high ML scores, binary garbage/background is easier, while real-world TACO-style data is extremely difficult without class mapping. Therefore, the next scientifically correct step is not YOLO training, but building a controlled balanced taxonomy from external datasets.
