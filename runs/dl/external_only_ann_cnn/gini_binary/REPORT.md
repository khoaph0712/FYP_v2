# ANN/CNN Baseline Report

## Purpose
- YOLO is paused for this experiment.
- This run trains object-crop classifiers only: ANN and/or CNN.
- Each model saves weights, logs, curves, confusion matrix, and classification report for end-user/report inspection.

## Dataset
- Data YAML: `C:\FYP_v2\prepared_datasets\gini_binary\data.yaml`
- Source filter: see `run_config.json`.
- Classes: background, garbage
- Train objects: 615
- Validation objects: 93
- Test objects: 95

## Models
- **ANN:** flattened RGB crop pixels; baseline that does not preserve local spatial structure.
- **CNN:** convolutional image classifier; better suited for image texture/shape because it preserves spatial locality.

## Outputs
- `training_log.csv/json`: epoch-wise train/val loss and accuracy.
- `training_curves_*.png`: train/val loss and accuracy figure.
- `*.pt` and `*_full.pt`: saved model weights/full model.
- `confusion_*.png`: classification confusion matrix. No `background` class is used here because this is crop classification, not object detection.
- `classification_report.json`: precision, recall, F1-score, and support by class.

## Results
| Model | Test Accuracy | Test F1-macro | Best Val Accuracy |
|---|---:|---:|---:|
| ann | 0.6211 | 0.3831 | 0.6452 |
| cnn | 0.7895 | 0.7883 | 0.7742 |

## Confusion Matrix Background Note
- For ANN/CNN crop classification, every sample already has a real class, so the matrix only contains dataset classes.
- In YOLO detection confusion matrices, `background` usually means unmatched predictions or missed ground-truth boxes; it is not an eighth trash class.