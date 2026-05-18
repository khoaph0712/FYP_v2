# ANN/CNN Baseline Report

## Purpose
- YOLO is paused for this experiment.
- This run trains object-crop classifiers only: ANN and/or CNN.
- Each model saves weights, logs, curves, confusion matrix, and classification report for end-user/report inspection.

## Dataset
- Data YAML: `C:\FYP_v2\prepared_datasets\taco_official_partial\data.yaml`
- Source filter: see `run_config.json`.
- Classes: Aluminium foil, Other plastic bottle, Clear plastic bottle, Glass bottle, Plastic bottle cap, Metal bottle cap, Broken glass, Aerosol, Drink can, Other carton, Drink carton, Corrugated carton, Meal carton, Paper cup, Disposable plastic cup, Foam cup, Food waste, Plastic lid, Other plastic, Tissues, Wrapping paper, Normal paper, Paper bag, Plastic film, Garbage bag, Other plastic wrapper, Single-use carrier bag, Crisp packet, Spread tub, Disposable food container, Foam food container, Plastic utensils, Pop tab, Rope & strings, Scrap metal, Plastic straw, Styrofoam piece, Unlabeled litter, Cigarette
- Train objects: 970
- Validation objects: 224
- Test objects: 222

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
| ann | 0.0225 | 0.0020 | 0.0402 |
| cnn | 0.0586 | 0.0124 | 0.0580 |

## Confusion Matrix Background Note
- For ANN/CNN crop classification, every sample already has a real class, so the matrix only contains dataset classes.
- In YOLO detection confusion matrices, `background` usually means unmatched predictions or missed ground-truth boxes; it is not an eighth trash class.