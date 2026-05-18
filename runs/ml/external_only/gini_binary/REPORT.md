# Feature + ML Analysis Report

## Dataset and classes (examiner checklist)
- **Dataset:** YOLO-format merged dataset at `C:\FYP_v2\prepared_datasets\gini_binary` (config: `C:\FYP_v2\prepared_datasets\gini_binary\data.yaml`).
- **Classes defined in `data.yaml`:** **2** — background, garbage.
- **Classes used in this ML run:** **2** — background, garbage.
- **Class balance:** The **raw** dataset splits can be **imbalanced** (different sources merged into `merged_dataset_v3`). When per-class caps are enabled, this script samples **up to N object crops per class** on train/test so counts are **intentionally balanced at the crop level** where enough boxes exist; see `class_support.json` for exact train/test counts. Rare classes may still fall **below** the cap.

## Handcrafted features used for ML (not raw pixels)
Each object crop is resized to 64x64 internally, then summarized into **637** floats.
- **Why 637 features?** The vector is the fixed concatenation of `8` spatial + `9` frequency/FFT + `44` color + `576` HOG features. This gives a lecturer-explainable representation of brightness/edges, texture frequencies, color distribution, and local shape/orientation instead of giving raw pixels directly to classical ML.
### Spatial domain (8 features)
1. `mean_intensity`
2. `std_intensity`
3. `p10_intensity`
4. `p50_intensity`
5. `p90_intensity`
6. `grad_mean`
7. `grad_std`
8. `edge_density`
### Frequency domain (9 features; FFT radial bins + `high_freq_energy`)
9. `fft_bin_1`
10. `fft_bin_2`
11. `fft_bin_3`
12. `fft_bin_4`
13. `fft_bin_5`
14. `fft_bin_6`
15. `fft_bin_7`
16. `fft_bin_8`
17. `high_freq_energy`
### Color domain (44 features)
- HSV histograms plus BGR/HSV mean and standard deviation.
### HOG texture/shape domain (576 features)
- Histogram of Oriented Gradients descriptor from each 64x64 crop.

## Models trained on extracted features
| Model | Role |
|---|---|
| `decision_tree` | **DecisionTreeClassifier** baseline; easiest tree model to explain. |
| `logreg` | Logistic Regression on **StandardScaler**-normalized features (linear baseline). |
| `linear_svm` | **Linear SVM** on scaled features, suitable for the larger color+HOG vector. |
| `extra_trees` | **ExtraTreesClassifier** (350 trees) - stronger high-dimensional classical baseline. |
| `rf` | **RandomForestClassifier** (250 trees) - tree baseline + feature importance for charts. |
| `xgboost` | **XGBoost** gradient-boosted tree baseline requested in lecturer notes. |

## Figures to include in the thesis / lecturer report
**Classical ML (this folder)**
- `chart_model_comparison.png` — Accuracy & F1-macro across ML models (no epoch-wise loss; ML is not trained by gradient descent here).
- `confusion_decision_tree.png`, `confusion_linear_svm.png`, `confusion_rf.png`, `confusion_xgboost.png` - requested ML confusion matrices where available.
- `classification_reports.json` - precision, recall, F1-score, and support for every class/model.
- `chart_domain_importance.png` — spatial/frequency/color/HOG contribution (RF feature importance).
- `ml/frequency_analysis/` — spatial/frequency summary CSVs + optional spectrum plots.
**Deep learning (separate runs; loss / training curves)**
- `runs/dl/trash_yolov8n_v3/results.png` — Ultralytics training curves (loss, mAP, precision, recall).
- `runs/dl/trash_yolov8n_v3/results.csv` — numeric log for custom plots.
- Run `python scripts/plot_training.py` → writes `training_curves.png` next to the chosen run’s `quality_check/`.
- `runs/dl/dl_baseline/training_loss.png` — tiny CNN baseline loss vs epoch on crops.

## Scope
- Mobile is intentionally excluded in this stage.
- Focus: feature extraction + classical ML model comparison.
- Classes in this run: background, garbage

## Lecture workflow checklist
1. **Spatial + frequency domains** — each crop gets handcrafted **spatial** statistics (intensity, gradients, edges) and **frequency** descriptors (2D FFT radial energy bins + high-frequency summary).
2. **Comment how objects differ** — before judging ML scores, read the per-class notes below and `ml/frequency_analysis/spatial_summary.csv` + `frequency_summary.csv` (class-wise means).
3. **Extract features, then ML** — Decision Tree / SVM / RandomForest / XGBoost are trained **only** on the stacked 637-D feature vectors from step 1 (not raw pixels inside this script).

## Pipeline (implementation order)
1. Crop objects from YOLO boxes; build the fixed-length feature vector per crop.
2. Export domain CSVs + object-difference commentary (spatial/frequency/color/HOG).
3. Fit classical ML on `X_train`; evaluate on `X_test`.

## Data
- Train object crops: **665**
- Test object crops: **129**
- Counts come from a **per-class** cap when enabled (each class can reach the cap independently; this is not a single 4000-total budget split across classes).
- Classes: background, garbage

## Comments: how object classes differ (feature domains)
Each bullet compares that class’s **mean feature vector** to the **global mean** over all training crops: which domain (spatial / frequency / both) shows the largest shift, and which named descriptors move most.
- Compared to the dataset average, **garbage** differs more in the **spatial** domain than in frequency: strongest spatial shifts involve `grad_mean, p10_intensity, grad_std`; strongest frequency shifts involve `fft_bin_1, fft_bin_2, fft_bin_3`.
  - *Scores:* overall `16.6007`, spatial `11.7153`, frequency `0.0289`.
- Compared to the dataset average, **background** differs more in the **spatial** domain than in frequency: strongest spatial shifts involve `grad_mean, p10_intensity, grad_std`; strongest frequency shifts involve `fft_bin_1, fft_bin_2, fft_bin_3`.
  - *Scores:* overall `10.9980`, spatial `7.7613`, frequency `0.0191`.

## ML results (features → models)
| Model | Accuracy | F1-macro |
|---|---:|---:|
| rf | 0.8837 | 0.8516 |
| xgboost | 0.8760 | 0.8484 |
| extra_trees | 0.8682 | 0.8348 |
| decision_tree | 0.8062 | 0.7719 |
| logreg | 0.8140 | 0.7688 |
| linear_svm | 0.7829 | 0.7427 |

## Model choice rationale
- **Decision Tree:** easiest baseline to explain; shows whether simple feature thresholds can separate classes.
- **Logistic Regression:** simple linear baseline on standardized feature vectors.
- **Linear SVM:** scalable margin-based baseline for high-dimensional handcrafted descriptors.
- **Random Forest:** robust tree-based baseline and interpretable feature importance.
- **ExtraTrees:** tree ensemble baseline that often improves over RF on noisy, high-dimensional features.

- **XGBoost:** boosted tree ensemble that tests whether sequential error correction improves the handcrafted-feature baseline.

## Chart comments
- `chart_domain_importance.png`: compares spatial, frequency, color, and HOG contribution based on model feature importance (not raw magnitude, so it is scale-safe).
- `chart_model_comparison.png`: compares Accuracy/F1 across ML models to justify chosen baseline.
