# Feature + ML Analysis Report

## Dataset and classes (examiner checklist)
- **Dataset:** YOLO-format merged dataset at `C:\FYP_v2\prepared_datasets\taco_official_partial` (config: `C:\FYP_v2\prepared_datasets\taco_official_partial\data.yaml`).
- **Classes defined in `data.yaml`:** **60** — Aluminium foil, Battery, Aluminium blister pack, Carded blister pack, Other plastic bottle, Clear plastic bottle, Glass bottle, Plastic bottle cap, Metal bottle cap, Broken glass, Food Can, Aerosol, Drink can, Toilet tube, Other carton, Egg carton, Drink carton, Corrugated carton, Meal carton, Pizza box, Paper cup, Disposable plastic cup, Foam cup, Glass cup, Other plastic cup, Food waste, Glass jar, Plastic lid, Metal lid, Other plastic, Magazine paper, Tissues, Wrapping paper, Normal paper, Paper bag, Plastified paper bag, Plastic film, Six pack rings, Garbage bag, Other plastic wrapper, Single-use carrier bag, Polypropylene bag, Crisp packet, Spread tub, Tupperware, Disposable food container, Foam food container, Other plastic container, Plastic glooves, Plastic utensils, Pop tab, Rope & strings, Scrap metal, Shoe, Squeezable tube, Plastic straw, Paper straw, Styrofoam piece, Unlabeled litter, Cigarette.
- **Classes used in this ML run:** **42** — Aluminium foil, Other plastic bottle, Clear plastic bottle, Glass bottle, Plastic bottle cap, Metal bottle cap, Broken glass, Aerosol, Drink can, Other carton, Drink carton, Corrugated carton, Meal carton, Paper cup, Disposable plastic cup, Foam cup, Food waste, Plastic lid, Metal lid, Other plastic, Tissues, Wrapping paper, Normal paper, Paper bag, Plastic film, Garbage bag, Other plastic wrapper, Single-use carrier bag, Crisp packet, Spread tub, Disposable food container, Foam food container, Plastic glooves, Plastic utensils, Pop tab, Rope & strings, Scrap metal, Squeezable tube, Plastic straw, Styrofoam piece, Unlabeled litter, Cigarette.
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
- Classes in this run: Aluminium foil, Other plastic bottle, Clear plastic bottle, Glass bottle, Plastic bottle cap, Metal bottle cap, Broken glass, Aerosol, Drink can, Other carton, Drink carton, Corrugated carton, Meal carton, Paper cup, Disposable plastic cup, Foam cup, Food waste, Plastic lid, Metal lid, Other plastic, Tissues, Wrapping paper, Normal paper, Paper bag, Plastic film, Garbage bag, Other plastic wrapper, Single-use carrier bag, Crisp packet, Spread tub, Disposable food container, Foam food container, Plastic glooves, Plastic utensils, Pop tab, Rope & strings, Scrap metal, Squeezable tube, Plastic straw, Styrofoam piece, Unlabeled litter, Cigarette

## Lecture workflow checklist
1. **Spatial + frequency domains** — each crop gets handcrafted **spatial** statistics (intensity, gradients, edges) and **frequency** descriptors (2D FFT radial energy bins + high-frequency summary).
2. **Comment how objects differ** — before judging ML scores, read the per-class notes below and `ml/frequency_analysis/spatial_summary.csv` + `frequency_summary.csv` (class-wise means).
3. **Extract features, then ML** — Decision Tree / SVM / RandomForest / XGBoost are trained **only** on the stacked 637-D feature vectors from step 1 (not raw pixels inside this script).

## Pipeline (implementation order)
1. Crop objects from YOLO boxes; build the fixed-length feature vector per crop.
2. Export domain CSVs + object-difference commentary (spatial/frequency/color/HOG).
3. Fit classical ML on `X_train`; evaluate on `X_test`.

## Data
- Train object crops: **2367**
- Test object crops: **393**
- Counts come from a **per-class** cap when enabled (each class can reach the cap independently; this is not a single 4000-total budget split across classes).
- Classes: Aluminium foil, Other plastic bottle, Clear plastic bottle, Glass bottle, Plastic bottle cap, Metal bottle cap, Broken glass, Aerosol, Drink can, Other carton, Drink carton, Corrugated carton, Meal carton, Paper cup, Disposable plastic cup, Foam cup, Food waste, Plastic lid, Metal lid, Other plastic, Tissues, Wrapping paper, Normal paper, Paper bag, Plastic film, Garbage bag, Other plastic wrapper, Single-use carrier bag, Crisp packet, Spread tub, Disposable food container, Foam food container, Plastic glooves, Plastic utensils, Pop tab, Rope & strings, Scrap metal, Squeezable tube, Plastic straw, Styrofoam piece, Unlabeled litter, Cigarette

## Comments: how object classes differ (feature domains)
Each bullet compares that class’s **mean feature vector** to the **global mean** over all training crops: which domain (spatial / frequency / both) shows the largest shift, and which named descriptors move most.
- Compared to the dataset average, **Aluminium blister pack** differs more in the **spatial** domain than in frequency: strongest spatial shifts involve `grad_mean, grad_std, p90_intensity`; strongest frequency shifts involve `fft_bin_1, fft_bin_2, fft_bin_3`.
  - *Scores:* overall `55.2830`, spatial `53.5174`, frequency `0.0178`.
- Compared to the dataset average, **Scrap metal** differs more in the **spatial** domain than in frequency: strongest spatial shifts involve `grad_mean, grad_std, p10_intensity`; strongest frequency shifts involve `fft_bin_1, fft_bin_2, fft_bin_3`.
  - *Scores:* overall `48.6146`, spatial `48.6146`, frequency `0.1090`.
- Compared to the dataset average, **Battery** differs more in the **spatial** domain than in frequency: strongest spatial shifts involve `grad_std, p90_intensity, grad_mean`; strongest frequency shifts involve `fft_bin_1, fft_bin_2, fft_bin_3`.
  - *Scores:* overall `46.2870`, spatial `46.2870`, frequency `0.0531`.
- Compared to the dataset average, **Metal lid** differs more in the **spatial** domain than in frequency: strongest spatial shifts involve `p90_intensity, mean_intensity, p50_intensity`; strongest frequency shifts involve `fft_bin_1, fft_bin_2, fft_bin_3`.
  - *Scores:* overall `43.8364`, spatial `38.7741`, frequency `0.0564`.
- Compared to the dataset average, **Broken glass** differs more in the **spatial** domain than in frequency: strongest spatial shifts involve `p90_intensity, p50_intensity, grad_mean`; strongest frequency shifts involve `fft_bin_1, fft_bin_2, fft_bin_3`.
  - *Scores:* overall `43.5783`, spatial `42.6091`, frequency `0.0632`.
- Compared to the dataset average, **Glass jar** differs more in the **spatial** domain than in frequency: strongest spatial shifts involve `p10_intensity, grad_mean, std_intensity`; strongest frequency shifts involve `fft_bin_1, fft_bin_2, fft_bin_3`.
  - *Scores:* overall `43.1445`, spatial `19.5590`, frequency `0.1013`.
- Compared to the dataset average, **Shoe** differs more in the **spatial** domain than in frequency: strongest spatial shifts involve `p50_intensity, p90_intensity, mean_intensity`; strongest frequency shifts involve `fft_bin_1, fft_bin_2, fft_bin_3`.
  - *Scores:* overall `40.7596`, spatial `39.8198`, frequency `0.0366`.
- Compared to the dataset average, **Garbage bag** differs more in the **spatial** domain than in frequency: strongest spatial shifts involve `p50_intensity, mean_intensity, p90_intensity`; strongest frequency shifts involve `fft_bin_1, fft_bin_2, fft_bin_3`.
  - *Scores:* overall `39.9706`, spatial `37.0292`, frequency `0.0143`.
- Compared to the dataset average, **Toilet tube** differs more in the **spatial** domain than in frequency: strongest spatial shifts involve `grad_mean, grad_std, p90_intensity`; strongest frequency shifts involve `fft_bin_1, fft_bin_2, fft_bin_3`.
  - *Scores:* overall `35.2662`, spatial `26.5502`, frequency `0.0825`.
- Compared to the dataset average, **Carded blister pack** differs more in the **spatial** domain than in frequency: strongest spatial shifts involve `grad_mean, p90_intensity, grad_std`; strongest frequency shifts involve `fft_bin_2, fft_bin_1, fft_bin_3`.
  - *Scores:* overall `32.7640`, spatial `28.4166`, frequency `0.0267`.
- Compared to the dataset average, **Glass cup** differs more in the **spatial** domain than in frequency: strongest spatial shifts involve `p10_intensity, mean_intensity, p90_intensity`; strongest frequency shifts involve `fft_bin_1, high_freq_energy, fft_bin_2`.
  - *Scores:* overall `31.8148`, spatial `14.9225`, frequency `0.0111`.
- Compared to the dataset average, **Plastic glooves** differs more in the **spatial** domain than in frequency: strongest spatial shifts involve `grad_mean, p10_intensity, p50_intensity`; strongest frequency shifts involve `fft_bin_1, fft_bin_2, fft_bin_3`.
  - *Scores:* overall `31.0909`, spatial `28.0533`, frequency `0.1732`.
- Compared to the dataset average, **Tupperware** differs more in the **spatial** domain than in frequency: strongest spatial shifts involve `grad_mean, p90_intensity, grad_std`; strongest frequency shifts involve `fft_bin_1, fft_bin_3, fft_bin_4`.
  - *Scores:* overall `29.0919`, spatial `29.0919`, frequency `0.0311`.
- Compared to the dataset average, **Foam cup** differs more in the **spatial** domain than in frequency: strongest spatial shifts involve `p50_intensity, p90_intensity, mean_intensity`; strongest frequency shifts involve `fft_bin_1, fft_bin_2, fft_bin_3`.
  - *Scores:* overall `28.2324`, spatial `25.2311`, frequency `0.0477`.
- Compared to the dataset average, **Other plastic container** differs more in the **spatial** domain than in frequency: strongest spatial shifts involve `grad_mean, grad_std, p90_intensity`; strongest frequency shifts involve `fft_bin_1, fft_bin_3, fft_bin_4`.
  - *Scores:* overall `28.0032`, spatial `28.0032`, frequency `0.0224`.
- Compared to the dataset average, **Paper straw** differs more in the **spatial** domain than in frequency: strongest spatial shifts involve `grad_std, grad_mean, p10_intensity`; strongest frequency shifts involve `fft_bin_2, fft_bin_1, high_freq_energy`.
  - *Scores:* overall `27.4787`, spatial `27.4706`, frequency `0.0609`.
- Compared to the dataset average, **Other plastic cup** differs more in the **spatial** domain than in frequency: strongest spatial shifts involve `grad_std, p10_intensity, std_intensity`; strongest frequency shifts involve `fft_bin_2, fft_bin_1, high_freq_energy`.
  - *Scores:* overall `27.3047`, spatial `11.9349`, frequency `0.0469`.
- Compared to the dataset average, **Drink can** differs more in the **spatial** domain than in frequency: strongest spatial shifts involve `grad_mean, grad_std, p10_intensity`; strongest frequency shifts involve `fft_bin_1, fft_bin_3, fft_bin_4`.
  - *Scores:* overall `26.5365`, spatial `26.5365`, frequency `0.0076`.
- Compared to the dataset average, **Polypropylene bag** differs more in the **spatial** domain than in frequency: strongest spatial shifts involve `grad_std, p10_intensity, p50_intensity`; strongest frequency shifts involve `fft_bin_1, fft_bin_2, fft_bin_3`.
  - *Scores:* overall `24.7052`, spatial `16.2668`, frequency `0.0439`.
- Compared to the dataset average, **Food waste** differs more in the **spatial** domain than in frequency: strongest spatial shifts involve `grad_mean, p10_intensity, p50_intensity`; strongest frequency shifts involve `fft_bin_1, fft_bin_2, fft_bin_3`.
  - *Scores:* overall `24.4276`, spatial `13.9329`, frequency `0.0620`.
- Compared to the dataset average, **Magazine paper** differs more in the **spatial** domain than in frequency: strongest spatial shifts involve `grad_mean, grad_std, p10_intensity`; strongest frequency shifts involve `fft_bin_1, high_freq_energy, fft_bin_2`.
  - *Scores:* overall `24.2227`, spatial `24.2227`, frequency `0.1163`.
- Compared to the dataset average, **Egg carton** differs more in the **spatial** domain than in frequency: strongest spatial shifts involve `p90_intensity, grad_mean, p50_intensity`; strongest frequency shifts involve `fft_bin_1, high_freq_energy, fft_bin_2`.
  - *Scores:* overall `23.5275`, spatial `19.1103`, frequency `0.0167`.
- Compared to the dataset average, **Foam food container** differs more in the **spatial** domain than in frequency: strongest spatial shifts involve `grad_std, p50_intensity, p90_intensity`; strongest frequency shifts involve `fft_bin_1, fft_bin_3, high_freq_energy`.
  - *Scores:* overall `23.2932`, spatial `23.2932`, frequency `0.0127`.
- Compared to the dataset average, **Cigarette** differs more in the **spatial** domain than in frequency: strongest spatial shifts involve `grad_mean, grad_std, p10_intensity`; strongest frequency shifts involve `fft_bin_1, fft_bin_3, high_freq_energy`.
  - *Scores:* overall `22.4392`, spatial `22.4392`, frequency `0.0501`.
- Compared to the dataset average, **Food Can** differs more in the **spatial** domain than in frequency: strongest spatial shifts involve `p10_intensity, p50_intensity, mean_intensity`; strongest frequency shifts involve `fft_bin_1, high_freq_energy, fft_bin_3`.
  - *Scores:* overall `22.3574`, spatial `12.2794`, frequency `0.0052`.
- Compared to the dataset average, **Meal carton** differs more in the **spatial** domain than in frequency: strongest spatial shifts involve `grad_mean, p90_intensity, grad_std`; strongest frequency shifts involve `fft_bin_1, high_freq_energy, fft_bin_3`.
  - *Scores:* overall `21.4345`, spatial `21.4345`, frequency `0.0472`.
- Compared to the dataset average, **Aerosol** differs more in the **spatial** domain than in frequency: strongest spatial shifts involve `grad_mean, grad_std, p10_intensity`; strongest frequency shifts involve `fft_bin_1, fft_bin_3, fft_bin_4`.
  - *Scores:* overall `20.6966`, spatial `17.9402`, frequency `0.0231`.
- Compared to the dataset average, **Six pack rings** differs more in the **spatial** domain than in frequency: strongest spatial shifts involve `grad_std, p10_intensity, grad_mean`; strongest frequency shifts involve `high_freq_energy, fft_bin_2, fft_bin_1`.
  - *Scores:* overall `20.5893`, spatial `17.1527`, frequency `0.0116`.
- Compared to the dataset average, **Pizza box** differs more in the **spatial** domain than in frequency: strongest spatial shifts involve `p90_intensity, p10_intensity, mean_intensity`; strongest frequency shifts involve `fft_bin_1, fft_bin_2, fft_bin_3`.
  - *Scores:* overall `19.3221`, spatial `13.8362`, frequency `0.0321`.
- Compared to the dataset average, **Wrapping paper** differs more in the **spatial** domain than in frequency: strongest spatial shifts involve `grad_mean, grad_std, p90_intensity`; strongest frequency shifts involve `fft_bin_1, high_freq_energy, fft_bin_2`.
  - *Scores:* overall `19.2874`, spatial `17.1214`, frequency `0.0085`.
- Compared to the dataset average, **Glass bottle** differs more in the **spatial** domain than in frequency: strongest spatial shifts involve `p50_intensity, mean_intensity, p90_intensity`; strongest frequency shifts involve `fft_bin_1, high_freq_energy, fft_bin_4`.
  - *Scores:* overall `18.4852`, spatial `16.8163`, frequency `0.0233`.
- Compared to the dataset average, **Plastic bottle cap** differs more in the **spatial** domain than in frequency: strongest spatial shifts involve `grad_mean, grad_std, p10_intensity`; strongest frequency shifts involve `fft_bin_1, high_freq_energy, fft_bin_4`.
  - *Scores:* overall `18.2592`, spatial `16.7808`, frequency `0.0189`.
- Compared to the dataset average, **Plastic utensils** differs more in the **spatial** domain than in frequency: strongest spatial shifts involve `grad_mean, grad_std, p90_intensity`; strongest frequency shifts involve `fft_bin_1, fft_bin_2, fft_bin_3`.
  - *Scores:* overall `18.1989`, spatial `18.1989`, frequency `0.0181`.
- Compared to the dataset average, **Crisp packet** differs more in the **spatial** domain than in frequency: strongest spatial shifts involve `p90_intensity, mean_intensity, p10_intensity`; strongest frequency shifts involve `fft_bin_2, high_freq_energy, fft_bin_6`.
  - *Scores:* overall `17.8833`, spatial `14.3161`, frequency `0.0114`.
- Compared to the dataset average, **Paper cup** differs more in the **spatial** domain than in frequency: strongest spatial shifts involve `mean_intensity, p50_intensity, p10_intensity`; strongest frequency shifts involve `fft_bin_1, fft_bin_2, fft_bin_3`.
  - *Scores:* overall `17.3412`, spatial `13.3772`, frequency `0.0149`.
- Compared to the dataset average, **Styrofoam piece** differs more in the **spatial** domain than in frequency: strongest spatial shifts involve `p50_intensity, p90_intensity, mean_intensity`; strongest frequency shifts involve `fft_bin_1, fft_bin_2, fft_bin_3`.
  - *Scores:* overall `15.8018`, spatial `14.0609`, frequency `0.0195`.
- Compared to the dataset average, **Other plastic wrapper** differs more in the **spatial** domain than in frequency: strongest spatial shifts involve `grad_mean, grad_std, p10_intensity`; strongest frequency shifts involve `fft_bin_1, fft_bin_2, fft_bin_3`.
  - *Scores:* overall `15.0702`, spatial `14.8191`, frequency `0.0202`.
- Compared to the dataset average, **Rope & strings** differs more in the **spatial** domain than in frequency: strongest spatial shifts involve `grad_std, p90_intensity, p10_intensity`; strongest frequency shifts involve `fft_bin_1, high_freq_energy, fft_bin_2`.
  - *Scores:* overall `14.6842`, spatial `14.6842`, frequency `0.0374`.
- Compared to the dataset average, **Disposable food container** differs more in the **spatial** domain than in frequency: strongest spatial shifts involve `p90_intensity, p50_intensity, mean_intensity`; strongest frequency shifts involve `fft_bin_1, fft_bin_2, fft_bin_3`.
  - *Scores:* overall `14.3533`, spatial `12.2321`, frequency `0.0125`.
- Compared to the dataset average, **Other carton** differs more in the **spatial** domain than in frequency: strongest spatial shifts involve `p90_intensity, p50_intensity, grad_mean`; strongest frequency shifts involve `fft_bin_1, high_freq_energy, fft_bin_4`.
  - *Scores:* overall `13.1632`, spatial `12.5390`, frequency `0.0090`.
- Compared to the dataset average, **Spread tub** differs more in the **spatial** domain than in frequency: strongest spatial shifts involve `grad_mean, p90_intensity, std_intensity`; strongest frequency shifts involve `fft_bin_1, high_freq_energy, fft_bin_2`.
  - *Scores:* overall `11.3167`, spatial `7.8002`, frequency `0.0377`.
- Compared to the dataset average, **Clear plastic bottle** differs more in the **spatial** domain than in frequency: strongest spatial shifts involve `grad_std, p10_intensity, grad_mean`; strongest frequency shifts involve `fft_bin_1, high_freq_energy, fft_bin_4`.
  - *Scores:* overall `11.2789`, spatial `11.2789`, frequency `0.0266`.
- Compared to the dataset average, **Squeezable tube** differs more in the **spatial** domain than in frequency: strongest spatial shifts involve `p50_intensity, grad_std, mean_intensity`; strongest frequency shifts involve `fft_bin_1, high_freq_energy, fft_bin_4`.
  - *Scores:* overall `10.9949`, spatial `10.0012`, frequency `0.0395`.
- Compared to the dataset average, **Other plastic bottle** differs more in the **spatial** domain than in frequency: strongest spatial shifts involve `p90_intensity, grad_std, std_intensity`; strongest frequency shifts involve `fft_bin_1, fft_bin_2, fft_bin_3`.
  - *Scores:* overall `10.8042`, spatial `10.0422`, frequency `0.0116`.
- Compared to the dataset average, **Unlabeled litter** differs more in the **spatial** domain than in frequency: strongest spatial shifts involve `p10_intensity, p50_intensity, mean_intensity`; strongest frequency shifts involve `fft_bin_1, high_freq_energy, fft_bin_5`.
  - *Scores:* overall `10.6889`, spatial `10.0495`, frequency `0.0103`.
- Compared to the dataset average, **Plastic lid** differs more in the **spatial** domain than in frequency: strongest spatial shifts involve `p10_intensity, grad_mean, p50_intensity`; strongest frequency shifts involve `fft_bin_1, fft_bin_2, fft_bin_3`.
  - *Scores:* overall `10.6312`, spatial `10.6312`, frequency `0.0255`.
- Compared to the dataset average, **Corrugated carton** differs more in the **spatial** domain than in frequency: strongest spatial shifts involve `grad_std, grad_mean, p10_intensity`; strongest frequency shifts involve `fft_bin_2, fft_bin_1, high_freq_energy`.
  - *Scores:* overall `10.5452`, spatial `9.0857`, frequency `0.0097`.
- Compared to the dataset average, **Plastic straw** differs more in the **spatial** domain than in frequency: strongest spatial shifts involve `grad_mean, grad_std, p10_intensity`; strongest frequency shifts involve `fft_bin_1, fft_bin_2, high_freq_energy`.
  - *Scores:* overall `9.2877`, spatial `9.2877`, frequency `0.0151`.
- Compared to the dataset average, **Plastic film** differs more in the **spatial** domain than in frequency: strongest spatial shifts involve `grad_mean, p50_intensity, mean_intensity`; strongest frequency shifts involve `fft_bin_1, high_freq_energy, fft_bin_3`.
  - *Scores:* overall `9.1242`, spatial `7.9274`, frequency `0.0084`.
- Compared to the dataset average, **Aluminium foil** differs more in the **spatial** domain than in frequency: strongest spatial shifts involve `p10_intensity, grad_std, mean_intensity`; strongest frequency shifts involve `fft_bin_1, fft_bin_3, high_freq_energy`.
  - *Scores:* overall `8.9769`, spatial `8.0287`, frequency `0.0045`.
- Compared to the dataset average, **Single-use carrier bag** differs more in the **spatial** domain than in frequency: strongest spatial shifts involve `p90_intensity, p50_intensity, grad_std`; strongest frequency shifts involve `fft_bin_1, fft_bin_2, fft_bin_4`.
  - *Scores:* overall `8.8695`, spatial `8.7023`, frequency `0.0185`.
- Compared to the dataset average, **Pop tab** differs more in the **spatial** domain than in frequency: strongest spatial shifts involve `p10_intensity, p90_intensity, grad_std`; strongest frequency shifts involve `fft_bin_2, high_freq_energy, fft_bin_4`.
  - *Scores:* overall `8.6446`, spatial `5.0379`, frequency `0.0231`.
- Compared to the dataset average, **Tissues** differs more in the **spatial** domain than in frequency: strongest spatial shifts involve `p50_intensity, p10_intensity, grad_std`; strongest frequency shifts involve `high_freq_energy, fft_bin_1, fft_bin_5`.
  - *Scores:* overall `8.3052`, spatial `8.3052`, frequency `0.0047`.
- Compared to the dataset average, **Paper bag** differs more in the **spatial** domain than in frequency: strongest spatial shifts involve `grad_std, p10_intensity, std_intensity`; strongest frequency shifts involve `fft_bin_2, fft_bin_1, high_freq_energy`.
  - *Scores:* overall `8.0377`, spatial `3.7746`, frequency `0.0066`.
- Compared to the dataset average, **Normal paper** differs more in the **spatial** domain than in frequency: strongest spatial shifts involve `p50_intensity, grad_mean, p10_intensity`; strongest frequency shifts involve `fft_bin_1, high_freq_energy, fft_bin_4`.
  - *Scores:* overall `7.8249`, spatial `6.6251`, frequency `0.0054`.
- Compared to the dataset average, **Disposable plastic cup** differs more in the **spatial** domain than in frequency: strongest spatial shifts involve `p10_intensity, p50_intensity, mean_intensity`; strongest frequency shifts involve `fft_bin_1, fft_bin_3, high_freq_energy`.
  - *Scores:* overall `7.6057`, spatial `5.6910`, frequency `0.0175`.
- Compared to the dataset average, **Metal bottle cap** differs more in the **spatial** domain than in frequency: strongest spatial shifts involve `p50_intensity, mean_intensity, grad_mean`; strongest frequency shifts involve `fft_bin_1, high_freq_energy, fft_bin_2`.
  - *Scores:* overall `7.2802`, spatial `1.4503`, frequency `0.0178`.
- Compared to the dataset average, **Drink carton** differs more in the **spatial** domain than in frequency: strongest spatial shifts involve `grad_std, p50_intensity, p10_intensity`; strongest frequency shifts involve `fft_bin_2, high_freq_energy, fft_bin_1`.
  - *Scores:* overall `7.1114`, spatial `2.3863`, frequency `0.0079`.
- Compared to the dataset average, **Other plastic** differs more in the **spatial** domain than in frequency: strongest spatial shifts involve `p90_intensity, grad_mean, mean_intensity`; strongest frequency shifts involve `fft_bin_1, fft_bin_2, fft_bin_3`.
  - *Scores:* overall `6.3465`, spatial `5.4406`, frequency `0.0279`.

## ML results (features → models)
| Model | Accuracy | F1-macro |
|---|---:|---:|
| xgboost | 0.1399 | 0.0675 |
| logreg | 0.1145 | 0.0479 |
| extra_trees | 0.1349 | 0.0457 |
| rf | 0.1552 | 0.0411 |
| linear_svm | 0.0840 | 0.0368 |
| decision_tree | 0.0611 | 0.0278 |

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
