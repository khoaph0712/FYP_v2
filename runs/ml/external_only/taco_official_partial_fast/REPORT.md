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
- Train object crops: **1072**
- Test object crops: **227**
- Counts come from a **per-class** cap when enabled (each class can reach the cap independently; this is not a single 4000-total budget split across classes).
- Classes: Aluminium foil, Other plastic bottle, Clear plastic bottle, Glass bottle, Plastic bottle cap, Metal bottle cap, Broken glass, Aerosol, Drink can, Other carton, Drink carton, Corrugated carton, Meal carton, Paper cup, Disposable plastic cup, Foam cup, Food waste, Plastic lid, Metal lid, Other plastic, Tissues, Wrapping paper, Normal paper, Paper bag, Plastic film, Garbage bag, Other plastic wrapper, Single-use carrier bag, Crisp packet, Spread tub, Disposable food container, Foam food container, Plastic glooves, Plastic utensils, Pop tab, Rope & strings, Scrap metal, Squeezable tube, Plastic straw, Styrofoam piece, Unlabeled litter, Cigarette

## Comments: how object classes differ (feature domains)
Each bullet compares that class’s **mean feature vector** to the **global mean** over all training crops: which domain (spatial / frequency / both) shows the largest shift, and which named descriptors move most.
- Compared to the dataset average, **Aluminium blister pack** differs more in the **spatial** domain than in frequency: strongest spatial shifts involve `grad_mean, grad_std, p90_intensity`; strongest frequency shifts involve `fft_bin_2, fft_bin_1, high_freq_energy`.
  - *Scores:* overall `54.8707`, spatial `53.1330`, frequency `0.0161`.
- Compared to the dataset average, **Broken glass** differs more in the **spatial** domain than in frequency: strongest spatial shifts involve `p50_intensity, mean_intensity, p90_intensity`; strongest frequency shifts involve `fft_bin_1, fft_bin_2, fft_bin_3`.
  - *Scores:* overall `52.3747`, spatial `50.8792`, frequency `0.1114`.
- Compared to the dataset average, **Scrap metal** differs more in the **spatial** domain than in frequency: strongest spatial shifts involve `grad_mean, grad_std, p10_intensity`; strongest frequency shifts involve `fft_bin_1, fft_bin_2, fft_bin_3`.
  - *Scores:* overall `47.8512`, spatial `47.8512`, frequency `0.1068`.
- Compared to the dataset average, **Battery** differs more in the **spatial** domain than in frequency: strongest spatial shifts involve `grad_std, p90_intensity, grad_mean`; strongest frequency shifts involve `fft_bin_1, fft_bin_2, fft_bin_3`.
  - *Scores:* overall `45.9026`, spatial `45.9026`, frequency `0.0554`.
- Compared to the dataset average, **Metal lid** differs more in the **spatial** domain than in frequency: strongest spatial shifts involve `p90_intensity, mean_intensity, p10_intensity`; strongest frequency shifts involve `fft_bin_1, fft_bin_2, fft_bin_3`.
  - *Scores:* overall `42.6944`, spatial `37.9824`, frequency `0.0586`.
- Compared to the dataset average, **Glass jar** differs more in the **spatial** domain than in frequency: strongest spatial shifts involve `p10_intensity, grad_mean, std_intensity`; strongest frequency shifts involve `fft_bin_1, fft_bin_2, fft_bin_3`.
  - *Scores:* overall `41.7328`, spatial `19.3840`, frequency `0.1035`.
- Compared to the dataset average, **Shoe** differs more in the **spatial** domain than in frequency: strongest spatial shifts involve `p50_intensity, p90_intensity, mean_intensity`; strongest frequency shifts involve `fft_bin_1, fft_bin_3, fft_bin_2`.
  - *Scores:* overall `40.1170`, spatial `39.0256`, frequency `0.0389`.
- Compared to the dataset average, **Garbage bag** differs more in the **spatial** domain than in frequency: strongest spatial shifts involve `p50_intensity, mean_intensity, p90_intensity`; strongest frequency shifts involve `fft_bin_1, fft_bin_2, fft_bin_3`.
  - *Scores:* overall `39.3280`, spatial `36.2350`, frequency `0.0165`.
- Compared to the dataset average, **Toilet tube** differs more in the **spatial** domain than in frequency: strongest spatial shifts involve `grad_mean, grad_std, p90_intensity`; strongest frequency shifts involve `fft_bin_1, fft_bin_2, fft_bin_3`.
  - *Scores:* overall `34.9171`, spatial `26.9345`, frequency `0.0847`.
- Compared to the dataset average, **Carded blister pack** differs more in the **spatial** domain than in frequency: strongest spatial shifts involve `grad_mean, p90_intensity, grad_std`; strongest frequency shifts involve `fft_bin_2, fft_bin_1, fft_bin_3`.
  - *Scores:* overall `33.1131`, spatial `28.0323`, frequency `0.0258`.
- Compared to the dataset average, **Glass cup** differs more in the **spatial** domain than in frequency: strongest spatial shifts involve `p10_intensity, mean_intensity, p90_intensity`; strongest frequency shifts involve `fft_bin_1, fft_bin_2, high_freq_energy`.
  - *Scores:* overall `32.7891`, spatial `15.5941`, frequency `0.0075`.
- Compared to the dataset average, **Plastic glooves** differs more in the **spatial** domain than in frequency: strongest spatial shifts involve `grad_mean, p10_intensity, p50_intensity`; strongest frequency shifts involve `fft_bin_1, fft_bin_2, fft_bin_3`.
  - *Scores:* overall `30.4115`, spatial `28.5484`, frequency `0.1709`.
- Compared to the dataset average, **Tupperware** differs more in the **spatial** domain than in frequency: strongest spatial shifts involve `grad_mean, p90_intensity, grad_std`; strongest frequency shifts involve `fft_bin_1, fft_bin_3, high_freq_energy`.
  - *Scores:* overall `29.4762`, spatial `29.4762`, frequency `0.0347`.
- Compared to the dataset average, **Foam cup** differs more in the **spatial** domain than in frequency: strongest spatial shifts involve `p50_intensity, p90_intensity, mean_intensity`; strongest frequency shifts involve `fft_bin_1, fft_bin_2, fft_bin_3`.
  - *Scores:* overall `29.3403`, spatial `26.0253`, frequency `0.0499`.
- Compared to the dataset average, **Plastic bottle cap** differs more in the **spatial** domain than in frequency: strongest spatial shifts involve `grad_mean, grad_std, p10_intensity`; strongest frequency shifts involve `fft_bin_1, high_freq_energy, fft_bin_3`.
  - *Scores:* overall `29.1304`, spatial `29.1304`, frequency `0.0333`.
- Compared to the dataset average, **Cigarette** differs more in the **spatial** domain than in frequency: strongest spatial shifts involve `grad_mean, grad_std, p10_intensity`; strongest frequency shifts involve `fft_bin_1, fft_bin_2, fft_bin_3`.
  - *Scores:* overall `28.6830`, spatial `26.1446`, frequency `0.0669`.
- Compared to the dataset average, **Other plastic container** differs more in the **spatial** domain than in frequency: strongest spatial shifts involve `grad_mean, grad_std, p90_intensity`; strongest frequency shifts involve `fft_bin_1, fft_bin_3, fft_bin_2`.
  - *Scores:* overall `27.6188`, spatial `27.6188`, frequency `0.0191`.
- Compared to the dataset average, **Other plastic cup** differs more in the **spatial** domain than in frequency: strongest spatial shifts involve `grad_std, p10_intensity, std_intensity`; strongest frequency shifts involve `fft_bin_2, fft_bin_1, high_freq_energy`.
  - *Scores:* overall `27.5810`, spatial `11.9103`, frequency `0.0465`.
- Compared to the dataset average, **Paper straw** differs more in the **spatial** domain than in frequency: strongest spatial shifts involve `grad_std, grad_mean, p10_intensity`; strongest frequency shifts involve `fft_bin_2, fft_bin_1, high_freq_energy`.
  - *Scores:* overall `26.7611`, spatial `26.7072`, frequency `0.0605`.
- Compared to the dataset average, **Polypropylene bag** differs more in the **spatial** domain than in frequency: strongest spatial shifts involve `grad_std, p10_intensity, p50_intensity`; strongest frequency shifts involve `fft_bin_1, fft_bin_2, high_freq_energy`.
  - *Scores:* overall `24.3367`, spatial `15.2701`, frequency `0.0461`.
- Compared to the dataset average, **Drink can** differs more in the **spatial** domain than in frequency: strongest spatial shifts involve `grad_mean, grad_std, p90_intensity`; strongest frequency shifts involve `fft_bin_1, high_freq_energy, fft_bin_4`.
  - *Scores:* overall `24.2802`, spatial `24.2802`, frequency `0.0091`.
- Compared to the dataset average, **Magazine paper** differs more in the **spatial** domain than in frequency: strongest spatial shifts involve `grad_mean, grad_std, p10_intensity`; strongest frequency shifts involve `fft_bin_1, high_freq_energy, fft_bin_2`.
  - *Scores:* overall `23.9829`, spatial `23.9829`, frequency `0.1135`.
- Compared to the dataset average, **Food waste** differs more in the **spatial** domain than in frequency: strongest spatial shifts involve `grad_mean, p10_intensity, p50_intensity`; strongest frequency shifts involve `fft_bin_1, fft_bin_2, fft_bin_3`.
  - *Scores:* overall `23.5160`, spatial `13.4378`, frequency `0.0642`.
- Compared to the dataset average, **Foam food container** differs more in the **spatial** domain than in frequency: strongest spatial shifts involve `grad_std, p50_intensity, p90_intensity`; strongest frequency shifts involve `fft_bin_1, fft_bin_3, fft_bin_4`.
  - *Scores:* overall `23.4443`, spatial `23.4443`, frequency `0.0089`.
- Compared to the dataset average, **Egg carton** differs more in the **spatial** domain than in frequency: strongest spatial shifts involve `grad_mean, p90_intensity, grad_std`; strongest frequency shifts involve `fft_bin_1, fft_bin_2, high_freq_energy`.
  - *Scores:* overall `23.0505`, spatial `19.3831`, frequency `0.0139`.
- Compared to the dataset average, **Glass bottle** differs more in the **spatial** domain than in frequency: strongest spatial shifts involve `p50_intensity, mean_intensity, p90_intensity`; strongest frequency shifts involve `fft_bin_1, high_freq_energy, fft_bin_3`.
  - *Scores:* overall `22.2859`, spatial `20.3995`, frequency `0.0336`.
- Compared to the dataset average, **Meal carton** differs more in the **spatial** domain than in frequency: strongest spatial shifts involve `grad_mean, p50_intensity, p90_intensity`; strongest frequency shifts involve `fft_bin_1, high_freq_energy, fft_bin_3`.
  - *Scores:* overall `21.6183`, spatial `21.6183`, frequency `0.0434`.
- Compared to the dataset average, **Food Can** differs more in the **spatial** domain than in frequency: strongest spatial shifts involve `p10_intensity, p50_intensity, grad_std`; strongest frequency shifts involve `fft_bin_2, fft_bin_1, high_freq_energy`.
  - *Scores:* overall `20.8506`, spatial `9.3320`, frequency `0.0046`.
- Compared to the dataset average, **Six pack rings** differs more in the **spatial** domain than in frequency: strongest spatial shifts involve `grad_std, p10_intensity, grad_mean`; strongest frequency shifts involve `high_freq_energy, fft_bin_1, fft_bin_2`.
  - *Scores:* overall `20.4855`, spatial `17.9161`, frequency `0.0152`.
- Compared to the dataset average, **Aerosol** differs more in the **spatial** domain than in frequency: strongest spatial shifts involve `grad_mean, grad_std, p10_intensity`; strongest frequency shifts involve `fft_bin_1, fft_bin_3, fft_bin_4`.
  - *Scores:* overall `20.4821`, spatial `18.1801`, frequency `0.0264`.
- Compared to the dataset average, **Wrapping paper** differs more in the **spatial** domain than in frequency: strongest spatial shifts involve `grad_mean, p90_intensity, grad_std`; strongest frequency shifts involve `fft_bin_2, fft_bin_1, high_freq_energy`.
  - *Scores:* overall `19.8319`, spatial `16.7371`, frequency `0.0057`.
- Compared to the dataset average, **Styrofoam piece** differs more in the **spatial** domain than in frequency: strongest spatial shifts involve `p50_intensity, p90_intensity, grad_std`; strongest frequency shifts involve `fft_bin_1, fft_bin_2, high_freq_energy`.
  - *Scores:* overall `19.4431`, spatial `17.4623`, frequency `0.0140`.
- Compared to the dataset average, **Other plastic** differs more in the **spatial** domain than in frequency: strongest spatial shifts involve `grad_std, grad_mean, p90_intensity`; strongest frequency shifts involve `fft_bin_1, high_freq_energy, fft_bin_4`.
  - *Scores:* overall `18.7773`, spatial `18.7773`, frequency `0.0380`.
- Compared to the dataset average, **Crisp packet** differs more in the **spatial** domain than in frequency: strongest spatial shifts involve `p90_intensity, mean_intensity, p10_intensity`; strongest frequency shifts involve `fft_bin_2, high_freq_energy, fft_bin_1`.
  - *Scores:* overall `18.2588`, spatial `14.9877`, frequency `0.0100`.
- Compared to the dataset average, **Pizza box** differs more in the **spatial** domain than in frequency: strongest spatial shifts involve `p90_intensity, p10_intensity, grad_mean`; strongest frequency shifts involve `fft_bin_1, fft_bin_2, fft_bin_3`.
  - *Scores:* overall `17.8824`, spatial `13.1724`, frequency `0.0344`.
- Compared to the dataset average, **Plastic utensils** differs more in the **spatial** domain than in frequency: strongest spatial shifts involve `grad_mean, grad_std, p90_intensity`; strongest frequency shifts involve `fft_bin_1, fft_bin_2, fft_bin_3`.
  - *Scores:* overall `17.8145`, spatial `17.8145`, frequency `0.0159`.
- Compared to the dataset average, **Paper cup** differs more in the **spatial** domain than in frequency: strongest spatial shifts involve `p10_intensity, p50_intensity, grad_mean`; strongest frequency shifts involve `fft_bin_1, fft_bin_2, high_freq_energy`.
  - *Scores:* overall `17.1968`, spatial `13.9663`, frequency `0.0143`.
- Compared to the dataset average, **Unlabeled litter** differs more in the **spatial** domain than in frequency: strongest spatial shifts involve `p50_intensity, mean_intensity, p90_intensity`; strongest frequency shifts involve `fft_bin_1, high_freq_energy, fft_bin_4`.
  - *Scores:* overall `16.0833`, spatial `13.7665`, frequency `0.0182`.
- Compared to the dataset average, **Other carton** differs more in the **spatial** domain than in frequency: strongest spatial shifts involve `p50_intensity, p90_intensity, mean_intensity`; strongest frequency shifts involve `fft_bin_1, fft_bin_2, high_freq_energy`.
  - *Scores:* overall `15.9034`, spatial `15.5558`, frequency `0.0237`.
- Compared to the dataset average, **Plastic film** differs more in the **spatial** domain than in frequency: strongest spatial shifts involve `grad_mean, p50_intensity, grad_std`; strongest frequency shifts involve `fft_bin_1, fft_bin_2, fft_bin_3`.
  - *Scores:* overall `15.7429`, spatial `14.4746`, frequency `0.0244`.
- Compared to the dataset average, **Rope & strings** differs more in the **spatial** domain than in frequency: strongest spatial shifts involve `grad_std, p90_intensity, p10_intensity`; strongest frequency shifts involve `fft_bin_1, high_freq_energy, fft_bin_2`.
  - *Scores:* overall `15.1793`, spatial `15.1793`, frequency `0.0347`.
- Compared to the dataset average, **Plastic straw** differs more in the **spatial** domain than in frequency: strongest spatial shifts involve `grad_mean, grad_std, p50_intensity`; strongest frequency shifts involve `fft_bin_1, fft_bin_3, high_freq_energy`.
  - *Scores:* overall `14.1701`, spatial `14.0013`, frequency `0.0400`.
- Compared to the dataset average, **Disposable food container** differs more in the **spatial** domain than in frequency: strongest spatial shifts involve `p90_intensity, grad_mean, mean_intensity`; strongest frequency shifts involve `fft_bin_1, fft_bin_2, fft_bin_3`.
  - *Scores:* overall `13.3001`, spatial `11.5268`, frequency `0.0103`.
- Compared to the dataset average, **Metal bottle cap** differs more in the **spatial** domain than in frequency: strongest spatial shifts involve `grad_mean, p50_intensity, mean_intensity`; strongest frequency shifts involve `fft_bin_1, high_freq_energy, fft_bin_2`.
  - *Scores:* overall `12.1857`, spatial `3.3798`, frequency `0.0104`.
- Compared to the dataset average, **Spread tub** differs more in the **spatial** domain than in frequency: strongest spatial shifts involve `grad_mean, p90_intensity, std_intensity`; strongest frequency shifts involve `fft_bin_1, high_freq_energy, fft_bin_2`.
  - *Scores:* overall `11.3408`, spatial `7.8981`, frequency `0.0341`.
- Compared to the dataset average, **Other plastic bottle** differs more in the **spatial** domain than in frequency: strongest spatial shifts involve `p90_intensity, grad_mean, grad_std`; strongest frequency shifts involve `fft_bin_1, fft_bin_2, fft_bin_3`.
  - *Scores:* overall `10.8599`, spatial `10.4849`, frequency `0.0052`.
- Compared to the dataset average, **Clear plastic bottle** differs more in the **spatial** domain than in frequency: strongest spatial shifts involve `grad_std, p10_intensity, p90_intensity`; strongest frequency shifts involve `fft_bin_1, high_freq_energy, fft_bin_4`.
  - *Scores:* overall `10.7204`, spatial `10.7204`, frequency `0.0104`.
- Compared to the dataset average, **Squeezable tube** differs more in the **spatial** domain than in frequency: strongest spatial shifts involve `p50_intensity, grad_std, mean_intensity`; strongest frequency shifts involve `fft_bin_1, high_freq_energy, fft_bin_4`.
  - *Scores:* overall `10.7096`, spatial `10.3275`, frequency `0.0357`.
- Compared to the dataset average, **Aluminium foil** differs more in the **spatial** domain than in frequency: strongest spatial shifts involve `grad_mean, p50_intensity, mean_intensity`; strongest frequency shifts involve `fft_bin_1, fft_bin_2, fft_bin_3`.
  - *Scores:* overall `10.3405`, spatial `6.7001`, frequency `0.0238`.
- Compared to the dataset average, **Single-use carrier bag** differs more in the **spatial** domain than in frequency: strongest spatial shifts involve `p90_intensity, grad_std, p50_intensity`; strongest frequency shifts involve `fft_bin_1, fft_bin_2, fft_bin_3`.
  - *Scores:* overall `9.5645`, spatial `8.8803`, frequency `0.0251`.
- Compared to the dataset average, **Plastic lid** differs more in the **spatial** domain than in frequency: strongest spatial shifts involve `p10_intensity, p50_intensity, mean_intensity`; strongest frequency shifts involve `fft_bin_1, fft_bin_2, fft_bin_3`.
  - *Scores:* overall `9.1741`, spatial `8.9739`, frequency `0.0278`.
- Compared to the dataset average, **Corrugated carton** differs more in the **spatial** domain than in frequency: strongest spatial shifts involve `grad_std, grad_mean, p10_intensity`; strongest frequency shifts involve `fft_bin_1, high_freq_energy, fft_bin_5`.
  - *Scores:* overall `8.5540`, spatial `6.3226`, frequency `0.0096`.
- Compared to the dataset average, **Normal paper** differs more in the **spatial** domain than in frequency: strongest spatial shifts involve `grad_mean, p50_intensity, grad_std`; strongest frequency shifts involve `fft_bin_1, high_freq_energy, fft_bin_3`.
  - *Scores:* overall `8.5495`, spatial `7.3090`, frequency `0.0067`.
- Compared to the dataset average, **Tissues** differs more in the **spatial** domain than in frequency: strongest spatial shifts involve `p50_intensity, p10_intensity, grad_std`; strongest frequency shifts involve `fft_bin_1, high_freq_energy, fft_bin_5`.
  - *Scores:* overall `8.4692`, spatial `8.0772`, frequency `0.0085`.
- Compared to the dataset average, **Disposable plastic cup** differs more in the **spatial** domain than in frequency: strongest spatial shifts involve `p90_intensity, p50_intensity, mean_intensity`; strongest frequency shifts involve `fft_bin_1, high_freq_energy, fft_bin_3`.
  - *Scores:* overall `8.3661`, spatial `6.2837`, frequency `0.0245`.
- Compared to the dataset average, **Paper bag** differs more in the **spatial** domain than in frequency: strongest spatial shifts involve `p10_intensity, grad_std, std_intensity`; strongest frequency shifts involve `fft_bin_1, fft_bin_2, fft_bin_4`.
  - *Scores:* overall `7.5813`, spatial `3.0980`, frequency `0.0082`.
- Compared to the dataset average, **Pop tab** differs more in the **spatial** domain than in frequency: strongest spatial shifts involve `grad_mean, p10_intensity, p50_intensity`; strongest frequency shifts involve `fft_bin_2, high_freq_energy, fft_bin_4`.
  - *Scores:* overall `7.5437`, spatial `5.3585`, frequency `0.0270`.
- Compared to the dataset average, **Other plastic wrapper** differs more in the **spatial** domain than in frequency: strongest spatial shifts involve `p10_intensity, grad_std, p50_intensity`; strongest frequency shifts involve `fft_bin_1, fft_bin_2, fft_bin_3`.
  - *Scores:* overall `6.5641`, spatial `5.0193`, frequency `0.0150`.
- Compared to the dataset average, **Drink carton** differs more in the **spatial** domain than in frequency: strongest spatial shifts involve `grad_mean, p50_intensity, mean_intensity`; strongest frequency shifts involve `fft_bin_1, high_freq_energy, fft_bin_4`.
  - *Scores:* overall `5.9842`, spatial `2.1809`, frequency `0.0133`.

## ML results (features → models)
| Model | Accuracy | F1-macro |
|---|---:|---:|
| extra_trees | 0.0705 | 0.0461 |
| logreg | 0.0573 | 0.0376 |
| decision_tree | 0.0396 | 0.0364 |
| xgboost | 0.0485 | 0.0361 |
| rf | 0.0485 | 0.0357 |
| linear_svm | 0.0264 | 0.0174 |

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
