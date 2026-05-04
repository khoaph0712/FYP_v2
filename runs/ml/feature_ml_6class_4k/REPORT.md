# Feature + ML Analysis Report

## Scope

- Mobile is intentionally excluded in this stage.
- Focus: feature extraction + classical ML model comparison.
- Classes in this run: plastic, glass, metal, paper, cardboard, organic

## Pipeline

1. Extract object crops from YOLO labels.
2. Extract spatial-domain and frequency-domain features.
3. Train ML models first (LogReg, SVM-RBF, RandomForest).
4. Compare with accuracy/F1/confusion matrix.

## Data

- Train objects: **24000**
- Test objects: **4168**
- Classes: plastic, glass, metal, paper, cardboard, organic

## Model choice rationale

- **Logistic Regression:** simple linear baseline on standardized feature vectors.
- **SVM (RBF):** captures non-linear class boundaries from handcrafted features.
- **Random Forest:** robust tree-based baseline and interpretable feature importance.

## Results


| Model   | Accuracy | F1-macro |
| ------- | -------- | -------- |
| rf      | 0.4717   | 0.4478   |
| svm_rbf | 0.4561   | 0.4275   |
| logreg  | 0.3817   | 0.3420   |


## Class difference comments (objects)

Top distinct feature indices are reported to explain which classes differ most in spatial/frequency signatures.

- **organic**: distinctiveness `27.1808`, top feature idx `[6, 4, 5]`
- **paper**: distinctiveness `20.4569`, top feature idx `[3, 5, 0]`
- **metal**: distinctiveness `15.8436`, top feature idx `[5, 6, 2]`
- **glass**: distinctiveness `9.9506`, top feature idx `[3, 2, 6]`
- **plastic**: distinctiveness `9.4096`, top feature idx `[5, 2, 6]`
- **cardboard**: distinctiveness `6.3501`, top feature idx `[5, 2, 3]`

## Chart comments

- `chart_domain_importance.png`: compares spatial vs frequency contribution based on model feature importance (not raw magnitude, so it is scale-safe).
- `chart_model_comparison.png`: compares Accuracy/F1 across ML models to justify chosen baseline.