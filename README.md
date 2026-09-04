# AI610_TP

Titanic survival prediction project for the Agent Based Systems course.

This repository contains a K-Nearest Neighbors (KNN) workflow for predicting
whether a Titanic passenger survived. The project is implemented in both a
Jupyter notebook and a Python script, and it includes feature engineering,
preprocessing, cross-validation, validation metrics, visualizations, and
final test-set submission generation. A separate CLI script lets you predict
survival for a single passenger you describe interactively.

## What Is Included

- `code/knn.ipynb`: notebook version of the full analysis with EDA,
  visualizations, and written interpretation.
- `code/knn.py`: script version of the same pipeline for repeatable
  execution; also persists the trained model and saves plots as PNGs.
- `code/predict.py`: interactive CLI that loads the persisted model and
  estimates survival probability for a passenger you describe.
- `code/titanic_cleaning.ipynb`: standalone exploratory notebook that
  produced `data/titanic/cleaned_train.csv`; it is not part of the KNN
  training pipeline, which builds its own features from the raw data.
- `code/outputs/knn_submission.csv`: generated Kaggle-style submission file.
- `code/outputs/knn_metrics.json`: saved validation metrics and CV scores.
- `code/outputs/knn_model.joblib`: trained pipeline persisted for reuse by
  `predict.py`.
- `code/outputs/knn_cv_accuracy.png`, `knn_confusion_matrix.png`,
  `knn_metrics_summary.png`: saved visualizations of CV accuracy vs. `k`,
  the confusion matrix, and the validation metrics summary.
- `data/titanic/train.csv`: raw training data (used directly by KNN).
- `data/titanic/test.csv`: raw test data.
- `data/titanic/cleaned_train.csv`: pre-encoded training data produced by
  `titanic_cleaning.ipynb`; not consumed by the KNN pipeline.
- `data/titanic/gender_submission.csv`: example submission format.

## Approach

The model uses the following steps:

1. Load the raw Titanic training and test data (`train.csv` / `test.csv`).
   Both share the same raw columns (`Name`, `Sex`, `Embarked`, etc.), which
   the feature engineering step below depends on.
2. Engineer additional features:
   - `FamilySize` (`SibSp` + `Parch` + 1)
   - `IsAlone` (1 if `FamilySize` == 1)
   - passenger `Title` extracted from the name (e.g. "Mr", "Miss")
3. Preprocess numeric and categorical variables separately via a
   `ColumnTransformer`:
   - Numeric (`Pclass`, `Age`, `SibSp`, `Parch`, `Fare`, `FamilySize`,
     `IsAlone`): median imputation, then standardization (required since
     KNN is distance-based).
   - Categorical (`Sex`, `Embarked`, `Title`): most-frequent imputation,
     then one-hot encoding (unseen categories ignored).
4. Select the best `k` (odd values 3–21) using 5-fold stratified
   cross-validation, refitting the preprocessing + model pipeline per fold
   to avoid cross-fold leakage.
5. Evaluate the final model on a held-out validation split (never used
   during cross-validation).
6. Retrain on the full training data, generate predictions for the test
   set, and persist the fitted pipeline for reuse.

## Reported Findings

The saved evaluation metrics (`code/outputs/knn_metrics.json`) show the
following validation results:

- Best `k`: `15`
- Accuracy: `0.8101`
- Precision: `0.8070`
- Sensitivity / recall: `0.6667`
- Specificity: `0.9000`
- F1 score: `0.7302`
- Confusion matrix: `[[99, 11], [23, 46]]`

Interpretation:

- The model is strong at identifying non-survivors (high specificity) and
  maintains good precision when predicting survivors.
- Recall is lower than specificity, meaning some true survivors are still
  missed; the F1 score (which balances precision and recall) sits below
  both, confirming that trade-off.
- This behavior is consistent with distance-based models on mixed-feature
  tabular data, where class boundaries can overlap.
- Overall, KNN provides a solid, interpretable baseline for this task,
  with stable accuracy and few false positives, but reduced sensitivity to
  all survivor cases.

## Requirements

Install the core dependencies with:

```bash
pip install -r requirements.txt
```

Both `knn.ipynb` and `knn.py` use `matplotlib` for visualizations, and
`knn.py`/`predict.py` use `joblib` to persist/load the trained model. Make
sure these are installed in whichever interpreter you run.

## How To Run

### Run the notebook

Open `code/knn.ipynb` and run the cells from top to bottom.

### Run the script

From the project root, run:

```bash
python code/knn.py
```

Both entry points use the same pipeline and write their outputs to
`code/outputs/`.

### Predict survival for a single passenger

After running `code/knn.py` at least once (so a trained model is saved),
run:

```bash
python code/predict.py
```

and answer the interactive prompts (passenger class, sex, age, siblings/
spouses aboard, parents/children aboard, fare, port of embarkation, and
title). The script prints the estimated survival probability and predicted
outcome.

## Output Files

After running `code/knn.py` (or the notebook), the project produces:

- `code/outputs/knn_submission.csv`
- `code/outputs/knn_metrics.json`
- `code/outputs/knn_model.joblib`
- `code/outputs/knn_cv_accuracy.png`
- `code/outputs/knn_confusion_matrix.png`
- `code/outputs/knn_metrics_summary.png`

These files are regenerated each time the notebook or script is run.

## Project Layout

```text
AI610_TP/
├── README.md
├── requirements.txt
├── code/
│   ├── knn.ipynb
│   ├── knn.py
│   ├── predict.py
│   ├── titanic_cleaning.ipynb
│   └── outputs/
│       ├── knn_confusion_matrix.png
│       ├── knn_cv_accuracy.png
│       ├── knn_metrics.json
│       ├── knn_metrics_summary.png
│       ├── knn_model.joblib
│       └── knn_submission.csv
└── data/
    └── titanic/
        ├── cleaned_train.csv
        ├── gender_submission.csv
        ├── test.csv
        └── train.csv
```

