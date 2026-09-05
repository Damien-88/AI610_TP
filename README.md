# AI610_TP

Titanic survival prediction project for the Agent Based Systems course.

This repository contains a K-Nearest Neighbors (KNN) workflow for predicting
whether a Titanic passenger survived. The project is implemented in both a
Jupyter notebook and a Python script, and it includes feature engineering,
preprocessing, cross-validation, validation metrics, visualizations, and
final test-set submission generation. A separate CLI script lets you predict
survival for a single passenger you describe interactively.

## What Is Included

- `code/titanic_cleaning.ipynb`: standalone data-preparation notebook that
  cleans `data/titanic/train.csv` (drops `PassengerId`/`Name`/`Ticket`/`Cabin`,
  drops rows with missing `Age`/`Embarked`, encodes `Sex`/`Embarked`) and
  writes the result to `data/titanic/cleaned_train.csv`. This is the file
  the KNN pipeline actually trains on.
- `code/test_cleaning.py`: shared cleaning helpers
  (`drop_identifier_columns`, `encode_categoricals`, `clean_titanic_columns`)
  that mirror the steps in `titanic_cleaning.ipynb`. Used by that notebook
  itself, and by `knn.py`/`knn.ipynb`/`predict.py` to bring the still-raw
  `test.csv` (and interactively entered passengers) into the same schema
  as `cleaned_train.csv`.
- `code/knn.py`: script version of the KNN pipeline; trains on
  `cleaned_train.csv`, persists the trained model, and saves plots as PNGs.
- `code/knn.ipynb`: notebook version of the analysis with EDA,
  visualizations, and a written interpretation. It defines its own
  `build_features`/`make_pipeline` independently of `knn.py` (the two are
  kept as separate, self-contained implementations) but produces the same
  results since both consume `cleaned_train.csv` and `test_cleaning.py`.
- `code/predict.py`: interactive CLI that loads the persisted model and
  estimates survival probability for a passenger you describe.
- `code/outputs/knn_submission.csv`: generated Kaggle-style submission file.
- `code/outputs/knn_metrics.json`: saved validation metrics and CV scores.
- `code/outputs/knn_model.joblib`: trained pipeline persisted for reuse by
  `predict.py`.
- `code/outputs/knn_cv_accuracy.png`, `knn_confusion_matrix.png`,
  `knn_metrics_summary.png`: saved visualizations of CV accuracy vs. `k`,
  the confusion matrix, and the validation metrics summary.
- `data/titanic/train.csv`: raw training data. Consumed by
  `titanic_cleaning.ipynb` (to produce `cleaned_train.csv`) and by
  `knn.ipynb`'s exploratory data analysis section; not read by `knn.py`.
- `data/titanic/test.csv`: raw test data, cleaned on the fly via
  `test_cleaning.py` before being scored.
- `data/titanic/cleaned_train.csv`: pre-encoded, missing-value-free training
  data produced by `titanic_cleaning.ipynb`; this is what `knn.py` and
  `knn.ipynb` actually train on.
- `data/titanic/gender_submission.csv`: example submission format.

## Approach

`titanic_cleaning.ipynb` is a one-time step that turns `train.csv` into
`cleaned_train.csv`: it drops `PassengerId`/`Name`/`Ticket`/`Cabin`, drops
the small number of rows with missing `Age`/`Embarked`, maps `Sex` to 0/1,
and one-hot encodes `Embarked` into `Embarked_C`/`Embarked_Q`/`Embarked_S`.

The KNN pipeline (`knn.py` / `knn.ipynb`) then works as follows:

1. Load `cleaned_train.csv` directly as the training data (no re-cleaning),
   and separate the `Survived` label from the feature columns.
2. Load the still-raw `test.csv` and run it through `test_cleaning.py`'s
   `clean_titanic_columns` so it matches the training schema.
3. Engineer two additional features on both sets:
   - `FamilySize` (`SibSp` + `Parch` + 1)
   - `IsAlone` (1 if `FamilySize` == 1)
4. Preprocess all (now-numeric) features via a `ColumnTransformer` with a
   single branch: median imputation (handles the missing `Age`/`Fare`
   values still present in `test.csv`) followed by standardization
   (required since KNN is distance-based).
5. Select the best `k` (odd values 3–21) using 5-fold stratified
   cross-validation, refitting the preprocessing + model pipeline per fold
   to avoid cross-fold leakage.
6. Evaluate the final model on a held-out validation split (never used
   during cross-validation).
7. Retrain on the full training data, generate predictions for the test
   set, and persist the fitted pipeline for reuse.

`predict.py` reuses `clean_titanic_columns` and `build_features` so a
single interactively-entered passenger goes through the same steps as
`test.csv`.

## Reported Findings

The saved evaluation metrics (`code/outputs/knn_metrics.json`) show the
following validation results:

- Best `k`: `7`
- Accuracy: `0.7902`
- Precision: `0.7414`
- Sensitivity / recall: `0.7414`
- Specificity: `0.8235`
- F1 score: `0.7414`
- Confusion matrix: `[[70, 15], [15, 43]]`

Interpretation:

- The model is somewhat stronger at identifying non-survivors (specificity
  0.82) than survivors, though precision and recall are balanced (both
  0.7414), meaning false positives and false negatives occur at similar
  rates.
- This behavior is consistent with distance-based models on mixed-feature
  tabular data, where class boundaries can overlap.
- Overall, KNN provides a solid, interpretable baseline for this task.
  Its strength here is balanced precision/recall with reasonable overall
  accuracy, while its main limitation is comparatively lower specificity
  than a model tuned to favor non-survivor predictions.

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
spouses aboard, parents/children aboard, fare, and port of embarkation).
The script prints the estimated survival probability and predicted
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
│   ├── test_cleaning.py
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