# AI610_TP

Titanic survival prediction project for the Agent Based Systems course.

This repository contains a K-Nearest Neighbors (KNN) workflow for predicting
whether a Titanic passenger survived. The project is implemented in both a
Jupyter notebook and a Python script, and it includes feature engineering,
preprocessing, cross-validation, validation metrics, and final test-set
submission generation.

## What Is Included

- `code/knn.ipynb`: notebook version of the full analysis with plots and
  written interpretation.
- `code/knn.py`: script version of the same pipeline for repeatable execution.
- `code/outputs/knn_submission.csv`: generated Kaggle-style submission file.
- `code/outputs/knn_metrics.json`: saved validation metrics and CV scores.
- `data/titanic/train.csv`: training data.
- `data/titanic/test.csv`: test data.
- `data/titanic/gender_submission.csv`: example submission format.

## Approach

The model uses the following steps:

1. Load the Titanic training and test data.
2. Engineer additional features:
   - `FamilySize`
   - `IsAlone`
   - passenger `Title` extracted from the name
3. Preprocess numeric and categorical variables separately.
4. Standardize numeric features and one-hot encode categorical features.
5. Select the best `k` using 5-fold stratified cross-validation.
6. Evaluate the final model on a validation split.
7. Retrain on the full training data and generate predictions for the test set.

## Reported Findings

The saved evaluation metrics show the following results:

- Best `k`: `15`
- Validation accuracy: `0.8101`
- Precision: `0.8070`
- Sensitivity / recall: `0.6667`
- Specificity: `0.9000`
- Confusion matrix: `[[99, 11], [23, 46]]`

Interpretation:

- The model is strong at identifying non-survivors.
- Precision is high, so positive predictions are usually correct.
- Recall is lower than specificity, which means some survivors are still
  missed.
- Overall, the model is a solid baseline for a mixed-feature tabular dataset
  like Titanic.

## Requirements

Install the core dependencies with:

```bash
pip install -r requirements.txt
```

The notebook also uses `matplotlib` for visualizations. If it is not already
available in your environment, the notebook will install it on first run.

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

## Output Files

After execution, the project produces:

- `code/outputs/knn_submission.csv`
- `code/outputs/knn_metrics.json`

These files are regenerated each time the notebook or script is run.

## Project Layout

```text
AI610_TP/
├── README.md
├── requirements.txt
├── code/
│   ├── knn.ipynb
│   ├── knn.py
│   └── outputs/
│       ├── knn_metrics.json
│       └── knn_submission.csv
└── data/
    └── titanic/
        ├── gender_submission.csv
        ├── test.csv
        └── train.csv
```
