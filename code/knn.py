import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class Config:
	"""Configuration for K-Nearest Neighbors model training."""

	random_state: int = 42
	test_size: float = 0.2
	cv_folds: int = 5
	# Odd values only, to avoid tie votes in binary classification.
	candidate_k_values: tuple[int, ...] = (3, 5, 7, 9, 11, 13, 15, 17, 19, 21)


def build_features(df):
	"""Builds features for the Titanic dataset."""

	features = df.copy()

	# Add simple engineered features that help similarity-based models.
	features["FamilySize"] = features["SibSp"] + features["Parch"] + 1
	features["IsAlone"] = (features["FamilySize"] == 1).astype(int)

	# Extract the honorific (e.g. "Mr", "Miss") between the comma and period in "Last, Title. First".
	title_series = (
		features["Name"]
		.fillna("")
		.str.extract(r",\s*([^\.]+)\.", expand=False)
		.fillna("Unknown")
		.str.strip()
	)
	features["Title"] = title_series

	return features


def make_pipeline():
	"""Creates a scikit-learn pipeline for KNN classification."""

	numeric_features = ["Pclass", "Age", "SibSp", "Parch", "Fare", "FamilySize", "IsAlone"]
	categorical_features = ["Sex", "Embarked", "Title"]

	numeric_transformer = Pipeline(
		steps=[
			("imputer", SimpleImputer(strategy="median")),
			# KNN relies on distance, so features must be on a comparable scale.
			("scaler", StandardScaler()),
		]
	)

	categorical_transformer = Pipeline(
		steps=[
			("imputer", SimpleImputer(strategy="most_frequent")),
			# Ignore categories unseen in training (e.g. test-only Embarked values) instead of erroring.
			("encoder", OneHotEncoder(handle_unknown="ignore")),
		]
	)

	preprocessor = ColumnTransformer(
		transformers=[
			("num", numeric_transformer, numeric_features),
			("cat", categorical_transformer, categorical_features),
		]
	)

	return Pipeline(
		steps=[
			("preprocessor", preprocessor),
			("model", KNeighborsClassifier()),
		]
	)


def select_best_k(X_train, y_train, cfg):
	"""Selects the best k value for KNN using cross-validation."""

	# Stratify folds so each split preserves the overall survival rate.
	cv = StratifiedKFold(n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.random_state)
	scores_by_k = {}

	# Refit a fresh pipeline per k so the preprocessor (scaler/encoder) is
	# never fit on data outside its own fold, avoiding cross-fold leakage.
	for k in cfg.candidate_k_values:
		model = make_pipeline()
		model.set_params(model__n_neighbors=k)
		cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
		scores_by_k[k] = float(np.mean(cv_scores))

	best_k = max(scores_by_k, key=scores_by_k.get)
	return best_k, scores_by_k


def evaluate_predictions(y_true, y_pred):
	"""Evaluates predictions using common classification metrics."""

	cm = confusion_matrix(y_true, y_pred)
	# Relies on binary labels {0, 1} so ravel() order is [[TN, FP], [FN, TP]].
	tn, fp, fn, tp = cm.ravel()

	metrics = {
		"accuracy": float(accuracy_score(y_true, y_pred)),
		"precision": float(precision_score(y_true, y_pred, zero_division=0)),
		"sensitivity_recall": float(recall_score(y_true, y_pred, zero_division=0)),
		"specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
		"confusion_matrix": cm.tolist(),
		"true_negative": int(tn),
		"false_positive": int(fp),
		"false_negative": int(fn),
		"true_positive": int(tp),
	}
	return metrics


def main():
	"""Main function to train KNN model and evaluate on Titanic dataset."""
	cfg = Config()

	root_dir = Path(__file__).resolve().parents[1]
	data_dir = root_dir / "data" / "titanic"
	output_dir = root_dir / "code" / "outputs"
	output_dir.mkdir(parents=True, exist_ok=True)

	train_path = data_dir / "train.csv"
	test_path = data_dir / "test.csv"

	train_df = pd.read_csv(train_path)
	test_df = pd.read_csv(test_path)

	train_features = build_features(train_df)
	test_features = build_features(test_df)

	y = train_df["Survived"]
	X = train_features

	# Hold out a validation set (untouched by CV) purely to report an unbiased final metric.
	X_train, X_valid, y_train, y_valid = train_test_split(
		X,
		y,
		test_size=cfg.test_size,
		random_state=cfg.random_state,
		stratify=y,
	)

	best_k, cv_scores = select_best_k(X_train, y_train, cfg)

	final_model = make_pipeline()
	final_model.set_params(model__n_neighbors=best_k)
	final_model.fit(X_train, y_train)

	valid_predictions = final_model.predict(X_valid)
	evaluation = evaluate_predictions(y_valid, valid_predictions)
	evaluation["best_k"] = best_k
	evaluation["cv_accuracy_by_k"] = cv_scores

	print("KNN validation results")
	print(f"Best k from CV: {best_k}")
	print(f"Accuracy: {evaluation['accuracy']:.4f}")
	print(f"Precision: {evaluation['precision']:.4f}")
	print(f"Sensitivity (Recall): {evaluation['sensitivity_recall']:.4f}")
	print(f"Specificity: {evaluation['specificity']:.4f}")
	print("Confusion Matrix [[TN, FP], [FN, TP]]:")
	print(np.array(evaluation["confusion_matrix"]))

	# Retrain on full training data for the final test predictions.
	production_model = make_pipeline()
	production_model.set_params(model__n_neighbors=best_k)
	production_model.fit(X, y)

	test_predictions = production_model.predict(test_features)
	submission = pd.DataFrame(
		{
			"PassengerId": test_df["PassengerId"],
			"Survived": test_predictions.astype(int),
		}
	)
	submission_path = output_dir / "knn_submission.csv"
	# Matches the Kaggle Titanic competition's expected submission format.
	submission.to_csv(submission_path, index=False)

	evaluation_path = output_dir / "knn_metrics.json"
	with evaluation_path.open("w", encoding="utf-8") as f:
		json.dump(evaluation, f, indent=2)

	print(f"Saved submission file: {submission_path}")
	print(f"Saved evaluation metrics: {evaluation_path}")


if __name__ == "__main__":
	main()