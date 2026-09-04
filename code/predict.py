"""Interactive CLI for predicting Titanic survival with the trained KNN model."""

from pathlib import Path

import joblib
import pandas as pd

from knn import build_features

TITLE_OPTIONS = ["Mr", "Mrs", "Miss", "Master", "Dr", "Rev", "Unknown"]


def prompt_choice(label, options, default):
	"""Prompts for a value restricted to a fixed set of options."""

	options_display = "/".join(options)
	raw = input(f"{label} [{options_display}] (default={default}): ").strip()
	if not raw:
		return default
	# Case-insensitive match against the allowed options.
	for option in options:
		if raw.lower() == option.lower():
			return option
	print(f"Unrecognized value '{raw}', using default '{default}'.")
	return default


def prompt_number(label, default, cast=float):
	"""Prompts for a numeric value, falling back to a default on blank/invalid input."""

	raw = input(f"{label} (default={default}): ").strip()
	if not raw:
		return default
	try:
		return cast(raw)
	except ValueError:
		print(f"Invalid number '{raw}', using default '{default}'.")
		return default


def collect_passenger_from_input():
	"""Collects the raw Titanic feature columns needed by build_features via prompts."""

	print("Enter passenger details to estimate survival probability.\n")

	pclass = int(prompt_number("Passenger class (1, 2, or 3)", default=3, cast=int))
	sex = prompt_choice("Sex", ["male", "female"], default="male")
	age = prompt_number("Age", default=30.0)
	sibsp = int(prompt_number("Number of siblings/spouses aboard", default=0, cast=int))
	parch = int(prompt_number("Number of parents/children aboard", default=0, cast=int))
	fare = prompt_number("Fare paid", default=32.0)
	embarked = prompt_choice("Port of embarkation", ["C", "Q", "S"], default="S")
	title = prompt_choice("Title", TITLE_OPTIONS, default="Mr")

	# Name only needs to carry the title, since build_features extracts it via regex.
	# Title is also kept as its own column so it can be echoed back without re-parsing Name.
	return pd.DataFrame(
		[
			{
				"Name": f"Passenger, {title}. Input",
				"Pclass": pclass,
				"Sex": sex,
				"Age": age,
				"SibSp": sibsp,
				"Parch": parch,
				"Fare": fare,
				"Embarked": embarked,
				"Title": title,
			}
		]
	)


def passenger_summary(passenger_df):
	"""Formats the entered passenger fields so they can be echoed alongside the prediction."""

	row = passenger_df.iloc[0]
	fields = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Title"]
	lines = [f"  {field}: {row[field]}" for field in fields]
	return "Entered Passenger Details\n" + "\n".join(lines)


def load_model(model_path):
	"""Loads the trained pipeline saved by knn.py, raising a helpful error if missing."""

	if not model_path.exists():
		raise FileNotFoundError(
			f"No trained model found at {model_path}. Run knn.py first to train and save it."
		)
	return joblib.load(model_path)


def predict_survival(model, passenger_df):
	"""Returns (predicted_label, survival_probability) for a single passenger row."""

	features = build_features(passenger_df)
	prediction = model.predict(features)[0]
	probability = model.predict_proba(features)[0][1]
	return int(prediction), float(probability)


def main():
	root_dir = Path(__file__).resolve().parents[1]
	model_path = root_dir / "code" / "outputs" / "knn_model.joblib"

	model = load_model(model_path)
	passenger_df = collect_passenger_from_input()

	prediction, probability = predict_survival(model, passenger_df)

	print(f"\n{passenger_summary(passenger_df)}")
	print("\nPrediction Result")
	print(f"Estimated survival probability: {probability:.2%}")
	print(f"Predicted outcome: {'Survived' if prediction == 1 else 'Did not survive'}")


if __name__ == "__main__":
	main()
