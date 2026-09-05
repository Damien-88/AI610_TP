"""Shared Titanic column cleaning, matching the steps performed in titanic_cleaning.ipynb.

titanic_cleaning.ipynb produces the already-cleaned data/titanic/cleaned_train.csv, so training
code should just read that file directly. test.csv (and single-passenger records from
predict.py) are never run through that notebook, so this module exists to apply the same column
drops and categorical encoding to them without re-deriving the logic in multiple places.
"""

import pandas as pd

DROP_COLUMNS = ("PassengerId", "Name", "Ticket", "Cabin")
EMBARKED_DUMMY_COLUMNS = ("Embarked_C", "Embarked_Q", "Embarked_S")


def drop_identifier_columns(df):
	"""Drops the identifier/noise columns dropped in titanic_cleaning.ipynb section 2."""

	return df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])


def encode_categoricals(df):
	"""Encodes Sex/Embarked exactly as titanic_cleaning.ipynb section 6."""

	encoded = df.copy()
	encoded["Sex"] = encoded["Sex"].map({"male": 0, "female": 1})
	encoded = pd.get_dummies(encoded, columns=["Embarked"], prefix="Embarked", dtype=int)

	# A single-passenger record only produces a dummy column for its own port.
	for col in EMBARKED_DUMMY_COLUMNS:
		if col not in encoded.columns:
			encoded[col] = 0

	return encoded


def clean_titanic_columns(df):
	"""Applies the full column-drop + encoding pipeline to a raw Titanic-schema dataframe."""

	return encode_categoricals(drop_identifier_columns(df))
