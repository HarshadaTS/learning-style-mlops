from pathlib import Path
import pandas as pd

TARGET = "Learner_Type"
DROP_COLUMNS = [
    "Div",
    "ROLL. NO.",
    "NAME",
    "General_Merit_No.",
    "TW (25)",
    "Insem_Total",
    "insem_percent",
]


def load_data(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def preprocess_data(df: pd.DataFrame):
    missing_target = df[TARGET].isna().sum()
    if missing_target:
        df = df.dropna(subset=[TARGET]).copy()

    cols_to_drop = [c for c in DROP_COLUMNS if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    # The source dataset is numeric apart from the target after the
    # identifier columns above are removed. Keep this check explicit so
    # unexpected columns fail early instead of silently entering production.
    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(str)

    non_numeric = X.select_dtypes(exclude="number").columns.tolist()
    if non_numeric:
        raise ValueError(f"Unexpected non-numeric feature columns: {non_numeric}")

    if X.isna().any().any():
        missing = X.columns[X.isna().any()].tolist()
        raise ValueError(f"Missing values remain in features: {missing}")

    return X, y
