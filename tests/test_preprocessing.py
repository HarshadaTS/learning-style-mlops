from pathlib import Path
from src.preprocessing import load_data, preprocess_data


def test_preprocessing():
    root = Path(__file__).resolve().parents[1]
    df = load_data(root / "data.csv")
    X, y = preprocess_data(df)
    assert len(X) == len(y)
    assert "Learner_Type" not in X.columns
    assert X.select_dtypes(exclude="number").shape[1] == 0
