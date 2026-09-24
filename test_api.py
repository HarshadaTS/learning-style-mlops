import requests
import joblib
import pandas as pd

# -----------------------------
# Load model
# -----------------------------
model = joblib.load("models/learning_style_model.joblib")

FEATURE_COLUMNS = list(model.feature_names_in_)

print("Number of features:", len(FEATURE_COLUMNS))

print("\nModel features:")
for feature in FEATURE_COLUMNS:
    print(repr(feature))


# -----------------------------
# Load CSV
# -----------------------------
df = pd.read_csv("data.csv")

print("\nCSV columns:")
for column in df.columns:
    print(repr(column))


# -----------------------------
# Normalize column names
# -----------------------------

def normalize_name(name):
    return (
        str(name)
        .replace("\n", "\\n")
        .replace("\r", "")
        .strip()
    )


# Create mapping:
# normalized name -> original CSV name

csv_column_map = {
    normalize_name(column): column
    for column in df.columns
}


# -----------------------------
# Find model features in CSV
# -----------------------------

missing = [
    feature
    for feature in FEATURE_COLUMNS
    if normalize_name(feature) not in csv_column_map
]

if missing:
    print("\nERROR: These model features are missing from CSV:")

    for feature in missing:
        print(repr(feature))

    raise SystemExit


# -----------------------------
# Get actual CSV column names
# -----------------------------

CSV_FEATURE_COLUMNS = [
    csv_column_map[normalize_name(feature)]
    for feature in FEATURE_COLUMNS
]


print("\nAll 36 features matched successfully!")


# -----------------------------
# Select a complete student row
# -----------------------------

valid_rows = df.dropna(subset=CSV_FEATURE_COLUMNS)

if valid_rows.empty:
    raise SystemExit("No complete row found in dataset.")

student = valid_rows.iloc[0]


# -----------------------------
# Create JSON request
# -----------------------------

data = {}

for model_feature, csv_feature in zip(
    FEATURE_COLUMNS,
    CSV_FEATURE_COLUMNS
):
    data[model_feature] = float(student[csv_feature])


# -----------------------------
# Send request to Flask API
# -----------------------------

response = requests.post(
    "http://127.0.0.1:8000/predict",
    json=data
)


# -----------------------------
# Print response
# -----------------------------

print("\n==============================")
print("API RESPONSE")
print("==============================")

print("Status code:", response.status_code)
print("Response:", response.json())