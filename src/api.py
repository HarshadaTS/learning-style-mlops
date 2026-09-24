from flask import Flask, request, jsonify
import mlflow
import mlflow.sklearn
import pandas as pd
import os

app = Flask(__name__)

# --------------------------------------------------
# MLflow configuration
# --------------------------------------------------
# Use the environment variable if provided.
# Otherwise, use the local Docker → Windows host address.
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://host.docker.internal:5000"
)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

MODEL_URI = "models:/LearningStyleModel@champion"

print(f"Connected to MLflow server: {MLFLOW_TRACKING_URI}")
print("Loading champion model...")

model = mlflow.sklearn.load_model(MODEL_URI)

print("Champion model loaded successfully!")

FEATURE_COLUMNS = list(model.feature_names_in_)

print(f"Number of features: {len(FEATURE_COLUMNS)}")


# --------------------------------------------------
# Helper function
# --------------------------------------------------
def canonical_name(name):
    """
    Clean feature names so that small formatting
    differences do not cause input matching problems.
    """
    name = str(name)
    name = name.replace("\r", "")
    name = name.replace("\n", "")
    name = name.replace("\\\\n", "")
    name = name.replace("\\n", "")
    name = name.strip()

    return name


# --------------------------------------------------
# Health endpoint
# --------------------------------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "API is running"
    })


# --------------------------------------------------
# Prediction endpoint
# --------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():

    try:
        # Get JSON data from request
        data = request.get_json()

        if data is None:
            return jsonify({
                "error": "Request must contain JSON data"
            }), 400

        # Normalize input feature names
        normalized_data = {
            canonical_name(key): value
            for key, value in data.items()
        }

        # Check whether all required features are present
        missing_features = []

        for feature in FEATURE_COLUMNS:

            canonical = canonical_name(feature)

            if canonical not in normalized_data:
                missing_features.append(feature)

        # Return error if features are missing
        if missing_features:

            return jsonify({
                "error": "Missing features",
                "missing_features": missing_features
            }), 400

        # Create values in exactly the same order
        # as the model's training features
        values = []

        for feature in FEATURE_COLUMNS:

            canonical = canonical_name(feature)

            values.append(
                normalized_data[canonical]
            )

        # Create DataFrame for model
        input_data = pd.DataFrame(
            [values],
            columns=FEATURE_COLUMNS
        )

        # Make prediction
        prediction = model.predict(input_data)

        learner_type = str(prediction[0])

        # Return JSON response
        return jsonify({
            "prediction": learner_type
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


# --------------------------------------------------
# Start Flask application
# --------------------------------------------------
if __name__ == "__main__":

    print("Starting Flask API...")
    print("API URL: http://127.0.0.1:8000")

    app.run(
        host="0.0.0.0",
        port=8000,
        debug=True
    )