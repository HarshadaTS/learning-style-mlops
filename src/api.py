from flask import Flask, request, jsonify
import mlflow
import mlflow.sklearn
import pandas as pd


# ============================================================
# Create Flask application
# ============================================================

app = Flask(__name__)


# ============================================================
# Connect to MLflow Server
# ============================================================

mlflow.set_tracking_uri(
    "http://host.docker.internal:5000"
)

print("Connected to MLflow server")


# ============================================================
# Load Champion Model from MLflow Model Registry
# ============================================================

MODEL_URI = "models:/LearningStyleModel@champion"

print("Loading champion model...")

model = mlflow.sklearn.load_model(MODEL_URI)

print("Champion model loaded successfully!")


# ============================================================
# Get feature names from trained model
# ============================================================

FEATURE_COLUMNS = list(model.feature_names_in_)

print(f"Number of features: {len(FEATURE_COLUMNS)}")


# ============================================================
# Feature-name normalization
# ============================================================

def canonical_name(name):
    name = str(name)

    name = name.replace("\r", "")
    name = name.replace("\n", "")
    name = name.replace("\\\\n", "")
    name = name.replace("\\n", "")
    name = name.strip()

    return name


# ============================================================
# Health Check API
# ============================================================

@app.route("/health", methods=["GET"])
def health():

    return jsonify({
        "status": "API is running"
    })


# ============================================================
# Prediction API
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # Get JSON request
        data = request.get_json()

        if data is None:

            return jsonify({
                "error": "Request must contain JSON data"
            }), 400


        # Normalize incoming feature names
        normalized_data = {
            canonical_name(key): value
            for key, value in data.items()
        }


        # Check missing features
        missing_features = []

        for feature in FEATURE_COLUMNS:

            canonical = canonical_name(feature)

            if canonical not in normalized_data:

                missing_features.append(feature)


        if missing_features:

            return jsonify({
                "error": "Missing features",
                "missing_features": missing_features
            }), 400


        # Arrange values in exactly the same order
        # used during model training

        values = []

        for feature in FEATURE_COLUMNS:

            canonical = canonical_name(feature)

            values.append(
                normalized_data[canonical]
            )


        # Create DataFrame

        input_data = pd.DataFrame(
            [values],
            columns=FEATURE_COLUMNS
        )


        # Make prediction

        prediction = model.predict(
            input_data
        )


        learner_type = str(
            prediction[0]
        )


        # Return JSON response

        return jsonify({
            "prediction": learner_type
        })


    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


# ============================================================
# Start Flask Server
# ============================================================

if __name__ == "__main__":

    print("Starting Flask API...")
    print("API URL: http://127.0.0.1:8000")

    app.run(
        host="0.0.0.0",
        port=8000,
        debug=True
    )