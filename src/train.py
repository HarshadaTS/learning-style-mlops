from pathlib import Path
import json
import joblib
import pandas as pd

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.preprocessing import load_data, preprocess_data


# ============================================================
# 1. PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data.csv"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


# ============================================================
# 2. MLFLOW CONFIGURATION
# ============================================================

# MLflow server
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Experiment name
mlflow.set_experiment("Learning Style Classification")

# Name used in MLflow Model Registry
REGISTERED_MODEL_NAME = "LearningStyleModel"


# ============================================================
# 3. LOG MODEL PARAMETERS
# ============================================================

def log_model_parameters(name, model):
    """
    Log model-specific hyperparameters to MLflow.
    """

    mlflow.log_param("model_type", name)

    if name == "Decision Tree":

        mlflow.log_param(
            "criterion",
            model.get_params()["criterion"]
        )

        mlflow.log_param(
            "max_depth",
            model.get_params()["max_depth"]
        )

        mlflow.log_param(
            "random_state",
            model.get_params()["random_state"]
        )

    elif name == "Random Forest":

        mlflow.log_param(
            "bootstrap",
            model.get_params()["bootstrap"]
        )

        mlflow.log_param(
            "max_depth",
            model.get_params()["max_depth"]
        )

        mlflow.log_param(
            "max_features",
            model.get_params()["max_features"]
        )

        mlflow.log_param(
            "random_state",
            model.get_params()["random_state"]
        )

    elif name == "SVM":

        classifier = model.named_steps["classifier"]

        mlflow.log_param(
            "kernel",
            classifier.get_params()["kernel"]
        )

        mlflow.log_param(
            "C",
            classifier.get_params()["C"]
        )

        mlflow.log_param(
            "gamma",
            classifier.get_params()["gamma"]
        )

    elif name == "KNN":

        classifier = model.named_steps["classifier"]

        mlflow.log_param(
            "n_neighbors",
            classifier.get_params()["n_neighbors"]
        )

        mlflow.log_param(
            "weights",
            classifier.get_params()["weights"]
        )


# ============================================================
# 4. EVALUATION FUNCTION
# ============================================================

def evaluate(
    name,
    model,
    X_train,
    X_test,
    y_train,
    y_test
):
    """
    Train model and calculate evaluation metrics.
    """

    # Train
    model.fit(X_train, y_train)

    # Predict
    pred = model.predict(X_test)

    # Calculate metrics
    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            y_test,
            pred,
            average="weighted",
            zero_division=0
        )
    )

    result = {
        "model": name,

        "accuracy": accuracy_score(
            y_test,
            pred
        ),

        "precision_weighted": precision,

        "recall_weighted": recall,

        "f1_weighted": f1,

        # Keep these for later use
        "model_object": model,

        "predictions": pred
    }

    return result


# ============================================================
# 5. LOAD DATA
# ============================================================

print("\nLoading dataset...")

df = load_data(DATA_PATH)

print(
    f"Dataset loaded successfully: "
    f"{df.shape[0]} rows, {df.shape[1]} columns"
)


# ============================================================
# 6. PREPROCESS DATA
# ============================================================

print("\nPreprocessing dataset...")

X, y = preprocess_data(df)

print(f"Number of features: {X.shape[1]}")
print(f"Number of samples: {X.shape[0]}")


# ============================================================
# 7. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=0,
    stratify=y
)

print("\nTrain/Test split:")
print(f"Training rows: {len(X_train)}")
print(f"Testing rows: {len(X_test)}")


# ============================================================
# 8. DEFINE MODELS
# ============================================================

models = {

    "Decision Tree":
        DecisionTreeClassifier(
            criterion="entropy",
            max_depth=3,
            random_state=42
        ),

    "Random Forest":
        RandomForestClassifier(
            bootstrap=True,
            max_depth=10,
            max_features="sqrt",
            random_state=1
        ),

    "SVM":
        Pipeline([
            (
                "scaler",
                StandardScaler()
            ),

            (
                "classifier",
                SVC()
            )
        ]),

    "KNN":
        Pipeline([
            (
                "scaler",
                StandardScaler()
            ),

            (
                "classifier",
                KNeighborsClassifier()
            )
        ])
}


# ============================================================
# 9. TRAIN ALL MODELS + MLFLOW TRACKING
# ============================================================

results = []

print("\n========================================")
print("Training models...")
print("========================================")


for name, model in models.items():

    print(f"\nTraining {name}...")

    # Start MLflow run
    with mlflow.start_run(
        run_name=name
    ):

        # ----------------------------------------------------
        # Log common parameters
        # ----------------------------------------------------

        mlflow.log_param(
            "test_size",
            0.20
        )

        mlflow.log_param(
            "random_state_split",
            0
        )

        mlflow.log_param(
            "training_rows",
            len(X_train)
        )

        mlflow.log_param(
            "testing_rows",
            len(X_test)
        )

        mlflow.log_param(
            "number_of_features",
            X.shape[1]
        )

        # ----------------------------------------------------
        # Log model-specific parameters
        # ----------------------------------------------------

        log_model_parameters(
            name,
            model
        )

        # ----------------------------------------------------
        # Train + evaluate
        # ----------------------------------------------------

        result = evaluate(
            name,
            model,
            X_train,
            X_test,
            y_train,
            y_test
        )

        # ----------------------------------------------------
        # Log metrics
        # ----------------------------------------------------

        mlflow.log_metrics({

            "accuracy":
                result["accuracy"],

            "precision_weighted":
                result["precision_weighted"],

            "recall_weighted":
                result["recall_weighted"],

            "f1_weighted":
                result["f1_weighted"]

        })

        # ----------------------------------------------------
        # Classification report
        # ----------------------------------------------------

        report = classification_report(
            y_test,
            result["predictions"],
            zero_division=0
        )

        mlflow.log_text(
            report,
            "classification_report.txt"
        )

        # ----------------------------------------------------
        # Log trained model
        #
        # pickle is used because it worked correctly
        # with your current MLflow + Windows environment.
        # ----------------------------------------------------

        mlflow.sklearn.log_model(
            model,
            name="model",
            serialization_format="pickle"
        )

        # ----------------------------------------------------
        # Get MLflow Run ID
        # ----------------------------------------------------

        run_id = mlflow.active_run().info.run_id

        result["run_id"] = run_id

        # ----------------------------------------------------
        # Save result
        # ----------------------------------------------------

        results.append(result)

        # ----------------------------------------------------
        # Print result
        # ----------------------------------------------------

        print(
            f"{name}: "
            f"Accuracy={result['accuracy']:.4f}, "
            f"F1={result['f1_weighted']:.4f}"
        )


# ============================================================
# 10. SELECT MODEL
# ============================================================

print("\n========================================")
print("Model comparison")
print("========================================")


for result in results:

    print(
        f"{result['model']:<20}"
        f" Accuracy: {result['accuracy']:.4f}"
        f" F1: {result['f1_weighted']:.4f}"
    )


# Select using weighted F1 first,
# then accuracy if there is a tie.

best = max(
    results,
    key=lambda r: (
        r["f1_weighted"],
        r["accuracy"]
    )
)


print("\nSelected model:")
print(best["model"])

print(
    f"Weighted F1: "
    f"{best['f1_weighted']:.4f}"
)

print(
    f"Accuracy: "
    f"{best['accuracy']:.4f}"
)


# ============================================================
# 11. SAVE BEST MODEL LOCALLY
# ============================================================

best_model_path = (
    MODEL_DIR /
    "learning_style_model.joblib"
)

joblib.dump(
    best["model_object"],
    best_model_path
)

print(
    f"\nBest model saved to:\n"
    f"{best_model_path}"
)


# ============================================================
# 12. SAVE MODEL COMPARISON RESULTS
# ============================================================

comparison_results = []

for result in results:

    comparison_results.append({

        "model":
            result["model"],

        "accuracy":
            result["accuracy"],

        "precision_weighted":
            result["precision_weighted"],

        "recall_weighted":
            result["recall_weighted"],

        "f1_weighted":
            result["f1_weighted"],

        "run_id":
            result["run_id"]
    })


comparison_path = (
    RESULTS_DIR /
    "model_comparison.json"
)

with open(
    comparison_path,
    "w"
) as f:

    json.dump(
        comparison_results,
        f,
        indent=4
    )


print(
    f"Model comparison saved to:\n"
    f"{comparison_path}"
)


# ============================================================
# 13. SAVE BEST MODEL INFORMATION
# ============================================================

best_model_info = {

    "selected_model":
        best["model"],

    "accuracy":
        best["accuracy"],

    "precision_weighted":
        best["precision_weighted"],

    "recall_weighted":
        best["recall_weighted"],

    "f1_weighted":
        best["f1_weighted"],

    "run_id":
        best["run_id"],

    "model_path":
        str(best_model_path)
}


best_model_info_path = (
    RESULTS_DIR /
    "best_model.json"
)

with open(
    best_model_info_path,
    "w"
) as f:

    json.dump(
        best_model_info,
        f,
        indent=4
    )


print(
    f"Best model information saved to:\n"
    f"{best_model_info_path}"
)


# ============================================================
# 14. REGISTER BEST MODEL IN MLFLOW MODEL REGISTRY
# ============================================================

print("\n========================================")
print("Registering selected model...")
print("========================================")


# Get Run ID of selected model
best_run_id = best["run_id"]


# MLflow URI pointing to the model logged
# inside the selected run
model_uri = (
    f"runs:/{best_run_id}/model"
)


print(
    f"Model URI:\n"
    f"{model_uri}"
)


# Register model
registered_model = mlflow.register_model(
    model_uri=model_uri,
    name=REGISTERED_MODEL_NAME
)


print(
    f"\nRegistered model:"
    f" {REGISTERED_MODEL_NAME}"
)

print(
    f"Model version:"
    f" {registered_model.version}"
)


# ============================================================
# 15. CREATE "CHAMPION" ALIAS
# ============================================================

client = MlflowClient()


client.set_registered_model_alias(
    REGISTERED_MODEL_NAME,
    "champion",
    str(registered_model.version)
)


print(
    f"\nAlias 'champion' now points to "
    f"{REGISTERED_MODEL_NAME} "
    f"version {registered_model.version}"
)


# ============================================================
# 16. FINAL SUMMARY
# ============================================================

print("\n========================================")
print("TRAINING COMPLETED")
print("========================================")

print(
    f"Selected model : {best['model']}"
)

print(
    f"Accuracy       : {best['accuracy']:.4f}"
)

print(
    f"Weighted F1    : {best['f1_weighted']:.4f}"
)

print(
    f"MLflow Run ID  : {best['run_id']}"
)

print(
    f"Registry Model : {REGISTERED_MODEL_NAME}"
)

print(
    f"Model Version  : {registered_model.version}"
)

print(
    "Alias          : champion"
)

print("\nDone!")