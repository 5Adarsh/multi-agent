"""
iris_mlflow_demo.py
====================
Lightweight ML training demo using scikit-learn + MLflow on the Iris dataset.

Trains a RandomForestClassifier, computes accuracy/precision/recall/f1,
logs everything to MLflow, and saves the model artifact.
"""

import os
import sys

# Force stdout/stderr to UTF-8 to prevent MLflow emoji crash on cp1252
os.environ["PYTHONIOENCODING"] = "utf-8"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import warnings
warnings.filterwarnings("ignore")

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import mlflow
import mlflow.sklearn

def main():
    print("Training Started")

    iris = load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0, stratify=y,
    )

    model_name = "RandomForestClassifier"
    n_estimators = 200
    max_depth = 10
    random_state = 42

    if "MLFLOW_HTTP_REQUEST_TIMEOUT" not in os.environ:
        os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] = "2"
    if "MLFLOW_HTTP_REQUEST_MAX_RETRIES" not in os.environ:
        os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] = "0"

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")

    if tracking_uri.startswith("http://") or tracking_uri.startswith("https://"):
        import socket
        from urllib.parse import urlparse
        parsed = urlparse(tracking_uri)
        host = parsed.hostname or "127.0.0.1"
        port = int(parsed.port) if parsed.port else 80
        try:
            with socket.create_connection((host, port), timeout=1.0):
                pass
        except (socket.timeout, ConnectionRefusedError, OSError):
            print(f"MLflow server at {tracking_uri} unreachable, falling back to sqlite:///mlflow.db")
            tracking_uri = "sqlite:///mlflow.db"

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("IRIS-ML-DEMO")

    with mlflow.start_run():
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)

        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1  # Use available CPU cores
        )
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        accuracy  = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted")
        recall    = recall_score(y_test, y_pred, average="weighted")
        f1        = f1_score(y_test, y_pred, average="weighted")

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        mlflow.sklearn.log_model(clf, "iris_random_forest_model")

        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")

    print("Training Complete")

if __name__ == "__main__":
    main()
