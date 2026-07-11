import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn
import os

def main():
    # Parsing command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_estimators', type=int, default=100)
    parser.add_argument('--max_depth', type=int, default=6)
    parser.add_argument('--min_samples_split', type=int, default=2)
    args = parser.parse_args()

    # Load preprocessed dataset
    data_path = "titanic_preprocessing.csv"
    if not os.path.exists(data_path):
        # Check parent folder if not found (in case of docker/custom run env)
        if os.path.exists("../titanic_preprocessing.csv"):
            data_path = "../titanic_preprocessing.csv"
        else:
            raise FileNotFoundError(f"Dataset not found at {data_path}")
            
    df = pd.read_csv(data_path)
    X = df.drop(columns=['Survived'])
    y = df['Survived']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # MLflow start run
    # If MLFLOW_TRACKING_URI is set in environment, it will log there
    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        mlflow.log_param("min_samples_split", args.min_samples_split)
        mlflow.log_param("random_state", 42)

        # Train model
        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_split=args.min_samples_split,
            random_state=42
        )
        model.fit(X_train, y_train)

        # Evaluate model
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)

        # Log model
        mlflow.sklearn.log_model(model, "model")

        print("Training complete!")
        print(f"Metrics - Accuracy: {acc:.4f}, F1: {f1:.4f}")

if __name__ == "__main__":
    main()
