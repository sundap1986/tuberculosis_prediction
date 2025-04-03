import os
import yaml
import argparse
import mlflow
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_model(config_path):
    """Evaluates the trained model on the test dataset and logs results to MLflow."""
    # Load Configuration
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Load Paths
    testset_path = config['data_split']['testset_path']
    model_path = config['train']['model_path']
    metrics_path = config['evaluate']['metrics_path']

    target_column = config['featurize']['target_column']

    # Start MLflow Tracking
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("TB Prediction Model")

    with mlflow.start_run():
        # Load Test Data
        df_test = pd.read_csv(testset_path)
        X_test = df_test.drop(columns=[target_column])
        y_test = df_test[target_column]

        # Load Trained Model
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"🚨 Model file not found at {model_path}")
        
        model = joblib.load(model_path)

        # Make Predictions
        y_pred = model.predict(X_test)

        # Calculate Metrics
        test_accuracy = accuracy_score(y_test, y_pred)
        test_precision = precision_score(y_test, y_pred, average='macro')
        test_recall = recall_score(y_test, y_pred, average='macro')
        test_f1 = f1_score(y_test, y_pred, average='macro')

        # Log Metrics to MLflow
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("test_precision", test_precision)
        mlflow.log_metric("test_recall", test_recall)
        mlflow.log_metric("test_f1", test_f1)

        # Save Metrics
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        metrics = {
            "test_accuracy": test_accuracy,
            "test_precision": test_precision,
            "test_recall": test_recall,
            "test_f1": test_f1
        }
        with open(metrics_path, "w") as f:
            yaml.dump(metrics, f)

        print(f"✅ Evaluation metrics saved at: {metrics_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml", help="Path to config file")
    args = parser.parse_args()
    
    evaluate_model(args.config)
