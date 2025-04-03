import os
import yaml
import pandas as pd
import mlflow
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import argparse

def check_data_drift(config_path):
    """Checks data drift between training and test datasets using Evidently AI and logs results to MLflow."""
    # ✅ Load Config
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    train_path = config["data_split"]["trainset_path"]
    test_path = config["data_split"]["testset_path"]
    drift_report_path = config["evaluate"]["drift_report_path"]
    mlflow_tracking_uri = config.get("mlflow", {}).get("tracking_uri", "http://127.0.0.1:5000")

    # ✅ Load Datasets
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # ✅ Remove ID Column (if exists)
    if "Patient_ID" in train_df.columns:
        train_df = train_df.drop(columns=["Patient_ID"])
        test_df = test_df.drop(columns=["Patient_ID"])

    # ✅ Create Data Drift Report
    drift_report = Report(metrics=[DataDriftPreset()])
    drift_report.run(reference_data=train_df, current_data=test_df)

    # ✅ Ensure Report Directory Exists
    os.makedirs(os.path.dirname(drift_report_path), exist_ok=True)

    # ✅ Save Report
    drift_report.save_html(drift_report_path)
    print(f"✅ Data drift report saved at: {drift_report_path}")

    # ✅ Log Report to MLflow
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    with mlflow.start_run():
        mlflow.log_artifact(drift_report_path)
        print("📊 Drift report logged to MLflow.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml", help="Path to config file")
    args = parser.parse_args()

    check_data_drift(args.config)
