import os
import pandas as pd
import yaml
import argparse
import mlflow
from sklearn.model_selection import train_test_split

def split_data(config_path):
    """Splits the processed dataset into training and test sets, logs details to MLflow."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Load Config Parameters
    processed_data_path = config['featurize']['processed_path']
    trainset_path = config['data_split']['trainset_path']
    testset_path = config['data_split']['testset_path']
    test_size = config['data_split']['test_size']
    target_column = config['featurize']['target_column']

    # Start MLflow Run
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    with mlflow.start_run():
        mlflow.log_param("processed_data_path", processed_data_path)
        mlflow.log_param("trainset_path", trainset_path)
        mlflow.log_param("testset_path", testset_path)
        mlflow.log_param("test_size", test_size)

        # Load Processed Data
        df = pd.read_csv(processed_data_path)

        # Split into Train & Test Sets
        X = df.drop(columns=[target_column])
        y = df[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

        # Combine X & y for saving
        train_df = X_train.copy()
        train_df[target_column] = y_train
        test_df = X_test.copy()
        test_df[target_column] = y_test

        # Ensure directories exist
        os.makedirs(os.path.dirname(trainset_path), exist_ok=True)
        os.makedirs(os.path.dirname(testset_path), exist_ok=True)

        # Save Train & Test Sets
        train_df.to_csv(trainset_path, index=False)
        test_df.to_csv(testset_path, index=False)
        print(f"✅ Train data saved at: {trainset_path}")
        print(f"✅ Test data saved at: {testset_path}")

        # Log the split data files as MLflow artifacts
        mlflow.log_artifact(trainset_path)
        mlflow.log_artifact(testset_path)

        # Log dataset stats
        mlflow.log_metric("train_size", len(train_df))
        mlflow.log_metric("test_size", len(test_df))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml", help="Path to config file")
    args = parser.parse_args()
    
    split_data(args.config)
