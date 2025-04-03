import os
import yaml
import argparse
import mlflow
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def train_model(config_path):
    """Trains a machine learning model on the tuberculosis dataset and logs results to MLflow."""
    # Load Configuration
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Load Paths & Parameters
    trainset_path = config['data_split']['trainset_path']
    model_path = config['train']['model_path']
    target_column = config['featurize']['target_column']

    # Hyperparameters
    n_estimators = config['train']['n_estimators']
    max_depth = config['train']['max_depth']
    min_samples_split = config['train']['min_samples_split']
    min_samples_leaf = config['train']['min_samples_leaf']
    random_state = config['base']['random_state']

    # Start MLflow Tracking
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("TB Prediction Model")

    with mlflow.start_run():
        # Load Train Data
        df_train = pd.read_csv(trainset_path)
        X_train = df_train.drop(columns=[target_column])
        y_train = df_train[target_column]

        # Initialize Model
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state
        )

        # Train the Model
        model.fit(X_train, y_train)
        
        # Make Predictions on Train Set
        y_train_pred = model.predict(X_train)

        # Calculate Metrics
        train_accuracy = accuracy_score(y_train, y_train_pred)
        train_precision = precision_score(y_train, y_train_pred, average='macro')
        train_recall = recall_score(y_train, y_train_pred, average='macro')
        train_f1 = f1_score(y_train, y_train_pred, average='macro')

        # Log Model & Metrics to MLflow
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("min_samples_split", min_samples_split)
        mlflow.log_param("min_samples_leaf", min_samples_leaf)
        mlflow.log_param("random_state", random_state)

        mlflow.log_metric("train_accuracy", train_accuracy)
        mlflow.log_metric("train_precision", train_precision)
        mlflow.log_metric("train_recall", train_recall)
        mlflow.log_metric("train_f1", train_f1)

        # Save Model
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model, model_path)
        print(f"✅ Model saved at: {model_path}")

        # Log the model to MLflow
        mlflow.sklearn.log_model(model, "tuberculosis_model")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml", help="Path to config file")
    args = parser.parse_args()
    
    train_model(args.config)
