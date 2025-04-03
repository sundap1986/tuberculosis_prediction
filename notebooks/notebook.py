import os
import yaml
import argparse
import mlflow
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
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
    # Train and Test split
    from sklearn.model_selection import train_test_split
    X = df.drop(columns=['Patient_ID' , 'Class'] , axis=1)
    y = df["Class"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
    # Normalization or Standardization
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    # Initialize Model
    model = RandomForestClassifier(
      n_estimators=n_estimators,
      max_depth=max_depth,
      min_samples_split=min_samples_split,
      min_samples_leaf=min_samples_leaf,
      random_state=random_state
     )


     
    # Make Predictions on Train Set
    y_train_pred = model.predict(X_train)

    # Calculate Metrics
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_precision = precision_score(y_train, y_train_pred, average='macro')
    train_recall = recall_score(y_train, y_train_pred, average='macro')
    train_f1 = f1_score(y_train, y_train_pred, average='macro')

    # Save Model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"✅ Model saved at: {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml", help="Path to config file")
    args = parser.parse_args()
    
    train_model(args.config)
