import os
import json
import joblib
import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import yaml

PROC_PATH = 'data/processed/iris_processed.csv'
MODEL_PATH = 'models/model.joblib'
METRICS_PATH = 'reports/metrics.json'

def load_params():
    with open('params.yaml', 'r') as f:
        return yaml.safe_load(f)

def main():
    mlflow.set_tracking_uri('sqlite:///mlflow.db')
    mlflow.set_experiment('iris_classification')

    params = load_params()['train']
    df = pd.read_csv(PROC_PATH)
    X = df.drop(columns=['target'])
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=params['test_size'], random_state=params['random_state'], stratify=y
    )

    pipe = Pipeline(steps=[
        ('clf', LogisticRegression(C=params['C'], max_iter=params['max_iter']))
    ])

    with mlflow.start_run():
        mlflow.log_params(params)
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        acc = float(accuracy_score(y_test, preds))
        f1 = float(f1_score(y_test, preds, average='macro'))
        mlflow.log_metric('accuracy', acc)
        mlflow.log_metric('f1_macro', f1)

        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(pipe, MODEL_PATH)
        mlflow.log_artifact(MODEL_PATH, artifact_path='model')

        os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
        with open(METRICS_PATH, 'w') as f:
            json.dump({'accuracy': acc, 'f1_macro': f1}, f, indent=2)

        print(f'Training complete. Accuracy={acc:.3f}, F1_macro={f1:.3f}')
        print(f'Model saved to {MODEL_PATH} and metrics to {METRICS_PATH}.')

if __name__ == '__main__':
    main()
