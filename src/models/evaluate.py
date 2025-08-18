import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import yaml

PROC_PATH = 'data/processed/iris_processed.csv'
MODEL_PATH = 'models/model.joblib'
FIG_PATH = 'reports/figures/confusion_matrix.png'

def load_params():
    with open('params.yaml', 'r') as f:
        return yaml.safe_load(f)

def main():
    params = load_params()['train']
    df = pd.read_csv(PROC_PATH)
    X = df.drop(columns=['target'])
    y = df['target']

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=params['test_size'], random_state=params['random_state'], stratify=y
    )

    model = joblib.load(MODEL_PATH)
    preds = model.predict(X_test)

    disp = ConfusionMatrixDisplay.from_predictions(y_test, preds)
    os.makedirs(os.path.dirname(FIG_PATH), exist_ok=True)
    plt.savefig(FIG_PATH, bbox_inches='tight')
    print(f'Confusion matrix saved to {FIG_PATH}')

if __name__ == '__main__':
    main()
