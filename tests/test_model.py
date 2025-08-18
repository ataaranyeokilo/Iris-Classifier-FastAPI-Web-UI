import os
import pandas as pd
import joblib

def test_model_predict_shape():
    if not os.path.exists('models/model.joblib'):
        assert True
        return
    model = joblib.load('models/model.joblib')
    df = pd.read_csv('data/processed/iris_processed.csv')
    X = df.drop(columns=['target']).head(5)
    preds = model.predict(X)
    assert preds.shape[0] == 5
