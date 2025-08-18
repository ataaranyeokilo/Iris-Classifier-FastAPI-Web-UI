import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

RAW_PATH = 'data/raw/iris.csv'
PROC_PATH = 'data/processed/iris_processed.csv'

def main():
    df = pd.read_csv(RAW_PATH)
    X = df.drop(columns=['target', 'target_name'])
    y = df['target']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    processed = pd.concat([X_scaled, y], axis=1)
    os.makedirs(os.path.dirname(PROC_PATH), exist_ok=True)
    processed.to_csv(PROC_PATH, index=False)
    print(f'Saved processed dataset to {PROC_PATH}. Shape: {processed.shape}')

if __name__ == '__main__':
    main()
