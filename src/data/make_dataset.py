import os
import pandas as pd
from sklearn.datasets import load_iris

RAW_PATH = 'data/raw/iris.csv'

def main():
    iris = load_iris(as_frame=True)
    df = iris.frame
    df['target_name'] = df['target'].map({i:name for i, name in enumerate(iris.target_names)})
    os.makedirs(os.path.dirname(RAW_PATH), exist_ok=True)
    df.to_csv(RAW_PATH, index=False)
    print(f'Saved raw dataset to {RAW_PATH}. Shape: {df.shape}')

if __name__ == '__main__':
    main()
