# src/app/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Optional
import joblib
import pandas as pd  # pip install pandas

app = FastAPI(title="Iris Classifier API", version="1.0")

# --- CORS so a local website can call the API ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----- Model + labels -----
CLASS_NAMES = ["setosa", "versicolor", "virginica"]

# The exact column names used during training:
TRAINING_COLS = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]

# API -> training-name mapping
FEATURE_MAP = {
    "sepal_length": "sepal length (cm)",
    "sepal_width":  "sepal width (cm)",
    "petal_length": "petal length (cm)",
    "petal_width":  "petal width (cm)",
}

# Load model once at import (simple)
model = joblib.load("models/iris_model.pkl")

# ----- Schemas -----
class Iris(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

class PredictResponse(BaseModel):
    class_id: int
    class_name: str
    probabilities: Optional[Dict[str, float]] = None  # e.g. {"setosa": 0.02, ...}

# ----- Routes -----
@app.get("/")
def root():
    return {"message": "API is running. See /docs for usage."}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict(iris: Iris):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # Build row using the ORIGINAL training column names
    row = {
        FEATURE_MAP["sepal_length"]: iris.sepal_length,
        FEATURE_MAP["sepal_width"]:  iris.sepal_width,
        FEATURE_MAP["petal_length"]: iris.petal_length,
        FEATURE_MAP["petal_width"]:  iris.petal_width,
    }
    # Ensure exact names + order match training
    X = pd.DataFrame([row], columns=TRAINING_COLS)

    # Prediction
    class_id = int(model.predict(X)[0])
    class_name = CLASS_NAMES[class_id]

    # Probabilities (only if model supports it)
    probs = None
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)[0].tolist()
        # round to 4 dp for nicer API output
        probs = {CLASS_NAMES[i]: round(p[i], 4) for i in range(len(CLASS_NAMES))}

    return {"class_id": class_id, "class_name": class_name, "probabilities": probs}
