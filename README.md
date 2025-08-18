# Iris Classifier – FastAPI + Web UI

## Run backend
make serve  # FastAPI on http://127.0.0.1:8000

## Run frontend
python3 -m http.server 3000 --directory web
# Open http://127.0.0.1:3000

## Predict (curl)
curl -s -X POST http://127.0.0.1:8000/predict \
 -H "Content-Type: application/json" \
 -d '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'

Tech: scikit-learn, FastAPI, Uvicorn, Tailwind (CDN). Includes CORS + Swagger at `/docs`.
