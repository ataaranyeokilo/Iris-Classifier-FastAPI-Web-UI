# Iris Classifier — FastAPI + Web UI

A minimal end-to-end ML app that trains an Iris flower classifier, exposes a **FastAPI** prediction service, and ships with a lightweight **HTML/JS** web UI. GitHub Actions run tests and training on every push.

## Tech Stack

* **Python** (scikit-learn, pandas, numpy, mlflow)
* **FastAPI** + **Uvicorn**
* **TailwindCSS** (CDN) for frontend styling
* **PyTest** for tests
* **GitHub Actions** CI
* **Docker** (optional)

---

## Quickstart (Local)

### Run backend (API)

```bash
make serve
# FastAPI will start on http://127.0.0.1:8000
```

### Run frontend (static UI)

```bash
python3 -m http.server 3000 --directory web
# Open http://127.0.0.1:3000
```

---

## API

### Interactive Docs

Swagger UI available at:

```
http://127.0.0.1:8000/docs
```

### Example Prediction (curl)

```bash
curl -s -X POST http://127.0.0.1:8000/predict \
 -H "Content-Type: application/json" \
 -d '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'
```

**Response**

```json
{
  "species": "setosa",
  "proba": {"setosa": 0.99, "versicolor": 0.01, "virginica": 0.00}
}
```

---

## Project Structure

```
.
├── src/                  # app + training scripts
│   ├── app/              # FastAPI app (main.py, routes, schemas)
│   └── models/           # train.py, evaluate.py
├── models/               # saved model(s)
├── web/                  # static frontend (index.html, css/js)
├── tests/                # unit tests
├── Makefile              # make train / make serve
├── requirements.txt
├── Dockerfile
└── .github/workflows/    # CI config
```

---

## Training

```bash
# train model
make train

# or directly
python src/models/train.py
```

Artifacts are saved under `models/`.

---

## Tests

```bash
pytest -q
```

---

## Docker

```bash
# build
docker build -t iris-api .

# run
docker run -p 8000:8000 iris-api
```

---


## Roadmap

* Model versioning with MLflow registry
* Container publish via GHCR
* Improved UI (better styling + validation)

---


