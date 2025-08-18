# Makefile for ML Pipeline (macOS-friendly with python3)

.PHONY: data features train evaluate serve test mlflow docker-build docker-run install

data:
	python3 src/data/make_dataset.py

features:
	python3 src/features/build_features.py

train:
	python3 src/models/train.py

evaluate:
	python3 src/models/evaluate.py

serve:
	python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload --app-dir src


test:
	pytest tests

mlflow:
	mlflow ui

docker-build:
	docker build -t ml-pipeline .

docker-run:
	docker run -p 5000:5000 ml-pipeline

install:
	python3 -m pip install -r requirements.txt


