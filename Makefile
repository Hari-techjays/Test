.PHONY: help install install-train install-serve lint typecheck test test-cov
.PHONY: ingest preprocess train tune evaluate serve
.PHONY: docker-build-train docker-build-serve docker-up docker-down

help:
	@echo "Available targets:"
	@echo "  install-train   Install training dependencies"
	@echo "  install-serve   Install serving dependencies"
	@echo "  lint            Run ruff linter"
	@echo "  typecheck       Run mypy type checker"
	@echo "  test            Run tests"
	@echo "  test-cov        Run tests with coverage"
	@echo "  ingest          Download raw review data"
	@echo "  preprocess      Clean and preprocess data"
	@echo "  train           Train sentiment model"
	@echo "  tune            Run hyperparameter tuning"
	@echo "  evaluate        Evaluate model on test set"
	@echo "  serve           Start inference server"
	@echo "  docker-build-train  Build training Docker image"
	@echo "  docker-build-serve  Build serving Docker image"
	@echo "  docker-up       Start all services with docker-compose"
	@echo "  docker-down     Stop all services"

install-train:
	pip install -r requirements-train.txt

install-serve:
	pip install -r requirements-serve.txt

lint:
	ruff check src/ tests/

typecheck:
	mypy src/

test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=src --cov-report=term-missing

ingest:
	python -m src.data.ingest

preprocess:
	python -m src.data.preprocess

train:
	python -m src.model.train

tune:
	python -m src.model.tune

evaluate:
	python -m src.model.evaluate

serve:
	uvicorn src.serving.app:app --host 0.0.0.0 --port 8080 --reload

docker-build-train:
	docker build -f docker/Dockerfile.train -t sentiment-train .

docker-build-serve:
	docker build -f docker/Dockerfile.serve -t sentiment-serve .

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down
