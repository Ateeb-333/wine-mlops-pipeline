PYTHON ?= python
PIP ?= $(PYTHON) -m pip
TRACKING_URI ?= sqlite:///mlflow.db

.PHONY: install lint test train evaluate mlflow-ui clean clean-mlflow all

install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

lint:
	$(PYTHON) -m flake8 src/ tests/ --max-line-length=100

test:
	$(PYTHON) -m pytest tests/ -v

train:
	$(PYTHON) -m src.train --tracking-uri $(TRACKING_URI)

evaluate:
	$(PYTHON) -m src.evaluate --tracking-uri $(TRACKING_URI)

mlflow-ui:
	mlflow ui --backend-store-uri $(TRACKING_URI)

clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	rm -rf .pytest_cache .coverage htmlcov build dist *.egg-info search_results.csv

clean-mlflow:
	rm -rf mlruns mlartifacts mlflow.db

all: install lint test train evaluate
