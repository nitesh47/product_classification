.PHONY: setup-dev format lint

LINT_DIRS = predict_category

setup-dev:
	pip install poetry
	poetry install
	pre-commit install

#format:
#	poetry run isort $(LINT_DIRS)
#	poetry run black $(LINT_DIRS)

#lint:
#	poetry run isort -c $(LINT_DIRS)
#	poetry run flake8 $(LINT_DIRS)
