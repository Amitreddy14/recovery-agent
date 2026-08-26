.PHONY: install check test lint type contracts clean

install:
	pip install -e ".[dev]"

lint:
	ruff check src tests
	ruff format --check src tests

type:
	mypy src

contracts:
	lint-imports

test:
	pytest --cov=recovery --cov-report=term-missing

check: lint type contracts test

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache .coverage htmlcov
