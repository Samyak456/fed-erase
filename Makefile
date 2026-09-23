# =============================================================================
# FedErase — Makefile
# =============================================================================
# Usage: make <target>
# All targets assume the virtual environment is already activated.
# =============================================================================

.PHONY: install test test-unit test-integration test-slow lint format typecheck \
        run-phase1 coverage clean help

# Default target
help:
	@echo ""
	@echo "FedErase — available targets:"
	@echo ""
	@echo "  install          Install project + dev dependencies (editable mode)"
	@echo "  test             Run unit + integration tests"
	@echo "  test-unit        Run unit tests only (fast)"
	@echo "  test-integration Run integration tests only"
	@echo "  test-slow        Run slow tests (full training runs)"
	@echo "  lint             Run ruff linter"
	@echo "  format           Auto-format with ruff"
	@echo "  typecheck        Run mypy type checker on ml/"
	@echo "  coverage         Run tests with coverage report"
	@echo "  run-phase1       Run Phase 1 end-to-end training script"
	@echo "  clean            Remove build artifacts and caches"
	@echo ""

# ---------------------------------------------------------------------------
# Installation
# ---------------------------------------------------------------------------
install:
	pip install -e ".[dev]"

# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------
test:
	pytest tests/ -m "unit or integration" -v

test-unit:
	pytest tests/ -m unit -v

test-integration:
	pytest tests/ -m integration -v

test-slow:
	pytest tests/ -m slow -v

# ---------------------------------------------------------------------------
# Code quality
# ---------------------------------------------------------------------------
lint:
	ruff check .

format:
	ruff format .
	ruff check --fix .

typecheck:
	mypy ml/

# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------
coverage:
	pytest tests/ -m "unit or integration" \
		--cov=ml \
		--cov=contribution \
		--cov=unlearning \
		--cov=evaluation \
		--cov-report=term-missing \
		--cov-report=html:htmlcov

# ---------------------------------------------------------------------------
# Running experiments
# ---------------------------------------------------------------------------
run-phase1:
	python scripts/run_phase1.py --config experiments/configs/phase1_baseline.yaml

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage coverage.xml
	rm -rf build dist *.egg-info
