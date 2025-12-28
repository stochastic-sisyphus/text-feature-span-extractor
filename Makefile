# Makefile for text-feature-span-extract pipeline
#
# Usage:
#   make run FILE=path/to/document.pdf  # Extract single document
#   make test                           # Run all tests
#   make lint                           # Run linting
#   make clean                          # Clean up generated data

.PHONY: help install install-dev test test-golden lint format clean run pipeline status artifacts

# Default target
help:
	@echo "Available targets:"
	@echo "  install      Install package and dependencies"
	@echo "  install-dev  Install with development dependencies"
	@echo "  test         Run all tests"
	@echo "  test-golden  Run golden tests only"
	@echo "  lint         Run linting (ruff, mypy)"
	@echo "  format       Format code (black, isort)"
	@echo "  clean        Clean generated data and cache"
	@echo "  run FILE=x   Extract single PDF file"
	@echo "  pipeline     Run full pipeline on seed_pdfs/"
	@echo "  status       Show pipeline status"
	@echo "  artifacts    Copy outputs to artifacts/ directory"
	@echo ""
	@echo "Examples:"
	@echo "  make run FILE=./seed_pdfs/1.1.0_30531904920240731.pdf"
	@echo "  make pipeline"
	@echo "  make test"

# Installation targets
install:
	python3 -m pip install -e .

install-dev:
	python3 -m pip install -e .[dev]

# Test targets
test:
	python3 -m pytest -v

test-golden:
	python3 -m pytest tests/golden/ -v

# Linting and formatting
lint:
	ruff check src/ tests/ scripts/
	# mypy src/  # Disabled for frozen pre-training snapshot

format:
	black src/ tests/ scripts/
	isort src/ tests/ scripts/
	ruff check --fix src/ tests/ scripts/

# Clean up
clean:
	rm -rf data/ingest/
	rm -rf data/tokens/
	rm -rf data/candidates/
	rm -rf data/predictions/
	rm -rf data/review/
	rm -rf data/logs/
	rm -rf data/models/
	rm -rf artifacts/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Pipeline operations
run:
	@if [ -z "$(FILE)" ]; then \
		echo "Error: FILE parameter required. Usage: make run FILE=path/to/document.pdf"; \
		exit 1; \
	fi
	@echo "Extracting: $(FILE)"
	@mkdir -p artifacts
	@# Create temporary directory for single file processing
	@TEMP_DIR=$$(mktemp -d) && \
	cp "$(FILE)" "$$TEMP_DIR/" && \
	python scripts/run_pipeline.py pipeline --seed-folder "$$TEMP_DIR" --verbose && \
	rm -rf "$$TEMP_DIR"
	@echo "✓ Extraction complete. Outputs saved to data/predictions/"
	@make artifacts

pipeline:
	@echo "Running full pipeline on seed_pdfs/"
	python scripts/run_pipeline.py pipeline --seed-folder seed_pdfs/ --verbose
	@make artifacts

status:
	python scripts/run_pipeline.py status

# Copy outputs to artifacts directory for easy access
artifacts:
	@echo "Copying outputs to artifacts/"
	@mkdir -p artifacts/predictions
	@mkdir -p artifacts/reports
	@if [ -d data/predictions/ ]; then \
		cp data/predictions/*.json artifacts/predictions/ 2>/dev/null || true; \
	fi
	@if [ -d data/logs/ ]; then \
		cp data/logs/*.json artifacts/reports/ 2>/dev/null || true; \
	fi
	@echo "✓ Artifacts ready in artifacts/"

# Development workflow targets
dev-setup: install-dev
	@echo "Development environment ready"

ci-test: lint test test-golden
	@echo "✓ All CI checks passed"

# Quick check for determinism
determinism-check:
	@echo "Testing determinism with seed PDFs..."
	@make clean
	@make pipeline > run1.log 2>&1
	@cp -r data/predictions artifacts/run1
	@make clean
	@make pipeline > run2.log 2>&1
	@cp -r data/predictions artifacts/run2
	@echo "Comparing outputs..."
	@if diff -r artifacts/run1 artifacts/run2 > /dev/null; then \
		echo "✓ Outputs are deterministic"; \
		rm -rf artifacts/run1 artifacts/run2 run1.log run2.log; \
	else \
		echo "✗ Outputs differ between runs"; \
		echo "Check artifacts/run1 vs artifacts/run2"; \
		exit 1; \
	fi
