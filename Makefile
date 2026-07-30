# Makefile for InvoiceX deployment
#
# Usage:
#   make up       Start all services
#   make down     Stop all services
#   make logs     Tail service logs
#   make build    Rebuild images
#   make deploy   Full deployment (build + up)
#   make status   Show service status
#   make lint     Run linting on src/
#   make format   Format src/

.PHONY: help up down logs build deploy status lint format clean backup backup-restore

help:
	@echo "Available targets:"
	@echo "  up       Start all services (docker compose up -d)"
	@echo "  down     Stop all services"
	@echo "  logs     Tail service logs"
	@echo "  build    Rebuild Docker images"
	@echo "  deploy   Full deployment (build + start)"
	@echo "  status   Show service status"
	@echo "  lint     Run linting (ruff, mypy) on src/"
	@echo "  format   Format code (ruff) in src/"
	@echo "  clean    Clean generated data and cache"

up:
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f

build:
	docker compose build

deploy:
	@if [ ! -f .env ]; then \
		echo "ERROR: .env not found. Copy .env.template to .env and fill in values."; \
		exit 1; \
	fi
	@echo "Building and starting services..."
	docker compose up -d --build
	@echo ""
	@echo "Service URLs (set INVOICEX_PUBLIC_URL in .env for your domain):"
	@echo "  - Grafana UI: $${INVOICEX_PUBLIC_URL}"
	@echo "  - API:        $${INVOICEX_PUBLIC_URL}/api/v1"
	@echo "  - MLflow:     $${INVOICEX_PUBLIC_URL}/mlflow"
	@echo ""
	@echo "Check logs: docker compose logs -f"

status:
	docker compose ps

lint:
	ruff check src/
	mypy src/

format:
	ruff format src/
	ruff check --fix src/

clean:
	rm -rf data/ingest/ data/tokens/ data/candidates/ data/predictions/ data/review/ data/logs/ data/models/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

backup:
	docker exec invoicex-postgres /backups/pg-backup.sh

backup-restore:
	@echo "Restore instructions:"
	@echo "  gunzip < backups/mlflow_YYYYMMDD_HHMMSS.sql.gz | docker exec -i invoicex-postgres psql -U invoicex mlflow"
	@echo "  gunzip < backups/grafana_YYYYMMDD_HHMMSS.sql.gz | docker exec -i invoicex-postgres psql -U invoicex grafana"
