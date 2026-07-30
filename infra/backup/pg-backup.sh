#!/bin/sh
# Restore:
#   gunzip < backups/mlflow_YYYYMMDD_HHMMSS.sql.gz | docker exec -i invoicex-postgres psql -U invoicex mlflow
#   gunzip < backups/grafana_YYYYMMDD_HHMMSS.sql.gz | docker exec -i invoicex-postgres psql -U invoicex grafana
# PostgreSQL backup script for InvoiceX databases (mlflow + grafana)
#
# Runs inside the postgres container, writes timestamped dumps to /backups/
# Keeps last 7 days of backups (auto-deletes older)
#
# Setup (run on host):
#   chmod +x infra/backup/pg-backup.sh
#   crontab -e
#   # Add: 0 2 * * * docker exec invoicex-postgres /backups/pg-backup.sh

set -euo pipefail

BACKUP_DIR="/backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RETENTION_DAYS=7

# Ensure backup directory exists
mkdir -p "$BACKUP_DIR"

echo "[$(date -Iseconds)] Starting PostgreSQL backup"

# Dump mlflow database
echo "[$(date -Iseconds)] Backing up mlflow database..."
pg_dump -U "${POSTGRES_USER:-invoicex}" -d mlflow \
  | gzip > "$BACKUP_DIR/mlflow_${TIMESTAMP}.sql.gz"

# Dump grafana database
echo "[$(date -Iseconds)] Backing up grafana database..."
pg_dump -U "${POSTGRES_USER:-invoicex}" -d grafana \
  | gzip > "$BACKUP_DIR/grafana_${TIMESTAMP}.sql.gz"

# Delete backups older than retention period
echo "[$(date -Iseconds)] Cleaning up backups older than ${RETENTION_DAYS} days..."
find "$BACKUP_DIR" -name "*.sql.gz" -type f -mtime +${RETENTION_DAYS} -delete

# Print summary
BACKUP_COUNT=$(find "$BACKUP_DIR" -name "*.sql.gz" -type f | wc -l)
BACKUP_SIZE=$(du -sh "$BACKUP_DIR" | cut -f1)
echo "[$(date -Iseconds)] Backup complete. Total backups: ${BACKUP_COUNT}, disk usage: ${BACKUP_SIZE}"
