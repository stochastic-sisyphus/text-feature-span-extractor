#!/bin/sh
set -e

# Create the grafana and invoicex databases.
# mlflow DB is auto-created via POSTGRES_DB env var.
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    SELECT 'CREATE DATABASE grafana'
    WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'grafana')\gexec

    GRANT ALL PRIVILEGES ON DATABASE grafana TO $POSTGRES_USER;

    SELECT 'CREATE DATABASE invoicex'
    WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'invoicex')\gexec

    GRANT ALL PRIVILEGES ON DATABASE invoicex TO $POSTGRES_USER;
EOSQL

# Run Wave E schema (4 JSONB tables + GIN indexes + RPCs + PostgREST roles)
# against the invoicex database — separate from mlflow and grafana.
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "invoicex" \
    -f /docker-entrypoint-initdb.d/init.sql

echo "init-db: databases ready (mlflow, grafana, invoicex)"
