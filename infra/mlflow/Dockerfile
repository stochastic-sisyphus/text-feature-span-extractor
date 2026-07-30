# Custom MLflow image with PostgreSQL support
FROM ghcr.io/mlflow/mlflow:v3.1.4

# Install psycopg2 for PostgreSQL backend
RUN pip install --no-cache-dir "psycopg2-binary>=2.9.0,<3.0" "azure-storage-blob>=12.19.0" "azure-identity>=1.16.1"

# Healthcheck and entrypoint inherited from base image
