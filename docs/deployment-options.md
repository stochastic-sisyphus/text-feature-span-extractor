# Deployment options

## Defaults

```bash
cp .env.template .env
docker compose up -d
```

| Setting | Default | Effect |
|---|---|---|
| `INVOICEX_DOCUMENT_SOURCE` | `none` | no ingest, valid values `none` `sharepoint` |
| `INVOICEX_OUTPUT_BACKEND` | `postgres` | results in Postgres, valid values `postgres` `dataverse` |
| `INVOICEX_PDF_SOURCE` | `local` | Go plugin PDF proxy only, reads `<INVOICEX_PDF_LOCAL_DIR>/<sha256>.pdf`, other values `http` `sharepoint` |
| `INVOICEX_CONNECTOR_MODE` | `local` | `azure` fills sharepoint and dataverse defaults |

- Stack: postgres, postgrest, worker, mlflow, grafana, nginx
- No local-folder ingest in `src`
- `INVOICEX_PDF_SOURCE` is the plugin proxy, separate from the Python ingest setting
- With `none` rows and jobs can be created by RPC (`ingest_doc`, `fn_pgqueuer_enqueue`)
- PDF bytes enter the worker only through `SharePointConnector.download` in `reextract`, the only `download` call site in `src`
- `sharepoint_wake` is scheduled `*/15 * * * *` and raises when SharePoint is unconfigured
- `INVOICEX_OUTPUT_BACKEND=dataverse` is validated config only, `DataverseConnector` has no call site in `src`
- `docker compose --profile observability up -d` adds loki, tempo, prometheus, otel-collector, alertmanager

## Access

- `web_anon` holds EXECUTE on write RPCs including `ingest_doc` `append_contract_schema` `submit_label` `fn_pgqueuer_enqueue`
- PostgREST runs `PGRST_DB_ANON_ROLE=web_anon` with no JWT setting
- nginx `/api/rpc/` proxies to postgrest with no `auth_request` or key header, only `/mlflow/` has `auth_basic`
- Restrict network access (VPN or private subnet) before exposing port 80

## SharePoint path

```bash
INVOICEX_DOCUMENT_SOURCE=sharepoint
```

| Variable | Note |
|---|---|
| `AZURE_TENANT_ID` | required |
| `SHAREPOINT_SITE_ID` | path A, with `SHAREPOINT_DRIVE_ID` |
| `SHAREPOINT_HOSTNAME` + `SHAREPOINT_SITE_PATH` | path B, runtime resolution |
| `SHAREPOINT_FOLDER` | folder scope |
| `SHAREPOINT_CLIENT_ID` `SHAREPOINT_CLIENT_SECRET` | service principal, else `AZURE_CLIENT_ID` or managed identity via `AZURE_MANAGED_IDENTITY_CLIENT_ID` |

- `queue.py` imports `azure.sharepoint._connector.SharePointConnector` directly in `sharepoint_wake` and `reextract`
- Plugin PDF proxy for SharePoint docs: `INVOICEX_PDF_SOURCE=sharepoint`, or `http` with `INVOICEX_PDF_FETCH_URL_TEMPLATE`

## Deploy targets

| Workflow | Target |
|---|---|
| `.github/workflows/deploy-vm.yml` | any Docker host over SSH |
| `.github/workflows/deploy-azure.yml` | Azure Container Apps, manual trigger |
