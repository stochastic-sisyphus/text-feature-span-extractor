# TLS Configuration

## Production (Azure Container Apps)

TLS is terminated at the Azure Container Apps ingress layer, not by nginx.

- **Azure ingress**: Automatic managed HTTPS certificates, terminates TLS
- **nginx reverse proxy**: Runs HTTP on port 80 internally, behind the ingress
- **Traffic flow**: `Internet HTTPS → Azure ingress → nginx:80 → services`

The nginx config forwards the `X-Forwarded-Proto` header from Azure to downstream services (Grafana, FastAPI) so they know the original protocol was HTTPS.

Azure Container Apps provides:
- Automatic certificate provisioning and renewal
- TLS 1.2+ enforced at ingress
- HTTPS endpoint: `https://<app-name>.azurecontainerapps.io`

No additional TLS config needed in nginx for production.

## Local Development (HTTPS)

For local HTTPS testing, generate a self-signed certificate:

```bash
# Generate self-signed cert (valid 365 days)
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout infra/nginx/ssl/nginx.key \
  -out infra/nginx/ssl/nginx.crt \
  -subj "/CN=localhost"

# Create volume mount in docker-compose.yml
volumes:
  - ./infra/nginx/ssl:/etc/nginx/ssl:ro
```

Add to `nginx.conf`:

```nginx
server {
    listen 443 ssl;
    ssl_certificate /etc/nginx/ssl/nginx.crt;
    ssl_certificate_key /etc/nginx/ssl/nginx.key;
    # ... rest of config
}
```

Browser will show certificate warning (expected for self-signed certs).
