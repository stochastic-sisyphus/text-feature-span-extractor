#!/usr/bin/env bash
# init-vm.sh — One-time VM setup before first deploy.
# Run this on the Azure VM after SSH-ing in via WireGuard (az ssh vm).
# Usage: git clone <repo> && cd <repo> && bash scripts/init-vm.sh
set -euo pipefail

APP_DIR="/opt/invoicex"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=== InvoiceX VM Init ==="

# 1. Create application directory
sudo mkdir -p "$APP_DIR"
sudo chown "$(whoami):$(whoami)" "$APP_DIR"
echo "OK: Created $APP_DIR"

# 2. Pre-create ALL bind-mount source directories as the deploy user.
# Docker auto-creates missing bind-mount sources as root, which then blocks
# SCP writes. Owning them here prevents that entirely.
mkdir -p \
    "$APP_DIR/backups" \
    "$APP_DIR/data" \
    "$APP_DIR/infra/nginx" \
    "$APP_DIR/infra/postgres" \
    "$APP_DIR/infra/grafana/provisioning" \
    "$APP_DIR/infra/grafana/dashboards" \
    "$APP_DIR/grafana-plugins/invoicex-labeling-app/dist"
echo "OK: Created bind-mount directories (deploy-user owned)"

# 3. Create .env from template (if not exists)
if [ ! -f "$APP_DIR/.env" ]; then
    if [ -f "$REPO_ROOT/.env.template" ]; then
        cp "$REPO_ROOT/.env.template" "$APP_DIR/.env"
        echo "OK: Copied .env.template -> $APP_DIR/.env"
        echo "  NOTE: Edit $APP_DIR/.env with production secrets before deploying"
    else
        echo "WARN: .env.template not found in repo — create $APP_DIR/.env manually"
    fi
else
    echo "OK: .env already exists (skipped)"
fi

# 4. Generate fresh .htpasswd for MLflow basic auth
HTPASSWD_FILE="$APP_DIR/infra/nginx/.htpasswd"
if [ ! -f "$HTPASSWD_FILE" ]; then
    mkdir -p "$APP_DIR/infra/nginx"
    MLFLOW_PASS=$(openssl rand -base64 12)
    htpasswd -nbB mlflow "$MLFLOW_PASS" > "$HTPASSWD_FILE"
    echo "OK: Generated fresh .htpasswd for MLflow"
    printf '%s\n' "$MLFLOW_PASS" > "$APP_DIR/backups/.mlflow-pass"
    chmod 600 "$APP_DIR/backups/.mlflow-pass"
    echo "  MLflow credentials saved to $APP_DIR/backups/.mlflow-pass"
    echo "  NOTE: Read that file for the password, then delete it"
else
    echo "OK: .htpasswd already exists (skipped)"
fi

# 5. Copy infra config from repo
# schema/ was removed (no migration layer; schema lives in infra/postgres/init.sql)
for item in docker-compose.yml docker-compose.prod.yml infra/; do
    if [ -e "$REPO_ROOT/$item" ]; then
        cp -r "$REPO_ROOT/$item" "$APP_DIR/"
        echo "OK: Copied $item -> $APP_DIR/"
    else
        echo "WARN: $item not found in repo (skipped)"
    fi
done

# 6. Install Docker if missing
if ! command -v docker &>/dev/null; then
    echo "Installing Docker..."
    sudo install -m 0755 -d /etc/apt/keyrings
    sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
    sudo chmod a+r /etc/apt/keyrings/docker.asc
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    sudo apt-get update
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
    sudo usermod -aG docker "$(whoami)"
    echo "OK: Docker installed (re-login for group changes)"
else
    echo "OK: Docker already installed ($(docker --version))"
fi

# 7. Pull images
cd "$APP_DIR"
if [ -f docker-compose.yml ]; then
    docker compose pull 2>/dev/null || echo "WARN: Image pull failed — deploy will pull on first run"
fi

# 8. Install systemd service
if [ -f "$REPO_ROOT/infra/invoicex.service" ]; then
    sudo cp "$REPO_ROOT/infra/invoicex.service" /etc/systemd/system/
    sudo systemctl daemon-reload
    sudo systemctl enable invoicex
    echo "OK: Installed systemd service (enabled on boot)"
else
    echo "WARN: invoicex.service not found (skipped)"
fi

# 9. Install logrotate config
if [ -f "$REPO_ROOT/infra/logrotate.conf" ]; then
    sudo cp "$REPO_ROOT/infra/logrotate.conf" /etc/logrotate.d/invoicex
    echo "OK: Installed logrotate config"
else
    echo "WARN: logrotate.conf not found (skipped)"
fi

echo ""
echo "=== Init complete ==="
echo "Next steps:"
echo "  1. Edit $APP_DIR/.env — set INVOICEX_PUBLIC_URL to the VM's WireGuard IP"
echo "  2. Push code & trigger deploy-vm.yml workflow"
echo "  3. Start service: sudo systemctl start invoicex"
echo "  4. Check status: sudo systemctl status invoicex"
