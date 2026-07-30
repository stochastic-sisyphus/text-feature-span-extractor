#!/bin/sh
# Render infra/alertmanager.yml.template into a shared volume that alertmanager
# mounts at /etc/alertmanager/alertmanager.yml. Runs as a one-shot init service
# (see docker-compose.yml service 'alertmanager-config').
#
# Optional env:
#   ALERT_WEBHOOK_URL — webhook endpoint for all receivers (Slack incoming
#                       webhook, generic HTTP receiver, etc.). When unset or
#                       empty, a placeholder URL is substituted so alertmanager
#                       still starts and exposes alert state in its UI;
#                       webhook POSTs will fail harmlessly until a real URL
#                       is configured.
#
# Behavior:
#   - Missing/empty ALERT_WEBHOOK_URL -> LOUD warning, substitute a .invalid
#     placeholder, exit 0. Alertmanager starts and serves its UI; external
#     webhook delivery is disabled until the env var is set.
#   - All other ${VAR} references in the template expand via envsubst; any
#     unset var becomes an empty string, which is the standard envsubst contract.
#
# Why sh (not bash): the alpine base image ships ash, not bash.

set -eu

TEMPLATE="${TEMPLATE:-/etc/alertmanager/alertmanager.yml.template}"
OUTPUT="${OUTPUT:-/rendered/alertmanager.yml}"

if [ ! -f "$TEMPLATE" ]; then
    echo "[alertmanager-render] FATAL: template not found at $TEMPLATE" >&2
    exit 1
fi

if [ -z "${ALERT_WEBHOOK_URL:-}" ]; then
    echo "==========================================================" >&2
    echo "[alertmanager-render] WARNING: ALERT_WEBHOOK_URL is unset." >&2
    echo "[alertmanager-render] Alertmanager will start but external" >&2
    echo "[alertmanager-render] webhook notifications will NOT be" >&2
    echo "[alertmanager-render] delivered. Alerts remain visible in" >&2
    echo "[alertmanager-render] the Alertmanager UI" >&2
    echo "[alertmanager-render] (http://alertmanager:9093 in the" >&2
    echo "[alertmanager-render] docker network). To enable webhook" >&2
    echo "[alertmanager-render] delivery, set ALERT_WEBHOOK_URL in" >&2
    echo "[alertmanager-render] the VM .env file and redeploy." >&2
    echo "==========================================================" >&2
    # RFC 2606 reserves .invalid; no DNS resolution, POST fails cleanly.
    ALERT_WEBHOOK_URL="http://webhook-not-configured.invalid/"
    export ALERT_WEBHOOK_URL
fi

# Restrict envsubst to the vars we actually template so unrelated $VAR-looking
# strings in the config (unlikely, but possible in future receivers) are not
# silently stripped.
envsubst '${ALERT_WEBHOOK_URL}' < "$TEMPLATE" > "$OUTPUT"

echo "[alertmanager-render] rendered $TEMPLATE -> $OUTPUT"
echo "[alertmanager-render] ALERT_WEBHOOK_URL length: ${#ALERT_WEBHOOK_URL} chars"
