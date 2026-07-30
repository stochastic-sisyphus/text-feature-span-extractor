#!/usr/bin/env bash
# deploy-stack.sh — Pull verified images and start the compose stack on the VM.
# Runs on the VM, invoked via ssh. Token is read from stdin (never argv).
#
# Required env vars:
#   DEPLOY_PATH      — absolute path to the deploy directory (e.g. /opt/invoicex)
#   COMPOSE_FILES    — compose file flags (e.g. "-f docker-compose.yml -f docker-compose.prod.yml --profile observability")
#   EXPECTED_IMAGES  — newline- or space-separated list of full image refs to verify after pull
#   GHCR_USER        — GitHub username for ghcr.io login
#
# Optional env vars:
#   STACK_WAIT_TIMEOUT — seconds to wait for --wait (default: 120)
#
# Stdin:
#   GHCR_TOKEN — passed via pipe or herestring, never as a positional argument
#
# Exit codes:
#   0  — success
#   10 — login failed
#   20 — pull failed
#   30 — expected image not present after pull
#   40 — stack up failed

set -euo pipefail

# ── Defaults ────────────────────────────────────────────────────────────────
STACK_WAIT_TIMEOUT="${STACK_WAIT_TIMEOUT:-120}"

# ── Validate required env ────────────────────────────────────────────────────
for var in DEPLOY_PATH COMPOSE_FILES EXPECTED_IMAGES GHCR_USER; do
  if [ -z "${!var:-}" ]; then
    printf 'ERROR: required env var %s is not set\n' "$var" >&2
    exit 1
  fi
done

# ── Read token from stdin (never argv) ──────────────────────────────────────
read -r GHCR_TOKEN

# ── Build compose command array ──────────────────────────────────────────────
# COMPOSE_FILES is intentionally word-split; shellcheck disable SC2206 is
# intentional — we want the flags split into discrete array elements.
# shellcheck disable=SC2206
read -ra COMPOSE_ARGS <<< "$COMPOSE_FILES"
COMPOSE_CMD=(docker compose "${COMPOSE_ARGS[@]}")

# ── Step 1: Login ────────────────────────────────────────────────────────────
echo "==> Logging in to ghcr.io"
if ! printf '%s' "$GHCR_TOKEN" | docker login ghcr.io -u "$GHCR_USER" --password-stdin; then
  printf 'ERROR: docker login failed (exit 10)\n' >&2
  exit 10
fi

# ── Step 2: cd to deploy path ────────────────────────────────────────────────
echo "==> Changing to deploy path: $DEPLOY_PATH"
if ! cd "$DEPLOY_PATH"; then
  printf 'ERROR: cannot cd to DEPLOY_PATH=%s\n' "$DEPLOY_PATH" >&2
  docker logout ghcr.io || true
  exit 1
fi

# ── Step 3: Pull images ──────────────────────────────────────────────────────
echo "==> Pulling images"
if ! "${COMPOSE_CMD[@]}" pull; then
  printf 'ERROR: compose pull failed (exit 20)\n' >&2
  docker logout ghcr.io || true
  exit 20
fi

# ── Step 4: Verify expected images are locally present ───────────────────────
echo "==> Verifying expected images"
missing_images=()
while IFS= read -r img; do
  # skip blank tokens from space- or newline-separated input
  [ -z "$img" ] && continue
  printf '  -> checking %s\n' "$img"
  if ! docker image inspect "$img" >/dev/null 2>&1; then
    missing_images+=("$img")
  fi
done <<< "$(printf '%s\n' "${EXPECTED_IMAGES}" | tr ' ' '\n')"

if [ "${#missing_images[@]}" -gt 0 ]; then
  printf 'ERROR: the following images were not found locally after pull (exit 30):\n' >&2
  for img in "${missing_images[@]}"; do
    printf '  missing: %s\n' "$img" >&2
  done
  docker logout ghcr.io || true
  exit 30
fi
echo "  -> all expected images present"

# ── Step 5: Start stack ──────────────────────────────────────────────────────
echo "==> Starting stack (--pull never --no-build --wait-timeout ${STACK_WAIT_TIMEOUT}s)"
if ! "${COMPOSE_CMD[@]}" up -d \
    --wait \
    --wait-timeout "$STACK_WAIT_TIMEOUT" \
    --pull never \
    --no-build; then
  # ── Step 6: Dump state on failure ─────────────────────────────────────────
  printf 'ERROR: compose up failed (exit 40)\n' >&2
  echo "==> Stack status at time of failure:" >&2
  "${COMPOSE_CMD[@]}" ps >&2 || true
  echo "==> Container logs at time of failure (last 100 lines per service):" >&2
  "${COMPOSE_CMD[@]}" logs --tail=100 --no-color >&2 || true
  docker logout ghcr.io || true
  exit 40
fi

# Surgical nginx restart so mounted-config changes propagate without a
# stack-wide --force-recreate. Runs only after a successful compose up.
"${COMPOSE_CMD[@]}" restart nginx

# ── Step 8: Logout (never fails the deploy) ──────────────────────────────────
echo "==> Logging out of ghcr.io"
docker logout ghcr.io || true

echo "==> Deploy complete"
exit 0
