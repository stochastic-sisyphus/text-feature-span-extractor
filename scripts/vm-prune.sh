#!/usr/bin/env bash
# vm-prune.sh — Docker disk hygiene on the VM.
# Prunes all unneeded images (tagged and dangling), stopped containers,
# and stale builder cache. Fails with exit 50 if free disk remains below
# DISK_FLOOR_GB after pruning.
#
# Optional env vars:
#   DISK_FLOOR_GB  — minimum free GB required after prune (default: 5)
#   PRUNE_UNTIL    — age filter for image prune (default: 24h)
#   BUILDER_UNTIL  — age filter for builder prune (default: 168h)
#   DOCKER_ROOT    — docker data root to measure disk on (default: /var/lib/docker)
#
# Exit codes:
#   0  — ok
#   50 — disk floor not met after prune

set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────────────────────
DISK_FLOOR_GB="${DISK_FLOOR_GB:-5}"
PRUNE_UNTIL="${PRUNE_UNTIL:-24h}"
BUILDER_UNTIL="${BUILDER_UNTIL:-168h}"
DOCKER_ROOT="${DOCKER_ROOT:-/var/lib/docker}"

# ── Helper: measure free GB on the filesystem hosting DOCKER_ROOT ─────────────
# NOTE: df --output is GNU coreutils (Linux). This script targets the Ubuntu VM;
# it is not portable to macOS. tail -n1 discards the header line; tr strips the
# trailing "G" suffix leaving a plain integer.
free_gb() {
  df --output=avail -BG "$DOCKER_ROOT" | tail -n1 | tr -dc '0-9'
}

# ── Step 1: Measure before ───────────────────────────────────────────────────
echo "==> Measuring disk before prune"
before_gb="$(free_gb)"
printf '  -> before: %s GB free on %s\n' "$before_gb" "$DOCKER_ROOT"

# ── Step 2: Prune all images unused for longer than PRUNE_UNTIL ──────────────
# -a prunes ALL unused images (tagged + dangling). Without -a, only dangling
# images are removed; old tagged images like ghcr.io/.../invoicex:abc123
# would never be reaped because they still have a tag.
echo "==> Pruning images (all, until=${PRUNE_UNTIL})"
docker image prune -af --filter "until=${PRUNE_UNTIL}"

# ── Step 3: Prune stopped containers ─────────────────────────────────────────
echo "==> Pruning stopped containers"
docker container prune -f

# ── Step 4: Prune builder cache ───────────────────────────────────────────────
echo "==> Pruning build cache (until=${BUILDER_UNTIL})"
docker builder prune -f --filter "until=${BUILDER_UNTIL}"

# ── Step 5: Measure after ────────────────────────────────────────────────────
echo "==> Measuring disk after prune"
after_gb="$(free_gb)"
printf '  -> after:  %s GB free on %s\n' "$after_gb" "$DOCKER_ROOT"

# ── Step 6: Enforce disk floor ───────────────────────────────────────────────
if [ "$after_gb" -lt "$DISK_FLOOR_GB" ]; then
  printf 'ERROR: disk floor not met: %s GB < %s GB (exit 50)\n' \
    "$after_gb" "$DISK_FLOOR_GB" >&2
  exit 50
fi

echo "==> Prune complete (${after_gb} GB free, floor ${DISK_FLOOR_GB} GB)"
exit 0
