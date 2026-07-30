"""Module-level constants for azure.sharepoint package."""

from __future__ import annotations

GRAPH_BASE_URL = "https://graph.microsoft.com/v1.0"
GRAPH_SCOPE = "https://graph.microsoft.com/.default"
# Refresh token 5 minutes before actual expiry
TOKEN_REFRESH_BUFFER_SECONDS = 300
