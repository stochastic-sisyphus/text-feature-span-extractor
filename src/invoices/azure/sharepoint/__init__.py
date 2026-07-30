"""azure.sharepoint sub-package — Microsoft Graph SharePoint connector.

Public surface (backward-compatible re-exports for callers using
``from invoices.azure.sharepoint import …``).
"""

from ._connector import SharePointConnector
from ._constants import (
    GRAPH_BASE_URL,
    GRAPH_SCOPE,
    TOKEN_REFRESH_BUFFER_SECONDS,
)
from ._errors import SharePointError
from ._types import RemovedDocument

__all__ = [
    "GRAPH_BASE_URL",
    "GRAPH_SCOPE",
    "TOKEN_REFRESH_BUFFER_SECONDS",
    "RemovedDocument",
    "SharePointConnector",
    "SharePointError",
]
