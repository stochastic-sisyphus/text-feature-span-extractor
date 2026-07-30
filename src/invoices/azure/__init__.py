"""SharePoint/Dataverse adapter implementing the vendor-neutral ingest protocols.

This package is one concrete implementation of the ``DocumentSource``/``DataStore``
protocols defined in ``invoices.ingest`` — it's an optional adapter, not a core
pipeline dependency. Swap in a different adapter for a different backend.

Core Components:
    - config: Azure/SharePoint/Dataverse configuration management
    - sharepoint: SharePoint document library connector
    - dataverse: Dataverse table connector
"""

from invoices.ingest.base import (
    DataStore,
    Document,
    DocumentSource,
)

from .config import AzureConfig, get_azure_config

__all__ = [
    # Configuration
    "AzureConfig",
    "DataStore",
    # Data classes
    "Document",
    # Protocols
    "DocumentSource",
    "get_azure_config",
]
