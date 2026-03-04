"""Azure service abstractions for invoice extraction pipeline.

This package provides protocol-based interfaces for Azure services,
with swappable implementations for local development and production.

Core Components:
    - base: Protocol definitions (DocumentSource, DataStore, Ledger)
    - config: Azure configuration management
    - sharepoint: SharePoint document library connector
    - dataverse: Dataverse table connector

Usage:
    from invoices.azure import DocumentSource, DataStore, ConnectorMode
    from invoices.azure.config import get_azure_config

    config = get_azure_config()
    if config.mode == ConnectorMode.LOCAL:
        from invoices.mocks import MockSharePointConnector
        connector = MockSharePointConnector()
    else:
        from invoices.azure.sharepoint import SharePointConnector
        connector = SharePointConnector(config.sharepoint)
"""

from .base import (
    ConnectorMode,
    DataStore,
    Document,
    DocumentSource,
    Ledger,
    LedgerEntry,
)
from .config import AzureConfig, get_azure_config
from .ledger import DataverseLedger

__all__ = [
    # Enums
    "ConnectorMode",
    # Data classes
    "Document",
    "LedgerEntry",
    # Protocols
    "DocumentSource",
    "DataStore",
    "Ledger",
    # Implementations
    "DataverseLedger",
    # Configuration
    "AzureConfig",
    "get_azure_config",
]
