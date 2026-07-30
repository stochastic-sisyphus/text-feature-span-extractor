"""Vendor-neutral ingest/storage protocols.

Defines the ``DocumentSource``, ``DataStore``, and ``Ledger`` protocols the
core pipeline depends on. Concrete adapters (SharePoint, Dataverse, or any
other backend) live in their own packages and implement these protocols —
see ``invoices.azure`` for the reference SharePoint/Dataverse adapter.
"""

from .base import DataStore, Document, DocumentSource, Ledger, LedgerEntry

__all__ = ["DataStore", "Document", "DocumentSource", "Ledger", "LedgerEntry"]
