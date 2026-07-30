"""Public dataclasses for SharePoint connector."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class RemovedDocument:
    """A document that was removed from SharePoint, as reported by Graph delta.

    Attributes:
        id: Graph API drive item ID of the removed document.
        reason: ``'changed'`` (moved to recycle bin, restorable) or
            ``'deleted'`` (permanently deleted, not restorable).
    """

    id: str
    reason: Literal["changed", "deleted"]
