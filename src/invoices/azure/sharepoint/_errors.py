"""SharePoint-specific error type."""

from __future__ import annotations

from invoices.exceptions import IntegrationError


class SharePointError(IntegrationError):
    """SharePoint-specific error."""

    def __init__(
        self,
        operation: str,
        reason: str,
        status_code: int | None = None,
        request_id: str | None = None,
    ):
        message = f"SharePoint {operation} failed: {reason}"
        super().__init__(
            message,
            operation=operation,
            reason=reason,
            status_code=status_code,
            request_id=request_id,
        )
        self.operation = operation
        self.reason = reason
        self.status_code = status_code
        self.request_id = request_id
