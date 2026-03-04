"""Dataverse table connector for invoice data persistence.

Provides CRUD operations for Dataverse tables used to store
extracted invoice data and manage the review workflow.

Usage:
    from invoices.azure.dataverse import DataverseConnector
    from invoices.azure.config import get_azure_config

    config = get_azure_config()
    async with DataverseConnector(config.dataverse) as connector:
        record_id = await connector.create("new_tem", staging_record)
        await connector.promote_to_production(record_id)
"""

from datetime import datetime, timezone
from typing import Any

from invoices.azure.base import DataStore
from invoices.azure.config import DataverseConfig
from invoices.azure.record_builder import build_production_record, build_staging_record
from invoices.exceptions import IntegrationError
from invoices.logging import get_logger

logger = get_logger(__name__)


class DataverseError(IntegrationError):
    """Dataverse-specific error."""

    def __init__(
        self,
        operation: str,
        reason: str,
        status_code: int | None = None,
        request_id: str | None = None,
        table: str | None = None,
    ):
        message = f"Dataverse {operation} failed: {reason}"
        super().__init__(
            message,
            operation=operation,
            reason=reason,
            status_code=status_code,
            request_id=request_id,
            table=table,
        )
        self.operation = operation
        self.reason = reason
        self.status_code = status_code
        self.request_id = request_id
        self.table = table


class DataverseConnector(DataStore):
    """Dataverse connector using Web API.

    Implements the DataStore protocol for Dataverse tables.

    Auth: When client_secret is set, uses ClientSecretCredential (service
    principal). Otherwise, falls back to DefaultAzureCredential (managed
    identity in Azure, CLI/env credentials locally).

    Usage:
        config = DataverseConfig(
            environment_url="https://org.crm.dynamics.com",
            tenant_id="...",
            client_id="...",
            client_secret="...",  # omit for managed identity
        )
        async with DataverseConnector(config) as connector:
            await connector.create("new_tem", record)
    """

    def __init__(self, config: DataverseConfig):
        """Initialize the Dataverse connector.

        Args:
            config: Dataverse configuration
        """
        self.config = config
        self._session: Any = None
        self._token: str | None = None
        self._token_expires: datetime | None = None

    async def __aenter__(self) -> "DataverseConnector":
        """Enter async context and initialize session."""
        await self._init_session()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit async context and cleanup."""
        await self._cleanup()

    async def _init_session(self) -> None:
        """Initialize the HTTP session and acquire token."""
        try:
            import httpx
        except ImportError as e:
            raise DataverseError(
                "init",
                f"httpx not installed. Run: pip install httpx. Error: {e}",
            ) from e

        if not self.config.is_configured():
            raise DataverseError(
                "init",
                "Dataverse not properly configured. Check environment variables.",
            )

        try:
            if not self.config.environment_url:
                raise DataverseError(
                    "init",
                    "DATAVERSE_ENVIRONMENT_URL is required but not set. "
                    "Cannot fall back to localhost.",
                )
            base_url: str | httpx.URL = self.config.environment_url
            self._session = httpx.AsyncClient(
                base_url=base_url,
                timeout=30.0,
            )
            await self._acquire_token()
            logger.info(
                "dataverse_session_initialized",
                environment_url=self.config.environment_url,
            )
        except Exception as e:
            raise DataverseError("init", str(e)) from e

    async def _cleanup(self) -> None:
        """Cleanup session resources."""
        if self._session:
            try:
                await self._session.aclose()
            except Exception as e:
                logger.warning("dataverse_session_cleanup_failed", error=str(e))
            self._session = None
        self._token = None
        self._token_expires = None

    async def _acquire_token(self) -> None:
        """Acquire OAuth2 access token.

        Uses ClientSecretCredential when client_secret is set (local dev
        with service principal), otherwise falls back to
        DefaultAzureCredential (managed identity in Azure).
        """
        credential = None
        try:
            if self.config.client_secret:
                # Service principal auth (local dev)
                try:
                    from azure.identity.aio import (  # type: ignore[import-not-found]
                        ClientSecretCredential,
                    )
                except ImportError as e:
                    raise DataverseError(
                        "auth",
                        "azure-identity not installed. Run: pip install azure-identity",
                    ) from e

                credential = ClientSecretCredential(
                    tenant_id=self.config.tenant_id,
                    client_id=self.config.client_id,
                    client_secret=self.config.client_secret,
                )
                auth_method = "client_secret"
            else:
                # Managed identity / DefaultAzureCredential (Azure)
                try:
                    from azure.identity.aio import (  # type: ignore[import-not-found]
                        DefaultAzureCredential,
                    )
                except ImportError as e:
                    raise DataverseError(
                        "auth",
                        "azure-identity not installed. Run: pip install azure-identity",
                    ) from e

                kwargs: dict[str, str] = {}
                if self.config.managed_identity_client_id:
                    kwargs["managed_identity_client_id"] = (
                        self.config.managed_identity_client_id
                    )
                credential = DefaultAzureCredential(**kwargs)
                auth_method = "default_credential"

            # Dataverse resource scope
            scope = f"{self.config.environment_url}/.default"
            token = await credential.get_token(scope)
            self._token = token.token
            self._token_expires = datetime.fromtimestamp(
                token.expires_on, tz=timezone.utc
            )

            logger.debug("dataverse_token_acquired", method=auth_method)

        except DataverseError:
            raise
        except Exception as e:
            raise DataverseError("auth", str(e)) from e
        finally:
            if credential is not None:
                await credential.close()

    async def _ensure_token(self) -> None:
        """Ensure we have a valid token, refreshing if needed."""
        if self._token is None or (
            self._token_expires and datetime.now(timezone.utc) >= self._token_expires
        ):
            await self._acquire_token()

    def _get_headers(self) -> dict[str, str]:
        """Get request headers with authorization."""
        return {
            "Authorization": f"Bearer {self._token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
            "OData-MaxVersion": "4.0",
            "OData-Version": "4.0",
            "Prefer": "return=representation",
        }

    async def create(self, table: str, record: dict[str, Any]) -> str:
        """Create a new record in a table.

        Args:
            table: Table name (e.g., "new_tem")
            record: Record data

        Returns:
            ID of the created record
        """
        if self._session is None:
            raise DataverseError("create", "Session not initialized", table=table)

        await self._ensure_token()

        try:
            # Dataverse uses plural table names in API
            table_set = self._get_entity_set_name(table)
            url = f"/api/data/v9.2/{table_set}"

            response = await self._session.post(
                url,
                headers=self._get_headers(),
                json=record,
            )

            if response.status_code not in (200, 201, 204):
                error_detail = self._parse_error(response)
                raise DataverseError(
                    "create",
                    error_detail,
                    status_code=response.status_code,
                    table=table,
                )

            # Extract ID from response or OData-EntityId header
            if response.status_code == 201 and response.content:
                data = response.json()
                record_id = data.get(f"{table}id") or data.get("id")
            else:
                entity_id = response.headers.get("OData-EntityId", "")
                # Extract GUID from URL like: .../new_tems(guid)
                record_id = entity_id.split("(")[-1].rstrip(")")

            logger.info("dataverse_record_created", table=table, record_id=record_id)
            return record_id  # type: ignore[no-any-return]

        except DataverseError:
            raise
        except Exception as e:
            raise DataverseError("create", str(e), table=table) from e

    async def get(self, table: str, record_id: str) -> dict[str, Any] | None:
        """Get a record by ID.

        Args:
            table: Table name
            record_id: Record GUID

        Returns:
            Record data or None if not found
        """
        if self._session is None:
            raise DataverseError("get", "Session not initialized", table=table)

        await self._ensure_token()

        try:
            table_set = self._get_entity_set_name(table)
            url = f"/api/data/v9.2/{table_set}({record_id})"

            response = await self._session.get(
                url,
                headers=self._get_headers(),
            )

            if response.status_code == 404:
                return None

            if response.status_code != 200:
                error_detail = self._parse_error(response)
                raise DataverseError(
                    "get",
                    error_detail,
                    status_code=response.status_code,
                    table=table,
                )

            return response.json()  # type: ignore[no-any-return]

        except DataverseError:
            raise
        except Exception as e:
            raise DataverseError("get", str(e), table=table) from e

    async def update(
        self,
        table: str,
        record_id: str,
        fields: dict[str, Any],
    ) -> bool:
        """Update a record.

        Args:
            table: Table name
            record_id: Record GUID
            fields: Fields to update

        Returns:
            True if updated, False if not found
        """
        if self._session is None:
            raise DataverseError("update", "Session not initialized", table=table)

        await self._ensure_token()

        try:
            table_set = self._get_entity_set_name(table)
            url = f"/api/data/v9.2/{table_set}({record_id})"

            response = await self._session.patch(
                url,
                headers=self._get_headers(),
                json=fields,
            )

            if response.status_code == 404:
                return False

            if response.status_code not in (200, 204):
                error_detail = self._parse_error(response)
                raise DataverseError(
                    "update",
                    error_detail,
                    status_code=response.status_code,
                    table=table,
                )

            logger.info("dataverse_record_updated", table=table, record_id=record_id)
            return True

        except DataverseError:
            raise
        except Exception as e:
            raise DataverseError("update", str(e), table=table) from e

    async def query(
        self,
        table: str,
        filter_expr: str | None = None,
        order_by: str | None = None,
        top: int | None = None,
    ) -> list[dict[str, Any]]:
        """Query records with OData filtering.

        Args:
            table: Table name
            filter_expr: OData filter expression (e.g., "status eq 'pending'")
            order_by: Order by expression (e.g., "created_at desc")
            top: Maximum records to return

        Returns:
            List of matching records
        """
        if self._session is None:
            raise DataverseError("query", "Session not initialized", table=table)

        await self._ensure_token()

        try:
            table_set = self._get_entity_set_name(table)
            url = f"/api/data/v9.2/{table_set}"

            params: dict[str, str] = {}
            if filter_expr:
                params["$filter"] = filter_expr
            if order_by:
                params["$orderby"] = order_by
            if top:
                params["$top"] = str(top)

            response = await self._session.get(
                url,
                headers=self._get_headers(),
                params=params,
            )

            if response.status_code != 200:
                error_detail = self._parse_error(response)
                raise DataverseError(
                    "query",
                    error_detail,
                    status_code=response.status_code,
                    table=table,
                )

            data = response.json()
            return data.get("value", [])  # type: ignore[no-any-return]

        except DataverseError:
            raise
        except Exception as e:
            raise DataverseError("query", str(e), table=table) from e

    async def delete(self, table: str, record_id: str) -> bool:
        """Delete a record.

        Args:
            table: Table name
            record_id: Record GUID

        Returns:
            True if deleted, False if not found
        """
        if self._session is None:
            raise DataverseError("delete", "Session not initialized", table=table)

        await self._ensure_token()

        try:
            table_set = self._get_entity_set_name(table)
            url = f"/api/data/v9.2/{table_set}({record_id})"

            response = await self._session.delete(
                url,
                headers=self._get_headers(),
            )

            if response.status_code == 404:
                return False

            if response.status_code != 204:
                error_detail = self._parse_error(response)
                raise DataverseError(
                    "delete",
                    error_detail,
                    status_code=response.status_code,
                    table=table,
                )

            logger.info("dataverse_record_deleted", table=table, record_id=record_id)
            return True

        except DataverseError:
            raise
        except Exception as e:
            raise DataverseError("delete", str(e), table=table) from e

    async def health_check(self) -> dict[str, Any]:
        """Check Dataverse connector health.

        Returns:
            Health status dictionary
        """
        if self._session is None:
            return {
                "status": "unhealthy",
                "message": "Session not initialized",
                "details": {"environment_url": self.config.environment_url},
            }

        try:
            await self._ensure_token()

            # Try to query the WhoAmI endpoint
            response = await self._session.get(
                "/api/data/v9.2/WhoAmI",
                headers=self._get_headers(),
            )

            if response.status_code == 200:
                data = response.json()
                return {
                    "status": "healthy",
                    "message": "Connected to Dataverse",
                    "details": {
                        "environment_url": self.config.environment_url,
                        "user_id": data.get("UserId"),
                        "organization_id": data.get("OrganizationId"),
                    },
                }
            else:
                return {
                    "status": "degraded",
                    "message": f"WhoAmI returned {response.status_code}",
                    "details": {"environment_url": self.config.environment_url},
                }

        except Exception as e:
            return {
                "status": "unhealthy",
                "message": str(e),
                "details": {"environment_url": self.config.environment_url},
            }

    # =========================================================================
    # Invoice-Specific Methods
    # =========================================================================

    async def create_staging_record(
        self,
        document_id: str,
        extraction_result: dict[str, Any],
        sharepoint_id: str | None = None,
    ) -> str:
        """Create a staging record from extraction results.

        Args:
            document_id: Internal document ID (fs:...)
            extraction_result: Pipeline extraction result
            sharepoint_id: Source SharePoint item ID

        Returns:
            Created record ID
        """
        record = build_staging_record(document_id, extraction_result, sharepoint_id)
        return await self.create(self.config.staging_table, record)

    async def promote_to_production(
        self,
        staging_id: str,
        approved_by: str,
    ) -> str:
        """Promote a staging record to production.

        Args:
            staging_id: Staging record ID
            approved_by: Approver identifier

        Returns:
            Production record ID
        """
        # Get staging record
        staging_record = await self.get(self.config.staging_table, staging_id)
        if staging_record is None:
            raise DataverseError(
                "promote",
                f"Staging record not found: {staging_id}",
                table=self.config.staging_table,
            )

        # Create production record
        production_record = build_production_record(
            staging_record, staging_id, approved_by
        )

        production_id = await self.create(
            self.config.production_table,
            production_record,
        )

        # Update staging record status
        await self.update(
            self.config.staging_table,
            staging_id,
            {
                "status": "approved",
                "reviewed_by": approved_by,
                "reviewed_at": datetime.now(timezone.utc).isoformat(),
            },
        )

        logger.info(
            "dataverse_record_promoted",
            staging_id=staging_id,
            production_id=production_id,
            approved_by=approved_by,
        )

        return production_id

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _get_entity_set_name(self, table: str) -> str:
        """Get the OData entity set name for a table.

        Dataverse uses plural names for entity sets.
        """
        # Simple pluralization - could be enhanced
        if table.endswith("s"):
            return table + "es"
        elif table.endswith("y"):
            return table[:-1] + "ies"
        else:
            return table + "s"

    def _parse_error(self, response: Any) -> str:
        """Parse error details from response."""
        try:
            data = response.json()
            error = data.get("error", {})
            return error.get("message", str(data))  # type: ignore[no-any-return]
        except Exception:
            return response.text if hasattr(response, "text") else "Unknown error"
