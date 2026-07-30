"""Azure configuration management.

Provides centralized configuration for Azure service connectors with
environment variable support.

Usage:
    from invoices.azure.config import get_azure_config

    config = get_azure_config()
    print(f"sharepoint={config.sharepoint.is_configured()}")
"""

import os
import threading
from dataclasses import dataclass, field
from typing import Any

from invoices.exceptions import MissingConfigurationError
from invoices.logging import get_logger

logger = get_logger(__name__)


def _env_str(key: str, default: str | None = None) -> str | None:
    """Get string from environment variable."""
    return os.environ.get(key, default)


def _env_bool(key: str, default: bool = False) -> bool:
    """Get boolean from environment variable."""
    value = os.environ.get(key)
    if value is None:
        return default
    return value.lower() in ("true", "1", "yes", "on")


@dataclass
class SharePointConfig:
    """Configuration for SharePoint connector via Microsoft Graph API.

    Auth: Uses ManagedIdentityCredential via VM_MANAGED_IDENTITY_CLIENT_ID.
    Falls back to DefaultAzureCredential when running locally.
    client_id / client_secret are not populated from env in this deployment
    but remain as runtime slots for service-principal auth code paths in
    the connector.

    Attributes:
        site_id: SharePoint site ID (GUID)
        hostname: SharePoint hostname (e.g. tenant.sharepoint.com)
        site_path: Site-relative path (e.g. /sites/invoices)
        tenant_id: Azure AD tenant ID (ENTRA_TENANT_ID)
        client_id: Runtime-only; not env-loaded. For service principal paths.
        client_secret: Runtime-only; not env-loaded. For service principal paths.
        managed_identity_client_id: Managed identity client ID (Azure only)
        drive_id: Resolved at runtime by _resolve_drive_id; None until resolved
        folder_path: Folder path within the drive (SHAREPOINT_FOLDER)
    """

    site_id: str | None = None
    hostname: str | None = None
    site_path: str | None = None
    tenant_id: str | None = None
    client_id: str | None = None
    client_secret: str | None = None
    managed_identity_client_id: str | None = None
    drive_id: str | None = None
    folder_path: str = ""

    @classmethod
    def from_environment(cls) -> "SharePointConfig":
        """Load configuration from environment variables."""
        return cls(
            site_id=_env_str("SHAREPOINT_SITE_ID"),
            hostname=_env_str("SHAREPOINT_HOSTNAME"),
            site_path=_env_str("SHAREPOINT_SITE_PATH"),
            tenant_id=_env_str("ENTRA_TENANT_ID"),
            managed_identity_client_id=(
                _env_str("VM_MANAGED_IDENTITY_CLIENT_ID")
                or _env_str("AZURE_MANAGED_IDENTITY_CLIENT_ID")
            ),
            folder_path=_env_str("SHAREPOINT_FOLDER") or "",
        )

    def is_configured(self) -> bool:
        """Check if SharePoint is properly configured.

        With managed identity, client_secret is not required — only
        site identification and tenant_id are mandatory. Site can be
        identified by site_id (GUID) or hostname + site_path (resolved
        at runtime).
        """
        has_site = bool(self.site_id) or bool(self.hostname and self.site_path)
        return bool(has_site and self.tenant_id)

    def validate(self) -> list[str]:
        """Validate configuration and return list of errors."""
        errors = []
        has_site = bool(self.site_id) or bool(self.hostname and self.site_path)
        if not has_site:
            errors.append(
                "SHAREPOINT_SITE_ID or both SHAREPOINT_HOSTNAME + "
                "SHAREPOINT_SITE_PATH are required"
            )
        if not self.tenant_id:
            errors.append("AZURE_TENANT_ID is required")
        if self.client_secret and not self.client_id:
            errors.append(
                "SHAREPOINT_CLIENT_ID or AZURE_CLIENT_ID is required "
                "when client_secret is set"
            )
        return errors


@dataclass
class DataverseConfig:
    """Configuration for Dataverse connector.

    Attributes:
        environment_url: Dataverse environment URL
        tenant_id: Azure AD tenant ID
        client_id: App registration client ID
        client_secret: App registration client secret
        staging_table: Staging table name (invoicex_staging)
        production_table: Production table name (invoicex_production)
    """

    environment_url: str | None = None
    tenant_id: str | None = None
    client_id: str | None = None
    client_secret: str | None = None
    managed_identity_client_id: str | None = None
    staging_table: str = "invoicex_staging"
    production_table: str = "invoicex_production"

    @classmethod
    def from_environment(cls) -> "DataverseConfig":
        """Load configuration from environment variables."""
        return cls(
            environment_url=_env_str("DATAVERSE_ENVIRONMENT_URL")
            or _env_str("DATAVERSE_ENV_URL"),
            tenant_id=_env_str("AZURE_TENANT_ID"),
            client_id=_env_str("DATAVERSE_CLIENT_ID") or _env_str("AZURE_CLIENT_ID"),
            client_secret=_env_str("DATAVERSE_CLIENT_SECRET")
            or _env_str("AZURE_CLIENT_SECRET"),
            managed_identity_client_id=(
                _env_str("VM_MANAGED_IDENTITY_CLIENT_ID")
                or _env_str("AZURE_MANAGED_IDENTITY_CLIENT_ID")
            ),
            staging_table=_env_str("DATAVERSE_STAGING_TABLE", "invoicex_staging")
            or "invoicex_staging",
            production_table=_env_str(
                "DATAVERSE_PRODUCTION_TABLE", "invoicex_production"
            )
            or "invoicex_production",
        )

    def is_configured(self) -> bool:
        """Check if Dataverse is properly configured.

        With managed identity, client_secret is not required — only
        environment_url and tenant_id are mandatory. client_id is
        needed for service principal auth but optional for managed identity.
        """
        return bool(self.environment_url and self.tenant_id)

    def validate(self) -> list[str]:
        """Validate configuration and return list of errors."""
        errors = []
        if not self.environment_url:
            errors.append(
                "DATAVERSE_ENVIRONMENT_URL (or DATAVERSE_ENV_URL) is required"
            )
        if not self.tenant_id:
            errors.append("AZURE_TENANT_ID is required")
        if not self.client_secret and not self.client_id:
            # With managed identity, neither client_id nor client_secret
            # is strictly required (identity comes from the VM/container).
            # But if using service principal, both are needed.
            pass
        elif self.client_secret and not self.client_id:
            errors.append(
                "DATAVERSE_CLIENT_ID or AZURE_CLIENT_ID is required when client_secret is set"
            )
        return errors


@dataclass
class OpenTelemetryConfig:
    """Configuration for OpenTelemetry observability.

    Attributes:
        enabled: Whether OTEL is enabled
        endpoint: OTEL collector endpoint
        service_name: Service name for tracing
        service_version: Service version
    """

    enabled: bool = False
    endpoint: str | None = None
    service_name: str = "invoicex"
    service_version: str = "1.0.0"

    @classmethod
    def from_environment(cls) -> "OpenTelemetryConfig":
        """Load configuration from environment variables."""
        return cls(
            enabled=_env_bool("OTEL_ENABLED", False),
            endpoint=_env_str("OTEL_EXPORTER_ENDPOINT"),
            service_name=_env_str("OTEL_SERVICE_NAME", "invoicex") or "invoicex",
            service_version=_env_str("OTEL_SERVICE_VERSION", "1.0.0") or "1.0.0",
        )


@dataclass
class AzureConfig:
    """Master configuration for all Azure services.

    Provides centralized access to all Azure service configurations.

    Attributes:
        sharepoint: SharePoint configuration
        dataverse: Dataverse configuration
        otel: OpenTelemetry configuration
    """

    sharepoint: SharePointConfig = field(default_factory=SharePointConfig)
    dataverse: DataverseConfig = field(default_factory=DataverseConfig)
    otel: OpenTelemetryConfig = field(default_factory=OpenTelemetryConfig)

    @classmethod
    def from_environment(cls) -> "AzureConfig":
        """Load configuration from environment variables."""
        return cls(
            sharepoint=SharePointConfig.from_environment(),
            dataverse=DataverseConfig.from_environment(),
            otel=OpenTelemetryConfig.from_environment(),
        )

    def validate(self) -> None:
        """Validate configuration and raise if invalid.

        Validates whichever services are actually configured.

        Raises:
            MissingConfigurationError: If required config is missing
        """
        errors: list[str] = []

        if self.sharepoint.is_configured():
            errors.extend(self.sharepoint.validate())

        if self.dataverse.is_configured():
            errors.extend(self.dataverse.validate())

        if errors:
            raise MissingConfigurationError(
                key="azure_config",
                description="; ".join(errors),
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary for logging/debugging."""
        return {
            "sharepoint_configured": self.sharepoint.is_configured(),
            "dataverse_configured": self.dataverse.is_configured(),
            "otel_enabled": self.otel.enabled,
        }


# Global singleton
_azure_config: AzureConfig | None = None
_AZURE_CONFIG_LOCK = threading.Lock()


def get_azure_config(reload: bool = False) -> AzureConfig:
    """Get the Azure configuration singleton.

    Args:
        reload: Force reload from environment variables

    Returns:
        AzureConfig instance
    """
    global _azure_config

    with _AZURE_CONFIG_LOCK:
        if _azure_config is None or reload:
            _azure_config = AzureConfig.from_environment()
            logger.info(
                "azure_config_loaded",
                sharepoint_configured=_azure_config.sharepoint.is_configured(),
                dataverse_configured=_azure_config.dataverse.is_configured(),
            )

        return _azure_config


def reset_azure_config() -> None:
    """Reset the configuration singleton (for testing)."""
    global _azure_config
    _azure_config = None
