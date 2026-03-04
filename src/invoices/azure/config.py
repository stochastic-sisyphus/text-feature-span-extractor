"""Azure configuration management.

Provides centralized configuration for Azure service connectors with
environment variable support and automatic mode detection.

Usage:
    from invoices.azure.config import get_azure_config

    config = get_azure_config()
    print(f"Running in {config.mode.value} mode")
"""

import os
from dataclasses import dataclass, field
from typing import Any

from invoices.azure.base import ConnectorMode
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


def _env_int(key: str, default: int) -> int:
    """Get int from environment variable."""
    value = os.environ.get(key)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


@dataclass
class SharePointConfig:
    """Configuration for SharePoint connector via Microsoft Graph API.

    Auth: When client_secret is set, uses ClientSecretCredential (service
    principal). Otherwise, falls back to DefaultAzureCredential (managed
    identity in Azure, CLI/env credentials locally).

    Attributes:
        site_id: SharePoint site ID (GUID)
        tenant_id: Azure AD tenant ID
        client_id: App registration client ID
        client_secret: App registration client secret (omit for managed identity)
        managed_identity_client_id: Managed identity client ID (Azure only)
        drive_id: SharePoint drive ID (omit for default document library)
        folder_path: Folder path within the drive
        poll_interval_seconds: Interval for fallback polling
    """

    site_id: str | None = None
    hostname: str | None = None
    site_path: str | None = None
    tenant_id: str | None = None
    client_id: str | None = None
    client_secret: str | None = None
    managed_identity_client_id: str | None = None
    drive_id: str | None = None
    folder_path: str = "AI PDF Files - DEV"
    poll_interval_seconds: int = 86400  # 24 hours

    @classmethod
    def from_environment(cls) -> "SharePointConfig":
        """Load configuration from environment variables."""
        return cls(
            site_id=_env_str("SHAREPOINT_SITE_ID"),
            hostname=_env_str("SHAREPOINT_HOSTNAME"),
            site_path=_env_str("SHAREPOINT_SITE_PATH"),
            tenant_id=_env_str("AZURE_TENANT_ID"),
            client_id=_env_str("SHAREPOINT_CLIENT_ID") or _env_str("AZURE_CLIENT_ID"),
            client_secret=_env_str("SHAREPOINT_CLIENT_SECRET")
            or _env_str("AZURE_CLIENT_SECRET"),
            managed_identity_client_id=_env_str("AZURE_MANAGED_IDENTITY_CLIENT_ID"),
            drive_id=_env_str("SHAREPOINT_DRIVE_ID"),
            folder_path=_env_str("SHAREPOINT_FOLDER_PATH")
            or _env_str("SHAREPOINT_FOLDER")
            or "AI PDF Files - DEV",
            poll_interval_seconds=_env_int("SHAREPOINT_POLL_INTERVAL", 86400),
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
        staging_table: Staging table name (new_tem)
        production_table: Production table name (tem)
    """

    environment_url: str | None = None
    tenant_id: str | None = None
    client_id: str | None = None
    client_secret: str | None = None
    managed_identity_client_id: str | None = None
    staging_table: str = "new_tem"
    production_table: str = "tem"

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
            managed_identity_client_id=_env_str("AZURE_MANAGED_IDENTITY_CLIENT_ID"),
            staging_table=_env_str("DATAVERSE_STAGING_TABLE", "new_tem") or "new_tem",
            production_table=_env_str("DATAVERSE_PRODUCTION_TABLE", "tem") or "tem",
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

    Provides centralized access to all Azure service configurations
    with automatic mode detection based on available credentials.

    Attributes:
        mode: Operating mode (LOCAL, AZURE, HYBRID)
        sharepoint: SharePoint configuration
        dataverse: Dataverse configuration
        otel: OpenTelemetry configuration
        data_dir: Local data directory for mock implementations
    """

    mode: ConnectorMode = ConnectorMode.LOCAL
    sharepoint: SharePointConfig = field(default_factory=SharePointConfig)
    dataverse: DataverseConfig = field(default_factory=DataverseConfig)
    otel: OpenTelemetryConfig = field(default_factory=OpenTelemetryConfig)
    data_dir: str = "data"

    @classmethod
    def from_environment(cls) -> "AzureConfig":
        """Load configuration from environment variables.

        Automatically detects mode based on available credentials:
        - AZURE: Both SharePoint and Dataverse configured
        - HYBRID: Only one of SharePoint or Dataverse configured
        - LOCAL: Neither configured
        """
        sharepoint = SharePointConfig.from_environment()
        dataverse = DataverseConfig.from_environment()
        otel = OpenTelemetryConfig.from_environment()

        # Check for explicit mode override
        mode_str = (_env_str("INVOICEX_CONNECTOR_MODE", "") or "").lower()
        if mode_str == "azure":
            mode = ConnectorMode.AZURE
        elif mode_str == "hybrid":
            mode = ConnectorMode.HYBRID
        elif mode_str == "local":
            mode = ConnectorMode.LOCAL
        else:
            # Auto-detect based on available credentials
            sp_configured = sharepoint.is_configured()
            dv_configured = dataverse.is_configured()

            if sp_configured and dv_configured:
                mode = ConnectorMode.AZURE
            elif sp_configured or dv_configured:
                mode = ConnectorMode.HYBRID
            else:
                mode = ConnectorMode.LOCAL

        # Share DATA_DIR from central Config to avoid duplicate env reads
        from invoices.config import Config as _main_cfg

        return cls(
            mode=mode,
            sharepoint=sharepoint,
            dataverse=dataverse,
            otel=otel,
            data_dir=_main_cfg.DATA_DIR,
        )

    def validate(self) -> None:
        """Validate configuration and raise if invalid.

        Raises:
            MissingConfigurationError: If required config is missing
            InvalidConfigurationError: If config values are invalid
        """
        errors: list[str] = []

        if self.mode == ConnectorMode.AZURE:
            # Both services must be configured
            errors.extend(self.sharepoint.validate())
            errors.extend(self.dataverse.validate())
        elif self.mode == ConnectorMode.HYBRID:
            # At least one service must be configured
            sp_errors = self.sharepoint.validate()
            dv_errors = self.dataverse.validate()
            if sp_errors and dv_errors:
                errors.append(
                    "At least one Azure service must be configured for HYBRID mode"
                )

        if errors:
            raise MissingConfigurationError(
                key="azure_config",
                description="; ".join(errors),
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary for logging/debugging."""
        return {
            "mode": self.mode.value,
            "sharepoint_configured": self.sharepoint.is_configured(),
            "dataverse_configured": self.dataverse.is_configured(),
            "otel_enabled": self.otel.enabled,
            "data_dir": self.data_dir,
        }


# Global singleton
_azure_config: AzureConfig | None = None


def get_azure_config(reload: bool = False) -> AzureConfig:
    """Get the Azure configuration singleton.

    Args:
        reload: Force reload from environment variables

    Returns:
        AzureConfig instance
    """
    global _azure_config

    if _azure_config is None or reload:
        _azure_config = AzureConfig.from_environment()
        logger.info(
            "azure_config_loaded",
            mode=_azure_config.mode.value,
            sharepoint_configured=_azure_config.sharepoint.is_configured(),
            dataverse_configured=_azure_config.dataverse.is_configured(),
        )

    return _azure_config


def reset_azure_config() -> None:
    """Reset the configuration singleton (for testing)."""
    global _azure_config
    _azure_config = None
