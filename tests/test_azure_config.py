"""Tests for Azure configuration management."""

import os
from unittest.mock import patch

import pytest

from invoices.azure.base import ConnectorMode
from invoices.azure.config import (
    AzureConfig,
    DataverseConfig,
    OpenTelemetryConfig,
    SharePointConfig,
)
from invoices.exceptions import MissingConfigurationError


class TestSharePointConfig:
    """Tests for SharePointConfig."""

    def test_from_environment_with_no_vars(self) -> None:
        """Test loading with no environment variables set."""
        with patch.dict(os.environ, {}, clear=True):
            config = SharePointConfig.from_environment()
            assert config.site_id is None
            assert config.tenant_id is None
            assert config.client_id is None
            assert config.client_secret is None
            assert config.drive_id is None
            assert config.folder_path == "AI PDF Files - DEV"
            assert config.poll_interval_seconds == 86400

    def test_from_environment_with_vars(self) -> None:
        """Test loading with environment variables set."""
        env = {
            "SHAREPOINT_SITE_ID": "site-guid-123",
            "AZURE_TENANT_ID": "tenant-guid-456",
            "SHAREPOINT_CLIENT_ID": "client-guid-789",
            "SHAREPOINT_CLIENT_SECRET": "secret-abc",
            "SHAREPOINT_DRIVE_ID": "drive-guid-012",
            "SHAREPOINT_FOLDER": "AI PDF Files - PROD",
            "SHAREPOINT_POLL_INTERVAL": "3600",
        }
        with patch.dict(os.environ, env, clear=True):
            config = SharePointConfig.from_environment()
            assert config.site_id == "site-guid-123"
            assert config.tenant_id == "tenant-guid-456"
            assert config.client_id == "client-guid-789"
            assert config.client_secret == "secret-abc"
            assert config.drive_id == "drive-guid-012"
            assert config.folder_path == "AI PDF Files - PROD"
            assert config.poll_interval_seconds == 3600

    def test_from_environment_falls_back_to_azure_credentials(self) -> None:
        """Test fallback to AZURE_CLIENT_ID when SHAREPOINT_CLIENT_ID not set."""
        env = {
            "SHAREPOINT_SITE_ID": "site-guid-123",
            "AZURE_TENANT_ID": "tenant-guid-456",
            "AZURE_CLIENT_ID": "azure-client-id",
            "AZURE_CLIENT_SECRET": "azure-secret",
        }
        with patch.dict(os.environ, env, clear=True):
            config = SharePointConfig.from_environment()
            assert config.client_id == "azure-client-id"
            assert config.client_secret == "azure-secret"

    def test_is_configured_requires_site_id_and_tenant_id(self) -> None:
        """Test is_configured needs site identification and tenant_id."""
        assert not SharePointConfig().is_configured()
        assert not SharePointConfig(site_id="x").is_configured()
        assert not SharePointConfig(tenant_id="x").is_configured()
        assert SharePointConfig(site_id="x", tenant_id="y").is_configured()
        # hostname + site_path is an alternative to site_id
        assert SharePointConfig(
            hostname="x.sharepoint.com", site_path="/sites/Y", tenant_id="z"
        ).is_configured()
        # hostname alone (missing site_path) is not enough
        assert not SharePointConfig(
            hostname="x.sharepoint.com", tenant_id="z"
        ).is_configured()

    def test_validate_returns_errors_for_missing_fields(self) -> None:
        """Test validate returns errors for missing required fields."""
        config = SharePointConfig()
        errors = config.validate()
        assert len(errors) == 2
        assert any(
            "SHAREPOINT_SITE_ID" in e or "SHAREPOINT_HOSTNAME" in e for e in errors
        )
        assert any("AZURE_TENANT_ID" in e for e in errors)

    def test_from_environment_with_hostname_vars(self) -> None:
        """Test loading hostname + site_path from environment."""
        env = {
            "SHAREPOINT_HOSTNAME": "contoso.sharepoint.com",
            "SHAREPOINT_SITE_PATH": "/sites/invoices",
            "AZURE_TENANT_ID": "tenant-guid-456",
        }
        with patch.dict(os.environ, env, clear=True):
            config = SharePointConfig.from_environment()
            assert config.hostname == "contoso.sharepoint.com"
            assert config.site_path == "/sites/invoices"
            assert config.site_id is None
            assert config.is_configured()

    def test_validate_requires_client_id_when_secret_set(self) -> None:
        """Test validate catches client_secret without client_id."""
        config = SharePointConfig(site_id="x", tenant_id="y", client_secret="secret")
        errors = config.validate()
        assert len(errors) == 1
        assert "CLIENT_ID" in errors[0]


class TestDataverseConfig:
    """Tests for DataverseConfig."""

    def test_from_environment_with_no_vars(self) -> None:
        """Test loading with no environment variables set."""
        with patch.dict(os.environ, {}, clear=True):
            config = DataverseConfig.from_environment()
            assert config.environment_url is None
            assert config.staging_table == "new_tem"
            assert config.production_table == "tem"

    def test_from_environment_with_vars(self) -> None:
        """Test loading with environment variables set."""
        env = {
            "DATAVERSE_ENV_URL": "https://org.crm.dynamics.com",
            "AZURE_TENANT_ID": "tenant123",
            "DATAVERSE_CLIENT_ID": "dv_client",
            "DATAVERSE_CLIENT_SECRET": "dv_secret",
            "DATAVERSE_STAGING_TABLE": "custom_staging",
            "DATAVERSE_PRODUCTION_TABLE": "custom_prod",
        }
        with patch.dict(os.environ, env, clear=True):
            config = DataverseConfig.from_environment()
            assert config.environment_url == "https://org.crm.dynamics.com"
            assert config.tenant_id == "tenant123"
            assert config.client_id == "dv_client"
            assert config.client_secret == "dv_secret"
            assert config.staging_table == "custom_staging"
            assert config.production_table == "custom_prod"

    def test_fallback_to_azure_credentials(self) -> None:
        """Test fallback to AZURE_CLIENT_ID when DATAVERSE_CLIENT_ID not set."""
        env = {
            "DATAVERSE_ENV_URL": "https://org.crm.dynamics.com",
            "AZURE_TENANT_ID": "tenant123",
            "AZURE_CLIENT_ID": "azure_client",
            "AZURE_CLIENT_SECRET": "azure_secret",
        }
        with patch.dict(os.environ, env, clear=True):
            config = DataverseConfig.from_environment()
            assert config.client_id == "azure_client"
            assert config.client_secret == "azure_secret"


class TestOpenTelemetryConfig:
    """Tests for OpenTelemetryConfig."""

    def test_from_environment_with_vars(self) -> None:
        """Test loading from environment."""
        env = {
            "OTEL_ENABLED": "true",
            "OTEL_EXPORTER_ENDPOINT": "http://collector:4317",
            "OTEL_SERVICE_NAME": "my-service",
            "OTEL_SERVICE_VERSION": "2.0.0",
        }
        with patch.dict(os.environ, env, clear=True):
            config = OpenTelemetryConfig.from_environment()
            assert config.enabled is True
            assert config.endpoint == "http://collector:4317"
            assert config.service_name == "my-service"
            assert config.service_version == "2.0.0"


class TestAzureConfig:
    """Tests for AzureConfig."""

    def test_mode_auto_detection_local(self) -> None:
        """Test mode auto-detection when no credentials are set."""
        with patch.dict(os.environ, {}, clear=True):
            config = AzureConfig.from_environment()
            assert config.mode == ConnectorMode.LOCAL

    def test_mode_auto_detection_azure(self) -> None:
        """Test mode auto-detection when all credentials are set."""
        env = {
            "SHAREPOINT_SITE_ID": "site-guid-123",
            "AZURE_TENANT_ID": "tenant-guid-456",
            "DATAVERSE_ENV_URL": "https://org.crm.dynamics.com",
        }
        with patch.dict(os.environ, env, clear=True):
            config = AzureConfig.from_environment()
            assert config.mode == ConnectorMode.AZURE

    def test_mode_auto_detection_hybrid(self) -> None:
        """Test mode auto-detection when only SharePoint is configured."""
        env = {
            "SHAREPOINT_SITE_ID": "site-guid-123",
            "AZURE_TENANT_ID": "tenant-guid-456",
        }
        with patch.dict(os.environ, env, clear=True):
            config = AzureConfig.from_environment()
            assert config.mode == ConnectorMode.HYBRID

    def test_mode_explicit_override(self) -> None:
        """Test explicit mode override via environment variable."""
        env = {
            "INVOICEX_CONNECTOR_MODE": "local",
            "SHAREPOINT_SITE_ID": "site-guid-123",
            "AZURE_TENANT_ID": "tenant-guid-456",
        }
        with patch.dict(os.environ, env, clear=True):
            config = AzureConfig.from_environment()
            assert config.mode == ConnectorMode.LOCAL

    def test_validate_azure_mode_requires_all_configs(self) -> None:
        """Test validation in AZURE mode requires all configurations."""
        config = AzureConfig(mode=ConnectorMode.AZURE)
        with pytest.raises(MissingConfigurationError):
            config.validate()
