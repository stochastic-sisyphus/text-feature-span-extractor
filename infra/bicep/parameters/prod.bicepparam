// =============================================================================
// Production Environment Parameters
// =============================================================================
// Parameter values for the production environment deployment
//
// IMPORTANT: Sensitive values should be passed via:
//   - Azure Key Vault references
//   - GitHub Secrets (for CI/CD)
//   - Secure parameter files (not committed to source control)
//
// Usage:
//   az deployment group create \
//     --resource-group rg-invoicex-prod \
//     --template-file main.bicep \
//     --parameters parameters/prod.bicepparam \
//     --parameters grafanaApiKey=@secretfile.txt
// =============================================================================

using '../main.bicep'

// Environment settings
param environment = 'prod'
param location = 'eastus'
param baseName = 'invoicex'

// Container configuration - use specific version tags in production
param imageTag = 'v0.2.0'
param useManagedIdentity = true

// SharePoint integration
// Set these values for your production SharePoint site
param sharepointSiteId = '' // e.g., 'contoso.sharepoint.com,guid1,guid2'
param sharepointDriveId = '' // e.g., 'b!...'

// Dataverse integration
param dataverseEnvironmentUrl = '' // e.g., 'https://org.crm.dynamics.com'

// Grafana Cloud endpoints
// SECURITY: These should be passed via secure methods, not hardcoded
param grafanaPrometheusEndpoint = ''
param grafanaLokiEndpoint = ''
param grafanaTempoEndpoint = ''
param grafanaInstanceId = ''
param grafanaApiKey = ''

// Extraction thresholds (stricter for production)
param confidenceAutoApprove = 0.85
param noneBias = 0.05
