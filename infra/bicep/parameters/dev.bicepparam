// =============================================================================
// Development Environment Parameters
// =============================================================================
// Parameter values for the development environment deployment
//
// Usage:
//   az deployment group create \
//     --resource-group rg-invoicex-dev \
//     --template-file main.bicep \
//     --parameters parameters/dev.bicepparam
// =============================================================================

using '../main.bicep'

// Environment settings
param environment = 'dev'
param location = 'eastus'
param baseName = 'invoicex'

// Container configuration
param imageTag = 'latest'
param useManagedIdentity = true

// SharePoint integration (leave empty for local development)
param sharepointSiteId = ''
param sharepointDriveId = ''

// Dataverse integration (leave empty for local development)
param dataverseEnvironmentUrl = ''

// Grafana Cloud endpoints (set via environment variables or Key Vault)
// These are intentionally empty - set them during deployment or use Key Vault references
param grafanaPrometheusEndpoint = ''
param grafanaLokiEndpoint = ''
param grafanaTempoEndpoint = ''
param grafanaInstanceId = ''
param grafanaApiKey = ''

// Extraction thresholds (more lenient for dev)
param confidenceAutoApprove = 0.80
param noneBias = 0.05
