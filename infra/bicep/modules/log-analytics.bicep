// =============================================================================
// Log Analytics Workspace Module
// =============================================================================
// Creates a Log Analytics workspace for centralized logging and monitoring
// =============================================================================

@description('Name of the Log Analytics workspace')
param name string

@description('Azure region for the resource')
param location string

@description('Resource tags')
param tags object = {}

@description('Retention period in days')
@minValue(30)
@maxValue(730)
param retentionInDays int = 30

@description('SKU for the workspace')
@allowed(['Free', 'PerGB2018', 'PerNode', 'Premium', 'Standalone', 'Standard'])
param sku string = 'PerGB2018'

// -----------------------------------------------------------------------------
// Resources
// -----------------------------------------------------------------------------

resource logAnalyticsWorkspace 'Microsoft.OperationalInsights/workspaces@2022-10-01' = {
  name: name
  location: location
  tags: tags
  properties: {
    sku: {
      name: sku
    }
    retentionInDays: retentionInDays
    features: {
      enableLogAccessUsingOnlyResourcePermissions: true
    }
    workspaceCapping: {
      dailyQuotaGb: -1 // No daily cap
    }
    publicNetworkAccessForIngestion: 'Enabled'
    publicNetworkAccessForQuery: 'Enabled'
  }
}

// -----------------------------------------------------------------------------
// Outputs
// -----------------------------------------------------------------------------

@description('The resource ID of the Log Analytics workspace')
output workspaceId string = logAnalyticsWorkspace.id

@description('The name of the Log Analytics workspace')
output workspaceName string = logAnalyticsWorkspace.name

@description('The primary shared key for the workspace')
output workspaceKey string = logAnalyticsWorkspace.listKeys().primarySharedKey

@description('The customer ID (workspace ID) for the workspace')
output customerId string = logAnalyticsWorkspace.properties.customerId
