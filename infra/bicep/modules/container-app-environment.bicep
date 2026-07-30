// =============================================================================
// Container App Environment Module
// =============================================================================
// Creates an Azure Container Apps Environment
// =============================================================================

@description('Name of the Container App Environment')
param name string

@description('Azure region for the resource')
param location string

@description('Resource tags')
param tags object = {}

@description('Log Analytics Workspace ID')
param logAnalyticsWorkspaceId string

@description('Log Analytics Workspace shared key')
@secure()
param logAnalyticsWorkspaceKey string

@description('Enable zone redundancy')
param zoneRedundant bool = false

// -----------------------------------------------------------------------------
// Resources
// -----------------------------------------------------------------------------

resource containerAppEnvironment 'Microsoft.App/managedEnvironments@2023-05-01' = {
  name: name
  location: location
  tags: tags
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: reference(logAnalyticsWorkspaceId, '2022-10-01').customerId
        sharedKey: logAnalyticsWorkspaceKey
      }
    }
    zoneRedundant: zoneRedundant
    workloadProfiles: [
      {
        name: 'Consumption'
        workloadProfileType: 'Consumption'
      }
    ]
  }
}

// -----------------------------------------------------------------------------
// Outputs
// -----------------------------------------------------------------------------

@description('The resource ID of the Container App Environment')
output environmentId string = containerAppEnvironment.id

@description('The name of the Container App Environment')
output environmentName string = containerAppEnvironment.name

@description('The default domain of the Container App Environment')
output defaultDomain string = containerAppEnvironment.properties.defaultDomain

@description('The static IP of the Container App Environment')
output staticIp string = containerAppEnvironment.properties.staticIp
