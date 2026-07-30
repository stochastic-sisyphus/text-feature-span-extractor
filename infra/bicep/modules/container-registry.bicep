// =============================================================================
// Container Registry Module
// =============================================================================
// Creates an Azure Container Registry for Docker images
// =============================================================================

@description('Name of the Container Registry')
param name string

@description('Azure region for the resource')
param location string

@description('Resource tags')
param tags object = {}

@description('SKU for the Container Registry')
@allowed(['Basic', 'Standard', 'Premium'])
param sku string = 'Basic'

@description('Enable admin user for the registry')
param adminUserEnabled bool = false

@description('Enable zone redundancy (Premium SKU only)')
param zoneRedundancy bool = false

// -----------------------------------------------------------------------------
// Resources
// -----------------------------------------------------------------------------

resource containerRegistry 'Microsoft.ContainerRegistry/registries@2023-07-01' = {
  name: name
  location: location
  tags: tags
  sku: {
    name: sku
  }
  properties: {
    adminUserEnabled: adminUserEnabled
    publicNetworkAccess: 'Enabled'
    zoneRedundancy: sku == 'Premium' && zoneRedundancy ? 'Enabled' : 'Disabled'
    policies: {
      retentionPolicy: {
        days: 7
        status: sku == 'Premium' ? 'enabled' : 'disabled'
      }
    }
  }
}

// -----------------------------------------------------------------------------
// Outputs
// -----------------------------------------------------------------------------

@description('The resource ID of the Container Registry')
output registryId string = containerRegistry.id

@description('The name of the Container Registry')
output registryName string = containerRegistry.name

@description('The login server of the Container Registry')
output loginServer string = containerRegistry.properties.loginServer
