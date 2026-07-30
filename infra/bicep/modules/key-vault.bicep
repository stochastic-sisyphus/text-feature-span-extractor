// =============================================================================
// Key Vault Module
// =============================================================================
// Creates an Azure Key Vault for secrets management
// =============================================================================

@description('Name of the Key Vault')
param name string

@description('Azure region for the resource')
param location string

@description('Resource tags')
param tags object = {}

@description('Enable purge protection')
param enablePurgeProtection bool = false

@description('Enable soft delete')
param enableSoftDelete bool = true

@description('Soft delete retention in days')
@minValue(7)
@maxValue(90)
param softDeleteRetentionInDays int = 7

@description('SKU for the Key Vault')
@allowed(['standard', 'premium'])
param sku string = 'standard'

@description('Enable RBAC authorization')
param enableRbacAuthorization bool = true

// -----------------------------------------------------------------------------
// Resources
// -----------------------------------------------------------------------------

resource keyVault 'Microsoft.KeyVault/vaults@2023-07-01' = {
  name: name
  location: location
  tags: tags
  properties: {
    sku: {
      family: 'A'
      name: sku
    }
    tenantId: subscription().tenantId
    enabledForDeployment: false
    enabledForDiskEncryption: false
    enabledForTemplateDeployment: true
    enableSoftDelete: enableSoftDelete
    softDeleteRetentionInDays: softDeleteRetentionInDays
    enablePurgeProtection: enablePurgeProtection ? true : null
    enableRbacAuthorization: enableRbacAuthorization
    publicNetworkAccess: 'Disabled'
    networkAcls: {
      defaultAction: 'Deny'
      bypass: 'AzureServices'
    }
  }
}

// -----------------------------------------------------------------------------
// Outputs
// -----------------------------------------------------------------------------

@description('The resource ID of the Key Vault')
output keyVaultId string = keyVault.id

@description('The name of the Key Vault')
output keyVaultName string = keyVault.name

@description('The URI of the Key Vault')
output keyVaultUri string = keyVault.properties.vaultUri
