// =============================================================================
// Storage Account Module
// =============================================================================
// Creates an Azure Storage Account with blob containers
// =============================================================================

@description('Name of the Storage Account')
param name string

@description('Azure region for the resource')
param location string

@description('Resource tags')
param tags object = {}

@description('SKU for the Storage Account')
@allowed(['Standard_LRS', 'Standard_GRS', 'Standard_RAGRS', 'Standard_ZRS', 'Premium_LRS'])
param sku string = 'Standard_LRS'

@description('Storage account kind')
@allowed(['StorageV2', 'BlobStorage', 'BlockBlobStorage'])
param kind string = 'StorageV2'

@description('Blob containers to create')
param containers array = []

@description('Enable hierarchical namespace (Data Lake Storage Gen2)')
param enableHns bool = false

@description('Enable blob versioning')
param enableVersioning bool = true

// -----------------------------------------------------------------------------
// Resources
// -----------------------------------------------------------------------------

resource storageAccount 'Microsoft.Storage/storageAccounts@2023-01-01' = {
  name: name
  location: location
  tags: tags
  sku: {
    name: sku
  }
  kind: kind
  properties: {
    accessTier: 'Hot'
    allowBlobPublicAccess: false
    allowSharedKeyAccess: true
    isHnsEnabled: enableHns
    minimumTlsVersion: 'TLS1_2'
    supportsHttpsTrafficOnly: true
    publicNetworkAccess: 'Enabled'
    networkAcls: {
      defaultAction: 'Allow'
      bypass: 'AzureServices'
    }
  }
}

resource blobServices 'Microsoft.Storage/storageAccounts/blobServices@2023-01-01' = {
  parent: storageAccount
  name: 'default'
  properties: {
    isVersioningEnabled: enableVersioning
    deleteRetentionPolicy: {
      enabled: true
      days: 7
    }
    containerDeleteRetentionPolicy: {
      enabled: true
      days: 7
    }
  }
}

resource blobContainers 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-01-01' = [for container in containers: {
  parent: blobServices
  name: container
  properties: {
    publicAccess: 'None'
  }
}]

// -----------------------------------------------------------------------------
// Outputs
// -----------------------------------------------------------------------------

@description('The resource ID of the Storage Account')
output storageAccountId string = storageAccount.id

@description('The name of the Storage Account')
output storageAccountName string = storageAccount.name

@description('The primary blob endpoint')
output blobEndpoint string = storageAccount.properties.primaryEndpoints.blob

@description('The primary connection string')
output connectionString string = 'DefaultEndpointsProtocol=https;AccountName=${storageAccount.name};EndpointSuffix=${environment().suffixes.storage};AccountKey=${storageAccount.listKeys().keys[0].value}'
