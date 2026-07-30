// =============================================================================
// Invoice Extraction Pipeline - Azure Infrastructure
// =============================================================================
// Main Bicep template for deploying the InvoiceX pipeline to Azure
//
// This template orchestrates the deployment of:
// - Log Analytics workspace for monitoring
// - Key Vault for secrets management
// - Container Registry for Docker images
// - Storage Account for models and artifacts
// - Container App Environment
// - Container App for the extraction pipeline
//
// Usage:
//   az deployment group create \
//     --resource-group rg-invoicex-dev \
//     --template-file main.bicep \
//     --parameters parameters/dev.bicepparam
// =============================================================================

// -----------------------------------------------------------------------------
// Parameters
// -----------------------------------------------------------------------------

@description('Deployment environment (dev, staging, prod)')
@allowed(['dev', 'staging', 'prod'])
param environment string = 'dev'

@description('Azure region for resources')
param location string = resourceGroup().location

@description('Base name for resources (will be prefixed with environment)')
@minLength(3)
@maxLength(20)
param baseName string = 'invoicex'

@description('Container image tag to deploy')
param imageTag string = 'latest'

@description('Enable managed identity for secure resource access')
param useManagedIdentity bool = true

@description('SharePoint Site ID for document intake (optional)')
param sharepointSiteId string = ''

@description('SharePoint Drive ID for document library (optional)')
param sharepointDriveId string = ''

@description('Dataverse environment URL (optional)')
param dataverseEnvironmentUrl string = ''

@description('OTLP exporter endpoint (default: sidecar collector on localhost, correct for Container Apps)')
param otelExporterEndpoint string = 'http://localhost:4317'

@description('Grafana Cloud Prometheus endpoint (optional)')
@secure()
param grafanaPrometheusEndpoint string = ''

@description('Grafana Cloud Loki endpoint (optional)')
@secure()
param grafanaLokiEndpoint string = ''

@description('Grafana Cloud Tempo endpoint (optional)')
@secure()
param grafanaTempoEndpoint string = ''

@description('Grafana Cloud instance ID (optional)')
@secure()
param grafanaInstanceId string = ''

@description('Grafana Cloud API key (optional)')
@secure()
param grafanaApiKey string = ''

@description('Auto-approve confidence threshold')
@minValue(0.0)
@maxValue(1.0)
param confidenceAutoApprove float = 0.85

@description('None bias for decoder abstention')
@minValue(0.0)
@maxValue(100.0)
param noneBias float = 0.05

// -----------------------------------------------------------------------------
// Variables
// -----------------------------------------------------------------------------

// Resource naming convention: {env}-{baseName}-{resourceType}
var resourcePrefix = '${environment}-${baseName}'

// Common tags for all resources
var tags = {
  Environment: environment
  Application: 'InvoiceX'
  ManagedBy: 'Bicep'
  Repository: 'invoicex'
}

// Container configuration
var containerConfig = {
  cpu: environment == 'prod' ? '1.0' : '0.5'
  memory: environment == 'prod' ? '2Gi' : '1Gi'
  minReplicas: environment == 'prod' ? 1 : 0
  maxReplicas: environment == 'prod' ? 10 : 3
}

// -----------------------------------------------------------------------------
// Module Deployments
// -----------------------------------------------------------------------------

// Log Analytics Workspace
module logAnalytics 'modules/log-analytics.bicep' = {
  name: 'deploy-log-analytics'
  params: {
    name: '${resourcePrefix}-logs'
    location: location
    tags: tags
    retentionInDays: environment == 'prod' ? 90 : 30
  }
}

// Key Vault for secrets
module keyVault 'modules/key-vault.bicep' = {
  name: 'deploy-key-vault'
  params: {
    name: replace('${resourcePrefix}-kv', '-', '')
    location: location
    tags: tags
    enablePurgeProtection: environment == 'prod'
    enableSoftDelete: true
    softDeleteRetentionInDays: environment == 'prod' ? 90 : 7
  }
}

// Container Registry
module containerRegistry 'modules/container-registry.bicep' = {
  name: 'deploy-container-registry'
  params: {
    name: replace('${resourcePrefix}acr', '-', '')
    location: location
    tags: tags
    sku: environment == 'prod' ? 'Standard' : 'Basic'
    adminUserEnabled: false
  }
}

// Storage Account for models and artifacts
module storage 'modules/storage.bicep' = {
  name: 'deploy-storage'
  params: {
    name: replace('${resourcePrefix}st', '-', '')
    location: location
    tags: tags
    sku: environment == 'prod' ? 'Standard_GRS' : 'Standard_LRS'
    containers: [
      'models'
      'artifacts'
      'documents'
      'exports'
    ]
  }
}

// Container App Environment
module containerAppEnvironment 'modules/container-app-environment.bicep' = {
  name: 'deploy-container-app-environment'
  params: {
    name: '${resourcePrefix}-env'
    location: location
    tags: tags
    logAnalyticsWorkspaceId: logAnalytics.outputs.workspaceId
    logAnalyticsWorkspaceKey: logAnalytics.outputs.workspaceKey
  }
}

// Container App
module containerApp 'modules/container-app.bicep' = {
  name: 'deploy-container-app'
  params: {
    name: '${resourcePrefix}-app'
    location: location
    tags: tags
    containerAppEnvironmentId: containerAppEnvironment.outputs.environmentId
    containerRegistryName: containerRegistry.outputs.registryName
    containerRegistryLoginServer: containerRegistry.outputs.loginServer
    imageTag: imageTag
    useManagedIdentity: useManagedIdentity
    keyVaultName: keyVault.outputs.keyVaultName
    storageAccountName: storage.outputs.storageAccountName

    // Application configuration
    environment: environment
    sharepointSiteId: sharepointSiteId
    sharepointDriveId: sharepointDriveId
    dataverseEnvironmentUrl: dataverseEnvironmentUrl
    otelExporterEndpoint: otelExporterEndpoint
    confidenceAutoApprove: confidenceAutoApprove
    noneBias: noneBias

    // Grafana Cloud configuration
    grafanaPrometheusEndpoint: grafanaPrometheusEndpoint
    grafanaLokiEndpoint: grafanaLokiEndpoint
    grafanaTempoEndpoint: grafanaTempoEndpoint
    grafanaInstanceId: grafanaInstanceId
    grafanaApiKey: grafanaApiKey

    // Scaling configuration
    cpuCore: containerConfig.cpu
    memorySize: containerConfig.memory
    minReplicas: containerConfig.minReplicas
    maxReplicas: containerConfig.maxReplicas
  }
}

// -----------------------------------------------------------------------------
// Role Assignments (when using managed identity)
// -----------------------------------------------------------------------------

// Grant Container App access to Key Vault secrets
module keyVaultRoleAssignment 'modules/role-assignment.bicep' = if (useManagedIdentity) {
  name: 'deploy-kv-role-assignment'
  params: {
    principalId: containerApp.outputs.managedIdentityPrincipalId
    roleDefinitionId: '4633458b-17de-408a-b874-0445c86b69e6' // Key Vault Secrets User
    principalType: 'ServicePrincipal'
  }
  scope: resourceGroup()
}

// Grant Container App access to Storage Blob Data
module storageRoleAssignment 'modules/role-assignment.bicep' = if (useManagedIdentity) {
  name: 'deploy-storage-role-assignment'
  params: {
    principalId: containerApp.outputs.managedIdentityPrincipalId
    roleDefinitionId: 'ba92f5b4-2d11-453d-a403-e96b0029c9fe' // Storage Blob Data Contributor
    principalType: 'ServicePrincipal'
  }
  scope: resourceGroup()
}

// Grant Container App access to Container Registry
module acrRoleAssignment 'modules/role-assignment.bicep' = if (useManagedIdentity) {
  name: 'deploy-acr-role-assignment'
  params: {
    principalId: containerApp.outputs.managedIdentityPrincipalId
    roleDefinitionId: '7f951dda-4ed3-4680-a7ca-43fe172d538d' // AcrPull
    principalType: 'ServicePrincipal'
  }
  scope: resourceGroup()
}

// -----------------------------------------------------------------------------
// Outputs
// -----------------------------------------------------------------------------

@description('Container App FQDN for API access')
output containerAppFqdn string = containerApp.outputs.fqdn

@description('Container App URL')
output containerAppUrl string = 'https://${containerApp.outputs.fqdn}'

@description('Container Registry login server')
output containerRegistryLoginServer string = containerRegistry.outputs.loginServer

@description('Key Vault URI for secrets access')
output keyVaultUri string = keyVault.outputs.keyVaultUri

@description('Storage Account name')
output storageAccountName string = storage.outputs.storageAccountName

@description('Storage Account blob endpoint')
output storageBlobEndpoint string = storage.outputs.blobEndpoint

@description('Log Analytics Workspace ID')
output logAnalyticsWorkspaceId string = logAnalytics.outputs.workspaceId

@description('Managed Identity Principal ID (if enabled)')
output managedIdentityPrincipalId string = useManagedIdentity ? containerApp.outputs.managedIdentityPrincipalId : ''

@description('Resource Group name')
output resourceGroupName string = resourceGroup().name

@description('Deployment environment')
output deploymentEnvironment string = environment
