// =============================================================================
// Container App Module
// =============================================================================
// Creates an Azure Container App for the InvoiceX extraction pipeline
// =============================================================================

@description('Name of the Container App')
param name string

@description('Azure region for the resource')
param location string

@description('Resource tags')
param tags object = {}

@description('Container App Environment ID')
param containerAppEnvironmentId string

@description('Container Registry name')
param containerRegistryName string

@description('Container Registry login server')
param containerRegistryLoginServer string

@description('Container image tag')
param imageTag string = 'latest'

@description('Use managed identity for authentication')
param useManagedIdentity bool = true

@description('Key Vault name for secrets')
param keyVaultName string

@description('Storage Account name')
param storageAccountName string

// Application configuration
@description('Deployment environment')
param environment string

@description('SharePoint Site ID')
param sharepointSiteId string = ''

@description('SharePoint Drive ID')
param sharepointDriveId string = ''

@description('Dataverse environment URL')
param dataverseEnvironmentUrl string = ''

@description('OTLP exporter endpoint (default: sidecar collector on localhost)')
param otelExporterEndpoint string = 'http://localhost:4317'

@description('Confidence auto-approve threshold')
param confidenceAutoApprove float = 0.85

@description('None bias for decoder')
param noneBias float = 0.05

// Grafana Cloud configuration
@description('Grafana Cloud Prometheus endpoint')
@secure()
param grafanaPrometheusEndpoint string = ''

@description('Grafana Cloud Loki endpoint')
@secure()
param grafanaLokiEndpoint string = ''

@description('Grafana Cloud Tempo endpoint')
@secure()
param grafanaTempoEndpoint string = ''

@description('Grafana Cloud instance ID')
@secure()
param grafanaInstanceId string = ''

@description('Grafana Cloud API key')
@secure()
param grafanaApiKey string = ''

// Scaling configuration
@description('CPU cores for the container')
param cpuCore string = '0.5'

@description('Memory size for the container')
param memorySize string = '1Gi'

@description('Minimum number of replicas')
param minReplicas int = 0

@description('Maximum number of replicas')
param maxReplicas int = 3

// -----------------------------------------------------------------------------
// Resources
// -----------------------------------------------------------------------------

// User-assigned managed identity
resource managedIdentity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = if (useManagedIdentity) {
  name: '${name}-identity'
  location: location
  tags: tags
}

// Container App
resource containerApp 'Microsoft.App/containerApps@2023-05-01' = {
  name: name
  location: location
  tags: tags
  identity: useManagedIdentity ? {
    type: 'UserAssigned'
    userAssignedIdentities: {
      '${managedIdentity.id}': {}
    }
  } : {
    type: 'None'
  }
  properties: {
    managedEnvironmentId: containerAppEnvironmentId
    workloadProfileName: 'Consumption'
    configuration: {
      activeRevisionsMode: 'Single'
      ingress: {
        external: true
        targetPort: 8080
        transport: 'http'
        allowInsecure: false
        traffic: [
          {
            latestRevision: true
            weight: 100
          }
        ]
        corsPolicy: {
          allowedOrigins: ['*']
          allowedMethods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS']
          allowedHeaders: ['*']
          maxAge: 86400
        }
      }
      registries: useManagedIdentity ? [
        {
          server: containerRegistryLoginServer
          identity: managedIdentity.id
        }
      ] : []
      secrets: [
        {
          name: 'grafana-prometheus-endpoint'
          value: grafanaPrometheusEndpoint
        }
        {
          name: 'grafana-loki-endpoint'
          value: grafanaLokiEndpoint
        }
        {
          name: 'grafana-tempo-endpoint'
          value: grafanaTempoEndpoint
        }
        {
          name: 'grafana-instance-id'
          value: grafanaInstanceId
        }
        {
          name: 'grafana-api-key'
          value: grafanaApiKey
        }
      ]
    }
    template: {
      containers: [
        {
          name: 'invoicex'
          image: '${containerRegistryLoginServer}/invoicex:${imageTag}'
          command: ['uvicorn', 'invoices.api:get_app', '--factory', '--host', '0.0.0.0', '--port', '8080']
          resources: {
            cpu: json(cpuCore)
            memory: memorySize
          }
          env: [
            // Application configuration
            {
              name: 'INVOICEX_LOG_LEVEL'
              value: environment == 'prod' ? 'INFO' : 'DEBUG'
            }
            {
              name: 'INVOICEX_LOG_FORMAT'
              value: 'json'
            }
            {
              name: 'MODEL_ID'
              value: 'xgboost-v2.1.0'
            }
            {
              name: 'INVOICEX_CONFIDENCE_AUTO_APPROVE'
              value: string(confidenceAutoApprove)
            }
            {
              name: 'INVOICEX_NONE_BIAS'
              value: string(noneBias)
            }
            {
              name: 'DEPLOYMENT_ENVIRONMENT'
              value: environment
            }
            // Connector configuration
            {
              name: 'INVOICEX_CONNECTOR_MODE'
              value: 'azure'
            }
            {
              name: 'SHAREPOINT_SITE_ID'
              value: sharepointSiteId
            }
            {
              name: 'SHAREPOINT_DRIVE_ID'
              value: sharepointDriveId
            }
            {
              name: 'DATAVERSE_ENVIRONMENT_URL'
              value: dataverseEnvironmentUrl
            }
            // Azure resource references
            {
              name: 'AZURE_KEY_VAULT_NAME'
              value: keyVaultName
            }
            {
              name: 'AZURE_STORAGE_ACCOUNT_NAME'
              value: storageAccountName
            }
            // Grafana Cloud OTEL configuration
            {
              name: 'OTEL_EXPORTER_OTLP_ENDPOINT'
              value: otelExporterEndpoint
            }
            {
              name: 'OTEL_SERVICE_NAME'
              value: 'invoicex'
            }
            {
              name: 'OTEL_RESOURCE_ATTRIBUTES'
              value: 'service.version=0.2.0,deployment.environment=${environment}'
            }
            {
              name: 'GRAFANA_CLOUD_PROMETHEUS_ENDPOINT'
              secretRef: 'grafana-prometheus-endpoint'
            }
            {
              name: 'GRAFANA_CLOUD_LOKI_ENDPOINT'
              secretRef: 'grafana-loki-endpoint'
            }
            {
              name: 'GRAFANA_CLOUD_TEMPO_ENDPOINT'
              secretRef: 'grafana-tempo-endpoint'
            }
            {
              name: 'GRAFANA_CLOUD_PROMETHEUS_USER'
              secretRef: 'grafana-instance-id'
            }
            {
              name: 'GRAFANA_CLOUD_PROMETHEUS_PASSWORD'
              secretRef: 'grafana-api-key'
            }
            {
              name: 'GRAFANA_CLOUD_LOKI_USER'
              secretRef: 'grafana-instance-id'
            }
            {
              name: 'GRAFANA_CLOUD_LOKI_PASSWORD'
              secretRef: 'grafana-api-key'
            }
            {
              name: 'GRAFANA_CLOUD_TEMPO_USER'
              secretRef: 'grafana-instance-id'
            }
            {
              name: 'GRAFANA_CLOUD_TEMPO_PASSWORD'
              secretRef: 'grafana-api-key'
            }
          ]
          probes: [
            {
              type: 'Liveness'
              httpGet: {
                path: '/health/live'
                port: 8080
                scheme: 'HTTP'
              }
              initialDelaySeconds: 10
              periodSeconds: 30
              timeoutSeconds: 10
              failureThreshold: 3
            }
            {
              type: 'Readiness'
              httpGet: {
                path: '/health/ready'
                port: 8080
                scheme: 'HTTP'
              }
              initialDelaySeconds: 5
              periodSeconds: 10
              timeoutSeconds: 5
              failureThreshold: 3
            }
            {
              type: 'Startup'
              httpGet: {
                path: '/health/live'
                port: 8080
                scheme: 'HTTP'
              }
              initialDelaySeconds: 5
              periodSeconds: 5
              timeoutSeconds: 10
              failureThreshold: 30
            }
          ]
        }
        // OTEL Collector sidecar for Grafana Cloud
        {
          name: 'otel-collector'
          image: 'otel/opentelemetry-collector-contrib:0.91.0'
          args: ['--config=/etc/otel/config.yaml']
          resources: {
            cpu: json('0.25')
            memory: '256Mi'
          }
          env: [
            {
              name: 'GRAFANA_CLOUD_PROMETHEUS_ENDPOINT'
              secretRef: 'grafana-prometheus-endpoint'
            }
            {
              name: 'GRAFANA_CLOUD_PROMETHEUS_USER'
              secretRef: 'grafana-instance-id'
            }
            {
              name: 'GRAFANA_CLOUD_PROMETHEUS_PASSWORD'
              secretRef: 'grafana-api-key'
            }
            {
              name: 'GRAFANA_CLOUD_LOKI_ENDPOINT'
              secretRef: 'grafana-loki-endpoint'
            }
            {
              name: 'GRAFANA_CLOUD_LOKI_USER'
              secretRef: 'grafana-instance-id'
            }
            {
              name: 'GRAFANA_CLOUD_LOKI_PASSWORD'
              secretRef: 'grafana-api-key'
            }
            {
              name: 'GRAFANA_CLOUD_TEMPO_ENDPOINT'
              secretRef: 'grafana-tempo-endpoint'
            }
            {
              name: 'GRAFANA_CLOUD_TEMPO_USER'
              secretRef: 'grafana-instance-id'
            }
            {
              name: 'GRAFANA_CLOUD_TEMPO_PASSWORD'
              secretRef: 'grafana-api-key'
            }
            {
              name: 'DEPLOYMENT_ENVIRONMENT'
              value: environment
            }
          ]
        }
      ]
      scale: {
        minReplicas: minReplicas
        maxReplicas: maxReplicas
        rules: [
          {
            name: 'http-scale'
            http: {
              metadata: {
                concurrentRequests: '100'
              }
            }
          }
          {
            name: 'cpu-scale'
            custom: {
              type: 'cpu'
              metadata: {
                type: 'Utilization'
                value: '70'
              }
            }
          }
        ]
      }
    }
  }
}

// -----------------------------------------------------------------------------
// Outputs
// -----------------------------------------------------------------------------

@description('The resource ID of the Container App')
output containerAppId string = containerApp.id

@description('The name of the Container App')
output containerAppName string = containerApp.name

@description('The FQDN of the Container App')
output fqdn string = containerApp.properties.configuration.ingress.fqdn

@description('The latest revision FQDN')
output latestRevisionFqdn string = containerApp.properties.latestRevisionFqdn

@description('The managed identity principal ID')
output managedIdentityPrincipalId string = useManagedIdentity ? managedIdentity.properties.principalId : ''

@description('The managed identity client ID')
output managedIdentityClientId string = useManagedIdentity ? managedIdentity.properties.clientId : ''
