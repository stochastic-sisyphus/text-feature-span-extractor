// =============================================================================
// Role Assignment Module
// =============================================================================
// Creates an Azure RBAC role assignment
// =============================================================================

@description('Principal ID to assign the role to')
param principalId string

@description('Role definition ID (GUID only, not full resource ID)')
param roleDefinitionId string

@description('Principal type')
@allowed(['User', 'Group', 'ServicePrincipal', 'ForeignGroup', 'Device'])
param principalType string = 'ServicePrincipal'

@description('Description of the role assignment')
param description string = ''

// -----------------------------------------------------------------------------
// Resources
// -----------------------------------------------------------------------------

resource roleAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(subscription().id, resourceGroup().id, principalId, roleDefinitionId)
  properties: {
    principalId: principalId
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', roleDefinitionId)
    principalType: principalType
    description: description
  }
}

// -----------------------------------------------------------------------------
// Outputs
// -----------------------------------------------------------------------------

@description('The resource ID of the role assignment')
output roleAssignmentId string = roleAssignment.id

@description('The name of the role assignment')
output roleAssignmentName string = roleAssignment.name
