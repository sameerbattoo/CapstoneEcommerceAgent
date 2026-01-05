import json

def lambda_handler(event, context):
    # Get user attributes
    user_attributes = event['request']['userAttributes']
    tenant_id = user_attributes.get('custom:tenantId', 'N/A')
    print(f"Tenant_id: {tenant_id}")
    
    # Initialize response if not present
    if 'response' not in event:
        event['response'] = {}
    
    # Add to ID token (V1 style - works for both)
    event['response']['claimsOverrideDetails'] = {
        'claimsToAddOrOverride': {
            'tenantId': tenant_id
        }
    }
    
    # Add to Access token (V2 style)
    event['response']['claimsAndScopeOverrideDetails'] = {
        'accessTokenGeneration': {
            'claimsToAddOrOverride': {
                'tenantId': tenant_id
            }
        },
        'idTokenGeneration': {
            'claimsToAddOrOverride': {
                'tenantId': tenant_id
            }
        }
    }
    
    return event
