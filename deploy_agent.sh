#!/bin/bash
# Agent deployment to AgentCore Runtime

set -e

uv run agentcore configure -e main.py --vpc \
  --subnets subnet-0d924ff443cebd41d,subnet-0463efcbc95e2d5e1 \
  --security-groups sg-06d560db4091b375c \
  --name capstone_ecomm_agent \
  --region us-west-2 \
  --execution-role arn:aws:iam::175918693907:role/CapstoneECommerce-agentcore-execution-role \
  --disable-memory \
  --deployment-type container \
  --authorizer-config '{"type":"customJWTAuthorizer","discoveryUrl":"https://cognito-idp.us-west-2.amazonaws.com/us-west-2_5cqnrBvAg/.well-known/openid-configuration","allowedClients":"2nc8c09npb6vru63mgfl7kiqb1"}' \
  --request-header-allowlist Authorization,X-Amzn-Bedrock-AgentCore-Runtime-Custom-TenantId,X-Amzn-Bedrock-AgentCore-Runtime-Custom-ActorId \
  --non-interactive

uv run agentcore launch \
	--env AWS_REGION="us-west-2" \
  --env AWS_SECRET_NAME="capstone-ecommerce-agent-config"

echo "Agent deployment complete!"
