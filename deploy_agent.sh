#!/bin/bash
# Agent deployment to AgentCore Runtime

set -e

uv run agentcore configure -e main.py --vpc \
  --subnets subnet-0d924ff443cebd41d,subnet-0463efcbc95e2d5e1 \
  --security-groups sg-06d560db4091b375c \
  --name capstone_ecommerce_agent_cognito \
  --region us-west-2 \
  --execution-role arn:aws:iam::175918693907:role/CapstoneECommerce-agentcore-execution-role \
  --disable-memory \
  --authorizer-config '{"type":"customJWTAuthorizer","discoveryUrl":"https://cognito-idp.us-west-2.amazonaws.com/us-west-2_5cqnrBvAg/.well-known/openid-configuration","allowedClients":"2nc8c09npb6vru63mgfl7kiqb1"}' \
  --request-header-allowlist Authorization,X-Amzn-Bedrock-AgentCore-Runtime-Custom-Session-Id \
  --non-interactive

uv run agentcore launch \
	--env DB_HOST="capstone-ecommerce-db.cluster-cnuwbnhjyijb.us-west-2.rds.amazonaws.com" \
	--env DB_PORT="5432" \
	--env DB_NAME="ecommerce" \
	--env DB_USER="postgres" \
	--env DB_PASSWORD="iK_Jy(nrapXm3*AkdDtb1o5Z7pz:" \
	--env AWS_REGION="us-west-2" \
	--env BEDROCK_MODEL_ID="global.anthropic.claude-sonnet-4-5-20250929-v1:0" \
	--env BEDROCK_MODEL_ARN="arn:aws:bedrock:us-west-2:175918693907:inference-profile/global.anthropic.claude-sonnet-4-5-20250929-v1:0" \
	--env AGENTCORE_MEMORY_ID="capstone_ecommerce_mem-dnvxeI5Nfm" \
	--env KB_ID="H36GPUOAYX" \
	--env GATEWAY_URL="https://capstone-ecommerce-reviews-gateway-ohq6pd9lf5.gateway.bedrock-agentcore.us-west-2.amazonaws.com/mcp" \
    --env OTEL_EXPORTER_OTLP_ENDPOINT=https://xray.us-west-2.amazonaws.com/v1/traces

echo "Agent deployment complete!"
