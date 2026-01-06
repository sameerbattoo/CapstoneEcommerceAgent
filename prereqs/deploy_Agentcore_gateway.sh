#!/bin/bash

# Script to create AWS Bedrock AgentCore Gateway
# Gateway ID: capstone-ecommerce-reviews-gateway-oauth-jnnukgjvgo
# ARN: arn:aws:bedrock-agentcore:us-west-2:175918693907:gateway/capstone-ecommerce-reviews-gateway-oauth-jnnukgjvgo

set -e  # Exit on any error

# Configuration variables
GATEWAY_NAME="capstone-ecommerce-reviews-gateway-oauth"
GATEWAY_DESCRIPTION="E-commerce Reviews AgentCore Gateway with OAuth Authentication"
REGION="us-west-2"
ACCOUNT_ID="175918693907"
PROTOCOL_TYPE="MCP"
AUTHORIZER_TYPE="CUSTOM_JWT"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}[HEADER]${NC} $1"
}

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    print_error "AWS CLI is not installed. Please install it first."
    exit 1
fi

# Check if jq is installed for JSON parsing
if ! command -v jq &> /dev/null; then
    print_warning "jq is not installed. Some output formatting may be limited."
    JQ_AVAILABLE=false
else
    JQ_AVAILABLE=true
fi

# Check if bedrock-agentcore-starter-toolkit is available
if command -v bedrock-agentcore-starter-toolkit &> /dev/null; then
    USE_TOOLKIT=true
    print_status "Using Bedrock AgentCore starter toolkit CLI"
else
    USE_TOOLKIT=false
    print_status "Using AWS CLI for Bedrock AgentCore"
fi

# Check if AWS credentials are configured
if ! aws sts get-caller-identity &> /dev/null; then
    print_error "AWS credentials are not configured. Please run 'aws configure' first."
    exit 1
fi

# Verify we're in the correct region and account
CURRENT_ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
if [ "$CURRENT_ACCOUNT" != "$ACCOUNT_ID" ]; then
    print_warning "Current AWS account ($CURRENT_ACCOUNT) doesn't match expected account ($ACCOUNT_ID)"
    read -p "Do you want to continue? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

## Create the Gateway Role first

# Configuration variables
ROLE_NAME="CapstoneEcommerceProductReviewsGatewayRole"
ROLE_PATH="/service-role/"
ACCOUNT_ID="175918693907"
ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role${ROLE_PATH}${ROLE_NAME}"
ROLE_DESCRIPTION="IAM role for Capstone E-commerce Product Reviews AgentCore Gateway"

print_header "=== Creating IAM Role for Bedrock AgentCore Gateway ==="
print_status "Role Name: $ROLE_NAME"
print_status "Role Path: $ROLE_PATH"
print_status "Target ARN: $ROLE_ARN"

# Check if role already exists
print_status "Checking if IAM role already exists..."
if aws iam get-role --role-name "$ROLE_NAME" &> /dev/null; then
    print_warning "Role '$ROLE_NAME' already exists!"

    # Get existing role ARN
    EXISTING_ARN=$(aws iam get-role --role-name "$ROLE_NAME" --query 'Role.Arn' --output text)
    print_status "Existing role ARN: $EXISTING_ARN"

    if [ "$EXISTING_ARN" = "$ROLE_ARN" ]; then
        print_status "✅ Role already exists with the correct ARN"
        exit 0
    fi
fi

# Create trust policy document
print_status "Creating trust policy document..."
cat > trust-policy.json << 'EOF'
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "AmazonBedrockAgentCoreGatewayBasePolicyProd",
            "Effect": "Allow",
            "Principal": {
                "Service": [
                    "bedrock-agentcore.amazonaws.com"
                ]
            },
            "Action": "sts:AssumeRole",
            "Condition": {
                "StringEquals": {
                    "aws:SourceAccount": "175918693907"
                }
            }
        }
    ]
}
EOF

# Create custom policy for AgentCore Gateway operations
print_status "Creating custom policy document..."
cat > gateway-policy.json << 'EOF'
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "GetGateway",
            "Effect": "Allow",
            "Action": [
                "bedrock-agentcore:GetGateway"
            ],
            "Resource": [
                "arn:aws:bedrock-agentcore:us-west-2:175918693907:gateway/capstone-ecommerce-reviews-gateway-*"
            ]
        },
        {
            "Sid": "AmazonBedrockAgentCoreGatewayLambdaProd",
            "Effect": "Allow",
            "Action": [
                "lambda:InvokeFunction"
            ],
            "Resource": [
                "arn:aws:lambda:us-west-2:175918693907:function:CapstoneEcommerceProductReviewsAPI:$LATEST"
            ]
        },
        {
            "Effect": "Allow",
            "Action": [
                "logs:CreateLogGroup",
                "logs:CreateLogStream",
                "logs:PutLogEvents",
                "logs:DescribeLogGroups",
                "logs:DescribeLogStreams"
            ],
            "Resource": [
                "arn:aws:logs:*:175918693907:log-group:/aws/bedrock/agentcore/*",
                "arn:aws:logs:*:175918693907:log-group:/aws/lambda/*"
            ]
        },
        {
            "Effect": "Allow",
            "Action": [
                "kms:Decrypt",
                "kms:DescribeKey",
                "kms:GenerateDataKey"
            ],
            "Resource": "*",
            "Condition": {
                "StringEquals": {
                    "kms:ViaService": [
                        "bedrock-agentcore.*.amazonaws.com",
                        "lambda.*.amazonaws.com"
                    ]
                }
            }
        },
        {
            "Effect": "Allow",
            "Action": [
                "secretsmanager:GetSecretValue",
                "secretsmanager:DescribeSecret"
            ],
            "Resource": "arn:aws:secretsmanager:*:175918693907:secret:*"
        }
    ]
}
EOF

# Create the IAM role
print_status "Creating IAM role: $ROLE_NAME"

CREATE_ROLE_OUTPUT=$(aws iam create-role \
    --role-name "$ROLE_NAME" \
    --path "$ROLE_PATH" \
    --assume-role-policy-document file://trust-policy.json \
    --description "$ROLE_DESCRIPTION" \
    --output json)

# Create and attach custom policy
POLICY_NAME="CapstoneEcommerceGatewayPolicy"
POLICY_ARN="arn:aws:iam::${ACCOUNT_ID}:policy/service-role/${POLICY_NAME}"

print_status "Creating custom policy: $POLICY_NAME"
# Check if policy already exists
if aws iam get-policy --policy-arn "$POLICY_ARN" &> /dev/null; then
    print_status "Policy already exists. Creating new version..."
    
    # Create new policy version
    aws iam create-policy-version \
        --policy-arn "$POLICY_ARN" \
        --policy-document file://gateway-policy.json \
        --set-as-default > /dev/null
    
    if [ $? -eq 0 ]; then
        print_status "✅ Policy version created successfully"
    else
        print_error "Failed to create policy version"
        exit 1
    fi
else
    # Create new policy
    CREATE_POLICY_OUTPUT=$(aws iam create-policy \
        --policy-name "$POLICY_NAME" \
        --path "/service-role/" \
        --policy-document file://gateway-policy.json \
        --description "Custom policy for Capstone E-commerce AgentCore Gateway" \
        --output json)
    
    if [ $? -eq 0 ]; then
        print_status "✅ Custom policy created successfully"
        
        if [ "$JQ_AVAILABLE" = true ]; then
            CREATED_POLICY_ARN=$(echo "$CREATE_POLICY_OUTPUT" | jq -r '.Policy.Arn')
            print_status "Created policy ARN: $CREATED_POLICY_ARN"
        fi
    else
        print_error "Failed to create custom policy"
        exit 1
    fi
fi

# Attach AWS managed policy for Bedrock AgentCore
print_status "Attaching AWS managed policy for Bedrock AgentCore..."
aws iam attach-role-policy \
    --role-name "$ROLE_NAME" \
    --policy-arn "arn:aws:iam::aws:policy/service-role/AmazonBedrockAgentCoreGatewayServiceRolePolicy" 2>/dev/null || {
    print_warning "AWS managed policy for AgentCore may not exist yet. Continuing with custom policy only."
}

# Attach custom policy to role
print_status "Attaching custom policy to role..."
aws iam attach-role-policy \
    --role-name "$ROLE_NAME" \
    --policy-arn "$POLICY_ARN"

if [ $? -eq 0 ]; then
    print_status "✅ Custom policy attached successfully"
else
    print_error "Failed to attach custom policy"
    exit 1
fi

# Wait for role to be available
print_status "Waiting for role to be fully available..."
sleep 10

# Verify role creation and policies
print_status "Verifying role configuration..."

# Get role details
ROLE_DETAILS=$(aws iam get-role --role-name "$ROLE_NAME" --output json)
if [ "$JQ_AVAILABLE" = true ]; then
    FINAL_ARN=$(echo "$ROLE_DETAILS" | jq -r '.Role.Arn')
    CREATION_DATE=$(echo "$ROLE_DETAILS" | jq -r '.Role.CreateDate')
    print_status "Final role ARN: $FINAL_ARN"
    print_status "Creation date: $CREATION_DATE"
fi

# List attached policies
print_status "Attached policies:"
aws iam list-attached-role-policies --role-name "$ROLE_NAME" --output table

# Clean up temporary files
print_status "Cleaning up temporary files..."
rm -f trust-policy.json gateway-policy.json

# Output summary
print_header "=== IAM Role Creation Summary ==="
echo "Role Name: $ROLE_NAME"
echo "Role Path: $ROLE_PATH"
echo "Target ARN: $ROLE_ARN"
echo "Account ID: $ACCOUNT_ID"
echo "Custom Policy: $POLICY_ARN"

print_status "✅ IAM role creation completed successfully!"

# Create the AgentCore Gateway
print_status "Creating AgentCore Gateway: $GATEWAY_NAME"

# Create authorizer configuration for OAuth
cat > authorizer-config.json << EOF
{
    "customJWTAuthorizer": {
        "discoveryUrl": "https://cognito-idp.us-west-2.amazonaws.com/us-west-2_5cqnrBvAg/.well-known/openid-configuration",
        "allowedClients": ["2nc8c09npb6vru63mgfl7kiqb1"],
        "allowedScopes": ["capstone_agent_gateway/read"]
    }
}
EOF

if [ "$USE_TOOLKIT" = true ]; then
    # Using the starter toolkit CLI (simplified approach)
    print_status "Using bedrock-agentcore-starter-toolkit to create gateway..."
    
    GATEWAY_OUTPUT=$(bedrock-agentcore-starter-toolkit gateway create-mcp-gateway \
        --name "$GATEWAY_NAME" \
        --region "$REGION" \
        --role-arn "$ROLE_ARN" \
        --authorizer-config  file://authorizer-config.json \
        --enable_semantic_search false 2>&1)
    
    if [ $? -eq 0 ]; then
        print_status "Gateway created successfully using starter toolkit"
        echo "$GATEWAY_OUTPUT"
        
        # Extract gateway ID from output if possible
        GATEWAY_ID=$(echo "$GATEWAY_OUTPUT" | grep -o 'gateway/[a-zA-Z0-9]\{10\}' | cut -d'/' -f2 || echo "")
    else
        print_error "Failed to create gateway using starter toolkit"
        echo "$GATEWAY_OUTPUT"
        exit 1
    fi
else
    # Using AWS CLI directly
    print_status "Using AWS CLI to create gateway..."

    # Create the gateway
    GATEWAY_RESPONSE=$(aws bedrock-agentcore-control create-gateway \
        --name "$GATEWAY_NAME" \
        --description "$GATEWAY_DESCRIPTION" \
        --role-arn "$ROLE_ARN" \
        --protocol-type "$PROTOCOL_TYPE" \
        --authorizer-type "$AUTHORIZER_TYPE" \
        --authorizer-configuration file://authorizer-config.json \
        --region "$REGION" \
        --output json)

    if [ $? -eq 0 ]; then
        print_status "Gateway created successfully"
        
        # Extract gateway information
        GATEWAY_ARN=$(echo "$GATEWAY_RESPONSE" | jq -r '.gatewayArn')
        GATEWAY_ID=$(echo "$GATEWAY_ARN" | grep -o '[a-zA-Z0-9]\{10\}$')
        GATEWAY_URL=$(echo "$GATEWAY_RESPONSE" | jq -r '.gatewayUrl // empty')
        
        print_status "Gateway ARN: $GATEWAY_ARN"
        print_status "Gateway ID: $GATEWAY_ID"
        if [ -n "$GATEWAY_URL" ]; then
            print_status "Gateway URL: $GATEWAY_URL"
        fi
    else
        print_error "Failed to create gateway"
        echo "$GATEWAY_RESPONSE"
        exit 1
    fi
fi

# Clean up temporary file
rm authorizer-config.json

# Verify the gateway was created with the expected ID
if [ -n "$GATEWAY_ID" ] && [ "$GATEWAY_ID" = "capstone-ecommerce-reviews-gateway-oauth-jnnukgjvgo" ]; then
    print_status "✅ Gateway created with the expected ID: $GATEWAY_ID"
elif [ -n "$GATEWAY_ID" ]; then
    print_warning "⚠️  Gateway created but with different ID: $GATEWAY_ID"
    print_warning "Expected ID: capstone-ecommerce-reviews-gateway-oauth-jnnukgjvgo"
else
    print_warning "Could not extract gateway ID from response"
fi

# Output summary
print_status "=== AgentCore Gateway Creation Summary ==="
echo "Gateway Name: $GATEWAY_NAME"
echo "Region: $REGION"
echo "Protocol Type: $PROTOCOL_TYPE"
echo "Authorizer Type: $AUTHORIZER_TYPE"
echo "IAM Role: $ROLE_ARN"
if [ -n "$GATEWAY_ID" ]; then
    echo "Gateway ID: $GATEWAY_ID"
    echo "Expected ARN: arn:aws:bedrock-agentcore:$REGION:$ACCOUNT_ID:gateway/$GATEWAY_ID"
fi

print_status "AgentCore Gateway creation completed!"

print_header "=== Creating Targets for AgentCore Gateway ==="

# Array to store created target IDs
declare -a CREATED_TARGETS=()

cat > inline-lambda-schema.json << EOF
[
  {
    "description": "Get product reviews with optional filters. Returns reviews matching the specified criteria. All parameters are optional - if no parameters provided, returns all reviews (limited by top_rows). Use product_id for efficient queries. Supports filtering by customer, product name, email, rating, date range, or combinations thereof. All ID and name parameters accept comma-separated lists for multiple values (OR logic).",
    "inputSchema": {
      "properties": {
        "customer_email": {
          "description": "Filter reviews by customer email(s) (partial match, case-sensitive). Single email (e.g., 'john') or comma-separated list (e.g., 'john,jane,bob'). Uses OR logic for multiple values.",
          "type": "string"
        },
        "customer_id": {
          "description": "Filter reviews by customer ID(s). Single ID (e.g., '5') or comma-separated list (e.g., '1,5,10'). Uses OR logic for multiple values.",
          "type": "string"
        },
        "product_id": {
          "description": "Filter reviews by product ID(s). Single ID (e.g., '1') or comma-separated list (e.g., '1,5,10'). When provided, uses efficient DynamoDB Query operation.",
          "type": "string"
        },
        "product_name": {
          "description": "Filter reviews by product name(s) (partial match, case-sensitive). Single name (e.g., 'Laptop') or comma-separated list (e.g., 'Laptop,Mouse,Keyboard'). Uses OR logic for multiple values.",
          "type": "string"
        },
        "rating": {
          "description": "Filter reviews by rating value(s). Single rating (e.g., '5') or comma-separated list (e.g., '4,5'). Valid values: 1, 2, 3, 4, or 5 stars. Uses OR logic for multiple values.",
          "type": "string"
        },
        "review_date_from": {
          "description": "Start date for review date range filter in ISO format (YYYY-MM-DD). Example: 2024-01-01",
          "type": "string"
        },
        "review_date_to": {
          "description": "End date for review date range filter in ISO format (YYYY-MM-DD). Example: 2024-12-31",
          "type": "string"
        },
        "top_rows": {
          "description": "Limit the number of rows returned. Default: 20, Minimum: 1, Maximum: 20. Results are sorted by review_date descending before applying limit.",
          "type": "integer"
        }
      },
      "required": [],
      "type": "object"
    },
    "name": "get_product_reviews"
  }
]
EOF

# Function to create a Lambda target
create_lambda_target() {
    local target_name=$1
    local lambda_arn=$2
    local description=$3
    local inlinelambdaschema=$4
    
    print_status "Creating Lambda target: $target_name"
    
    if [ "$USE_TOOLKIT" = true ]; then
        # Using starter toolkit
        TARGET_OUTPUT=$(bedrock-agentcore-starter-toolkit gateway create-mcp-gateway-target \
            --gateway-arn "$GATEWAY_ARN" \
            --gateway-url "$GATEWAY_URL" \
            --role-arn "$ROLE_ARN" \
            --name "$target_name" \
            --target-type lambda \
            --target-payload "{\"lambdaArn\": \"$lambda_arn\", 
                "toolSchema": \"$inlinelambdaschema\"}" 2>&1)
        
        if [ $? -eq 0 ]; then
            print_status "✅ Lambda target '$target_name' created successfully"
            echo "$TARGET_OUTPUT"
            CREATED_TARGETS+=("$target_name")
        else
            print_error "Failed to create Lambda target '$target_name'"
            echo "$TARGET_OUTPUT"
        fi
    else
        # Using AWS CLI directly
        cat > lambda-target-config.json << EOF
{
    "name": "$target_name",
    "description": "$description",
    "targetConfiguration": {
        "lambdaTargetConfiguration": {
            "lambdaArn": "$lambda_arn"
        }
    }
}
EOF
        
        TARGET_RESPONSE=$(aws bedrock-agentcore-control create-gateway-target \
            --gateway-identifier "$GATEWAY_ID" \
            --cli-input-json file://lambda-target-config.json \
            --region "$REGION" \
            --output json 2>&1)
        
        if [ $? -eq 0 ]; then
            print_status "✅ Lambda target '$target_name' created successfully"
            if [ "$JQ_AVAILABLE" = true ]; then
                TARGET_ID=$(echo "$TARGET_RESPONSE" | jq -r '.targetId // empty')
                if [ -n "$TARGET_ID" ]; then
                    print_status "Target ID: $TARGET_ID"
                fi
            fi
            CREATED_TARGETS+=("$target_name")
        else
            print_error "Failed to create Lambda target '$target_name'"
            echo "$TARGET_RESPONSE"
        fi
        
        rm -f lambda-target-config.json
    fi
}

# Create Lambda targets for e-commerce reviews CRUD operations
print_header "Creating Lambda Targets for ProductReviewLambda"

# Create Reviews Lambda Target
create_lambda_target \
    "ProductReviewLambda" \
    "arn:aws:lambda:us-west-2:175918693907:function:CapstoneEcommerceProductReviewsAPI:$LATEST" \
    "Lambda function for getting reviews" \
    file://inline-lambda-schema.json

# Clean up temporary file
rm inline-lambda-schema.json