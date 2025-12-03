#!/bin/bash
# Deploy Lambda function for ProductReviews API

set -e

REGION="${AWS_REGION:-us-west-2}"
FUNCTION_NAME="CapstoneEcommerceProductReviewsAPI"
TABLE_NAME="ProductReviews"
ROLE_NAME="CapstoneEcommerceProductReviewsLambdaRole"

echo "Deploying Lambda function: $FUNCTION_NAME"

# Get AWS Account ID
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

# Create IAM role for Lambda if it doesn't exist
echo "Creating IAM role for Lambda..."

cat > /tmp/lambda-trust-policy.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "lambda.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF

# Create role (ignore error if exists)
aws iam create-role \
    --role-name $ROLE_NAME \
    --assume-role-policy-document file:///tmp/lambda-trust-policy.json \
    --region $REGION 2>/dev/null || echo "Role already exists"

# Create policy for DynamoDB access
cat > /tmp/lambda-policy.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "dynamodb:Query",
        "dynamodb:Scan",
        "dynamodb:GetItem",
        "dynamodb:BatchGetItem"
      ],
      "Resource": [
        "arn:aws:dynamodb:${REGION}:${ACCOUNT_ID}:table/${TABLE_NAME}",
        "arn:aws:dynamodb:${REGION}:${ACCOUNT_ID}:table/${TABLE_NAME}/index/*"
      ]
    },
    {
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": "arn:aws:logs:${REGION}:${ACCOUNT_ID}:*"
    }
  ]
}
EOF

# Attach policies
aws iam put-role-policy \
    --role-name $ROLE_NAME \
    --policy-name DynamoDBAccess \
    --policy-document file:///tmp/lambda-policy.json

echo "Waiting for IAM role to propagate..."
sleep 10

# Package Lambda function
echo "Packaging Lambda function..."
cd "$(dirname "$0")"
zip -q function.zip product_reviews_api.py

ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/${ROLE_NAME}"

# Check if function exists
if aws lambda get-function --function-name $FUNCTION_NAME --region $REGION 2>/dev/null; then
    echo "Updating existing Lambda function..."
    aws lambda update-function-code \
        --function-name $FUNCTION_NAME \
        --zip-file fileb://function.zip \
        --region $REGION
    
    aws lambda update-function-configuration \
        --function-name $FUNCTION_NAME \
        --environment "Variables={TABLE_NAME=$TABLE_NAME}" \
        --region $REGION
else
    echo "Creating new Lambda function..."
    aws lambda create-function \
        --function-name $FUNCTION_NAME \
        --runtime python3.11 \
        --role $ROLE_ARN \
        --handler product_reviews_api.lambda_handler \
        --zip-file fileb://function.zip \
        --timeout 30 \
        --memory-size 256 \
        --environment "Variables={TABLE_NAME=$TABLE_NAME}" \
        --region $REGION
fi

# Create or update function URL with AWS_IAM auth
echo "Configuring Lambda Function URL with AWS_IAM authentication..."
FUNCTION_URL=$(aws lambda create-function-url-config \
    --function-name $FUNCTION_NAME \
    --auth-type AWS_IAM \
    --cors "AllowOrigins=*,AllowMethods=GET,AllowHeaders=*" \
    --region $REGION \
    --query 'FunctionUrl' \
    --output text 2>/dev/null || \
    aws lambda get-function-url-config \
        --function-name $FUNCTION_NAME \
        --region $REGION \
        --query 'FunctionUrl' \
        --output text)

# Clean up
rm function.zip

echo ""
echo "Lambda function deployed successfully!"
echo "Function Name: $FUNCTION_NAME"
echo "Function URL: $FUNCTION_URL"
echo ""
echo "Example API calls:"
echo "  All reviews: curl '$FUNCTION_URL'"
echo "  By product: curl '$FUNCTION_URL?product_id=1'"
echo "  By customer: curl '$FUNCTION_URL?customer_id=5'"
echo "  Date range: curl '$FUNCTION_URL?review_date_from=2024-01-01&review_date_to=2024-12-31'"
echo "  Combined: curl '$FUNCTION_URL?product_id=1&customer_id=5&review_date_from=2024-01-01'"
