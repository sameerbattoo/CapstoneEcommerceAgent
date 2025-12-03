#!/bin/bash

# Variables - Update these values for your environment
VPC_ID="vpc-034baf740bf9b15fc"  # Your CapstoneECommerce VPC
SG_NAME="rds-postgres-serverless-sg"
SG_DESCRIPTION="Security group for RDS PostgreSQL Serverless v2 with Data API"
YOUR_IP="$(curl -s https://checkip.amazonaws.com)/32"  # Auto-detect your current IP
REGION="us-west-2"

echo "Creating security group for RDS PostgreSQL Serverless v2..."

# Create the security group
SG_ID=$(aws ec2 create-security-group \
    --group-name "$SG_NAME" \
    --description "$SG_DESCRIPTION" \
    --vpc-id "$VPC_ID" \
    --region "$REGION" \
    --query 'GroupId' \
    --output text)

echo "Security Group created: $SG_ID"

# Add inbound rules for PostgreSQL access
echo "Adding inbound rules..."

# Rule 1: PostgreSQL port 5432 from your current IP (for direct database access)
aws ec2 authorize-security-group-ingress \
    --group-id "$SG_ID" \
    --protocol tcp \
    --port 5432 \
    --cidr "$YOUR_IP" \
    --region "$REGION"

echo "Added rule: PostgreSQL (5432) from your IP ($YOUR_IP)"

# Rule 2: HTTPS port 443 for RDS Data API (from your IP)
aws ec2 authorize-security-group-ingress \
    --group-id "$SG_ID" \
    --protocol tcp \
    --port 443 \
    --cidr "$YOUR_IP" \
    --region "$REGION"

echo "Added rule: HTTPS (443) from your IP for RDS Data API"

# Rule 3: PostgreSQL from application subnet (if you have application servers)
# Uncomment and modify the CIDR block for your application subnet
# aws ec2 authorize-security-group-ingress \
#     --group-id "$SG_ID" \
#     --protocol tcp \
#     --port 5432 \
#     --cidr "10.0.1.0/24" \
#     --region "$REGION"

# Add tags to the security group
aws ec2 create-tags \
    --resources "$SG_ID" \
    --tags Key=Name,Value="$SG_NAME" \
           Key=Purpose,Value="RDS PostgreSQL Serverless v2" \
           Key=Environment,Value="Production" \
           Key=Project,Value="CapstoneECommerce" \
    --region "$REGION"

echo "Security group tagged successfully"

# Display the security group details
echo "Security Group Details:"
aws ec2 describe-security-groups \
    --group-ids "$SG_ID" \
    --region "$REGION" \
    --query 'SecurityGroups[0].{GroupId:GroupId,GroupName:GroupName,Description:Description,VpcId:VpcId}'

echo "Security Group ID: $SG_ID"
echo "Setup complete!"
