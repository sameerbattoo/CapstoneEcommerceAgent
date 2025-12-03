#!/bin/bash
# One-time infrastructure setup for AgentCore Runtime

set -e

# Configuration
REGION="${AWS_REGION:-us-west-2}"
PROJECT_NAME="${PROJECT_NAME:-CapstoneECommerce}"
VPC_CIDR="10.0.0.0/16"
PUBLIC_SUBNET_CIDR="10.0.1.0/24"
PRIVATE_SUBNET_1_CIDR="10.0.2.0/24"
PRIVATE_SUBNET_2_CIDR="10.0.3.0/24"

# Get availability zones
AZ1=$(aws ec2 describe-availability-zones --region $REGION \
    --query 'AvailabilityZones[0].ZoneName' --output text)
AZ2=$(aws ec2 describe-availability-zones --region $REGION \
    --query 'AvailabilityZones[1].ZoneName' --output text)

# Create VPC
VPC_ID=$(aws ec2 create-vpc \
    --cidr-block $VPC_CIDR \
    --region $REGION \
    --tag-specifications "ResourceType=vpc,Tags=[{Key=Name,Value=$PROJECT_NAME-vpc}]" \
    --query 'Vpc.VpcId' \
    --output text)

# Enable DNS (required for VPC endpoint resolution)
aws ec2 modify-vpc-attribute --vpc-id $VPC_ID --enable-dns-hostnames --region $REGION
aws ec2 modify-vpc-attribute --vpc-id $VPC_ID --enable-dns-support --region $REGION

# Create Internet Gateway
IGW_ID=$(aws ec2 create-internet-gateway \
    --region $REGION \
    --tag-specifications "ResourceType=internet-gateway,Tags=[{Key=Name,Value=$PROJECT_NAME-igw}]" \
    --query 'InternetGateway.InternetGatewayId' \
    --output text)
aws ec2 attach-internet-gateway --vpc-id $VPC_ID --internet-gateway-id $IGW_ID --region $REGION

# Create Subnets
PUBLIC_SUBNET_ID=$(aws ec2 create-subnet \
    --vpc-id $VPC_ID \
    --cidr-block $PUBLIC_SUBNET_CIDR \
    --availability-zone $AZ1 \
    --region $REGION \
    --tag-specifications "ResourceType=subnet,Tags=[{Key=Name,Value=$PROJECT_NAME-public}]" \
    --query 'Subnet.SubnetId' \
    --output text)

PRIVATE_SUBNET_1_ID=$(aws ec2 create-subnet \
    --vpc-id $VPC_ID \
    --cidr-block $PRIVATE_SUBNET_1_CIDR \
    --availability-zone $AZ1 \
    --region $REGION \
    --tag-specifications "ResourceType=subnet,Tags=[{Key=Name,Value=$PROJECT_NAME-private-1}]" \
    --query 'Subnet.SubnetId' \
    --output text)

PRIVATE_SUBNET_2_ID=$(aws ec2 create-subnet \
    --vpc-id $VPC_ID \
    --cidr-block $PRIVATE_SUBNET_2_CIDR \
    --availability-zone $AZ2 \
    --region $REGION \
    --tag-specifications "ResourceType=subnet,Tags=[{Key=Name,Value=$PROJECT_NAME-private-2}]" \
    --query 'Subnet.SubnetId' \
    --output text)

# Create NAT Gateway
EIP_ALLOC_ID=$(aws ec2 allocate-address \
    --domain vpc \
    --region $REGION \
    --query 'AllocationId' \
    --output text)

NAT_GW_ID=$(aws ec2 create-nat-gateway \
    --subnet-id $PUBLIC_SUBNET_ID \
    --allocation-id $EIP_ALLOC_ID \
    --region $REGION \
    --query 'NatGateway.NatGatewayId' \
    --output text)

# Wait for NAT Gateway to become available
aws ec2 wait nat-gateway-available --nat-gateway-ids $NAT_GW_ID --region $REGION

# Create Route Tables
PUBLIC_RT_ID=$(aws ec2 create-route-table \
    --vpc-id $VPC_ID \
    --region $REGION \
    --query 'RouteTable.RouteTableId' \
    --output text)
aws ec2 create-route --route-table-id $PUBLIC_RT_ID \
    --destination-cidr-block 0.0.0.0/0 --gateway-id $IGW_ID --region $REGION
aws ec2 associate-route-table --route-table-id $PUBLIC_RT_ID \
    --subnet-id $PUBLIC_SUBNET_ID --region $REGION

PRIVATE_RT_ID=$(aws ec2 create-route-table \
    --vpc-id $VPC_ID \
    --region $REGION \
    --query 'RouteTable.RouteTableId' \
    --output text)
aws ec2 create-route --route-table-id $PRIVATE_RT_ID \
    --destination-cidr-block 0.0.0.0/0 --nat-gateway-id $NAT_GW_ID --region $REGION
aws ec2 associate-route-table --route-table-id $PRIVATE_RT_ID \
    --subnet-id $PRIVATE_SUBNET_1_ID --region $REGION
aws ec2 associate-route-table --route-table-id $PRIVATE_RT_ID \
    --subnet-id $PRIVATE_SUBNET_2_ID --region $REGION

# Create Security Group
SG_ID=$(aws ec2 create-security-group \
    --group-name "$PROJECT_NAME-agentcore-sg" \
    --description "Security group for AgentCore Runtime" \
    --vpc-id $VPC_ID \
    --region $REGION \
    --query 'GroupId' \
    --output text)

# Allow HTTPS outbound (required for Bedrock API calls)
aws ec2 authorize-security-group-egress \
    --group-id $SG_ID \
    --ip-permissions IpProtocol=tcp,FromPort=443,ToPort=443,IpRanges='[{CidrIp=0.0.0.0/0}]' \
    --region $REGION

# Create IAM Execution Role
ROLE_NAME="$PROJECT_NAME-agentcore-execution-role"

cat > /tmp/trust-policy.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "bedrock-agentcore.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF

aws iam create-role \
    --role-name $ROLE_NAME \
    --assume-role-policy-document file:///tmp/trust-policy.json \
    --region $REGION

# Attach required policies
aws iam attach-role-policy \
    --role-name $ROLE_NAME \
    --policy-arn arn:aws:iam::aws:policy/AmazonBedrockFullAccess \
    --region $REGION

aws iam attach-role-policy \
    --role-name $ROLE_NAME \
    --policy-arn arn:aws:iam::aws:policy/CloudWatchLogsFullAccess \
    --region $REGION

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/${ROLE_NAME}"

# Wait for IAM role to propagate
sleep 10

# Save configuration for deployment script
cat > infrastructure.env <<EOF
export VPC_ID=$VPC_ID
export PRIVATE_SUBNET_1_ID=$PRIVATE_SUBNET_1_ID
export PRIVATE_SUBNET_2_ID=$PRIVATE_SUBNET_2_ID
export SECURITY_GROUP_ID=$SG_ID
export EXECUTION_ROLE_ARN=$ROLE_ARN
export REGION=$REGION
export PROJECT_NAME=$PROJECT_NAME
EOF

echo "Infrastructure setup complete!"
