# ECommerce Agent - Deployment Guide

## Prerequisites

### Required Tools
- **Python**: 3.13+
- **uv**: Python package manager (recommended) or pip
- **AWS CLI**: v2.x configured with appropriate credentials
- **Docker**: For container-based deployment
- **Git**: For version control

### AWS Account Requirements
- **IAM Permissions**: Administrator access or specific permissions for:
  - Bedrock, AgentCore, Cognito, RDS, DynamoDB, S3, Lambda, VPC, CloudWatch, Secrets Manager
- **Service Quotas**: Ensure sufficient quotas for:
  - Bedrock model invocations
  - AgentCore runtime instances
  - RDS instances
  - DynamoDB tables

### AWS Services Setup
- AWS Bedrock enabled in your region
- AWS Bedrock AgentCore access
- Model access granted for Claude 4.5 Sonnet/Haiku

---

## Deployment Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     AWS Cloud (us-west-2)                    │
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │                    VPC                              │    │
│  │  ┌──────────────┐         ┌──────────────┐        │    │
│  │  │ Private      │         │ Private      │        │    │
│  │  │ Subnet 1     │         │ Subnet 2     │        │    │
│  │  │              │         │              │        │    │
│  │  │ ┌──────────┐ │         │ ┌──────────┐ │        │    │
│  │  │ │AgentCore │ │         │ │   RDS    │ │        │    │
│  │  │ │ Runtime  │ │         │ │PostgreSQL│ │        │    │
│  │  │ └──────────┘ │         │ └──────────┘ │        │    │
│  │  │              │         │              │        │    │
│  │  │ ┌──────────┐ │         │ ┌──────────┐ │        │    │
│  │  │ │ElastiCache│ │         │ │          │ │        │    │
│  │  │ │ (Valkey) │ │         │ │          │ │        │    │
│  │  │ └──────────┘ │         │ └──────────┘ │        │    │
│  │  └──────────────┘         └──────────────┘        │    │
│  │         │                         │                │    │
│  │         └─────────┬───────────────┘                │    │
│  │                   │                                │    │
│  │            ┌──────▼──────┐                         │    │
│  │            │ NAT Gateway │                         │    │
│  │            └─────────────┘                         │    │
│  └────────────────────────────────────────────────────┘    │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │   Cognito    │  │  DynamoDB    │  │      S3      │    │
│  │  User Pool   │  │ProductReviews│  │  KB + Charts │    │
│  └──────────────┘  └──────────────┘  └──────────────┘    │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │   Lambda     │  │  CloudFront  │  │   Secrets    │    │
│  │ Reviews API  │  │     CDN      │  │   Manager    │    │
│  └──────────────┘  └──────────────┘  └──────────────┘    │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐                       │
│  │  CloudWatch  │  │   Bedrock    │                       │
│  │Logs + Metrics│  │Knowledge Base│                       │
│  └──────────────┘  └──────────────┘                       │
└─────────────────────────────────────────────────────────────┘
```

---

## Step-by-Step Deployment

### Phase 1: Infrastructure Setup

#### 1.1 Clone Repository
```bash
git clone https://github.com/your-org/capstone-ecommerce-agent.git
cd capstone-ecommerce-agent
```

#### 1.2 Install Dependencies
```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -r requirements.txt
```

#### 1.3 Configure AWS CLI
```bash
aws configure
# Enter your AWS Access Key ID
# Enter your AWS Secret Access Key
# Default region: us-west-2
# Default output format: json
```

#### 1.4 Set Up VPC and Networking
```bash
cd prereqs
./setup_infrastructure.sh
```

**This script creates**:
- VPC with CIDR 10.0.0.0/16
- 2 private subnets (10.0.1.0/24, 10.0.2.0/24)
- 2 public subnets (10.0.101.0/24, 10.0.102.0/24)
- Internet Gateway
- NAT Gateway (in public subnet)
- Route tables
- Security groups

**Expected Output**:
```
VPC ID: vpc-0abc123def456
Private Subnet 1: subnet-0d924ff443cebd41d
Private Subnet 2: subnet-0463efcbc95e2d5e1
Security Group: sg-06d560db4091b375c
NAT Gateway: nat-0xyz789
```

**Save these IDs** - you'll need them for AgentCore configuration.

#### 1.5 Create RDS PostgreSQL Database
```bash
# Create RDS instance
aws rds create-db-instance \
  --db-instance-identifier capstone-ecommerce-db \
  --db-instance-class db.t3.medium \
  --engine postgres \
  --engine-version 14.10 \
  --master-username admin \
  --master-user-password YourSecurePassword123! \
  --allocated-storage 100 \
  --vpc-security-group-ids sg-06d560db4091b375c \
  --db-subnet-group-name your-db-subnet-group \
  --backup-retention-period 7 \
  --multi-az \
  --publicly-accessible false

# Wait for instance to be available (5-10 minutes)
aws rds wait db-instance-available \
  --db-instance-identifier capstone-ecommerce-db

# Get endpoint
aws rds describe-db-instances \
  --db-instance-identifier capstone-ecommerce-db \
  --query 'DBInstances[0].Endpoint.Address' \
  --output text
```

**Expected Output**:
```
capstone-ecommerce-db.c1abc2def3gh.us-west-2.rds.amazonaws.com
```

#### 1.6 Configure RDS Security Group
```bash
./db_security_group_setup.sh
```

This allows inbound PostgreSQL traffic (port 5432) from AgentCore security group.

#### 1.7 Create Database Schemas and Load Data
```bash
cd database

# Create schemas for tenants
python create_schema_tables.py \
  --host capstone-ecommerce-db.c1abc2def3gh.us-west-2.rds.amazonaws.com \
  --port 5432 \
  --database ecommerce \
  --user admin \
  --password YourSecurePassword123! \
  --tenants tenanta,tenantb

# Populate with sample data
python populate_schema_data.py \
  --host capstone-ecommerce-db.c1abc2def3gh.us-west-2.rds.amazonaws.com \
  --port 5432 \
  --database ecommerce \
  --user admin \
  --password YourSecurePassword123! \
  --tenant tenanta

python populate_schema_data.py \
  --host capstone-ecommerce-db.c1abc2def3gh.us-west-2.rds.amazonaws.com \
  --port 5432 \
  --database ecommerce \
  --user admin \
  --password YourSecurePassword123! \
  --tenant tenantb
```

#### 1.8 Create DynamoDB Table
```bash
cd ../..
./prereqs/deploy_dynamo_db.sh
```

**This creates**:
- Table: ProductReviews
- Partition Key: product_id (Number)
- Sort Key: review_id (String)
- GSI: tenant_id-product_id-index
- Sample review data for both tenants

#### 1.9 Create S3 Buckets
```bash
# Knowledge Base bucket
aws s3 mb s3://capstone-ecommerce-kb-${AWS_ACCOUNT_ID} --region us-west-2

# Charts bucket
aws s3 mb s3://capstone-ecommerce-charts-${AWS_ACCOUNT_ID} --region us-west-2

# Upload product specification PDFs
aws s3 sync prereqs/kb_files/ s3://capstone-ecommerce-kb-${AWS_ACCOUNT_ID}/
```

#### 1.10 Create Bedrock Knowledge Base
```bash
# Via AWS Console:
# 1. Go to Amazon Bedrock → Knowledge Bases → Create
# 2. Name: ECommerce Product Specifications
# 3. Data source: S3
# 4. S3 URI: s3://capstone-ecommerce-kb-${AWS_ACCOUNT_ID}/
# 5. Embedding model: amazon.titan-embed-text-v2:0
# 6. Vector store: OpenSearch Serverless (auto-create)
# 7. Sync data source

# Or via CLI (requires additional setup)
# See AWS documentation for Knowledge Base CLI creation
```

**Save the Knowledge Base ID**: `KB123456789`

#### 1.11 Create ElastiCache (Valkey) Cluster
```bash
aws elasticache create-cache-cluster \
  --cache-cluster-id capstone-sqlagent-valkey-cache \
  --engine valkey \
  --cache-node-type cache.t3.medium \
  --num-cache-nodes 1 \
  --security-group-ids sg-06d560db4091b375c \
  --cache-subnet-group-name your-cache-subnet-group \
  --preferred-availability-zone us-west-2a

# Wait for cluster to be available
aws elasticache wait cache-cluster-available \
  --cache-cluster-id capstone-sqlagent-valkey-cache

# Get endpoint
aws elasticache describe-cache-clusters \
  --cache-cluster-id capstone-sqlagent-valkey-cache \
  --show-cache-node-info \
  --query 'CacheClusters[0].CacheNodes[0].Endpoint.Address' \
  --output text
```

**Expected Output**:
```
master.capstone-sqlagent-valkey-cache.8ot617.usw2.cache.amazonaws.com
```

---

### Phase 2: Authentication Setup

#### 2.1 Create Cognito User Pool
```bash
aws cognito-idp create-user-pool \
  --pool-name CapstoneECommerceUserPool \
  --policies '{
    "PasswordPolicy": {
      "MinimumLength": 8,
      "RequireUppercase": true,
      "RequireLowercase": true,
      "RequireNumbers": true,
      "RequireSymbols": true
    }
  }' \
  --auto-verified-attributes email \
  --mfa-configuration OPTIONAL \
  --region us-west-2
```

**Save User Pool ID**: `us-west-2_5cqnrBvAg`

#### 2.2 Create App Client
```bash
aws cognito-idp create-user-pool-client \
  --user-pool-id us-west-2_5cqnrBvAg \
  --client-name CapstoneECommerceWebApp \
  --explicit-auth-flows ALLOW_USER_PASSWORD_AUTH ALLOW_REFRESH_TOKEN_AUTH \
  --token-validity-units '{
    "AccessToken": "hours",
    "IdToken": "hours",
    "RefreshToken": "days"
  }' \
  --access-token-validity 1 \
  --id-token-validity 1 \
  --refresh-token-validity 30 \
  --region us-west-2
```

**Save Client ID**: `2nc8c09npb6vru63mgfl7kiqb1`

#### 2.3 Deploy Pre-Token Generation Lambda
```bash
cd lambda

# Create Lambda function
aws lambda create-function \
  --function-name CapstoneEcommerceCognitoPreTokenGeneration \
  --runtime python3.13 \
  --role arn:aws:iam::${AWS_ACCOUNT_ID}:role/LambdaExecutionRole \
  --handler CapstoneEcommerceCognitoPreTokenGeneration.lambda_handler \
  --zip-file fileb://CapstoneEcommerceCognitoPreTokenGeneration.zip \
  --region us-west-2

# Grant Cognito permission to invoke Lambda
aws lambda add-permission \
  --function-name CapstoneEcommerceCognitoPreTokenGeneration \
  --statement-id CognitoInvoke \
  --action lambda:InvokeFunction \
  --principal cognito-idp.amazonaws.com \
  --source-arn arn:aws:cognito-idp:us-west-2:${AWS_ACCOUNT_ID}:userpool/us-west-2_5cqnrBvAg
```

#### 2.4 Configure Cognito Trigger
```bash
aws cognito-idp update-user-pool \
  --user-pool-id us-west-2_5cqnrBvAg \
  --lambda-config '{
    "PreTokenGeneration": "arn:aws:lambda:us-west-2:${AWS_ACCOUNT_ID}:function:CapstoneEcommerceCognitoPreTokenGeneration"
  }' \
  --region us-west-2
```

#### 2.5 Create Test Users
```bash
# Tenant A user
aws cognito-idp admin-create-user \
  --user-pool-id us-west-2_5cqnrBvAg \
  --username john.doe \
  --user-attributes Name=email,Value=john.doe@tenanta.com \
  --temporary-password TempPass123! \
  --message-action SUPPRESS

# Tenant B user
aws cognito-idp admin-create-user \
  --user-pool-id us-west-2_5cqnrBvAg \
  --username jane.smith \
  --user-attributes Name=email,Value=jane.smith@tenantb.com \
  --temporary-password TempPass123! \
  --message-action SUPPRESS

# Set permanent passwords
aws cognito-idp admin-set-user-password \
  --user-pool-id us-west-2_5cqnrBvAg \
  --username john.doe \
  --password YourPassword123! \
  --permanent

aws cognito-idp admin-set-user-password \
  --user-pool-id us-west-2_5cqnrBvAg \
  --username jane.smith \
  --password YourPassword123! \
  --permanent
```

---

### Phase 3: Lambda Function Deployment

#### 3.1 Deploy Product Reviews API
```bash
cd lambda
./deploy_lambda.sh
```

**This script**:
1. Creates Lambda function with Python 3.13 runtime
2. Configures environment variables (TABLE_NAME)
3. Creates Function URL with AWS_IAM auth
4. Grants necessary IAM permissions

**Expected Output**:
```
Function URL: https://abc123def456.lambda-url.us-west-2.on.aws/
```

**Save this URL** - you'll need it for AgentCore Gateway configuration.

#### 3.2 Deploy AgentCore Gateway
```bash
cd ../prereqs
./deploy_Agentcore_gateway.sh
```

This creates an AgentCore Gateway that wraps the Lambda Function URL with fine-grained access control.

**Expected Output**:
```
Gateway URL: https://gateway-xyz123.execute-api.us-west-2.amazonaws.com/prod
```

---

### Phase 4: Secrets Manager Configuration

#### 4.1 Create Secret
```bash
aws secretsmanager create-secret \
  --name capstone-ecommerce-agent-config \
  --description "Configuration for ECommerce Agent" \
  --secret-string '{
    "AWS_REGION": "us-west-2",
    "BEDROCK_MODEL_ID": "anthropic.claude-3-5-sonnet-20241022-v2:0",
    "KB_ID": "KB123456789",
    "DB_HOST": "capstone-ecommerce-db.c1abc2def3gh.us-west-2.rds.amazonaws.com",
    "DB_PORT": "5432",
    "DB_NAME": "ecommerce",
    "DB_USER": "admin",
    "DB_PASSWORD": "YourSecurePassword123!",
    "AGENTCORE_MEMORY_ID": "memory-id-placeholder",
    "GATEWAY_URL": "https://gateway-xyz123.execute-api.us-west-2.amazonaws.com/prod",
    "CHART_S3_BUCKET": "capstone-ecommerce-charts-'${AWS_ACCOUNT_ID}'",
    "CLOUDFRONT_DOMAIN": "",
    "VALKEY_ENDPOINT": "master.capstone-sqlagent-valkey-cache.8ot617.usw2.cache.amazonaws.com",
    "VALKEY_PORT": "6379",
    "VALKEY_USE_TLS": "true",
    "VALKEY_PASSWORD": "",
    "CACHE_TTL_SECONDS": "36000",
    "CACHE_SIMILARITY_THRESHOLD": "0.99",
    "CACHE_SIMILARITY_THRESHOLD_MIN": "0.70",
    "BEDROCK_EMBED_MODEL": "amazon.titan-embed-text-v2:0",
    "METRICS_NAMESPACE": "CapstoneECommerceAgent/PerTenant",
    "ENABLE_METRICS_LOGGING": "true",
    "LOG_LEVEL": "WARNING"
  }' \
  --region us-west-2
```

---

### Phase 5: AgentCore Deployment

#### 5.1 Configure AgentCore
```bash
cd ..

uv run agentcore configure \
  -e main.py \
  --vpc \
  --subnets subnet-0d924ff443cebd41d,subnet-0463efcbc95e2d5e1 \
  --security-groups sg-06d560db4091b375c \
  --name capstone_ecomm_agent \
  --region us-west-2 \
  --execution-role arn:aws:iam::${AWS_ACCOUNT_ID}:role/CapstoneECommerce-agentcore-execution-role \
  --disable-memory \
  --authorizer-config '{
    "type": "customJWTAuthorizer",
    "discoveryUrl": "https://cognito-idp.us-west-2.amazonaws.com/us-west-2_5cqnrBvAg/.well-known/openid-configuration",
    "allowedClients": "2nc8c09npb6vru63mgfl7kiqb1"
  }' \
  --request-header-allowlist Authorization,X-Amzn-Bedrock-AgentCore-Runtime-Custom-TenantId,X-Amzn-Bedrock-AgentCore-Runtime-Custom-ActorId \
  --non-interactive
```

**This creates** `.bedrock_agentcore.yaml` configuration file.

#### 5.2 Create AgentCore Memory
```bash
# Via AWS Console:
# 1. Go to Amazon Bedrock → AgentCore → Memory → Create
# 2. Name: capstone_ecommerce_memory
# 3. Type: Conversation Memory
# 4. Event expiry: 30 days

# Save Memory ID and update Secrets Manager
aws secretsmanager update-secret \
  --secret-id capstone-ecommerce-agent-config \
  --secret-string '{
    ...
    "AGENTCORE_MEMORY_ID": "memory-XPTahoCqjM",
    ...
  }'
```

#### 5.3 Add Memory Strategies
Follow instructions in `prereqs/MEMORY_STRATEGY_SETUP.md`:
1. Add OrderFactsExtractor (Semantic Strategy)
2. Add CustomerPreferences (User Preference Strategy)

#### 5.4 Launch AgentCore
```bash
uv run agentcore launch \
  --env AWS_REGION="us-west-2" \
  --env AWS_SECRET_NAME="capstone-ecommerce-agent-config"
```

**This will**:
1. Build Docker container
2. Push to ECR
3. Deploy to AgentCore Runtime
4. Configure VPC networking
5. Set up JWT authentication

**Expected Output**:
```
✓ Container built successfully
✓ Pushed to ECR: 175918693907.dkr.ecr.us-west-2.amazonaws.com/bedrock-agentcore-capstone_ecomm_agent
✓ Agent deployed: capstone_ecomm_agent-iOd15C2EqB
✓ Agent ARN: arn:aws:bedrock-agentcore:us-west-2:175918693907:runtime/capstone_ecomm_agent-iOd15C2EqB
✓ Status: ACTIVE

Agent endpoint: https://runtime.bedrock-agentcore.us-west-2.amazonaws.com/agents/capstone_ecomm_agent-iOd15C2EqB/invoke
```

**Deployment time**: 10-15 minutes

---

### Phase 6: CloudFront Setup (Optional)

#### 6.1 Create CloudFront Distribution
Follow instructions in `prereqs/CLOUDFRONT_SETUP.md`:
1. Create distribution with S3 origin
2. Configure Origin Access Control (OAC)
3. Update S3 bucket policy
4. Copy CloudFront domain

#### 6.2 Update Configuration
```bash
aws secretsmanager update-secret \
  --secret-id capstone-ecommerce-agent-config \
  --secret-string '{
    ...
    "CLOUDFRONT_DOMAIN": "d1234567890abc.cloudfront.net",
    ...
  }'
```

#### 6.3 Restart Agent
```bash
uv run agentcore launch
```

---

### Phase 7: UI Deployment

#### 7.1 Configure UI Environment
```bash
cd ui

# Create .env file
cat > .env << EOF
AWS_REGION=us-west-2
COGNITO_USER_POOL_ID=us-west-2_5cqnrBvAg
COGNITO_CLIENT_ID=2nc8c09npb6vru63mgfl7kiqb1
AGENTCORE_AGENT_ID=capstone_ecomm_agent-iOd15C2EqB
AGENTCORE_AGENT_ARN=arn:aws:bedrock-agentcore:us-west-2:175918693907:runtime/capstone_ecomm_agent-iOd15C2EqB
AGENTCORE_MEMORY_ID=memory-XPTahoCqjM
EOF
```

#### 7.2 Run Locally (Testing)
```bash
cd ..
uv run streamlit run ui/orch_web_app_cognito.py
```

Access at: `http://localhost:8501`

#### 7.3 Deploy to ECS/Fargate (Production)
```bash
# Build Docker image
docker build -t capstone-ecommerce-ui:latest -f ui/Dockerfile .

# Push to ECR
aws ecr create-repository --repository-name capstone-ecommerce-ui
docker tag capstone-ecommerce-ui:latest ${AWS_ACCOUNT_ID}.dkr.ecr.us-west-2.amazonaws.com/capstone-ecommerce-ui:latest
docker push ${AWS_ACCOUNT_ID}.dkr.ecr.us-west-2.amazonaws.com/capstone-ecommerce-ui:latest

# Create ECS cluster, task definition, and service
# (See AWS ECS documentation for detailed steps)
```

---

## Verification & Testing

### Test 1: Database Connectivity
```bash
psql -h capstone-ecommerce-db.c1abc2def3gh.us-west-2.rds.amazonaws.com \
     -U admin -d ecommerce -c "SELECT * FROM tenanta.products LIMIT 5;"
```

### Test 2: DynamoDB Access
```bash
aws dynamodb query \
  --table-name ProductReviews \
  --index-name tenant_id-product_id-index \
  --key-condition-expression "tenant_id = :tid" \
  --expression-attribute-values '{":tid":{"S":"tenanta"}}'
```

### Test 3: Lambda Function
```bash
curl "https://abc123def456.lambda-url.us-west-2.on.aws/?product_id=1"
```

### Test 4: Cognito Authentication
```python
import boto3

cognito = boto3.client('cognito-idp', region_name='us-west-2')
response = cognito.initiate_auth(
    ClientId='2nc8c09npb6vru63mgfl7kiqb1',
    AuthFlow='USER_PASSWORD_AUTH',
    AuthParameters={
        'USERNAME': 'john.doe',
        'PASSWORD': 'YourPassword123!'
    }
)
print(response['AuthenticationResult']['AccessToken'])
```

### Test 5: AgentCore Invocation
```bash
# Get access token from Cognito (from Test 4)
ACCESS_TOKEN="eyJraWQiOiJ..."

# Invoke agent
curl -X POST \
  https://runtime.bedrock-agentcore.us-west-2.amazonaws.com/agents/capstone_ecomm_agent-iOd15C2EqB/invoke \
  -H "Authorization: Bearer ${ACCESS_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{
    "inputText": "Show me products in the Electronics category",
    "sessionId": "test-session-123"
  }'
```

---

## Troubleshooting

### Issue: AgentCore can't connect to RDS
**Solution**: Check security group rules allow inbound on port 5432 from AgentCore SG

### Issue: Lambda returns 403 Forbidden
**Solution**: Verify IAM role has DynamoDB permissions and tenant_id is being passed

### Issue: Knowledge Base returns no results
**Solution**: Ensure data source is synced and metadata files are correctly formatted

### Issue: Valkey connection timeout
**Solution**: Verify ElastiCache is in same VPC and security group allows inbound on port 6379

### Issue: Cognito authentication fails
**Solution**: Check user exists, password is correct, and app client is configured properly

---

## Deployment Checklist

- [ ] VPC and networking configured
- [ ] RDS PostgreSQL created and populated
- [ ] DynamoDB table created with sample data
- [ ] S3 buckets created (KB + Charts)
- [ ] Bedrock Knowledge Base created and synced
- [ ] ElastiCache (Valkey) cluster created
- [ ] Cognito User Pool and App Client created
- [ ] Pre-Token Generation Lambda deployed
- [ ] Product Reviews Lambda deployed
- [ ] AgentCore Gateway configured
- [ ] Secrets Manager secret created
- [ ] AgentCore Memory created with strategies
- [ ] IAM roles and permissions configured
- [ ] AgentCore agent deployed and active
- [ ] CloudFront distribution created (optional)
- [ ] Test users created
- [ ] All tests passing
- [ ] UI deployed and accessible

---

## Deployment Scripts Summary

| Script | Purpose | Location |
|--------|---------|----------|
| `setup_infrastructure.sh` | VPC, subnets, NAT, security groups | `prereqs/` |
| `deploy_dynamo_db.sh` | DynamoDB table creation | `prereqs/` |
| `db_security_group_setup.sh` | RDS security group rules | `prereqs/` |
| `create_schema_tables.py` | RDS schema creation | `prereqs/database/` |
| `populate_schema_data.py` | RDS data population | `prereqs/database/` |
| `deploy_lambda.sh` | Lambda function deployment | `lambda/` |
| `deploy_Agentcore_gateway.sh` | AgentCore Gateway setup | `prereqs/` |
| `deploy_agent.sh` | AgentCore agent deployment | Root |

---

**Document Version**: 2.0  
**Last Updated**: January 2026  
**Author**: Sameer Battoo (sbattoo@amazon.com)
