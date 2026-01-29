# ECommerce Agent - Complete Documentation Suite

## Overview

This documentation suite provides comprehensive technical specifications for the ECommerce Agent solution - a multi-tenant, intelligent e-commerce assistant powered by AWS Bedrock AgentCore.

**Version**: 2.0  
**Last Updated**: January 2026  
**Author**: Sameer Battoo (sbattoo@amazon.com)

---

## Architecture Diagram

![V2 Architecture](../generated-diagrams/Capstone-ECommerce-Agent%20V2.png)

---

## Documentation Structure

### 1. [Solution Overview](01-Solution-Overview.md)
**Purpose**: Executive summary and high-level architecture

**Contents**:
- Executive summary
- Key capabilities and features
- Architecture highlights
- V2 innovations (semantic caching, chart generation, memory strategies)
- Target use cases
- Success metrics

**Audience**: Technical leaders, architects, stakeholders

---

### 2. [Architecture Specification](02-Architecture-Specification.md)
**Purpose**: Detailed system architecture and component design

**Contents**:
- System architecture layers (presentation, orchestration, sub-agents, data, integration)
- AWS services configuration
- VPC and networking setup
- Multi-tenancy architecture
- Data flow diagrams
- Deployment architecture
- Scalability considerations

**Audience**: Solution architects, DevOps engineers

---

### 3. [Multi-Agent Design](03-Multi-Agent-Design.md)
**Purpose**: Detailed design of the multi-agent system

**Contents**:
- Agent hierarchy and responsibilities
- Orchestrator Agent design
- SQL Agent with semantic caching
- Knowledge Base Agent
- Chart Agent with Code Interpreter
- Product Reviews Tool (MCP)
- Agent communication patterns
- Performance metrics by agent

**Audience**: AI/ML engineers, developers

---

### 4. [Data Model & Schema](04-Data-Model-Schema.md)
**Purpose**: Complete data architecture and schema definitions

**Contents**:
- RDS PostgreSQL schema (structured data)
- DynamoDB schema (semi-structured data)
- S3 + Knowledge Base structure (unstructured data)
- Valkey cache structure (semantic caching)
- AgentCore Memory namespaces
- Multi-tenancy strategies per data layer
- Sample data and queries

**Audience**: Database administrators, data engineers

---

### 5. [Authentication & Authorization](05-Authentication-Authorization.md)
**Purpose**: Security architecture and access control

**Contents**:
- OAuth2/JWT authentication flow
- AWS Cognito configuration
- Pre-Token Generation Lambda
- JWT token structure and claims
- AgentCore JWT validation
- Fine-grained access control
- IAM roles and permissions
- Multi-tenancy security

**Audience**: Security engineers, IAM administrators

---

### 6. [Deployment Guide](06-Deployment-Guide.md)
**Purpose**: Step-by-step deployment instructions

**Contents**:
- Prerequisites and requirements
- Infrastructure setup (VPC, RDS, DynamoDB, S3, ElastiCache)
- Authentication setup (Cognito, Lambda triggers)
- Lambda function deployment
- Secrets Manager configuration
- AgentCore deployment
- CloudFront setup (optional)
- UI deployment
- Verification and testing
- Troubleshooting guide

**Audience**: DevOps engineers, deployment teams

---

### 7. [API Reference](07-API-Reference.md)
**Purpose**: Complete API documentation for all components

**Contents**:
- AgentCore Runtime API
- Product Reviews API (Lambda)
- Orchestrator Agent tools
- AgentCore Memory API
- Cognito Authentication API
- Python SDK usage examples
- Rate limits and error codes

**Audience**: Developers, integration engineers

---

### 8. [Performance & Optimization](08-Performance-Optimization.md)
**Purpose**: Performance optimization strategies and best practices

**Contents**:
- Performance targets and metrics
- Prompt caching implementation
- Semantic caching (SQL Agent)
- Schema caching
- Sliding window conversation manager
- Connection pooling
- Model selection strategy
- CloudFront CDN for charts
- Monitoring and profiling

**Audience**: Performance engineers, architects

---

### 9. [Monitoring & Observability](09-Monitoring-Observability.md)
**Purpose**: Comprehensive monitoring and observability setup

**Contents**:
- CloudWatch Metrics (latency, tokens, cost, cache)
- CloudWatch Logs and Log Insights
- CloudWatch Dashboards
- Alarms and alerting
- Distributed tracing with X-Ray
- Per-tenant metrics
- Monitoring best practices

**Audience**: SRE, operations teams

---

### 10. [Testing & Evaluation](10-Testing-Evaluation.md)
**Purpose**: Testing strategy and evaluation methodologies

**Contents**:
- Unit testing framework
- Integration testing
- End-to-end testing
- AWS Bedrock Model Evaluation
- RAG evaluation with RAGAS metrics
- Performance and load testing
- Testing checklists

**Audience**: QA engineers, test automation teams

---

## Quick Start Guide

### For Developers
1. Start with [Solution Overview](01-Solution-Overview.md) for context
2. Review [Multi-Agent Design](03-Multi-Agent-Design.md) for implementation details
3. Reference [API Reference](07-API-Reference.md) for integration
4. Follow [Testing & Evaluation](10-Testing-Evaluation.md) for quality assurance

### For Architects
1. Read [Solution Overview](01-Solution-Overview.md) for high-level design
2. Study [Architecture Specification](02-Architecture-Specification.md) for detailed architecture
3. Review [Data Model & Schema](04-Data-Model-Schema.md) for data design
4. Check [Performance & Optimization](08-Performance-Optimization.md) for scalability

### For DevOps/SRE
1. Follow [Deployment Guide](06-Deployment-Guide.md) for setup
2. Configure [Authentication & Authorization](05-Authentication-Authorization.md)
3. Set up [Monitoring & Observability](09-Monitoring-Observability.md)
4. Review [Performance & Optimization](08-Performance-Optimization.md) for tuning

### For Security Teams
1. Review [Authentication & Authorization](05-Authentication-Authorization.md)
2. Check [Architecture Specification](02-Architecture-Specification.md) for network security
3. Verify [Data Model & Schema](04-Data-Model-Schema.md) for data isolation
4. Audit IAM permissions in [Authentication & Authorization](05-Authentication-Authorization.md)

---

## Key Features Documented

### V2 Enhancements
✅ **Chart Generation**: Automatic visualization using AWS Code Interpreter  
✅ **Memory Strategies**: Short-term and long-term memory with semantic search  
✅ **Model Evaluation**: AWS Bedrock Model Evaluation integration  
✅ **RAG Evaluation**: RAGAS metrics for retrieval quality  
✅ **Multi-Tenancy**: Complete isolation at all data layers  
✅ **OAuth2 Authentication**: Fine-grained access control with JWT  
✅ **Semantic Caching**: Valkey-based caching for SQL queries  
✅ **Prompt Caching**: 90% cost savings on repeated prompts  
✅ **Per-Tenant Metrics**: CloudWatch metrics for cost tracking  
✅ **Sliding Window**: Optimized conversation context management  

### Core Capabilities
✅ **Multi-Agent Orchestration**: Specialized agents for different tasks  
✅ **Structured Data**: RDS PostgreSQL with schema-per-tenant  
✅ **Semi-Structured Data**: DynamoDB with attribute filtering  
✅ **Unstructured Data**: S3 + Bedrock Knowledge Base with metadata filtering  
✅ **Real-Time Streaming**: Server-sent events for responsive UI  
✅ **Cost Optimization**: Model selection, caching, connection pooling  
✅ **Observability**: Comprehensive metrics, logs, and dashboards  

---

## Technology Stack

### AWS Services
- **AWS Bedrock AgentCore**: Agent runtime and orchestration
- **AWS Bedrock Models**: Claude 4.5 Sonnet/Haiku
- **Amazon RDS PostgreSQL**: Structured data storage
- **Amazon DynamoDB**: Semi-structured data storage
- **Amazon S3**: Unstructured data storage
- **AWS Bedrock Knowledge Base**: Document retrieval
- **Amazon ElastiCache (Valkey)**: Semantic caching
- **AWS Cognito**: User authentication
- **AWS Lambda**: Serverless functions
- **Amazon CloudFront**: CDN for charts
- **AWS CloudWatch**: Monitoring and observability
- **AWS Secrets Manager**: Configuration management

### Frameworks & Libraries
- **Strands Agent SDK**: Agent framework
- **Streamlit**: Web UI framework
- **psycopg2**: PostgreSQL driver
- **boto3**: AWS SDK for Python
- **redis-py**: Redis/Valkey client
- **PyJWT**: JWT token handling
- **pytest**: Testing framework
- **RAGAS**: RAG evaluation

---

## Performance Benchmarks

| Metric | Target | Achieved |
|--------|--------|----------|
| End-to-End Latency | < 5s | 3-5s ✓ |
| SQL Query Latency | < 2s | 1.5-2.5s ✓ |
| KB Query Latency | < 2s | 1.5-2s ✓ |
| Chart Generation | < 4s | 3-4s ✓ |
| Token Cost per Query | < $0.01 | $0.005-$0.008 ✓ |
| Cache Hit Rate | > 40% | 45-55% ✓ |
| Model Accuracy | > 90% | 92% ✓ |
| RAGAS Faithfulness | > 0.90 | 0.92 ✓ |

---

## Support & Contribution

### Getting Help
- Review relevant documentation section
- Check troubleshooting guides in [Deployment Guide](06-Deployment-Guide.md)
- Review CloudWatch logs and metrics
- Contact: Sameer Battoo (sbattoo@amazon.com)

### Reporting Issues
- Provide detailed error messages
- Include relevant logs from CloudWatch
- Specify environment (dev/prod)
- Include steps to reproduce

### Contributing
- Follow existing documentation structure
- Include code examples where applicable
- Update version history
- Test all code samples

---

## Version History

### Version 2.0 (January 2026)
- Added chart generation with Code Interpreter
- Implemented memory strategies (short-term and long-term)
- Integrated AWS Bedrock Model Evaluation
- Added RAG evaluation with RAGAS metrics
- Implemented multi-tenancy across all data layers
- Added OAuth2/JWT authentication with fine-grained access control
- Implemented semantic caching with Valkey
- Added prompt caching for cost optimization
- Implemented per-tenant metrics tracking
- Added sliding window conversation management
- Optimized prompts using AWS Bedrock prompt optimization

### Version 1.0 (December 2024)
- Initial release with basic agent functionality
- SQL Agent for structured data
- Knowledge Base Agent for unstructured data
- Product Reviews API
- Basic authentication with Cognito
- CloudWatch monitoring

---

## License

[Your License Here]

---

## Contact

**Author**: Sameer Battoo  
**Email**: sbattoo@amazon.com  
**Role**: SUP Sr SA, US West EGD, SJ  
**Organization**: AWS

---

## Additional Resources

- [AWS Bedrock Documentation](https://docs.aws.amazon.com/bedrock/)
- [AWS AgentCore Documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/agents.html)
- [Strands Agent SDK](https://github.com/awslabs/strands)
- [RAGAS Documentation](https://docs.ragas.io/)
- [AWS Well-Architected Framework](https://aws.amazon.com/architecture/well-architected/)

---

**Last Updated**: January 29, 2026
