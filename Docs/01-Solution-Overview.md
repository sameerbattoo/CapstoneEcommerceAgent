# ECommerce Agent Solution - Overview

## Executive Summary

The ECommerce Agent is an intelligent, multi-tenant e-commerce assistant powered by AWS Bedrock AgentCore. It provides natural language interaction capabilities for customers to query product information, reviews, orders, and specifications across structured, semi-structured, and unstructured data sources.

## Version

**Current Version:** 2.0

## Key Capabilities

### Core Features
- **Natural Language Query Processing**: Customers can ask questions in plain English about products, orders, reviews, and specifications
- **Multi-Agent Orchestration**: Specialized agents handle different data types and tasks
- **Multi-Tenancy Support**: ISV-ready architecture with tenant isolation at all data layers
- **Secure Authentication**: OAuth2-based access control with AWS Cognito
- **Conversation Memory**: Short-term and long-term memory for personalized interactions
- **Visual Analytics**: Automatic chart generation from query results
- **Performance Optimization**: Semantic caching, prompt caching, and sliding window conversation management

### Data Access Patterns
1. **Structured Data (SQL)**: Product catalog, orders, customers via RDS PostgreSQL
2. **Semi-Structured Data (NoSQL)**: Product reviews via DynamoDB
3. **Unstructured Data (Documents)**: Product specifications via Bedrock Knowledge Base (S3)

## Architecture Highlights

### Multi-Agent System
- **Orchestrator Agent**: Routes requests and coordinates sub-agents
- **SQL Agent**: Handles structured data queries with semantic caching
- **Knowledge Base Agent**: Retrieves information from product specification PDFs
- **Chart Agent**: Generates visualizations using Code Interpreter

### AWS Services Used
- **AWS Bedrock AgentCore**: Agent runtime and orchestration
- **AWS Bedrock Models**: Claude 4.5 Sonnet/Haiku for LLM capabilities
- **Amazon RDS PostgreSQL**: Structured product and order data
- **Amazon DynamoDB**: Product reviews storage
- **Amazon S3**: Product specification documents and generated charts
- **AWS Cognito**: User authentication and authorization
- **Amazon ElastiCache (Valkey)**: Semantic caching for SQL queries
- **AWS Lambda**: Product reviews API with fine-grained access control
- **Amazon CloudFront**: CDN for chart delivery
- **AWS CloudWatch**: Metrics, logging, and observability

## Key Innovations (V2)

### 1. Visual Analytics
- Automatic chart generation from SQL query results
- Code Interpreter integration for secure Python execution
- CloudFront-backed chart delivery

### 2. Advanced Memory Management
- Strands Memory Hooks integration with AgentCore Short-Term Memory
- Long-term memory strategies for user preferences and facts
- Semantic search across memory data

### 3. Multi-Tenancy Architecture
- **Unstructured Data**: Metadata-based filtering in S3/Knowledge Base
- **Structured Data**: Schema-per-tenant in RDS PostgreSQL
- **Semi-Structured Data**: Tenant ID attribute filtering in DynamoDB
- **Authentication**: Tenant ID embedded in JWT access tokens

### 4. Performance Optimizations
- **Semantic Caching**: Valkey-based caching for similar SQL queries
- **Prompt Caching**: System prompt and tool caching for reduced input tokens
- **Sliding Window**: Last 10 conversations for context management
- **Schema Caching**: Background pre-loading of database schemas

### 5. Cost & Observability
- Per-tenant metrics tracking (latency, tokens, cost)
- Custom CloudWatch metrics for multi-tenant monitoring
- AWS Bedrock Model Evaluation integration
- RAG evaluation using RAGAS metrics

## Target Use Cases

1. **Customer Self-Service**: Customers query product information, track orders, read reviews
2. **Product Discovery**: Natural language search across specifications and features
3. **Order Management**: Check order status, shipment tracking, order history
4. **Review Analysis**: Aggregate and analyze product reviews by various criteria
5. **Data Visualization**: Automatic chart generation for sales trends, ratings, etc.

## Deployment Models

- **Cloud Deployment**: AWS Bedrock AgentCore Runtime with VPC integration
- **Container Deployment**: Docker-based deployment to ECS/Fargate
- **Local Development**: Streamlit UI for testing without authentication

## Security & Compliance

- OAuth2/JWT-based authentication
- Fine-grained access control at Lambda Gateway level
- VPC isolation for database access
- Tenant isolation at data layer
- PII handling through JWT claims
- Secrets management via AWS Secrets Manager

## Success Metrics

- Query response latency < 5 seconds
- Token cost reduction through caching (30-50% savings)
- Multi-tenant isolation validation
- Model evaluation scores (accuracy, relevance)
- RAG evaluation metrics (faithfulness, answer relevance)

## Document Structure

This documentation suite includes:
1. **Solution Overview** (this document)
2. **Architecture Specification**
3. **Multi-Agent Design**
4. **Data Model & Schema**
5. **Authentication & Authorization**
6. **Deployment Guide**
7. **API Reference**
8. **Performance & Optimization**
9. **Monitoring & Observability**
10. **Testing & Evaluation**

---

**Author**: Sameer Battoo (sbattoo@amazon.com)  
**Last Updated**: January 2026  
**Version**: 2.0
