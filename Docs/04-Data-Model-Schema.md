# ECommerce Agent - Data Model & Schema Specification

## Data Architecture Overview

The solution implements a multi-layered data architecture supporting three data paradigms:
1. **Structured Data**: RDS PostgreSQL (relational)
2. **Semi-Structured Data**: DynamoDB (NoSQL)
3. **Unstructured Data**: S3 + Bedrock Knowledge Base (documents)

Each layer implements multi-tenancy with different isolation strategies.

---

## 1. Structured Data - RDS PostgreSQL

### Multi-Tenancy Strategy
**Schema-per-Tenant**: Each tenant has a dedicated PostgreSQL schema

```sql
-- Tenant A Schema
CREATE SCHEMA tenanta;

-- Tenant B Schema
CREATE SCHEMA tenantb;
```

### Database Configuration
- **Engine**: PostgreSQL 14.x
- **Instance**: db.t3.medium (or higher)
- **Storage**: 100 GB SSD
- **Multi-AZ**: Enabled for production
- **Backup**: Automated daily backups (7-day retention)

### Schema Design

#### 1. Categories Table
```sql
CREATE TABLE {tenant_schema}.categories (
    category_id SERIAL PRIMARY KEY,
    category_name VARCHAR(100) NOT NULL UNIQUE,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Sample Data
INSERT INTO tenanta.categories (category_name, description) VALUES
('Electronics', 'Electronic devices and accessories'),
('Clothing', 'Apparel and fashion items'),
('Sports', 'Sports equipment and gear');
```

#### 2. Products Table
```sql
CREATE TABLE {tenant_schema}.products (
    product_id SERIAL PRIMARY KEY,
    product_name VARCHAR(200) NOT NULL,
    category_id INTEGER REFERENCES {tenant_schema}.categories(category_id),
    price DECIMAL(10, 2) NOT NULL CHECK (price >= 0),
    stock_quantity INTEGER NOT NULL DEFAULT 0 CHECK (stock_quantity >= 0),
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes
CREATE INDEX idx_products_category ON {tenant_schema}.products(category_id);
CREATE INDEX idx_products_name ON {tenant_schema}.products(product_name);

-- Sample Data
INSERT INTO tenanta.products (product_name, category_id, price, stock_quantity, description) VALUES
('Smartphone X', 1, 999.99, 50, 'Latest flagship smartphone'),
('Laptop Pro 15', 1, 1499.99, 30, 'Professional laptop'),
('Cotton T-Shirt', 2, 29.99, 200, 'Comfortable cotton t-shirt');
```

#### 3. Customers Table
```sql
CREATE TABLE {tenant_schema}.customers (
    customer_id SERIAL PRIMARY KEY,
    first_name VARCHAR(50) NOT NULL,
    last_name VARCHAR(50) NOT NULL,
    email VARCHAR(100) NOT NULL UNIQUE,
    phone VARCHAR(20),
    address TEXT,
    city VARCHAR(50),
    state VARCHAR(50),
    zip_code VARCHAR(10),
    country VARCHAR(50) DEFAULT 'USA',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes
CREATE INDEX idx_customers_email ON {tenant_schema}.customers(email);
CREATE INDEX idx_customers_name ON {tenant_schema}.customers(last_name, first_name);

-- Sample Data
INSERT INTO tenanta.customers (first_name, last_name, email, phone, city, state) VALUES
('John', 'Doe', 'john.doe@email.com', '555-0101', 'San Francisco', 'CA'),
('Jane', 'Smith', 'jane.smith@email.com', '555-0102', 'Seattle', 'WA');
```

#### 4. Orders Table
```sql
CREATE TABLE {tenant_schema}.orders (
    order_id SERIAL PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES {tenant_schema}.customers(customer_id),
    order_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    total_amount DECIMAL(10, 2) NOT NULL CHECK (total_amount >= 0),
    status VARCHAR(20) DEFAULT 'pending' CHECK (status IN ('pending', 'processing', 'shipped', 'delivered', 'cancelled')),
    shipping_address TEXT,
    billing_address TEXT,
    payment_method VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes
CREATE INDEX idx_orders_customer ON {tenant_schema}.orders(customer_id);
CREATE INDEX idx_orders_date ON {tenant_schema}.orders(order_date DESC);
CREATE INDEX idx_orders_status ON {tenant_schema}.orders(status);

-- Sample Data
INSERT INTO tenanta.orders (customer_id, order_date, total_amount, status, payment_method) VALUES
(1, '2024-12-01 10:30:00', 1029.98, 'delivered', 'credit_card'),
(2, '2024-12-15 14:20:00', 1499.99, 'shipped', 'paypal');
```

#### 5. Order Items Table
```sql
CREATE TABLE {tenant_schema}.order_items (
    order_item_id SERIAL PRIMARY KEY,
    order_id INTEGER NOT NULL REFERENCES {tenant_schema}.orders(order_id) ON DELETE CASCADE,
    product_id INTEGER NOT NULL REFERENCES {tenant_schema}.products(product_id),
    quantity INTEGER NOT NULL CHECK (quantity > 0),
    unit_price DECIMAL(10, 2) NOT NULL CHECK (unit_price >= 0),
    subtotal DECIMAL(10, 2) GENERATED ALWAYS AS (quantity * unit_price) STORED,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes
CREATE INDEX idx_order_items_order ON {tenant_schema}.order_items(order_id);
CREATE INDEX idx_order_items_product ON {tenant_schema}.order_items(product_id);

-- Sample Data
INSERT INTO tenanta.order_items (order_id, product_id, quantity, unit_price) VALUES
(1, 1, 1, 999.99),
(1, 3, 1, 29.99),
(2, 2, 1, 1499.99);
```

#### 6. Shipments Table
```sql
CREATE TABLE {tenant_schema}.shipments (
    shipment_id SERIAL PRIMARY KEY,
    order_id INTEGER NOT NULL REFERENCES {tenant_schema}.orders(order_id),
    tracking_number VARCHAR(100) UNIQUE,
    carrier VARCHAR(50),
    shipped_date TIMESTAMP,
    estimated_delivery TIMESTAMP,
    actual_delivery TIMESTAMP,
    status VARCHAR(20) DEFAULT 'preparing' CHECK (status IN ('preparing', 'in_transit', 'out_for_delivery', 'delivered', 'failed')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes
CREATE INDEX idx_shipments_order ON {tenant_schema}.shipments(order_id);
CREATE INDEX idx_shipments_tracking ON {tenant_schema}.shipments(tracking_number);
CREATE INDEX idx_shipments_status ON {tenant_schema}.shipments(status);

-- Sample Data
INSERT INTO tenanta.shipments (order_id, tracking_number, carrier, shipped_date, estimated_delivery, status) VALUES
(1, 'TRK123456789', 'FedEx', '2024-12-02 09:00:00', '2024-12-05 17:00:00', 'delivered'),
(2, 'TRK987654321', 'UPS', '2024-12-16 10:00:00', '2024-12-20 17:00:00', 'in_transit');
```

### Common Queries

#### 1. Customer Order History
```sql
SELECT 
    o.order_id,
    o.order_date,
    o.total_amount,
    o.status,
    s.tracking_number,
    s.carrier,
    s.status as shipment_status
FROM tenanta.orders o
LEFT JOIN tenanta.shipments s ON o.order_id = s.order_id
WHERE o.customer_id = 1
ORDER BY o.order_date DESC;
```

#### 2. Product Sales Summary
```sql
SELECT 
    p.product_name,
    c.category_name,
    SUM(oi.quantity) as total_sold,
    SUM(oi.subtotal) as total_revenue
FROM tenanta.order_items oi
JOIN tenanta.products p ON oi.product_id = p.product_id
JOIN tenanta.categories c ON p.category_id = c.category_id
GROUP BY p.product_name, c.category_name
ORDER BY total_revenue DESC;
```

#### 3. Orders by Status
```sql
SELECT 
    status,
    COUNT(*) as order_count,
    SUM(total_amount) as total_value
FROM tenanta.orders
WHERE order_date >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY status;
```

### Connection Pooling Configuration
```python
from psycopg2 import pool

db_pool = pool.ThreadedConnectionPool(
    minconn=2,
    maxconn=10,
    host="rds-endpoint.us-west-2.rds.amazonaws.com",
    port=5432,
    database="ecommerce",
    user="admin",
    password="password"
)

# Usage
conn = db_pool.getconn()
try:
    cursor = conn.cursor()
    cursor.execute(f"SELECT * FROM {tenant_id}.products LIMIT 10")
    results = cursor.fetchall()
finally:
    cursor.close()
    db_pool.putconn(conn)
```

---

## 2. Semi-Structured Data - DynamoDB

### Multi-Tenancy Strategy
**Attribute-based Filtering**: Single table with `tenant_id` attribute

### Table Configuration

#### ProductReviews Table
```json
{
  "TableName": "ProductReviews",
  "BillingMode": "PAY_PER_REQUEST",
  "AttributeDefinitions": [
    {"AttributeName": "product_id", "AttributeType": "N"},
    {"AttributeName": "review_id", "AttributeType": "S"},
    {"AttributeName": "tenant_id", "AttributeType": "S"}
  ],
  "KeySchema": [
    {"AttributeName": "product_id", "KeyType": "HASH"},
    {"AttributeName": "review_id", "KeyType": "RANGE"}
  ],
  "GlobalSecondaryIndexes": [
    {
      "IndexName": "tenant_id-product_id-index",
      "KeySchema": [
        {"AttributeName": "tenant_id", "KeyType": "HASH"},
        {"AttributeName": "product_id", "KeyType": "RANGE"}
      ],
      "Projection": {"ProjectionType": "ALL"}
    }
  ]
}
```

### Item Structure
```json
{
  "product_id": 1,
  "review_id": "rev_20241201_001",
  "tenant_id": "tenanta",
  "customer_id": 1,
  "customer_email": "john.doe@email.com",
  "product_name": "Smartphone X",
  "rating": 5,
  "title": "Excellent phone!",
  "comment": "Best smartphone I've ever owned. Great camera and battery life.",
  "review_date": "2024-12-01T15:30:00Z",
  "helpful_votes": 12,
  "verified_purchase": true,
  "images": ["https://s3.../review_img1.jpg"],
  "created_at": "2024-12-01T15:30:00Z",
  "updated_at": "2024-12-01T15:30:00Z"
}
```

### Access Patterns

#### 1. Query by Tenant and Product
```python
response = table.query(
    IndexName='tenant_id-product_id-index',
    KeyConditionExpression=Key('tenant_id').eq('tenanta') & Key('product_id').eq(1)
)
```

#### 2. Filter by Rating
```python
response = table.query(
    IndexName='tenant_id-product_id-index',
    KeyConditionExpression=Key('tenant_id').eq('tenanta'),
    FilterExpression=Attr('rating').gte(4)
)
```

#### 3. Filter by Date Range
```python
response = table.query(
    IndexName='tenant_id-product_id-index',
    KeyConditionExpression=Key('tenant_id').eq('tenanta'),
    FilterExpression=Attr('review_date').between('2024-01-01', '2024-12-31')
)
```

### Sample Data
```python
items = [
    {
        'product_id': 1,
        'review_id': 'rev_20241201_001',
        'tenant_id': 'tenanta',
        'customer_id': 1,
        'customer_email': 'john.doe@email.com',
        'product_name': 'Smartphone X',
        'rating': 5,
        'title': 'Excellent phone!',
        'comment': 'Best smartphone I\'ve ever owned.',
        'review_date': '2024-12-01T15:30:00Z',
        'helpful_votes': 12,
        'verified_purchase': True
    },
    {
        'product_id': 2,
        'review_id': 'rev_20241215_001',
        'tenant_id': 'tenanta',
        'customer_id': 2,
        'customer_email': 'jane.smith@email.com',
        'product_name': 'Laptop Pro 15',
        'rating': 4,
        'title': 'Great laptop for work',
        'comment': 'Fast and reliable. Battery could be better.',
        'review_date': '2024-12-15T10:20:00Z',
        'helpful_votes': 8,
        'verified_purchase': True
    }
]
```

---

## 3. Unstructured Data - S3 + Bedrock Knowledge Base

### Multi-Tenancy Strategy
**Metadata-based Filtering**: PDF files with associated metadata JSON files

### S3 Bucket Structure
```
s3://capstone-ecommerce-kb/
├── Smartphone X - Product Spec.pdf
├── Smartphone X - Product Spec.pdf.metadata.json
├── Laptop Pro 15 - Product Spec.pdf
├── Laptop Pro 15 - Product Spec.pdf.metadata.json
├── Cotton T-Shirt - Product Spec.pdf
└── Cotton T-Shirt - Product Spec.pdf.metadata.json
```

### Metadata File Format
```json
{
  "metadataAttributes": {
    "tenant_access": "tenanta",
    "product_id": "1",
    "product_name": "Smartphone X",
    "product_category": "Electronics",
    "document_type": "product_specification",
    "version": "1.0",
    "last_updated": "2024-12-01"
  }
}
```

### Knowledge Base Configuration
```json
{
  "knowledgeBaseId": "KB123456",
  "name": "ECommerce Product Specifications",
  "dataSource": {
    "type": "S3",
    "s3Configuration": {
      "bucketArn": "arn:aws:s3:::capstone-ecommerce-kb"
    }
  },
  "embeddingModel": "amazon.titan-embed-text-v2:0",
  "vectorStore": {
    "type": "OPENSEARCH_SERVERLESS"
  }
}
```

### Retrieval with Tenant Filtering
```python
response = bedrock_client.retrieve(
    knowledgeBaseId='KB123456',
    retrievalQuery={'text': 'What are the camera specifications?'},
    retrievalConfiguration={
        'vectorSearchConfiguration': {
            'numberOfResults': 5,
            'filter': {
                'equals': {
                    'key': 'tenant_access',
                    'value': 'tenanta'
                }
            }
        }
    }
)
```

### Sample PDF Content Structure

**Smartphone X - Product Spec.pdf**:
```
Product Specification: Smartphone X

Overview:
The Smartphone X is our flagship device featuring cutting-edge technology.

Technical Specifications:
- Display: 6.7" OLED, 120Hz refresh rate
- Processor: Octa-core 3.2GHz
- RAM: 12GB
- Storage: 256GB / 512GB
- Camera: 
  * Main: 50MP wide-angle
  * Ultra-wide: 12MP
  * Telephoto: 10MP with 3x optical zoom
- Battery: 5000mAh with fast charging
- Operating System: Android 14

Features:
- 5G connectivity
- Wireless charging
- Water resistant (IP68)
- Face unlock and fingerprint sensor

Warranty: 1 year manufacturer warranty
```

---

## 4. Cache Layer - Valkey (ElastiCache)

### Data Structure

#### SQL Cache Entry
```json
{
  "query": "Show me sales by category",
  "tenant_id": "tenanta",
  "sql": "SELECT category_name, SUM(total_amount) FROM tenanta.orders...",
  "results": [
    {"category_name": "Electronics", "total": 15000.00},
    {"category_name": "Clothing", "total": 8500.00}
  ],
  "chart_url": "https://cloudfront.net/charts/tenanta/20241226_123456.png",
  "embedding": [0.123, -0.456, ...],  // 1024-dimensional vector
  "timestamp": "2024-12-26T12:34:56Z",
  "ttl": 36000
}
```

#### Cache Key Format
```
sql_cache:{tenant_id}:{query_hash}
```

Example:
```
sql_cache:tenanta:a3f5b2c1d4e6f7g8h9i0
```

### Cache Operations

#### Store
```python
redis_client.setex(
    f"sql_cache:{tenant_id}:{hash(query)}",
    36000,  # 10 hours TTL
    json.dumps(cache_data)
)
```

#### Retrieve
```python
cache_key = f"sql_cache:{tenant_id}:*"
for key in redis_client.scan_iter(match=cache_key):
    cached_data = json.loads(redis_client.get(key))
    similarity = cosine_similarity(query_embedding, cached_data['embedding'])
    if similarity >= 0.99:
        return cached_data
```

---

## 5. Memory Layer - AgentCore Memory

### Namespace Structure

#### Short-Term Memory (Conversations)
```
/sessions/{session_id}/
├── turn_1
├── turn_2
└── turn_3
```

#### Long-Term Memory (Facts)
```
/users/{actorId}/facts/
├── fact_1: "User ordered iPhone 15 Pro, order #123456"
├── fact_2: "Shipment delivered on June 5, 2025"
└── fact_3: "User reported slow performance issue"
```

#### Long-Term Memory (Preferences)
```
/users/{actorId}/preferences/
├── pref_1: "User prefers ThinkPad laptops"
├── pref_2: "User is interested in electronics"
└── pref_3: "User likes detailed product specifications"
```

### Memory Item Structure
```json
{
  "memoryId": "mem_abc123",
  "namespace": "/users/john_doe-email_com/facts",
  "content": "User ordered Smartphone X on December 1, 2024. Order ID: 1001",
  "metadata": {
    "order_id": "1001",
    "product_name": "Smartphone X",
    "order_date": "2024-12-01",
    "extracted_at": "2024-12-01T15:30:00Z"
  },
  "createdAt": "2024-12-01T15:30:00Z",
  "updatedAt": "2024-12-01T15:30:00Z"
}
```

---

## Data Flow Summary

```
User Query: "Show me my recent orders and reviews for Smartphone X"
    ↓
Orchestrator Agent
    ↓
    ├─→ SQL Agent → RDS (tenanta.orders) → Order data
    │
    └─→ Reviews Tool → Lambda → DynamoDB (tenant_id filter) → Review data
    ↓
Aggregate Results
    ↓
Response: "You have 2 orders. Latest order #1001 delivered on Dec 5.
           Smartphone X has 4.5 star rating with 150 reviews."
```

---

**Document Version**: 2.0  
**Last Updated**: January 2026  
**Author**: Sameer Battoo (sbattoo@amazon.com)
