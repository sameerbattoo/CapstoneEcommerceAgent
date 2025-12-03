#!/bin/bash

# Database Setup Script for AWS RDS PostgreSQL
# This script creates the sample e-commerce database

# Load environment variables
if [ -f ../.env ]; then
    export $(cat ../.env | grep -v '^#' | xargs)
fi

# Database connection parameters
DB_HOST=${DB_HOST:-localhost}
DB_PORT=${DB_PORT:-5432}
DB_NAME=${DB_NAME:-ecommerce}
DB_USER=${DB_USER:-sameer}

echo "=========================================="
echo "Database Setup Script"
echo "=========================================="
echo "Host: $DB_HOST"
echo "Port: $DB_PORT"
echo "Database: $DB_NAME"
echo "User: $DB_USER"
echo "=========================================="

# Check if psql is installed
if ! command -v psql &> /dev/null; then
    echo "Error: psql is not installed. Please install PostgreSQL client."
    exit 1
fi

# Create database if it doesn't exist
echo "Creating database if it doesn't exist..."
PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d postgres -c "CREATE DATABASE $DB_NAME;" 2>/dev/null || echo "Database already exists"

# Run schema creation
echo "Creating database schema..."
PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -f sample_schema.sql

if [ $? -eq 0 ]; then
    echo "✓ Schema created successfully"
else
    echo "✗ Failed to create schema"
    exit 1
fi

# Insert sample data
echo "Inserting sample data..."
PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -f sample_data.sql

if [ $? -eq 0 ]; then
    echo "✓ Sample data inserted successfully"
else
    echo "✗ Failed to insert sample data"
    exit 1
fi

# Verify data
echo ""
echo "Verifying data..."
PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -c "
SELECT 
    'categories' as table_name, COUNT(*) as row_count FROM categories
UNION ALL
SELECT 'customers', COUNT(*) FROM customers
UNION ALL
SELECT 'products', COUNT(*) FROM products
UNION ALL
SELECT 'orders', COUNT(*) FROM orders
UNION ALL
SELECT 'order_items', COUNT(*) FROM order_items;
"

echo ""
echo "=========================================="
echo "Database setup completed successfully!"
echo "=========================================="
echo ""
echo "You can now run the SQL Agent:"
echo "  CLI: python main.py cli"
echo "  Web: python main.py web"
