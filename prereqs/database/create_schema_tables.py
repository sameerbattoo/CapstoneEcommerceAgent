#!/usr/bin/env python3
"""
Generate SQL Script to Create E-commerce Database Tables Under a Specific Schema

This script generates a SQL file that creates all tables, primary keys, foreign keys,
and indexes from the sample_schema.sql file under a specified PostgreSQL schema.

Usage:
    python create_schema_tables.py <schema_name> [output_file]
    
Example:
    python create_schema_tables.py tenant_a
    python create_schema_tables.py tenant_a tenant_a_schema.sql
    python create_schema_tables.py tenant_b > tenant_b_schema.sql
"""

import sys
import os


def generate_sql_header(schema_name):
    """Generate SQL header with comments"""
    return f"""-- ============================================================
-- E-commerce Database Schema Creation Script
-- Schema: {schema_name}
-- Generated: {os.popen('date').read().strip()}
-- ============================================================

-- Create schema if it doesn't exist
CREATE SCHEMA IF NOT EXISTS {schema_name};

-- Set search path to the new schema
SET search_path TO {schema_name}, public;

-- Drop existing tables (in reverse dependency order)
DROP TABLE IF EXISTS {schema_name}.shipments CASCADE;
DROP TABLE IF EXISTS {schema_name}.shipping_methods CASCADE;
DROP TABLE IF EXISTS {schema_name}.customer_addresses CASCADE;
DROP TABLE IF EXISTS {schema_name}.order_items CASCADE;
DROP TABLE IF EXISTS {schema_name}.orders CASCADE;
DROP TABLE IF EXISTS {schema_name}.products CASCADE;
DROP TABLE IF EXISTS {schema_name}.customers CASCADE;
DROP TABLE IF EXISTS {schema_name}.categories CASCADE;

"""


def generate_create_tables_sql(schema_name):
    """Generate SQL for creating all tables"""
    
    sql_statements = []
    
    # Categories table
    sql_statements.append(f"""-- ============================================================
-- Table: {schema_name}.categories
-- ============================================================
CREATE TABLE {schema_name}.categories (
    category_id SERIAL PRIMARY KEY,
    category_name VARCHAR(100) NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
""")
    
    # Customers table
    sql_statements.append(f"""-- ============================================================
-- Table: {schema_name}.customers
-- ============================================================
CREATE TABLE {schema_name}.customers (
    customer_id SERIAL PRIMARY KEY,
    first_name VARCHAR(50) NOT NULL,
    last_name VARCHAR(50) NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    phone VARCHAR(20),
    address TEXT,
    city VARCHAR(50),
    state VARCHAR(50),
    country VARCHAR(50) DEFAULT 'USA',
    postal_code VARCHAR(20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
""")
    
    # Products table
    sql_statements.append(f"""-- ============================================================
-- Table: {schema_name}.products
-- ============================================================
CREATE TABLE {schema_name}.products (
    product_id SERIAL PRIMARY KEY,
    product_name VARCHAR(200) NOT NULL,
    category_id INTEGER REFERENCES {schema_name}.categories(category_id),
    description TEXT,
    price DECIMAL(10, 2) NOT NULL,
    stock_quantity INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
""")
    
    # Orders table
    sql_statements.append(f"""-- ============================================================
-- Table: {schema_name}.orders
-- ============================================================
CREATE TABLE {schema_name}.orders (
    order_id SERIAL PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES {schema_name}.customers(customer_id),
    order_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(20) DEFAULT 'pending',
    total_amount DECIMAL(10, 2),
    shipping_address TEXT,
    shipping_city VARCHAR(50),
    shipping_state VARCHAR(50),
    shipping_country VARCHAR(50),
    shipping_postal_code VARCHAR(20),
    notes TEXT
);
""")
    
    # Order Items table
    sql_statements.append(f"""-- ============================================================
-- Table: {schema_name}.order_items
-- ============================================================
CREATE TABLE {schema_name}.order_items (
    order_item_id SERIAL PRIMARY KEY,
    order_id INTEGER NOT NULL REFERENCES {schema_name}.orders(order_id) ON DELETE CASCADE,
    product_id INTEGER NOT NULL REFERENCES {schema_name}.products(product_id),
    quantity INTEGER NOT NULL,
    unit_price DECIMAL(10, 2) NOT NULL,
    subtotal DECIMAL(10, 2) NOT NULL
);
""")
    
    # Customer Addresses table
    sql_statements.append(f"""-- ============================================================
-- Table: {schema_name}.customer_addresses
-- ============================================================
CREATE TABLE {schema_name}.customer_addresses (
    address_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    customer_id BIGINT NOT NULL,
    address_type VARCHAR(20) NOT NULL,
    full_name VARCHAR(200) NOT NULL,
    phone_number VARCHAR(50),
    address_line1 VARCHAR(255) NOT NULL,
    address_line2 VARCHAR(255),
    city VARCHAR(100) NOT NULL,
    state_province VARCHAR(100),
    postal_code VARCHAR(20),
    country_code CHAR(2) NOT NULL,
    is_default BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_customer_addresses_customer
        FOREIGN KEY (customer_id) REFERENCES {schema_name}.customers(customer_id)
);
""")
    
    # Shipping Methods table
    sql_statements.append(f"""-- ============================================================
-- Table: {schema_name}.shipping_methods
-- ============================================================
CREATE TABLE {schema_name}.shipping_methods (
    shipping_method_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    code VARCHAR(50) NOT NULL UNIQUE,
    display_name VARCHAR(100) NOT NULL,
    base_cost NUMERIC(10,2),
    is_active BOOLEAN NOT NULL DEFAULT TRUE
);
""")
    
    # Shipments table
    sql_statements.append(f"""-- ============================================================
-- Table: {schema_name}.shipments
-- ============================================================
CREATE TABLE {schema_name}.shipments (
    shipment_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    order_id BIGINT NOT NULL,
    shipping_method_id BIGINT NOT NULL,
    shipping_address_id BIGINT NOT NULL,
    carrier_name VARCHAR(100),
    tracking_number VARCHAR(100),
    shipment_status VARCHAR(50) NOT NULL DEFAULT 'created',
    shipped_at TIMESTAMP,
    delivered_at TIMESTAMP,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_shipments_order
        FOREIGN KEY (order_id) REFERENCES {schema_name}.orders(order_id),
    CONSTRAINT fk_shipments_method
        FOREIGN KEY (shipping_method_id) REFERENCES {schema_name}.shipping_methods(shipping_method_id),
    CONSTRAINT fk_shipments_address
        FOREIGN KEY (shipping_address_id) REFERENCES {schema_name}.customer_addresses(address_id)
);
""")
    
    return "\n".join(sql_statements)


def generate_create_indexes_sql(schema_name):
    """Generate SQL for creating all indexes"""
    
    sql_statements = [f"""-- ============================================================
-- Indexes for Schema: {schema_name}
-- ============================================================

-- Orders indexes
CREATE INDEX idx_orders_customer_id ON {schema_name}.orders(customer_id);
CREATE INDEX idx_orders_order_date ON {schema_name}.orders(order_date);
CREATE INDEX idx_orders_status ON {schema_name}.orders(status);
CREATE INDEX idx_orders_customer_date ON {schema_name}.orders(customer_id, order_date DESC);

-- Order Items indexes
CREATE INDEX idx_order_items_order_id ON {schema_name}.order_items(order_id);
CREATE INDEX idx_order_items_product_id ON {schema_name}.order_items(product_id);

-- Products indexes
CREATE INDEX idx_products_category_id ON {schema_name}.products(category_id);
CREATE INDEX idx_products_name ON {schema_name}.products(product_name);
CREATE INDEX idx_products_price ON {schema_name}.products(price);

-- Customers indexes
CREATE INDEX idx_customers_email ON {schema_name}.customers(email);
CREATE INDEX idx_customers_name ON {schema_name}.customers(last_name, first_name);
CREATE INDEX idx_customers_city_state ON {schema_name}.customers(city, state);

-- Customer Addresses indexes
CREATE INDEX idx_customer_addresses_customer_id ON {schema_name}.customer_addresses(customer_id);
CREATE INDEX idx_customer_addresses_type ON {schema_name}.customer_addresses(address_type);
CREATE INDEX idx_customer_addresses_default ON {schema_name}.customer_addresses(customer_id, is_default) WHERE is_default = TRUE;
CREATE INDEX idx_customer_addresses_postal ON {schema_name}.customer_addresses(postal_code);
CREATE INDEX idx_customer_addresses_country ON {schema_name}.customer_addresses(country_code);

-- Shipments indexes
CREATE INDEX idx_shipments_order_id ON {schema_name}.shipments(order_id);
CREATE INDEX idx_shipments_tracking ON {schema_name}.shipments(tracking_number);
CREATE INDEX idx_shipments_status ON {schema_name}.shipments(shipment_status);
CREATE INDEX idx_shipments_shipped_at ON {schema_name}.shipments(shipped_at);
CREATE INDEX idx_shipments_delivered_at ON {schema_name}.shipments(delivered_at);
CREATE INDEX idx_shipments_method ON {schema_name}.shipments(shipping_method_id);
CREATE INDEX idx_shipments_address ON {schema_name}.shipments(shipping_address_id);
"""]
    
    return "\n".join(sql_statements)


def generate_sql_footer(schema_name):
    """Generate SQL footer with summary"""
    return f"""
-- ============================================================
-- Schema Creation Complete: {schema_name}
-- ============================================================
-- Tables created: 8
--   1. categories
--   2. customers
--   3. products
--   4. orders
--   5. order_items
--   6. customer_addresses
--   7. shipping_methods
--   8. shipments
--
-- Indexes created: 24
-- Foreign keys: 8
-- ============================================================
-- To execute this script:
--   psql -U postgres -d ecommerce -f {schema_name}_schema.sql
-- ============================================================
"""


def main():
    """Main execution function"""
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python create_schema_tables.py <schema_name> [output_file]")
        print("\nExample:")
        print("  python create_schema_tables.py tenant_a")
        print("  python create_schema_tables.py tenant_a tenant_a_schema.sql")
        print("  python create_schema_tables.py tenant_b > tenant_b_schema.sql")
        sys.exit(1)
    
    schema_name = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) == 3 else None
    
    # Validate schema name (basic validation)
    if not schema_name.replace('_', '').isalnum():
        print(f"Error: Invalid schema name '{schema_name}'", file=sys.stderr)
        print("Schema name must contain only letters, numbers, and underscores", file=sys.stderr)
        sys.exit(1)
    
    # Generate SQL
    sql_script = []
    sql_script.append(generate_sql_header(schema_name))
    sql_script.append(generate_create_tables_sql(schema_name))
    sql_script.append(generate_create_indexes_sql(schema_name))
    sql_script.append(generate_sql_footer(schema_name))
    
    full_sql = "\n".join(sql_script)
    
    # Output SQL
    if output_file:
        with open(output_file, 'w') as f:
            f.write(full_sql)
        print(f"✅ SQL script generated: {output_file}", file=sys.stderr)
        print(f"\nTo execute:", file=sys.stderr)
        print(f"  psql -U postgres -d ecommerce -f {output_file}", file=sys.stderr)
    else:
        print(full_sql)


if __name__ == "__main__":
    main()

