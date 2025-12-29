-- ============================================================
-- E-commerce Database Schema Creation Script
-- Schema: tenantA
-- Generated: Mon Dec 22 23:35:21 PST 2025
-- ============================================================

-- Create schema if it doesn't exist
CREATE SCHEMA IF NOT EXISTS tenantA;

-- Set search path to the new schema
SET search_path TO tenantA, public;

-- Drop existing tables (in reverse dependency order)
DROP TABLE IF EXISTS tenantA.shipments CASCADE;
DROP TABLE IF EXISTS tenantA.shipping_methods CASCADE;
DROP TABLE IF EXISTS tenantA.customer_addresses CASCADE;
DROP TABLE IF EXISTS tenantA.order_items CASCADE;
DROP TABLE IF EXISTS tenantA.orders CASCADE;
DROP TABLE IF EXISTS tenantA.products CASCADE;
DROP TABLE IF EXISTS tenantA.customers CASCADE;
DROP TABLE IF EXISTS tenantA.categories CASCADE;


-- ============================================================
-- Table: tenantA.categories
-- ============================================================
CREATE TABLE tenantA.categories (
    category_id SERIAL PRIMARY KEY,
    category_name VARCHAR(100) NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================
-- Table: tenantA.customers
-- ============================================================
CREATE TABLE tenantA.customers (
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

-- ============================================================
-- Table: tenantA.products
-- ============================================================
CREATE TABLE tenantA.products (
    product_id SERIAL PRIMARY KEY,
    product_name VARCHAR(200) NOT NULL,
    category_id INTEGER REFERENCES tenantA.categories(category_id),
    description TEXT,
    price DECIMAL(10, 2) NOT NULL,
    stock_quantity INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================
-- Table: tenantA.orders
-- ============================================================
CREATE TABLE tenantA.orders (
    order_id SERIAL PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES tenantA.customers(customer_id),
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

-- ============================================================
-- Table: tenantA.order_items
-- ============================================================
CREATE TABLE tenantA.order_items (
    order_item_id SERIAL PRIMARY KEY,
    order_id INTEGER NOT NULL REFERENCES tenantA.orders(order_id) ON DELETE CASCADE,
    product_id INTEGER NOT NULL REFERENCES tenantA.products(product_id),
    quantity INTEGER NOT NULL,
    unit_price DECIMAL(10, 2) NOT NULL,
    subtotal DECIMAL(10, 2) NOT NULL
);

-- ============================================================
-- Table: tenantA.customer_addresses
-- ============================================================
CREATE TABLE tenantA.customer_addresses (
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
        FOREIGN KEY (customer_id) REFERENCES tenantA.customers(customer_id)
);

-- ============================================================
-- Table: tenantA.shipping_methods
-- ============================================================
CREATE TABLE tenantA.shipping_methods (
    shipping_method_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    code VARCHAR(50) NOT NULL UNIQUE,
    display_name VARCHAR(100) NOT NULL,
    base_cost NUMERIC(10,2),
    is_active BOOLEAN NOT NULL DEFAULT TRUE
);

-- ============================================================
-- Table: tenantA.shipments
-- ============================================================
CREATE TABLE tenantA.shipments (
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
        FOREIGN KEY (order_id) REFERENCES tenantA.orders(order_id),
    CONSTRAINT fk_shipments_method
        FOREIGN KEY (shipping_method_id) REFERENCES tenantA.shipping_methods(shipping_method_id),
    CONSTRAINT fk_shipments_address
        FOREIGN KEY (shipping_address_id) REFERENCES tenantA.customer_addresses(address_id)
);

-- ============================================================
-- Indexes for Schema: tenantA
-- ============================================================

-- Orders indexes
CREATE INDEX idx_orders_customer_id ON tenantA.orders(customer_id);
CREATE INDEX idx_orders_order_date ON tenantA.orders(order_date);
CREATE INDEX idx_orders_status ON tenantA.orders(status);
CREATE INDEX idx_orders_customer_date ON tenantA.orders(customer_id, order_date DESC);
CREATE INDEX idx_orders_status_date ON tenantA.orders(status, order_date DESC);
CREATE INDEX idx_orders_date_status ON tenantA.orders(order_date DESC, status);
CREATE INDEX idx_orders_customer_status ON tenantA.orders(customer_id, status, order_date DESC);

-- Order Items indexes
CREATE INDEX idx_order_items_order_id ON tenantA.order_items(order_id);
CREATE INDEX idx_order_items_product_id ON tenantA.order_items(product_id);
CREATE INDEX idx_order_items_product_qty ON tenantA.order_items(product_id, quantity);
CREATE INDEX idx_order_items_order_product ON tenantA.order_items(order_id, product_id, quantity);

-- Products indexes
CREATE INDEX idx_products_category_id ON tenantA.products(category_id);
CREATE INDEX idx_products_name ON tenantA.products(product_name);
CREATE INDEX idx_products_price ON tenantA.products(price);
CREATE INDEX idx_products_category_price ON tenantA.products(category_id, price);
CREATE INDEX idx_products_stock ON tenantA.products(stock_quantity) WHERE stock_quantity < 50;

-- Customers indexes
CREATE INDEX idx_customers_email ON tenantA.customers(email);
CREATE INDEX idx_customers_name ON tenantA.customers(last_name, first_name);
CREATE INDEX idx_customers_city_state ON tenantA.customers(city, state);
CREATE INDEX idx_customers_state ON tenantA.customers(state);

-- Customer Addresses indexes
CREATE INDEX idx_customer_addresses_customer_id ON tenantA.customer_addresses(customer_id);
CREATE INDEX idx_customer_addresses_type ON tenantA.customer_addresses(address_type);
CREATE INDEX idx_customer_addresses_default ON tenantA.customer_addresses(customer_id, is_default) WHERE is_default = TRUE;
CREATE INDEX idx_customer_addresses_postal ON tenantA.customer_addresses(postal_code);
CREATE INDEX idx_customer_addresses_country ON tenantA.customer_addresses(country_code);

-- Shipments indexes
CREATE INDEX idx_shipments_order_id ON tenantA.shipments(order_id);
CREATE INDEX idx_shipments_tracking ON tenantA.shipments(tracking_number);
CREATE INDEX idx_shipments_status ON tenantA.shipments(shipment_status);
CREATE INDEX idx_shipments_shipped_at ON tenantA.shipments(shipped_at);
CREATE INDEX idx_shipments_delivered_at ON tenantA.shipments(delivered_at);
CREATE INDEX idx_shipments_method ON tenantA.shipments(shipping_method_id);
CREATE INDEX idx_shipments_address ON tenantA.shipments(shipping_address_id);
CREATE INDEX idx_shipments_status_shipped ON tenantA.shipments(shipment_status, shipped_at DESC);
CREATE INDEX idx_shipments_order_status ON tenantA.shipments(order_id, shipment_status);


-- ============================================================
-- Schema Creation Complete: tenantA
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
-- Indexes created: 35
-- Foreign keys: 8
-- ============================================================
-- To execute this script:
--   psql -U postgres -d ecommerce -f tenantA_schema.sql
-- ============================================================
