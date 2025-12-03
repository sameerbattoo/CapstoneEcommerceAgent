-- Sample E-commerce Database Schema
-- This creates a simple e-commerce database with customers, products, orders, and order items

-- Drop tables if they exist (for clean setup)
-- Drop in reverse order of dependencies
DROP TABLE IF EXISTS shipment_items CASCADE;
DROP TABLE IF EXISTS shipments CASCADE;
DROP TABLE IF EXISTS shipping_methods CASCADE;
DROP TABLE IF EXISTS customer_addresses CASCADE;
DROP TABLE IF EXISTS order_items CASCADE;
DROP TABLE IF EXISTS orders CASCADE;
DROP TABLE IF EXISTS products CASCADE;
DROP TABLE IF EXISTS customers CASCADE;
DROP TABLE IF EXISTS categories CASCADE;

-- Categories table
CREATE TABLE categories (
    category_id SERIAL PRIMARY KEY,
    category_name VARCHAR(100) NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Customers table
CREATE TABLE customers (
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

-- Products table
CREATE TABLE products (
    product_id SERIAL PRIMARY KEY,
    product_name VARCHAR(200) NOT NULL,
    category_id INTEGER REFERENCES categories(category_id),
    description TEXT,
    price DECIMAL(10, 2) NOT NULL,
    stock_quantity INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Orders table
CREATE TABLE orders (
    order_id SERIAL PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(customer_id),
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

-- Order Items table
CREATE TABLE order_items (
    order_item_id SERIAL PRIMARY KEY,
    order_id INTEGER NOT NULL REFERENCES orders(order_id) ON DELETE CASCADE,
    product_id INTEGER NOT NULL REFERENCES products(product_id),
    quantity INTEGER NOT NULL,
    unit_price DECIMAL(10, 2) NOT NULL,
    subtotal DECIMAL(10, 2) NOT NULL
);

-- Create indexes for better query performance

-- Existing table indexes
CREATE INDEX idx_orders_customer_id ON orders(customer_id);
CREATE INDEX idx_orders_order_date ON orders(order_date);
CREATE INDEX idx_orders_status ON orders(status);
CREATE INDEX idx_orders_customer_date ON orders(customer_id, order_date DESC);

CREATE INDEX idx_order_items_order_id ON order_items(order_id);
CREATE INDEX idx_order_items_product_id ON order_items(product_id);

CREATE INDEX idx_products_category_id ON products(category_id);
CREATE INDEX idx_products_name ON products(product_name);
CREATE INDEX idx_products_price ON products(price);

CREATE INDEX idx_customers_email ON customers(email);
CREATE INDEX idx_customers_name ON customers(last_name, first_name);
CREATE INDEX idx_customers_city_state ON customers(city, state);

CREATE TABLE customer_addresses (
    address_id        BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    customer_id       BIGINT NOT NULL,
    address_type      VARCHAR(20) NOT NULL, -- 'billing','shipping','other'
    full_name         VARCHAR(200) NOT NULL,
    phone_number      VARCHAR(50),
    address_line1     VARCHAR(255) NOT NULL,
    address_line2     VARCHAR(255),
    city              VARCHAR(100) NOT NULL,
    state_province    VARCHAR(100),
    postal_code       VARCHAR(20),
    country_code      CHAR(2) NOT NULL,
    is_default        BOOLEAN NOT NULL DEFAULT FALSE,
    created_at        TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at        TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT fk_customer_addresses_customer
        FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

-- Indexes for customer_addresses
CREATE INDEX idx_customer_addresses_customer_id ON customer_addresses(customer_id);
CREATE INDEX idx_customer_addresses_type ON customer_addresses(address_type);
CREATE INDEX idx_customer_addresses_default ON customer_addresses(customer_id, is_default) WHERE is_default = TRUE;
CREATE INDEX idx_customer_addresses_postal ON customer_addresses(postal_code);
CREATE INDEX idx_customer_addresses_country ON customer_addresses(country_code);

CREATE TABLE shipping_methods (
    shipping_method_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    code               VARCHAR(50) NOT NULL UNIQUE, -- 'standard','express', etc.
    display_name       VARCHAR(100) NOT NULL,
    base_cost          NUMERIC(10,2),
    is_active          BOOLEAN NOT NULL DEFAULT TRUE
);

CREATE TABLE shipments (
    shipment_id        BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    order_id           BIGINT NOT NULL,
    shipping_method_id BIGINT NOT NULL,
    shipping_address_id BIGINT NOT NULL,
    carrier_name       VARCHAR(100),
    tracking_number    VARCHAR(100),
    shipment_status    VARCHAR(50) NOT NULL DEFAULT 'created', -- or separate status table
    shipped_at         TIMESTAMP,
    delivered_at       TIMESTAMP,
    created_at         TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT fk_shipments_order
        FOREIGN KEY (order_id) REFERENCES orders(order_id),
    CONSTRAINT fk_shipments_method
        FOREIGN KEY (shipping_method_id) REFERENCES shipping_methods(shipping_method_id),
    CONSTRAINT fk_shipments_address
        FOREIGN KEY (shipping_address_id) REFERENCES customer_addresses(address_id)
);

-- Indexes for shipments
CREATE INDEX idx_shipments_order_id ON shipments(order_id);
CREATE INDEX idx_shipments_tracking ON shipments(tracking_number);
CREATE INDEX idx_shipments_status ON shipments(shipment_status);
CREATE INDEX idx_shipments_shipped_at ON shipments(shipped_at);
CREATE INDEX idx_shipments_delivered_at ON shipments(delivered_at);
CREATE INDEX idx_shipments_method ON shipments(shipping_method_id);
CREATE INDEX idx_shipments_address ON shipments(shipping_address_id);

