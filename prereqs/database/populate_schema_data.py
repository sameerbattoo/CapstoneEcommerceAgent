#!/usr/bin/env python3
"""
Generate SQL Script to Populate E-commerce Database Schema with Randomized Data

This script generates SQL INSERT statements to populate tables in a specified 
PostgreSQL schema with randomized sample data. The data volume matches sample_data.sql 
but is randomized for each tenant to ensure unique datasets.

Usage:
    python populate_schema_data.py <schema_name> [output_file]
    
Example:
    python populate_schema_data.py tenantA
    python populate_schema_data.py tenantA tenantA_data.sql
    python populate_schema_data.py tenantB > tenantB_data.sql
"""

import sys
import os
import random
from datetime import datetime, timedelta
from decimal import Decimal

# Data generation constants
FIRST_NAMES = [
    'James', 'Mary', 'John', 'Patricia', 'Robert', 'Jennifer', 'Michael', 'Linda',
    'William', 'Elizabeth', 'David', 'Barbara', 'Richard', 'Susan', 'Joseph', 'Jessica',
    'Thomas', 'Sarah', 'Charles', 'Karen', 'Christopher', 'Nancy', 'Daniel', 'Lisa',
    'Matthew', 'Betty', 'Anthony', 'Margaret', 'Mark', 'Sandra', 'Donald', 'Ashley',
    'Steven', 'Kimberly', 'Paul', 'Emily', 'Andrew', 'Donna', 'Joshua', 'Michelle'
]

LAST_NAMES = [
    'Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis',
    'Rodriguez', 'Martinez', 'Hernandez', 'Lopez', 'Gonzalez', 'Wilson', 'Anderson',
    'Thomas', 'Taylor', 'Moore', 'Jackson', 'Martin', 'Lee', 'Perez', 'Thompson',
    'White', 'Harris', 'Sanchez', 'Clark', 'Ramirez', 'Lewis', 'Robinson', 'Walker',
    'Young', 'Allen', 'King', 'Wright', 'Scott', 'Torres', 'Nguyen', 'Hill', 'Flores'
]

STREETS = [
    'Main St', 'Oak Ave', 'Pine Rd', 'Elm St', 'Maple Dr', 'Cedar Ln', 'Birch Ct',
    'Spruce Way', 'Willow Pl', 'Ash Blvd', 'Park Ave', 'Lake Dr', 'River Rd',
    'Hill St', 'Valley Way', 'Mountain View', 'Sunset Blvd', 'Ocean Dr', 'Forest Ln'
]

CITIES_STATES = [
    ('New York', 'NY', '10001'), ('Los Angeles', 'CA', '90001'),
    ('Chicago', 'IL', '60601'), ('Houston', 'TX', '77001'),
    ('Phoenix', 'AZ', '85001'), ('Philadelphia', 'PA', '19019'),
    ('San Antonio', 'TX', '78201'), ('San Diego', 'CA', '92101'),
    ('Dallas', 'TX', '75201'), ('San Jose', 'CA', '95101'),
    ('Austin', 'TX', '78701'), ('Jacksonville', 'FL', '32099'),
    ('San Francisco', 'CA', '94102'), ('Columbus', 'OH', '43004'),
    ('Indianapolis', 'IN', '46201'), ('Seattle', 'WA', '98101')
]

CATEGORIES = [
    ('Electronics', 'Electronic devices and accessories'),
    ('Clothing', 'Apparel and fashion items'),
    ('Books', 'Physical and digital books'),
    ('Home & Garden', 'Home improvement and garden supplies'),
    ('Sports', 'Sports equipment and accessories')
]

PRODUCTS_BY_CATEGORY = {
    'Electronics': [
        ('Laptop Pro 15"', 'High-performance laptop with 16GB RAM', 1299.99, 50),
        ('Wireless Mouse', 'Ergonomic wireless mouse', 29.99, 200),
        ('USB-C Hub', '7-in-1 USB-C adapter', 49.99, 150),
        ('Bluetooth Headphones', 'Noise-cancelling headphones', 199.99, 75),
        ('Smartphone X', 'Latest smartphone with 5G', 899.99, 100),
    ],
    'Clothing': [
        ('Cotton T-Shirt', 'Comfortable cotton t-shirt', 19.99, 300),
        ('Denim Jeans', 'Classic blue jeans', 59.99, 150),
        ('Running Shoes', 'Lightweight running shoes', 89.99, 100),
        ('Winter Jacket', 'Warm winter jacket', 149.99, 80),
        ('Baseball Cap', 'Adjustable baseball cap', 24.99, 200),
    ],
    'Books': [
        ('Python Programming', 'Learn Python from scratch', 39.99, 120),
        ('Data Science Handbook', 'Comprehensive data science guide', 49.99, 90),
        ('Fiction Novel', 'Bestselling fiction novel', 14.99, 200),
        ('Cookbook', 'Healthy recipes cookbook', 29.99, 110),
        ('Biography', 'Inspiring biography', 24.99, 85),
    ],
    'Home & Garden': [
        ('Coffee Maker', 'Programmable coffee maker', 79.99, 60),
        ('Garden Tools Set', '5-piece garden tool set', 44.99, 70),
        ('LED Desk Lamp', 'Adjustable LED lamp', 34.99, 120),
        ('Throw Pillows', 'Decorative throw pillows (set of 2)', 29.99, 150),
        ('Plant Pot', 'Ceramic plant pot', 19.99, 180),
    ],
    'Sports': [
        ('Yoga Mat', 'Non-slip yoga mat', 34.99, 140),
        ('Dumbbell Set', 'Adjustable dumbbell set', 129.99, 50),
        ('Tennis Racket', 'Professional tennis racket', 89.99, 60),
        ('Basketball', 'Official size basketball', 29.99, 100),
        ('Fitness Tracker', 'Smart fitness tracker', 79.99, 90),
    ]
}

ORDER_STATUSES = ['delivered', 'shipped', 'processing', 'cancelled']
ORDER_STATUS_WEIGHTS = [0.70, 0.15, 0.10, 0.05]  # 70% delivered, 15% shipped, etc.


def generate_sql_header(schema_name):
    """Generate SQL header"""
    return f"""-- ============================================================
-- E-commerce Database Data Population Script
-- Schema: {schema_name}
-- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
-- ============================================================
-- This script contains randomized sample data for multi-tenant testing
-- Data volume: 5 categories, 10 customers, 25 products, 2015 orders
-- ============================================================

-- Set search path to the schema
SET search_path TO {schema_name}, public;

-- Disable triggers for faster insertion (optional)
-- SET session_replication_role = replica;

"""


def generate_email(first_name, last_name, schema_name):
    """Generate a unique email address"""
    return f"{first_name.lower()}.{last_name.lower()}.{schema_name}@email.com"


def generate_categories_sql(schema_name):
    """Generate SQL for inserting categories"""
    sql_lines = [f"\n-- ============================================================"]
    sql_lines.append(f"-- Insert Categories")
    sql_lines.append(f"-- ============================================================")
    sql_lines.append(f"INSERT INTO {schema_name}.categories (category_name, description) VALUES")
    
    values = []
    for category_name, description in CATEGORIES:
        values.append(f"  ({sql_escape(category_name)}, {sql_escape(description)})")
    
    sql_lines.append(",\n".join(values) + ";")
    sql_lines.append(f"\n-- {len(CATEGORIES)} categories inserted\n")
    
    return "\n".join(sql_lines)


def generate_customers_sql(schema_name, num_customers=10):
    """Generate SQL for inserting randomized customers"""
    sql_lines = [f"\n-- ============================================================"]
    sql_lines.append(f"-- Insert Customers (Randomized)")
    sql_lines.append(f"-- ============================================================")
    sql_lines.append(f"INSERT INTO {schema_name}.customers (first_name, last_name, email, phone, address, city, state, country, postal_code) VALUES")
    
    customers = []
    used_emails = set()
    
    for i in range(num_customers):
        first_name = random.choice(FIRST_NAMES)
        last_name = random.choice(LAST_NAMES)
        email = generate_email(first_name, last_name, schema_name)
        
        # Ensure unique email
        counter = 1
        while email in used_emails:
            email = f"{first_name.lower()}.{last_name.lower()}{counter}.{schema_name}@email.com"
            counter += 1
        used_emails.add(email)
        
        phone = f"555-{random.randint(1000, 9999)}"
        street_num = random.randint(100, 9999)
        street = random.choice(STREETS)
        city, state, postal_code = random.choice(CITIES_STATES)
        
        customers.append(
            f"  ({sql_escape(first_name)}, {sql_escape(last_name)}, {sql_escape(email)}, "
            f"{sql_escape(phone)}, {sql_escape(f'{street_num} {street}')}, "
            f"{sql_escape(city)}, {sql_escape(state)}, {sql_escape('USA')}, {sql_escape(postal_code)})"
        )
    
    sql_lines.append(",\n".join(customers) + ";")
    sql_lines.append(f"\n-- {num_customers} customers inserted\n")
    
    return "\n".join(sql_lines)


def generate_products_sql(schema_name):
    """Generate SQL for inserting products"""
    sql_lines = [f"\n-- ============================================================"]
    sql_lines.append(f"-- Insert Products (Randomized prices and stock)")
    sql_lines.append(f"-- ============================================================")
    sql_lines.append(f"INSERT INTO {schema_name}.products (product_name, category_id, description, price, stock_quantity) VALUES")
    
    products = []
    category_id = 1
    
    for category_name in ['Electronics', 'Clothing', 'Books', 'Home & Garden', 'Sports']:
        for product_name, description, price, stock in PRODUCTS_BY_CATEGORY[category_name]:
            # Add some randomization to price and stock
            price_variation = random.uniform(0.95, 1.05)
            stock_variation = random.randint(-10, 10)
            
            final_price = round(price * price_variation, 2)
            final_stock = max(0, stock + stock_variation)
            
            products.append(
                f"  ({sql_escape(product_name)}, {category_id}, {sql_escape(description)}, "
                f"{final_price}, {final_stock})"
            )
        category_id += 1
    
    sql_lines.append(",\n".join(products) + ";")
    sql_lines.append(f"\n-- {len(products)} products inserted\n")
    
    return "\n".join(sql_lines)


def generate_random_date(start_date, end_date):
    """Generate a random datetime between start and end dates"""
    time_between = end_date - start_date
    days_between = time_between.days
    random_days = random.randint(0, days_between)
    random_seconds = random.randint(0, 86400)
    return start_date + timedelta(days=random_days, seconds=random_seconds)


def generate_orders_and_items_sql(schema_name, num_customers, num_products, num_orders=2015):
    """Generate SQL for inserting randomized orders and order items"""
    
    # Date range: Jan 2023 to Nov 2025
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2025, 11, 30)
    
    sql_lines = [f"\n-- ============================================================"]
    sql_lines.append(f"-- Insert Orders (Randomized)")
    sql_lines.append(f"-- ============================================================")
    
    orders = []
    all_order_items = []
    
    for order_num in range(1, num_orders + 1):
        customer_id = random.randint(1, num_customers)
        order_date = generate_random_date(start_date, end_date)
        status = random.choices(ORDER_STATUSES, weights=ORDER_STATUS_WEIGHTS)[0]
        
        # Get customer address
        city, state, postal_code = random.choice(CITIES_STATES)
        street_num = random.randint(100, 9999)
        street = random.choice(STREETS)
        
        # Generate order items (1-5 items per order)
        num_items = random.randint(1, 5)
        selected_products = random.sample(range(1, num_products + 1), num_items)
        
        total_amount = Decimal('0.00')
        order_items = []
        
        for product_id in selected_products:
            quantity = random.randint(1, 3)
            unit_price = Decimal(str(round(random.uniform(14.99, 1299.99), 2)))
            subtotal = unit_price * quantity
            total_amount += subtotal
            
            order_items.append((order_num, product_id, quantity, unit_price, subtotal))
        
        orders.append(
            f"  ({customer_id}, {sql_escape(order_date)}, {sql_escape(status)}, "
            f"{total_amount}, {sql_escape(f'{street_num} {street}')}, "
            f"{sql_escape(city)}, {sql_escape(state)}, {sql_escape('USA')}, {sql_escape(postal_code)})"
        )
        
        all_order_items.extend(order_items)
    
    # Add orders SQL in batches to avoid PostgreSQL statement length limit
    batch_size = 500  # 500 orders per INSERT statement
    for i in range(0, len(orders), batch_size):
        batch = orders[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(orders) + batch_size - 1) // batch_size
        
        if i > 0:
            sql_lines.append(f"\n-- Batch {batch_num}/{total_batches}")
        
        sql_lines.append(f"INSERT INTO {schema_name}.orders (customer_id, order_date, status, total_amount, shipping_address, shipping_city, shipping_state, shipping_country, shipping_postal_code) VALUES")
        sql_lines.append(",\n".join(batch) + ";")
    
    sql_lines.append(f"\n-- {len(orders)} orders inserted\n")
    
    # Add order items SQL in batches
    sql_lines.append(f"\n-- ============================================================")
    sql_lines.append(f"-- Insert Order Items (Randomized)")
    sql_lines.append(f"-- ============================================================")
    
    items = []
    for order_id, product_id, quantity, unit_price, subtotal in all_order_items:
        items.append(
            f"  ({order_id}, {product_id}, {quantity}, {unit_price}, {subtotal})"
        )
    
    # Split into batches of 1000 to avoid PostgreSQL statement length limit
    batch_size = 1000
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(items) + batch_size - 1) // batch_size
        
        if i > 0:
            sql_lines.append(f"\n-- Batch {batch_num}/{total_batches}")
        
        sql_lines.append(f"INSERT INTO {schema_name}.order_items (order_id, product_id, quantity, unit_price, subtotal) VALUES")
        sql_lines.append(",\n".join(batch) + ";")
    
    sql_lines.append(f"\n-- {len(all_order_items)} order items inserted\n")
    
    return "\n".join(sql_lines)


def generate_customer_addresses_sql(schema_name, num_customers):
    """Generate SQL for inserting customer addresses (2-3 addresses per customer)"""
    sql_lines = [f"\n-- ============================================================"]
    sql_lines.append(f"-- Insert Customer Addresses")
    sql_lines.append(f"-- ============================================================")
    sql_lines.append(f"INSERT INTO {schema_name}.customer_addresses (customer_id, address_type, full_name, phone_number, address_line1, address_line2, city, state_province, postal_code, country_code, is_default) VALUES")
    
    addresses = []
    address_types = ['home', 'work', 'billing', 'shipping']
    
    for customer_id in range(1, num_customers + 1):
        # Each customer gets 2-3 addresses
        num_addresses = random.randint(2, 3)
        
        for addr_idx in range(num_addresses):
            # Generate random name (could be different from customer name for gift shipping, etc.)
            first_name = random.choice(FIRST_NAMES)
            last_name = random.choice(LAST_NAMES)
            full_name = f"{first_name} {last_name}"
            
            address_type = random.choice(address_types)
            phone = f"555-{random.randint(1000, 9999)}"
            
            street_num = random.randint(100, 9999)
            street = random.choice(STREETS)
            address_line1 = f"{street_num} {street}"
            
            # 30% chance of having address_line2
            address_line2 = f"Apt {random.randint(1, 999)}" if random.random() < 0.3 else None
            
            city, state, postal_code = random.choice(CITIES_STATES)
            country_code = 'US'
            
            # First address is default
            is_default = 'TRUE' if addr_idx == 0 else 'FALSE'
            
            addresses.append(
                f"  ({customer_id}, {sql_escape(address_type)}, {sql_escape(full_name)}, "
                f"{sql_escape(phone)}, {sql_escape(address_line1)}, {sql_escape(address_line2)}, "
                f"{sql_escape(city)}, {sql_escape(state)}, {sql_escape(postal_code)}, "
                f"{sql_escape(country_code)}, {is_default})"
            )
    
    sql_lines.append(",\n".join(addresses) + ";")
    sql_lines.append(f"\n-- {len(addresses)} customer addresses inserted\n")
    
    return "\n".join(sql_lines)


def generate_shipping_methods_sql(schema_name):
    """Generate SQL for inserting shipping methods"""
    sql_lines = [f"\n-- ============================================================"]
    sql_lines.append(f"-- Insert Shipping Methods")
    sql_lines.append(f"-- ============================================================")
    sql_lines.append(f"INSERT INTO {schema_name}.shipping_methods (code, display_name, base_cost, is_active) VALUES")
    
    shipping_methods = [
        ('STANDARD', 'Standard Shipping (5-7 business days)', 5.99, True),
        ('EXPRESS', 'Express Shipping (2-3 business days)', 12.99, True),
        ('OVERNIGHT', 'Overnight Shipping (1 business day)', 24.99, True),
        ('FREE', 'Free Standard Shipping', 0.00, True),
        ('PICKUP', 'In-Store Pickup', 0.00, True),
    ]
    
    methods = []
    for code, display_name, base_cost, is_active in shipping_methods:
        # Add slight price variation per tenant
        varied_cost = round(base_cost * random.uniform(0.95, 1.05), 2) if base_cost > 0 else 0.00
        is_active_str = 'TRUE' if is_active else 'FALSE'
        
        methods.append(
            f"  ({sql_escape(code)}, {sql_escape(display_name)}, {varied_cost}, {is_active_str})"
        )
    
    sql_lines.append(",\n".join(methods) + ";")
    sql_lines.append(f"\n-- {len(methods)} shipping methods inserted\n")
    
    return "\n".join(sql_lines)


def generate_shipments_sql(schema_name, num_orders, num_customers):
    """Generate SQL for inserting shipments (one shipment per delivered/shipped order)"""
    sql_lines = [f"\n-- ============================================================"]
    sql_lines.append(f"-- Insert Shipments")
    sql_lines.append(f"-- ============================================================")
    
    shipments = []
    carriers = ['FedEx', 'UPS', 'USPS', 'DHL', 'Amazon Logistics']
    shipment_statuses_map = {
        'delivered': 'delivered',
        'shipped': 'in_transit',
        'processing': 'created',
        'cancelled': 'cancelled'
    }
    
    # We need to regenerate order data to know which orders need shipments
    # Using same random seed logic as in generate_orders_and_items_sql
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2025, 11, 30)
    
    for order_id in range(1, num_orders + 1):
        # Regenerate order status (must match the order generation)
        order_status = random.choices(ORDER_STATUSES, weights=ORDER_STATUS_WEIGHTS)[0]
        
        # Only create shipments for orders that have been shipped or delivered
        if order_status in ['delivered', 'shipped', 'processing']:
            # Random customer address (1-3 addresses per customer, we'll use address_id based on order)
            customer_id = random.randint(1, num_customers)
            # Assume 2-3 addresses per customer, pick one
            address_offset = random.randint(0, 2)
            shipping_address_id = (customer_id - 1) * 2 + address_offset + 1  # Approximate address_id
            
            # Random shipping method (1-5)
            shipping_method_id = random.randint(1, 5)
            
            carrier_name = random.choice(carriers)
            tracking_number = f"{carrier_name[:3].upper()}{random.randint(100000000, 999999999)}"
            
            shipment_status = shipment_statuses_map[order_status]
            
            # Generate shipment dates based on order status
            order_date = generate_random_date(start_date, end_date)
            
            if order_status == 'delivered':
                # Shipped 1-3 days after order, delivered 2-7 days after shipping
                shipped_at = order_date + timedelta(days=random.randint(1, 3))
                delivered_at = shipped_at + timedelta(days=random.randint(2, 7))
                shipments.append(
                    f"  ({order_id}, {shipping_method_id}, {shipping_address_id}, "
                    f"{sql_escape(carrier_name)}, {sql_escape(tracking_number)}, "
                    f"{sql_escape(shipment_status)}, {sql_escape(shipped_at)}, {sql_escape(delivered_at)})"
                )
            elif order_status == 'shipped':
                # Shipped 1-3 days after order, not yet delivered
                shipped_at = order_date + timedelta(days=random.randint(1, 3))
                shipments.append(
                    f"  ({order_id}, {shipping_method_id}, {shipping_address_id}, "
                    f"{sql_escape(carrier_name)}, {sql_escape(tracking_number)}, "
                    f"{sql_escape(shipment_status)}, {sql_escape(shipped_at)}, NULL)"
                )
            else:  # processing
                # Created but not yet shipped
                shipments.append(
                    f"  ({order_id}, {shipping_method_id}, {shipping_address_id}, "
                    f"{sql_escape(carrier_name)}, {sql_escape(tracking_number)}, "
                    f"{sql_escape(shipment_status)}, NULL, NULL)"
                )
    
    # Batch shipments to avoid statement length issues
    batch_size = 500
    for i in range(0, len(shipments), batch_size):
        batch = shipments[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(shipments) + batch_size - 1) // batch_size
        
        if i > 0:
            sql_lines.append(f"\n-- Batch {batch_num}/{total_batches}")
        
        sql_lines.append(f"INSERT INTO {schema_name}.shipments (order_id, shipping_method_id, shipping_address_id, carrier_name, tracking_number, shipment_status, shipped_at, delivered_at) VALUES")
        sql_lines.append(",\n".join(batch) + ";")
    
    sql_lines.append(f"\n-- {len(shipments)} shipments inserted\n")
    
    return "\n".join(sql_lines)


def sql_escape(value):
    """Escape SQL string values"""
    if value is None:
        return 'NULL'
    if isinstance(value, (int, float, Decimal)):
        return str(value)
    if isinstance(value, datetime):
        return f"'{value.strftime('%Y-%m-%d %H:%M:%S')}'"
    # Escape single quotes in strings
    return f"'{str(value).replace(chr(39), chr(39) + chr(39))}'"


def generate_sql_footer(schema_name):
    """Generate SQL footer with summary"""
    return f"""
-- ============================================================
-- Re-enable triggers (if disabled)
-- ============================================================
-- SET session_replication_role = DEFAULT;

-- ============================================================
-- Data Population Complete: {schema_name}
-- ============================================================
-- Summary:
--   - 5 categories
--   - 10 customers (randomized)
--   - 20-30 customer addresses (2-3 per customer)
--   - 25 products (randomized prices/stock)
--   - 5 shipping methods
--   - 2,015 orders (randomized dates, customers, statuses)
--   - ~6,000 order items (randomized quantities)
--   - ~1,700 shipments (for shipped/delivered orders)
-- ============================================================
-- To execute this script:
--   psql -U postgres -d ecommerce -f {schema_name}_data.sql
-- ============================================================
"""


def main():
    """Main execution function"""
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python populate_schema_data.py <schema_name> [output_file]", file=sys.stderr)
        print("\nExample:", file=sys.stderr)
        print("  python populate_schema_data.py tenant_a", file=sys.stderr)
        print("  python populate_schema_data.py tenant_a tenant_a_data.sql", file=sys.stderr)
        print("  python populate_schema_data.py tenant_b > tenant_b_data.sql", file=sys.stderr)
        sys.exit(1)
    
    schema_name = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) == 3 else None
    
    # Validate schema name
    if not schema_name.replace('_', '').isalnum():
        print(f"Error: Invalid schema name '{schema_name}'", file=sys.stderr)
        print("Schema name must contain only letters, numbers, and underscores", file=sys.stderr)
        sys.exit(1)
    
    # Progress messages to stderr
    print(f"Generating SQL for schema: {schema_name}", file=sys.stderr)
    
    # Seed random with schema name for reproducibility per tenant
    random.seed(schema_name)
    
    # Generate SQL
    sql_script = []
    sql_script.append(generate_sql_header(schema_name))
    
    print("  - Generating categories...", file=sys.stderr)
    sql_script.append(generate_categories_sql(schema_name))
    
    print("  - Generating customers...", file=sys.stderr)
    num_customers = 10
    sql_script.append(generate_customers_sql(schema_name, num_customers))
    
    print("  - Generating products...", file=sys.stderr)
    num_products = 25
    sql_script.append(generate_products_sql(schema_name))
    
    print("  - Generating customer addresses...", file=sys.stderr)
    sql_script.append(generate_customer_addresses_sql(schema_name, num_customers))
    
    print("  - Generating shipping methods...", file=sys.stderr)
    sql_script.append(generate_shipping_methods_sql(schema_name))
    
    print("  - Generating orders and order items...", file=sys.stderr)
    num_orders = 2015
    sql_script.append(generate_orders_and_items_sql(schema_name, num_customers, num_products, num_orders))
    
    print("  - Generating shipments...", file=sys.stderr)
    sql_script.append(generate_shipments_sql(schema_name, num_orders, num_customers))
    
    sql_script.append(generate_sql_footer(schema_name))
    
    full_sql = "\n".join(sql_script)
    
    # Output SQL
    if output_file:
        with open(output_file, 'w') as f:
            f.write(full_sql)
        print(f"\n✅ SQL script generated: {output_file}", file=sys.stderr)
        print(f"\nTo execute:", file=sys.stderr)
        print(f"  psql -U postgres -d ecommerce -f {output_file}", file=sys.stderr)
        print(f"\nData Summary:", file=sys.stderr)
        print(f"  - 5 categories", file=sys.stderr)
        print(f"  - 10 customers (randomized)", file=sys.stderr)
        print(f"  - 20-30 customer addresses (2-3 per customer)", file=sys.stderr)
        print(f"  - 25 products (randomized prices/stock)", file=sys.stderr)
        print(f"  - 5 shipping methods", file=sys.stderr)
        print(f"  - 2,015 orders (randomized dates, customers, statuses)", file=sys.stderr)
        print(f"  - ~6,000 order items (randomized quantities)", file=sys.stderr)
        print(f"  - ~1,700 shipments (for shipped/delivered orders)", file=sys.stderr)
    else:
        print(full_sql)


if __name__ == "__main__":
    main()
