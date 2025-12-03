"""
Lambda function to query ProductReviews DynamoDB table
Supports filtering by product_id, customer_id, and review_date range
"""

import json
import boto3
from boto3.dynamodb.conditions import Key, Attr
from decimal import Decimal
from datetime import datetime
import os

# Initialize DynamoDB resource
dynamodb = boto3.resource('dynamodb')
table_name = os.environ.get('TABLE_NAME', 'ProductReviews')
table = dynamodb.Table(table_name)
MAX_ROWS_FOR_DYNAMODB_RESULT_DISPLAY = 20


class DecimalEncoder(json.JSONEncoder):
    """Helper class to convert Decimal to float for JSON serialization"""
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        return super(DecimalEncoder, self).default(obj)


def lambda_handler(event, context):
    """
    Lambda handler to query product reviews
    
    Query Parameters (all optional):
        - product_id: Filter by specific product ID(s) - comma-separated for multiple
        - product_name: Filter by product name(s) (partial match) - comma-separated for multiple
        - customer_id: Filter by specific customer ID(s) - comma-separated for multiple
        - customer_email: Filter by customer email(s) (partial match) - comma-separated for multiple
        - review_date_from: Start date for review date range (ISO format: YYYY-MM-DD)
        - review_date_to: End date for review date range (ISO format: YYYY-MM-DD)
        - top_rows: Limit number of rows returned (default: 100, max: 1000)
    
    Returns:
        JSON response with reviews matching the filters
    """
    
    try:
        # Extract query parameters from event      
        product_id = event.get('product_id')
        product_name = event.get('product_name')
        customer_id = event.get('customer_id')
        customer_email = event.get('customer_email')
        review_date_from = event.get('review_date_from')
        review_date_to = event.get('review_date_to')
        top_rows = event.get('top_rows')
        
        # Parse comma-separated lists
        product_ids = [int(x.strip()) for x in product_id.split(',')] if product_id else None
        product_names = [x.strip() for x in product_name.split(',')] if product_name else None
        customer_ids = [int(x.strip()) for x in customer_id.split(',')] if customer_id else None
        customer_emails = [x.strip() for x in customer_email.split(',')] if customer_email else None
        
        # Parse top_rows as integer with default of MAX_ROWS_FOR_DYNAMODB_RESULT_DISPLAY
        try:
            top_rows = int(top_rows) if top_rows else MAX_ROWS_FOR_DYNAMODB_RESULT_DISPLAY
            top_rows = max(1, min(top_rows, MAX_ROWS_FOR_DYNAMODB_RESULT_DISPLAY))  # Limit between 1 and 1000
        except ValueError:
            top_rows = MAX_ROWS_FOR_DYNAMODB_RESULT_DISPLAY
        
        print(f"Inside lambda_handler: Query parameters: product_ids={product_ids}, product_names={product_names}, "
              f"customer_ids={customer_ids}, customer_emails={customer_emails}, "
              f"review_date_from={review_date_from}, review_date_to={review_date_to}, top_rows={top_rows}")
        
        # Build the query/scan based on parameters
        if product_ids and len(product_ids) == 1:
            # Use Query operation when single product_id is provided (most efficient)
            response = query_by_product(
                product_ids[0],
                customer_ids=customer_ids,
                customer_emails=customer_emails,
                product_names=product_names,
                date_from=review_date_from,
                date_to=review_date_to
            )
        elif product_ids and len(product_ids) > 1:
            # Multiple product_ids: query each and combine results
            all_items = []
            for pid in product_ids:
                response = query_by_product(
                    pid,
                    customer_ids=customer_ids,
                    customer_emails=customer_emails,
                    product_names=product_names,
                    date_from=review_date_from,
                    date_to=review_date_to
                )
                all_items.extend(response.get('Items', []))
            response = {'Items': all_items}
        else:
            # Use Scan operation when no product_id (less efficient but flexible)
            response = scan_with_filters(
                customer_ids=customer_ids,
                customer_emails=customer_emails,
                product_names=product_names,
                date_from=review_date_from,
                date_to=review_date_to
            )
        
        items = response.get('Items', [])
        original_row_count = len(items)

        print(f"Rows retrieved before limit: {original_row_count}")
        
        # Sort by review_date descending
        items.sort(key=lambda x: x.get('review_date', ''), reverse=True)
        
        # Apply top_rows limit
        items = items[:top_rows]
        
        print(f"Rows returned after limit: {len(items)}")
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'count': len(items),
                'total_rows': original_row_count,
                'reviews': items
            }, cls=DecimalEncoder)
        }
        
    except ValueError as e:
        print(f"Validation error: {str(e)}")
        return {
            'statusCode': 400,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'error': 'Invalid parameter format',
                'message': str(e)
            })
        }
        
    except Exception as e:
        print(f"Error querying reviews: {str(e)}")
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'error': 'Internal server error',
                'message': str(e)
            })
        }


def query_by_product(product_id, customer_ids=None, customer_emails=None, product_names=None, date_from=None, date_to=None):
    """
    Query reviews for a specific product with optional filters
    Supports lists for customer_ids, customer_emails, and product_names
    """

    # Start with product_id key condition
    key_condition = Key('product_id').eq(product_id)
    
    # Build filter expression for additional filters
    filter_expressions = []
    
    if customer_ids:
        # Multiple customer IDs - use OR logic
        customer_filters = [Attr('customer_id').eq(cid) for cid in customer_ids]
        if len(customer_filters) == 1:
            filter_expressions.append(customer_filters[0])
        else:
            combined = customer_filters[0]
            for cf in customer_filters[1:]:
                combined = combined | cf
            filter_expressions.append(combined)
    
    if customer_emails:
        # Multiple customer emails - use OR logic with contains
        email_filters = [Attr('customer_email').contains(email) for email in customer_emails]
        if len(email_filters) == 1:
            filter_expressions.append(email_filters[0])
        else:
            combined = email_filters[0]
            for ef in email_filters[1:]:
                combined = combined | ef
            filter_expressions.append(combined)
    
    if product_names:
        # Multiple product names - use OR logic with contains
        name_filters = [Attr('product_name').contains(name) for name in product_names]
        if len(name_filters) == 1:
            filter_expressions.append(name_filters[0])
        else:
            combined = name_filters[0]
            for nf in name_filters[1:]:
                combined = combined | nf
            filter_expressions.append(combined)
    
    if date_from and date_to:
        filter_expressions.append(Attr('review_date').between(date_from, date_to))
    elif date_from:
        filter_expressions.append(Attr('review_date').gte(date_from))
    elif date_to:
        filter_expressions.append(Attr('review_date').lte(date_to))
    
    # Execute query
    query_params = {
        'KeyConditionExpression': key_condition
    }
    
    if filter_expressions:
        # Combine all filter expressions with AND
        combined_filter = filter_expressions[0]
        for expr in filter_expressions[1:]:
            combined_filter = combined_filter & expr
        query_params['FilterExpression'] = combined_filter
    
    print(f"Inside function - query_by_product. Executing Query with params: {combined_filter}")
    return table.query(**query_params)


def scan_with_filters(customer_ids=None, customer_emails=None, product_names=None, date_from=None, date_to=None):
    """
    Scan table with filters (used when product_id is not provided)
    Note: Scan is less efficient than Query but allows filtering without partition key
    Supports lists for customer_ids, customer_emails, and product_names
    """
    filter_expressions = []
    
    if customer_ids:
        # Multiple customer IDs - use OR logic
        customer_filters = [Attr('customer_id').eq(cid) for cid in customer_ids]
        if len(customer_filters) == 1:
            filter_expressions.append(customer_filters[0])
        else:
            combined = customer_filters[0]
            for cf in customer_filters[1:]:
                combined = combined | cf
            filter_expressions.append(combined)
    
    if customer_emails:
        # Multiple customer emails - use OR logic with contains
        email_filters = [Attr('customer_email').contains(email) for email in customer_emails]
        if len(email_filters) == 1:
            filter_expressions.append(email_filters[0])
        else:
            combined = email_filters[0]
            for ef in email_filters[1:]:
                combined = combined | ef
            filter_expressions.append(combined)
    
    if product_names:
        # Multiple product names - use OR logic with contains
        name_filters = [Attr('product_name').contains(name) for name in product_names]
        if len(name_filters) == 1:
            filter_expressions.append(name_filters[0])
        else:
            combined = name_filters[0]
            for nf in name_filters[1:]:
                combined = combined | nf
            filter_expressions.append(combined)
    
    if date_from and date_to:
        filter_expressions.append(Attr('review_date').between(date_from, date_to))
    elif date_from:
        filter_expressions.append(Attr('review_date').gte(date_from))
    elif date_to:
        filter_expressions.append(Attr('review_date').lte(date_to))
    
    scan_params = {}
    
    if filter_expressions:
        # Combine all filter expressions with AND
        combined_filter = filter_expressions[0]
        for expr in filter_expressions[1:]:
            combined_filter = combined_filter & expr
        scan_params['FilterExpression'] = combined_filter
    
    print(f"Inside function - scan_with_filters. Executing Scan with params: {combined_filter}")
    return table.scan(**scan_params)


# For local testing
if __name__ == "__main__":
    import sys
    
    # Test event
    test_event = {
        'product_name': 'Dumbbell Set,Plant Pot',
        'review_date_from': '2024-01-01',
        'top_rows': 2
    }
    
    try:
        print("Testing Lambda function locally...")
        print(f"Table: {table_name}")
        print(f"Region: {os.environ.get('AWS_REGION', 'us-west-2')}")
        print(f"Test parameters: {test_event}")
        print("-" * 50)
        
        result = lambda_handler(test_event, None)
        print(f"Status Code: {result['statusCode']}")
        print(f"Response Body:")
        print(json.dumps(json.loads(result['body']), indent=2))
    except Exception as e:
        print(f"Error during local testing: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
