#!/bin/bash
# Deploy DynamoDB table for product reviews and populate with sample data

set -e

REGION="${AWS_REGION:-us-west-2}"
TABLE_NAME="ProductReviews"

echo "Creating DynamoDB table: $TABLE_NAME"

# Create DynamoDB table
aws dynamodb create-table \
    --table-name $TABLE_NAME \
    --attribute-definitions \
        AttributeName=product_id,AttributeType=N \
        AttributeName=review_id,AttributeType=S \
    --key-schema \
        AttributeName=product_id,KeyType=HASH \
        AttributeName=review_id,KeyType=RANGE \
    --billing-mode PAY_PER_REQUEST \
    --region $REGION \
    --tags Key=Project,Value=CapstoneECommerce

echo "Waiting for table to be active..."
aws dynamodb wait table-exists --table-name $TABLE_NAME --region $REGION

echo "Table created successfully. Populating with sample data..."

# Generate and insert 2000 reviews using Python
python3 << 'EOF'
import boto3
import random
import uuid
from datetime import datetime, timedelta
from decimal import Decimal

dynamodb = boto3.resource('dynamodb', region_name='us-west-2')
table = dynamodb.Table('ProductReviews')

# Product IDs and names from sample_data.sql (1-25)
product_ids = list(range(1, 26))

# Product names mapping (from sample_data.sql)
product_names = {
    1: "Laptop Pro 15\"",
    2: "Wireless Mouse",
    3: "USB-C Hub",
    4: "Bluetooth Headphones",
    5: "Smartphone X",
    6: "Cotton T-Shirt",
    7: "Denim Jeans",
    8: "Running Shoes",
    9: "Winter Jacket",
    10: "Baseball Cap",
    11: "Python Programming",
    12: "Data Science Handbook",
    13: "Fiction Novel",
    14: "Cookbook",
    15: "Biography",
    16: "Coffee Maker",
    17: "Garden Tools Set",
    18: "LED Desk Lamp",
    19: "Throw Pillows",
    20: "Plant Pot",
    21: "Yoga Mat",
    22: "Dumbbell Set",
    23: "Tennis Racket",
    24: "Basketball",
    25: "Fitness Tracker"
}

# Customer IDs and emails from sample_data.sql (1-10)
customer_ids = list(range(1, 11))

# Customer emails mapping (from sample_data.sql)
customer_emails = {
    1: "john.doe@email.com",
    2: "jane.smith@email.com",
    3: "bob.johnson@email.com",
    4: "alice.williams@email.com",
    5: "charlie.brown@email.com",
    6: "diana.davis@email.com",
    7: "eve.martinez@email.com",
    8: "frank.garcia@email.com",
    9: "grace.rodriguez@email.com",
    10: "henry.wilson@email.com"
}

# Sample review titles and comments
review_titles = [
    "Excellent product!", "Great value for money", "Highly recommended",
    "Not what I expected", "Amazing quality", "Could be better",
    "Perfect for my needs", "Disappointed", "Exceeded expectations",
    "Good but not great", "Love it!", "Waste of money",
    "Best purchase ever", "Decent product", "Outstanding quality",
    "Not worth the price", "Fantastic!", "Average at best",
    "Superb quality", "Terrible experience", "Very satisfied",
    "Would not recommend", "Impressive", "Mediocre product",
    "Absolutely love it", "Poor quality", "Great purchase",
    "Not as described", "Wonderful product", "Regret buying this"
]

positive_comments = [
    "This product exceeded all my expectations. The quality is outstanding and it works perfectly.",
    "I'm very happy with this purchase. It arrived quickly and was exactly as described.",
    "Excellent quality and great value. Would definitely buy again and recommend to others.",
    "This is exactly what I was looking for. The build quality is solid and it performs great.",
    "Very impressed with this product. It's well-made and does everything I need it to do.",
    "Outstanding product! The attention to detail is remarkable and it works flawlessly.",
    "I've been using this for a few weeks now and I'm completely satisfied with the performance.",
    "Great product at a reasonable price. The quality is much better than I expected.",
    "This has made my life so much easier. Highly recommend to anyone looking for this type of product.",
    "Perfect! Exactly what I needed and the quality is top-notch."
]

negative_comments = [
    "Unfortunately, this product didn't meet my expectations. The quality is poor.",
    "Not worth the money. It broke after just a few uses and I'm very disappointed.",
    "The product description was misleading. What I received was not what I expected.",
    "Poor quality materials and construction. Would not recommend this to anyone.",
    "This is a waste of money. It doesn't work as advertised and feels cheap.",
    "Very disappointed with this purchase. The product arrived damaged and doesn't function properly.",
    "Not as described. The actual product is much lower quality than shown in the pictures.",
    "I regret buying this. It's poorly made and doesn't do what it's supposed to do.",
    "Terrible product. It stopped working after a week and customer service was unhelpful.",
    "Don't waste your money on this. There are much better alternatives available."
]

neutral_comments = [
    "It's okay. Does the job but nothing special. Average quality for the price.",
    "Decent product. It works as expected but there's room for improvement.",
    "It's alright. Not amazing but not terrible either. Gets the job done.",
    "Average product. It does what it's supposed to do but doesn't stand out.",
    "It's fine. Nothing to complain about but nothing to rave about either.",
    "Acceptable quality. It works but I've seen better products in this category.",
    "It's serviceable. Does the basic functions but lacks some features I'd like.",
    "Mediocre product. It's functional but the quality could be better.",
    "It's passable. Works as advertised but feels a bit cheap.",
    "Neutral feelings about this. It's neither great nor terrible."
]

# Generate start date (Jan 2023) and end date (Nov 2025)
start_date = datetime(2023, 1, 1)
end_date = datetime(2025, 11, 30)
date_range = (end_date - start_date).days

print("Generating 2000 product reviews...")

with table.batch_writer() as batch:
    for i in range(2000):
        product_id = random.choice(product_ids)
        customer_id = random.choice(customer_ids)
        
        # Generate random rating (1-5 stars)
        rating = random.randint(1, 5)
        
        # Select review content based on rating
        if rating >= 4:
            title = random.choice([t for t in review_titles if any(word in t.lower() for word in ['excellent', 'great', 'amazing', 'love', 'best', 'fantastic', 'superb', 'wonderful'])])
            comment = random.choice(positive_comments)
        elif rating <= 2:
            title = random.choice([t for t in review_titles if any(word in t.lower() for word in ['not', 'disappointed', 'waste', 'terrible', 'poor', 'regret'])])
            comment = random.choice(negative_comments)
        else:
            title = random.choice([t for t in review_titles if any(word in t.lower() for word in ['good', 'decent', 'average', 'okay', 'mediocre'])])
            comment = random.choice(neutral_comments)
        
        # Generate random review date
        random_days = random.randint(0, date_range)
        review_date = start_date + timedelta(days=random_days)
        
        # Generate helpful votes (0-100)
        helpful_votes = random.randint(0, 100)
        
        # Verified purchase (80% chance)
        verified_purchase = random.random() < 0.8
        
        review_item = {
            'product_id': product_id,
            'review_id': str(uuid.uuid4()),
            'customer_id': customer_id,
            'customer_email': customer_emails[customer_id],
            'product_name': product_names[product_id],
            'rating': Decimal(str(rating)),
            'title': title,
            'comment': comment,
            'review_date': review_date.isoformat(),
            'helpful_votes': helpful_votes,
            'verified_purchase': verified_purchase,
            'created_at': datetime.now().isoformat()
        }
        
        batch.put_item(Item=review_item)
        
        if (i + 1) % 100 == 0:
            print(f"Inserted {i + 1} reviews...")

print("Successfully inserted 2000 product reviews!")

# Query and display sample data
print("\nSample reviews from the table:")
response = table.scan(Limit=5)
for item in response['Items']:
    print(f"Product {item['product_id']}: {item['rating']} stars - {item['title']}")

EOF

echo ""
echo "DynamoDB table setup complete!"
echo "Table name: $TABLE_NAME"
echo "Region: $REGION"
echo "Total reviews: 2000"
