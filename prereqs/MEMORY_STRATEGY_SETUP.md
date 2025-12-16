# Adding Long-Term Memory Strategies to Existing Memory

Adding strategies for long-term memory functionality for existing memory `capstone_ecommerce_mempry-XPTahoCqjM` 

## Using AWS Console

### Step 1: Open AWS Console

1. Go to AWS Console → Amazon Bedrock
2. Navigate to **AgentCore** → **Memory**
3. Find your memory: `capstone_ecommerce_mem*`

### Step 2: Add Semantic Strategy

1. Click on your memory
2. Click **Add Strategy**
3. Select **Custom Semantic Strategy**
4. Configure:
   - **Name**: `OrderFactsExtractor`
   - **Description**: `Extracts facts about orders, shipments, and products`
   - **Namespace**: `/users/{actorId}/facts`
   - **Extraction Model**: `anthropic.claude-haiku-4-5-20251001-v1:0`
   - **Extraction Prompt**: 
     ```
     Extract factual information from the conversation including:
     - Order IDs and order details
     - Product names and specifications
     - Shipment status and tracking information
     - Purchase dates and delivery dates
     - Customer issues and resolutions
     - Product preferences and interests
     Focus on concrete facts that can be referenced in future conversations.
     ```
   - **Consolidation Model**: `anthropic.claude-haiku-4-5-20251001-v1:0`
   - **Consolidation Prompt**: 
     ```
     Consolidate and merge order and purchase facts, removing duplicates and keeping the most recent information.
     ```
   - **Execution Role**: Select your IAM role with bedrock:InvokeModel permissions

5. Click **Add Strategy**
6. Wait for status to become **ACTIVE** (1-2 minutes)

### Step 3: Add User Preference Strategy

1. Click **Add Strategy** again
2. Select **Custom User Preference Strategy**
3. Configure:
   - **Name**: `CustomerPreferences`
   - **Description**: `Captures customer preferences and behavior patterns`
   - **Namespace**: `/users/{actorId}/preferences`
   - **Extraction Model**: `anthropic.claude-haiku-4-5-20251001-v1:0`
   - **Extraction Prompt**: 
     ```
     Extract customer preferences and behavior patterns including:
     - Product categories of interest
     - Brand preferences
     - Shopping patterns
     - Communication preferences
     - Feature preferences
     Focus on understanding what the customer likes and how they prefer to interact.
     ```
   - **Consolidation Model**: `anthropic.claude-haiku-4-5-20251001-v1:0`
   - **Consolidation Prompt**: 
     ```
     Consolidate customer preferences, keeping the most recent and relevant preferences.
     ```
   - **Execution Role**: Same IAM role as above

4. Click **Add Strategy**
5. Wait for status to become **ACTIVE**

---


## What These Strategies Do

### OrderFactsExtractor (Semantic Strategy)
- **Automatically extracts** facts from conversations
- **Stores** in `/users/{actorId}/facts`
- **Examples of what it captures:**
  - "User ordered iPhone 15 Pro, order #123456"
  - "Shipment delivered on June 5, 2025"
  - "User reported slow performance issue"

### CustomerPreferences (User Preference Strategy)
- **Automatically learns** user preferences
- **Stores** in `/users/{actorId}/preferences`
- **Examples of what it captures:**
  - "User prefers ThinkPad laptops"
  - "User is interested in electronics"
  - "User likes detailed product specifications"

### How They Work Together
1. User has conversation with agent
2. Strategies automatically extract facts and preferences
3. Information stored in respective namespaces
4. In future sessions, hooks retrieve this information
5. Agent has full context of user's history

---
