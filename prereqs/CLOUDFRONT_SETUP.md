# CloudFront Setup for Chart Images

## Overview

Using CloudFront in front of S3 solves the pre-signed URL expiration problem by providing permanent URLs that never expire.

## Architecture Comparison

### Before (S3 Pre-signed URLs):
```
Browser → S3 Pre-signed URL (expires in 7 days) → S3 Bucket
```
**Problem**: URLs expire after 7 days, causing 403 errors

### After (CloudFront):
```
Browser → CloudFront URL (never expires) → S3 Bucket (private, OAC protected)
```
**Benefits**:
- ✅ URLs never expire
- ✅ Better performance (CDN caching)
- ✅ Lower S3 costs (fewer direct requests)
- ✅ S3 bucket stays private (secured by OAC)
- ✅ Global edge locations for faster delivery

---

## Step-by-Step Setup

### Step 1: Create CloudFront Distribution

#### Via AWS Console:

1. **Navigate to CloudFront**
   - Go to AWS Console → CloudFront → Create Distribution

2. **Origin Settings**
   - **Origin Domain**: Select your S3 bucket from dropdown
     - Example: `your-chart-bucket.s3.us-west-2.amazonaws.com`
   - **Origin Path**: Leave empty (or use `/charts` if you want to restrict to that folder)
   - **Name**: Auto-filled (keep default)
   - **Origin Access**: Select **"Origin access control settings (recommended)"**
   - Click **"Create new OAC"**:
     - **Name**: `chart-bucket-oac`
     - **Description**: "OAC for chart images bucket"
     - **Sign requests**: Yes (default)
     - **Origin type**: S3
     - Click **"Create"**

3. **Default Cache Behavior Settings**
   - **Path Pattern**: Default (*)
   - **Compress Objects Automatically**: Yes
   - **Viewer Protocol Policy**: **Redirect HTTP to HTTPS**
   - **Allowed HTTP Methods**: **GET, HEAD, OPTIONS**
   - **Cache Policy**: **CachingOptimized** (or create custom)
   - **Origin Request Policy**: **CORS-S3Origin** (important for CORS headers)
   - **Response Headers Policy**: None (optional)

4. **Function Associations** (Optional)
   - Leave empty for now

5. **Settings**
   - **Price Class**: Use all edge locations (or choose based on budget)
   - **AWS WAF**: Do not enable (unless needed)
   - **Alternate Domain Names (CNAMEs)**: Leave empty (or add custom domain if you have one)
   - **Custom SSL Certificate**: Default CloudFront Certificate
   - **Supported HTTP Versions**: HTTP/2, HTTP/3
   - **Default Root Object**: Leave empty
   - **Standard Logging**: Off (or enable if you want access logs)
   - **IPv6**: On

6. **Create Distribution**
   - Click **"Create distribution"**
   - Wait 5-10 minutes for deployment (Status: "Deploying" → "Enabled")

7. **Copy Distribution Domain**
   - After deployment, copy the **Distribution domain name**
   - Example: `d1234567890abc.cloudfront.net`

### Step 2: Update S3 Bucket Policy

After creating the distribution, CloudFront will show you a banner with the required S3 bucket policy.

1. **Go to S3 Console** → Your bucket → Permissions → Bucket Policy
2. **Replace or add** this policy (CloudFront provides the exact ARN):

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "AllowCloudFrontServicePrincipal",
            "Effect": "Allow",
            "Principal": {
                "Service": "cloudfront.amazonaws.com"
            },
            "Action": "s3:GetObject",
            "Resource": "arn:aws:s3:::YOUR-BUCKET-NAME/*",
            "Condition": {
                "StringEquals": {
                    "AWS:SourceArn": "arn:aws:cloudfront::YOUR-ACCOUNT-ID:distribution/YOUR-DISTRIBUTION-ID"
                }
            }
        }
    ]
}
```

**Important**: Replace:
- `YOUR-BUCKET-NAME` with your actual bucket name
- `YOUR-ACCOUNT-ID` with your AWS account ID
- `YOUR-DISTRIBUTION-ID` with your CloudFront distribution ID (starts with `E`)

3. **Save changes**

### Step 3: Update Environment Variables

Add the CloudFront domain to your `.env` file:

```bash
# Add this line
CLOUDFRONT_DOMAIN=d1234567890abc.cloudfront.net
```

**Note**: Use only the domain name, not the full URL (no `https://`)

### Step 4: Deploy and Test

1. **Restart your application** to load the new environment variable

2. **Generate a new chart** through your Streamlit app

3. **Verify the URL format**:
   - Old format: `https://your-bucket.s3.amazonaws.com/charts/...?X-Amz-Algorithm=...`
   - New format: `https://d1234567890abc.cloudfront.net/charts/20251226_123456_abc12345.png`

4. **Test URL persistence**:
   - The CloudFront URL should work indefinitely (no expiration)
   - Images are cached at edge locations for better performance

---

## How It Works

### Code Changes Made:

1. **`agent/sql_agent.py`**:
   - Added `cloudfront_domain` parameter to `ChartGeneratorTool.__init__()`
   - Modified `_upload_to_s3()` to generate CloudFront URLs when domain is configured
   - Falls back to pre-signed URLs if CloudFront is not configured
   - Added `CacheControl` header for optimal caching

2. **`main.py`**:
   - Added `cloudfront_domain` to config loading
   - Passed `cloudfront_domain` to SQLAgent initialization

### URL Generation Logic:

```python
if self.cloudfront_domain:
    # Use CloudFront URL (no expiration)
    chart_url = f"https://{self.cloudfront_domain}/{s3_key}"
else:
    # Fallback to pre-signed S3 URL (7-day expiration)
    chart_url = self.s3_client.generate_presigned_url(...)
```

### S3 Upload with Cache Headers:

```python
self.s3_client.put_object(
    Bucket=bucket_name,
    Key=s3_key,
    Body=chart_bytes,
    ContentType='image/png',
    CacheControl='max-age=31536000'  # Cache for 1 year at edge locations
)
```

---

## Benefits Summary

| Feature | S3 Pre-signed URLs | CloudFront URLs |
|---------|-------------------|-----------------|
| **URL Expiration** | 7 days max | Never expires |
| **Performance** | Direct S3 access | Cached at edge locations |
| **Global Delivery** | Single region | 400+ edge locations |
| **Cost** | Higher (more S3 requests) | Lower (cached responses) |
| **Security** | Temporary access | S3 stays private (OAC) |
| **HTTPS** | Yes | Yes (enforced) |
| **Custom Domain** | No | Yes (with ACM certificate) |

---

## Troubleshooting

### Issue: 403 Forbidden from CloudFront

**Cause**: S3 bucket policy not updated or incorrect

**Solution**:
1. Verify bucket policy includes CloudFront OAC
2. Check the `AWS:SourceArn` matches your distribution
3. Ensure bucket doesn't have "Block all public access" enabled for OAC

### Issue: Images not loading

**Cause**: CloudFront distribution not fully deployed

**Solution**:
1. Check distribution status in CloudFront console (should be "Enabled")
2. Wait 5-10 minutes after creation
3. Try invalidating cache: CloudFront → Invalidations → Create `/charts/*`

### Issue: Old images still using S3 URLs

**Cause**: Application not restarted after adding `CLOUDFRONT_DOMAIN`

**Solution**:
1. Restart your Streamlit app
2. Generate a new chart to test
3. Check logs for "Generated CloudFront URL" message

---

## Optional: Custom Domain Setup

If you want to use your own domain (e.g., `charts.yourdomain.com`):

1. **Request ACM Certificate** (in us-east-1 region):
   - AWS Certificate Manager → Request certificate
   - Add domain: `charts.yourdomain.com`
   - Validate via DNS or email

2. **Update CloudFront Distribution**:
   - Alternate Domain Names (CNAMEs): `charts.yourdomain.com`
   - Custom SSL Certificate: Select your ACM certificate

3. **Update DNS**:
   - Add CNAME record: `charts.yourdomain.com` → `d1234567890abc.cloudfront.net`

4. **Update `.env`**:
   ```bash
   CLOUDFRONT_DOMAIN=charts.yourdomain.com
   ```

---

## Cost Considerations

### CloudFront Pricing (as of 2024):
- **Data Transfer Out**: ~$0.085/GB (first 10 TB/month)
- **HTTP/HTTPS Requests**: $0.0075 per 10,000 requests
- **Free Tier**: 1 TB data transfer out, 10M requests/month (first 12 months)

### S3 Pricing Comparison:
- **Without CloudFront**: Every image load = 1 S3 GET request ($0.0004 per 1,000)
- **With CloudFront**: First load = S3 GET, subsequent loads = cached (free)

**Savings**: For frequently accessed images, CloudFront can reduce costs by 80-90%

---

## Maintenance

### Cache Invalidation (if needed):
```bash
aws cloudfront create-invalidation \
  --distribution-id YOUR-DISTRIBUTION-ID \
  --paths "/charts/*"
```

### S3 Lifecycle Policy (optional):
Delete old charts after 30 days to save storage:

```json
{
    "Rules": [
        {
            "Id": "DeleteOldCharts",
            "Status": "Enabled",
            "Prefix": "charts/",
            "Expiration": {
                "Days": 30
            }
        }
    ]
}
```

---

## Summary

✅ **Code updated** to support CloudFront URLs
✅ **Backward compatible** - falls back to pre-signed URLs if CloudFront not configured
✅ **No expiration** - URLs work indefinitely
✅ **Better performance** - global CDN caching
✅ **Lower costs** - reduced S3 requests
✅ **Secure** - S3 bucket stays private with OAC

**Next Steps**:
1. Create CloudFront distribution (10 minutes)
2. Update S3 bucket policy
3. Add `CLOUDFRONT_DOMAIN` to `.env`
4. Restart app and test!
