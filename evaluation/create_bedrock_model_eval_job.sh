#!/bin/bash

# Amazon Bedrock Model Evaluation Creation Script
# Region: us-west-2
# Evaluation Type: Automated Model-as-a-Judge

# Set variables - Replace these with your actual values
EVALUATION_NAME="capstone-model-evaluation-job-haiku45-20260108xxx"
EVALUATION_DESCRIPTION="Bedrock Model Eval job for Capstone project for Haiku 4.5"
JUDGE_MODEL_ID="anthropic.claude-3-5-sonnet-20241022-v2:0"  # Replace with your judge model
CANDIDATE_MODEL_ID="us.anthropic.claude-haiku-4-5-20251001-v1:0"  # Replace with your candidate model
DATASET_S3_URI="s3://capstone-eval/agent_evaluation_dataset.jsonl"
OUTPUT_S3_URI="s3://capstone-eval/output/capstone-model-evaluation-job-haiku45-20260108225842/"
EVALUATION_ROLE_ARN="arn:aws:iam::175918693907:role/service-role/Amazon-Bedrock-IAM-Role-20251218T163287"
REGION="us-west-2"

# Optional: Set custom metrics (uncomment and modify as needed)
# CUSTOM_METRICS='[{"name":"accuracy","description":"Measures correctness"}]'

echo "Creating Bedrock Model Evaluation..."
echo "Evaluation Name: $EVALUATION_NAME"
echo "Judge Model: $JUDGE_MODEL_ID"
echo "Candidate Model: $CANDIDATE_MODEL_ID"
echo "Region: $REGION"

# Create the evaluation job JSON configuration
cat > eval_job_config.json << EOF
{
  "jobName": "$EVALUATION_NAME",
  "jobDescription": "$EVALUATION_DESCRIPTION",
  "roleArn": "$EVALUATION_ROLE_ARN",
  "applicationType": "ModelEvaluation",
  "evaluationConfig": {
    "automated": {
      "datasetMetricConfigs": [
        {
          "taskType": "General",
          "dataset": {
            "name": "agent_evaluation_dataset",
            "datasetLocation": {
              "s3Uri": "$DATASET_S3_URI"
            }
          },
          "metricNames": [
            "Builtin.Correctness",
            "Builtin.Completeness",
            "Builtin.Faithfulness",
            "Builtin.Relevance",
            "Builtin.Coherence",
            "Builtin.FollowingInstructions",
            "Builtin.Harmfulness"
          ]
        }
      ],
      "evaluatorModelConfig": {
        "bedrockEvaluatorModels": [
          {
            "modelIdentifier": "$JUDGE_MODEL_ID"
          }
        ]
      }
    }
  },
  "inferenceConfig": {
    "models": [
      {
        "bedrockModel": {
          "modelIdentifier": "$CANDIDATE_MODEL_ID",
          "inferenceParams": "{\"inferenceConfig\":{\"maxTokens\":512,\"temperature\":0,\"stopSequences\":[\"stop\"]},\"additionalModelRequestFields\":{\"top_k\": 250}}"
        }
      }
    ]
  },
  "outputDataConfig": {
    "s3Uri": "$OUTPUT_S3_URI"
  }
}
EOF

echo "Generated evaluation job configuration:"
cat eval_job_config.json

# Create the evaluation job
echo "Submitting evaluation job to AWS Bedrock..."
aws bedrock create-evaluation-job --cli-input-json file://eval_job_config.json --region "$REGION"

# Check if the command was successful
if [ $? -eq 0 ]; then
    echo "✅ Evaluation job created successfully!"
    
    # Extract job ARN from the response
    JOB_ARN=$(aws bedrock create-evaluation-job --cli-input-json file://eval_job_config.json --region "$REGION" --query 'jobArn' --output text 2>/dev/null || echo "")
    
    if [ -n "$JOB_ARN" ]; then
        echo "Job ARN: $JOB_ARN"
        echo "You can monitor the job status with:"
        echo "aws bedrock get-evaluation-job --job-identifier $JOB_ARN --region $REGION"
    else
        echo "You can monitor the job status with:"
        echo "aws bedrock get-evaluation-job --job-identifier $EVALUATION_NAME --region $REGION"
    fi
    
    # Clean up temporary config file
    rm -f eval_job_config.json
    echo "Cleaned up temporary configuration file."
else
    echo "❌ Failed to create evaluation job. Please check your configuration."
    echo "Configuration file saved as eval_job_config.json for debugging."
    exit 1
fi

echo "Script completed."
