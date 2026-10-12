#!/usr/bin/env bash
# Deploy the stack, then train and deploy the model.
#   ./deploy.sh my-unique-plc-bucket
set -euo pipefail

BUCKET="${1:?usage: ./deploy.sh <globally-unique-bucket-name> [stack-name]}"
STACK="${2:-hvac-valve-optimiser}"
REGION="${AWS_REGION:-$(aws configure get region)}"
: "${REGION:?set AWS_REGION or run aws configure}"

echo "==> building SAM application"
sam build --template infra/template.yaml --build-dir .aws-sam

echo "==> deploying stack $STACK to $REGION"
sam deploy \
  --template-file .aws-sam/build/template.yaml \
  --stack-name "$STACK" \
  --region "$REGION" \
  --capabilities CAPABILITY_NAMED_IAM \
  --resolve-s3 \
  --no-fail-on-empty-changeset \
  --parameter-overrides DataBucketName="$BUCKET"

ROLE=$(aws cloudformation describe-stacks --stack-name "$STACK" --region "$REGION" \
  --query "Stacks[0].Outputs[?OutputKey=='SageMakerRoleArn'].OutputValue" --output text)
API=$(aws cloudformation describe-stacks --stack-name "$STACK" --region "$REGION" \
  --query "Stacks[0].Outputs[?OutputKey=='DashboardEndpoint'].OutputValue" --output text)

echo "==> stack ready"
echo "    bucket         s3://$BUCKET"
echo "    sagemaker role $ROLE"
echo "    dashboard API  $API"
echo
echo "Next:"
echo "  aws s3 cp ./plc_data s3://$BUCKET/training/ --recursive"
echo "  python scripts/launch.py --bucket $BUCKET --role $ROLE"
