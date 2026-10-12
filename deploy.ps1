# PowerShell deployment script for Windows
param (
    [Parameter(Mandatory=$true, Position=0)]
    [string]$Bucket,

    [Parameter(Mandatory=$false, Position=1)]
    [string]$Stack = "hvac-valve-optimiser",

    [Parameter(Mandatory=$false)]
    [string]$Region = $env:AWS_REGION
)

$ErrorActionPreference = "Stop"

if (-not $Region) {
    $Region = aws configure get region
}
if (-not $Region) {
    Write-Error "AWS Region not configured. Set AWS_REGION or run 'aws configure'."
}

Write-Host "==> building SAM application"
sam build --template infra/template.yaml --build-dir .aws-sam

Write-Host "==> deploying stack $Stack to $Region"
sam deploy `
  --template-file .aws-sam/build/template.yaml `
  --stack-name "$Stack" `
  --region "$Region" `
  --capabilities CAPABILITY_NAMED_IAM `
  --resolve-s3 `
  --no-fail-on-empty-changeset `
  --parameter-overrides DataBucketName="$Bucket"

$Role = aws cloudformation describe-stacks --stack-name "$Stack" --region "$Region" `
  --query "Stacks[0].Outputs[?OutputKey=='SageMakerRoleArn'].OutputValue" --output text
$Api = aws cloudformation describe-stacks --stack-name "$Stack" --region "$Region" `
  --query "Stacks[0].Outputs[?OutputKey=='DashboardEndpoint'].OutputValue" --output text

Write-Host "==> stack ready"
Write-Host "    bucket         s3://$Bucket"
Write-Host "    sagemaker role $Role"
Write-Host "    dashboard API  $Api"
Write-Host ""
Write-Host "Next:"
Write-Host "  aws s3 cp ./plc_data s3://$Bucket/training/ --recursive"
Write-Host "  python scripts/launch.py --bucket $Bucket --role $Role"
