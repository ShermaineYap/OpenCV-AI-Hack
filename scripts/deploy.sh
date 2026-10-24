#!/usr/bin/env bash
# Deploy BoilGuard from a machine with Docker, AWS SAM CLI and AWS credentials.
#   ./scripts/deploy.sh [region] [alert-email]
set -euo pipefail
REGION="${1:-us-east-1}"
EMAIL="${2:-}"
STACK="${STACK:-boilguard}"
cd "$(dirname "$0")/.."
sam build --template-file aws/template.yaml
OVR="EnableBedrock=1"; [ -n "$EMAIL" ] && OVR="$OVR AlertEmail=$EMAIL"
sam deploy --stack-name "$STACK" --region "$REGION" --resolve-s3 --resolve-image-repos \
  --capabilities CAPABILITY_IAM --no-confirm-changeset --no-fail-on-empty-changeset --parameter-overrides $OVR
URL=$(aws cloudformation describe-stacks --stack-name "$STACK" --region "$REGION" \
      --query "Stacks[0].Outputs[?OutputKey=='AppUrl'].OutputValue" --output text)
echo "BoilGuard is live at: $URL"
curl -sf "${URL}health" && echo
