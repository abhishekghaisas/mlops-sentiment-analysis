#!/bin/bash

# Blue-Green Deployment Script
set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

NAMESPACE="mlops"
NEW_VERSION=${1:-"v1.1.0"}

echo "=========================================="
echo "Blue-Green Deployment"
echo "Version: $NEW_VERSION"
echo "=========================================="

# Get current active version
get_active_version() {
    kubectl get service sentiment-api-service -n $NAMESPACE -o jsonpath='{.spec.selector.version}' 2>/dev/null || echo "blue"
}

# Get inactive version
get_inactive_version() {
    ACTIVE=$(get_active_version)
    if [ "$ACTIVE" = "blue" ]; then
        echo "green"
    else
        echo "blue"
    fi
}

ACTIVE=$(get_active_version)
INACTIVE=$(get_inactive_version)

echo -e "\n${YELLOW}Current active: ${ACTIVE}${NC}"
echo -e "${YELLOW}Deploying to: ${INACTIVE}${NC}"

# Step 1: Update inactive deployment
echo -e "\n${YELLOW}[Step 1/6] Updating ${INACTIVE} deployment...${NC}"
kubectl set image deployment/sentiment-api-${INACTIVE} \
    api=sentiment-api:${NEW_VERSION} \
    -n $NAMESPACE

# Step 2: Scale up inactive
echo -e "\n${YELLOW}[Step 2/6] Scaling up ${INACTIVE} to 3 replicas...${NC}"
kubectl scale deployment/sentiment-api-${INACTIVE} --replicas=3 -n $NAMESPACE

# Step 3: Wait for rollout
echo -e "\n${YELLOW}[Step 3/6] Waiting for ${INACTIVE} to be ready...${NC}"
kubectl rollout status deployment/sentiment-api-${INACTIVE} -n $NAMESPACE --timeout=120s

# Step 4: Test inactive environment
echo -e "\n${YELLOW}[Step 4/6] Testing ${INACTIVE} environment...${NC}"
kubectl run test-pod --rm -i --restart=Never --image=curlimages/curl -n $NAMESPACE -- \
    curl -f http://sentiment-api-${INACTIVE}-service/health && \
    echo -e "${GREEN}✓ Health check passed${NC}" || \
    { echo -e "${RED}✗ Health check failed${NC}"; exit 1; }

# Step 5: Switch traffic
echo -e "\n${YELLOW}[Step 5/6] Switching traffic to ${INACTIVE}...${NC}"
kubectl patch service sentiment-api-service -n $NAMESPACE -p \
    "{\"spec\":{\"selector\":{\"version\":\"${INACTIVE}\"}}}"

echo -e "${GREEN}✓ Traffic switched to ${INACTIVE}${NC}"
sleep 10

# Step 6: Scale down old version
echo -e "\n${YELLOW}[Step 6/6] Scaling down ${ACTIVE} (old version)...${NC}"
kubectl scale deployment/sentiment-api-${ACTIVE} --replicas=0 -n $NAMESPACE

echo -e "\n${GREEN}=========================================="
echo "✓ Deployment Complete!"
echo "Active version: ${INACTIVE}"
echo "==========================================${NC}"

# Show status
kubectl get pods -n $NAMESPACE | grep sentiment-api
kubectl get svc sentiment-api-service -n $NAMESPACE -o jsonpath='{.spec.selector.version}'
echo ""
