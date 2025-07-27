#!/bin/bash
"""
Deployment script for Iris API
Builds Docker image and deploys to Kubernetes

Author: Abhyudaya B Tharakan 22f3001492
"""

set -e

# Configuration
PROJECT_ID="${GOOGLE_CLOUD_PROJECT:-steady-triumph-447006-f8}"
CLUSTER_NAME="${GKE_CLUSTER:-iris-api-cluster}"
ZONE="${GKE_ZONE:-asia-south1-a}"
IMAGE_NAME="iris-api"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi
    
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed"
        exit 1
    fi
    
    if ! command -v gcloud &> /dev/null; then
        log_error "gcloud CLI is not installed"
        exit 1
    fi
    
    log_info "Prerequisites check passed"
}

# Build Docker image
build_image() {
    log_info "Building Docker image..."
    
    # Get current git commit hash for tagging
    if git rev-parse --git-dir > /dev/null 2>&1; then
        GIT_HASH=$(git rev-parse --short HEAD)
    else
        GIT_HASH="latest"
    fi
    
    IMAGE_TAG="gcr.io/$PROJECT_ID/$IMAGE_NAME:$GIT_HASH"
    IMAGE_LATEST="gcr.io/$PROJECT_ID/$IMAGE_NAME:latest"
    
    docker build -t "$IMAGE_TAG" .
    docker tag "$IMAGE_TAG" "$IMAGE_LATEST"
    
    log_info "Built image: $IMAGE_TAG"
}

# Push image to GCR
push_image() {
    log_info "Pushing image to Google Container Registry..."
    
    # Configure Docker for GCR
    gcloud auth configure-docker --quiet
    
    docker push "$IMAGE_TAG"
    docker push "$IMAGE_LATEST"
    
    log_info "Image pushed successfully"
}

# Create GKE cluster if it doesn't exist
create_cluster() {
    log_info "Checking if GKE cluster exists..."
    
    if gcloud container clusters describe "$CLUSTER_NAME" --zone="$ZONE" &> /dev/null; then
        log_info "Cluster $CLUSTER_NAME already exists"
    else
        log_info "Creating GKE cluster: $CLUSTER_NAME"
        gcloud container clusters create "$CLUSTER_NAME" \
            --zone="$ZONE" \
            --num-nodes=3 \
            --enable-autoscaling \
            --min-nodes=1 \
            --max-nodes=10 \
            --machine-type=e2-medium \
            --enable-autorepair \
            --enable-autoupgrade \
            --quiet
        
        log_info "Cluster created successfully"
    fi
    
    # Get cluster credentials
    gcloud container clusters get-credentials "$CLUSTER_NAME" --zone="$ZONE"
}

# Deploy to Kubernetes
deploy_to_k8s() {
    log_info "Deploying to Kubernetes..."
    
    # Update deployment with current project ID and image tag
    sed "s/PROJECT_ID/$PROJECT_ID/g" k8s/deployment.yaml > k8s/deployment-temp.yaml
    sed -i "s/:latest/:$GIT_HASH/g" k8s/deployment-temp.yaml
    
    # Apply Kubernetes manifests
    kubectl apply -f k8s/namespace.yaml
    kubectl apply -f k8s/configmap.yaml
    kubectl apply -f k8s/deployment-temp.yaml
    kubectl apply -f k8s/service.yaml
    kubectl apply -f k8s/hpa.yaml
    
    # Clean up temp file
    rm k8s/deployment-temp.yaml
    
    # Wait for deployment to be ready
    log_info "Waiting for deployment to be ready..."
    kubectl rollout status deployment/iris-api -n iris-api --timeout=300s
    
    log_info "Deployment completed successfully"
}

# Show deployment status
show_status() {
    log_info "Deployment Status:"
    echo ""
    echo "Pods:"
    kubectl get pods -n iris-api
    echo ""
    echo "Services:"
    kubectl get services -n iris-api
    echo ""
    echo "HPA:"
    kubectl get hpa -n iris-api
}

# Test deployment
test_deployment() {
    log_info "Testing deployment..."
    
    # Port forward to test the service
    kubectl port-forward service/iris-api-service 8080:80 -n iris-api &
    PF_PID=$!
    
    sleep 5
    
    # Test health endpoint
    if curl -f http://localhost:8080/health; then
        log_info "Health check passed"
    else
        log_error "Health check failed"
    fi
    
    # Clean up port forward
    kill $PF_PID
}

# Main deployment function
main() {
    log_info "Starting deployment of Iris API..."
    
    check_prerequisites
    build_image
    push_image
    create_cluster
    deploy_to_k8s
    show_status
    test_deployment
    
    log_info "Deployment completed successfully!"
    log_info "Access your API at: kubectl port-forward service/iris-api-service 8080:80 -n iris-api"
}

# Run main function
main "$@"