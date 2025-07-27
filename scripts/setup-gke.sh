#!/bin/bash
"""
GKE Cluster Setup Script
Sets up Google Kubernetes Engine cluster and configures gcloud CLI

Author: Abhyudaya B Tharakan 22f3001492
"""

set -e

# Configuration
PROJECT_ID="${GOOGLE_CLOUD_PROJECT:-steady-triumph-447006-f8}"
CLUSTER_NAME="${GKE_CLUSTER:-iris-api-cluster}"
ZONE="${GKE_ZONE:-asia-south1-a}"

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

# Check if gcloud is installed
check_gcloud() {
    if ! command -v gcloud &> /dev/null; then
        log_error "gcloud CLI is not installed. Please install it first:"
        echo "https://cloud.google.com/sdk/docs/install"
        exit 1
    fi
    log_info "gcloud CLI found"
}

# Configure gcloud
configure_gcloud() {
    log_info "Configuring gcloud..."
    
    # Set project
    gcloud config set project "$PROJECT_ID"
    
    # Set default zone
    gcloud config set compute/zone "$ZONE"
    
    # Enable required APIs
    log_info "Enabling required Google Cloud APIs..."
    gcloud services enable container.googleapis.com
    gcloud services enable containerregistry.googleapis.com
    
    log_info "gcloud configuration completed"
}

# Create service account for CI/CD
create_service_account() {
    log_info "Creating service account for CI/CD..."
    
    SA_NAME="iris-api-cicd"
    SA_EMAIL="$SA_NAME@$PROJECT_ID.iam.gserviceaccount.com"
    
    # Create service account if it doesn't exist
    if gcloud iam service-accounts describe "$SA_EMAIL" &> /dev/null; then
        log_info "Service account already exists: $SA_EMAIL"
    else
        gcloud iam service-accounts create "$SA_NAME" \
            --display-name="Iris API CI/CD Service Account"
        log_info "Created service account: $SA_EMAIL"
    fi
    
    # Grant necessary roles
    gcloud projects add-iam-policy-binding "$PROJECT_ID" \
        --member="serviceAccount:$SA_EMAIL" \
        --role="roles/container.developer"
    
    gcloud projects add-iam-policy-binding "$PROJECT_ID" \
        --member="serviceAccount:$SA_EMAIL" \
        --role="roles/storage.admin"
    
    gcloud projects add-iam-policy-binding "$PROJECT_ID" \
        --member="serviceAccount:$SA_EMAIL" \
        --role="roles/container.clusterAdmin"
    
    log_info "Service account configured with necessary permissions"
    
    # Create and download key
    KEY_FILE="$SA_NAME-key.json"
    if [ ! -f "$KEY_FILE" ]; then
        gcloud iam service-accounts keys create "$KEY_FILE" \
            --iam-account="$SA_EMAIL"
        log_info "Service account key saved to: $KEY_FILE"
        log_warn "Add this key to your GitHub secrets as GCP_SA_KEY"
    fi
}

# Create GKE cluster
create_cluster() {
    log_info "Creating GKE cluster: $CLUSTER_NAME"
    
    if gcloud container clusters describe "$CLUSTER_NAME" --zone="$ZONE" &> /dev/null; then
        log_info "Cluster already exists: $CLUSTER_NAME"
    else
        gcloud container clusters create "$CLUSTER_NAME" \
            --zone="$ZONE" \
            --num-nodes=3 \
            --enable-autoscaling \
            --min-nodes=1 \
            --max-nodes=10 \
            --disk-size=40GB \
            --machine-type=e2-medium \
            --enable-autorepair \
            --enable-autoupgrade \
            --enable-network-policy \
            --quiet
        
        log_info "Cluster created successfully"
    fi
}

# Configure kubectl
configure_kubectl() {
    log_info "Configuring kubectl..."
    
    gcloud container clusters get-credentials "$CLUSTER_NAME" --zone="$ZONE"
    
    # Test kubectl access
    if kubectl cluster-info &> /dev/null; then
        log_info "kubectl configured successfully"
    else
        log_error "Failed to configure kubectl"
        exit 1
    fi
}

# Install ingress controller
install_ingress() {
    log_info "Installing NGINX Ingress Controller..."
    
    # Add ingress-nginx helm repo
    helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx || true
    helm repo update
    
    # Install ingress controller
    helm upgrade --install ingress-nginx ingress-nginx/ingress-nginx \
        --namespace ingress-nginx \
        --create-namespace \
        --set controller.service.type=LoadBalancer
    
    log_info "Waiting for ingress controller to be ready..."
    kubectl wait --namespace ingress-nginx \
        --for=condition=ready pod \
        --selector=app.kubernetes.io/component=controller \
        --timeout=300s
    
    log_info "NGINX Ingress Controller installed successfully"
}

# Show cluster information
show_cluster_info() {
    log_info "Cluster Information:"
    echo ""
    echo "Project ID: $PROJECT_ID"
    echo "Cluster Name: $CLUSTER_NAME"
    echo "Zone: $ZONE"
    echo ""
    echo "Cluster Status:"
    gcloud container clusters describe "$CLUSTER_NAME" --zone="$ZONE" --format="value(status)"
    echo ""
    echo "Nodes:"
    kubectl get nodes
    echo ""
    echo "External IP (may take a few minutes to appear):"
    kubectl get services -n ingress-nginx
}

# Main setup function
main() {
    log_info "Setting up GKE cluster for Iris API..."
    
    check_gcloud
    configure_gcloud
    create_service_account
    create_cluster
    configure_kubectl
    
    # Check if helm is available for ingress setup
    if command -v helm &> /dev/null; then
        install_ingress
    else
        log_warn "Helm not found. Skipping ingress controller installation."
        log_warn "Install helm and run: helm upgrade --install ingress-nginx ingress-nginx/ingress-nginx --namespace ingress-nginx --create-namespace"
    fi
    
    show_cluster_info
    
    log_info "GKE setup completed successfully!"
    log_info "You can now deploy your application using: ./scripts/deploy.sh"
}

# Run main function
main "$@"