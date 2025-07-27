#!/bin/bash

# Week 6 - Iris API Deployment Script
# This script builds and deploys the Iris API to Kubernetes

set -e

# Configuration
PROJECT_ID="steady-triumph-447006-f8"
REGION="asia-south1"
IMAGE_NAME="iris-api"
NAMESPACE="iris-api"

echo "🚀 Starting Week 6 Iris API Deployment..."

# Build Docker image
echo "📦 Building Docker image..."
docker build -t gcr.io/${PROJECT_ID}/${IMAGE_NAME}:latest .
echo "✅ Docker image built successfully"

# Configure Docker for GCR
echo "🔐 Configuring Docker for Google Container Registry..."
gcloud auth configure-docker

# Push image to GCR
echo "📤 Pushing image to Google Container Registry..."
docker push gcr.io/${PROJECT_ID}/${IMAGE_NAME}:latest
echo "✅ Image pushed successfully"

# Create namespace if it doesn't exist
echo "🔧 Creating Kubernetes namespace..."
kubectl create namespace ${NAMESPACE} --dry-run=client -o yaml | kubectl apply -f -

# Deploy to Kubernetes
echo "🚢 Deploying to Kubernetes..."
kubectl apply -f k8s/ -n ${NAMESPACE}

# Wait for deployment to be ready
echo "⏳ Waiting for deployment to be ready..."
kubectl rollout status deployment/iris-api -n ${NAMESPACE} --timeout=300s

# Get service details
echo "📋 Deployment Details:"
kubectl get pods -n ${NAMESPACE}
kubectl get services -n ${NAMESPACE}
kubectl get hpa -n ${NAMESPACE}

echo "✅ Deployment completed successfully!"

# Test the deployment
echo "🧪 Testing the deployment..."
kubectl port-forward -n ${NAMESPACE} service/iris-api-service 8080:80 &
PORT_FORWARD_PID=$!

sleep 10

# Test health endpoint
echo "Testing health endpoint..."
curl -f http://localhost:8080/health && echo "✅ Health check passed" || echo "❌ Health check failed"

# Test prediction endpoint
echo "Testing prediction endpoint..."
curl -X POST "http://localhost:8080/predict" \
  -H "Content-Type: application/json" \
  -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}' \
  && echo "✅ Prediction test passed" || echo "❌ Prediction test failed"

# Clean up port forward
kill $PORT_FORWARD_PID

echo "🎉 Week 6 deployment completed and tested successfully!"