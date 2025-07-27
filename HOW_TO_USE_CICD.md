# How to Use the CI/CD Pipeline - Week 6 Iris MLOps

## 🎯 Overview

This repository contains a complete CI/CD pipeline for the Week 6 Iris Classification MLOps project. The pipeline automatically builds, tests, and deploys your application to Google Kubernetes Engine.

## 📁 Repository Structure

```
week6-iris-mlops/
├── .github/workflows/main.yml    # Main CI/CD pipeline
├── app.py                        # FastAPI application
├── hyperparameter_tuning.py      # ML training pipeline
├── Dockerfile                    # Container definition
├── k8s/                         # Kubernetes manifests
├── tests/                       # Unit tests
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

## 🚀 Quick Start Guide

### 1. Repository Setup

The repository is already configured and ready to use:
- **Repository URL**: https://github.com/ALikesToCode/week6-iris-mlops
- **CI/CD Pipeline**: Automatically triggered on push/PR
- **Secrets**: Already configured for GCP deployment

### 2. Triggering Deployments

The pipeline automatically triggers based on branch operations:

#### Development Deployment (Staging)
```bash
# Create and switch to develop branch
git checkout -b develop
git push origin develop

# Make changes and push to develop
git add .
git commit -m "Add new feature"
git push origin develop
```

#### Production Deployment
```bash
# Merge develop to main for production deployment
git checkout main
git merge develop
git push origin main
```

## 🔄 CI/CD Pipeline Stages

### Stage 1: Code Quality & Testing
- **Triggers**: On every push and pull request
- **Actions**:
  - Python dependency installation
  - Code formatting check (Black)
  - Import sorting check (isort)
  - Linting (flake8)
  - Unit tests (pytest)

### Stage 2: Container Build
- **Triggers**: On push to main/develop branches
- **Actions**:
  - Docker image building
  - Push to Google Artifact Registry
  - Image tagging with commit SHA
  - Build caching for optimization

### Stage 3: Development Deployment
- **Triggers**: On push to `develop` branch
- **Actions**:
  - Deploy to `development` namespace in GKE
  - Run smoke tests
  - Validate API endpoints

### Stage 4: Production Deployment
- **Triggers**: On push to `main` branch
- **Actions**:
  - Blue-green deployment to `iris-api` namespace
  - Production validation tests
  - Traffic switching
  - Cleanup of old deployment

## 🛠️ How to Use

### For Development Work

1. **Clone the repository**:
```bash
git clone https://github.com/ALikesToCode/week6-iris-mlops.git
cd week6-iris-mlops
```

2. **Create a feature branch**:
```bash
git checkout -b feature/your-feature-name
```

3. **Make your changes** and test locally:
```bash
pip install -r requirements.txt
python app.py  # Test locally
pytest tests/  # Run tests
```

4. **Push your changes**:
```bash
git add .
git commit -m "Descriptive commit message"
git push origin feature/your-feature-name
```

5. **Create a Pull Request** to `develop` branch

### For Production Release

1. **Merge to develop** first for staging tests:
```bash
git checkout develop
git merge feature/your-feature-name
git push origin develop
```

2. **Verify development deployment** by checking:
   - GitHub Actions logs
   - Application health in development namespace

3. **Promote to production**:
```bash
git checkout main
git merge develop
git push origin main
```

## 📊 Monitoring Your Deployments

### GitHub Actions Dashboard
- Go to: https://github.com/ALikesToCode/week6-iris-mlops/actions
- Monitor pipeline execution in real-time
- View logs for debugging

### Command Line Monitoring
```bash
# Check latest workflow runs
gh run list --limit 5

# View specific run details
gh run view <run-id>

# Watch running workflow
gh run watch <run-id>
```

### Kubernetes Monitoring
```bash
# Check pods in development
kubectl get pods -n development

# Check pods in production
kubectl get pods -n iris-api

# View logs
kubectl logs -n iris-api deployment/iris-api -f

# Check service status
kubectl get svc -n iris-api
```

## 🧪 Testing Your Changes

### Run Tests Locally
```bash
# Install test dependencies
pip install pytest pytest-cov httpx

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html
```

### Test API Endpoints
```bash
# Health check
curl http://localhost:8000/health

# Single prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'

# Batch prediction
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '[{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}]'
```

## 🔧 Configuration

### Environment Variables
The pipeline uses these environment variables (configured in workflow):
- `PROJECT_ID`: steady-triumph-447006-f8
- `REGION`: asia-south1
- `GAR_LOCATION`: asia-south1-docker.pkg.dev
- `SERVICE_NAME`: iris-api

### GitHub Secrets
These secrets are already configured:
- `GCP_SA_KEY`: Google Cloud service account key
- `MLFLOW_TRACKING_URI`: MLflow tracking server URI

### Kubernetes Namespaces
- `development`: For staging deployments
- `iris-api`: For production deployments

## 🚨 Troubleshooting

### Common Issues

#### 1. Pipeline Fails on Test Stage
```bash
# Check test failures
gh run view <run-id> --log

# Fix locally and re-push
pytest tests/ -v
git add .
git commit -m "Fix failing tests"
git push
```

#### 2. Docker Build Fails
```bash
# Test Docker build locally
docker build -t iris-api:local .
docker run -p 8000:8000 iris-api:local

# Check Dockerfile syntax and dependencies
```

#### 3. Deployment Timeout
```bash
# Check Kubernetes resources
kubectl describe deployment iris-api -n iris-api
kubectl get events -n iris-api --sort-by='.lastTimestamp'

# Check pod logs
kubectl logs -n iris-api deployment/iris-api
```

#### 4. Service Account Issues
```bash
# Verify GCP permissions
gcloud auth list
gcloud projects get-iam-policy steady-triumph-447006-f8
```

### Debug Commands

```bash
# View workflow file
cat .github/workflows/main.yml

# Check repository secrets
gh secret list

# View recent commits
git log --oneline -5

# Check current branch and status
git status
git branch -v
```

## 📈 Advanced Usage

### Custom Deployment Environments

To add a new environment (e.g., staging):

1. **Update workflow** to add new job:
```yaml
deploy-staging:
  name: Deploy to Staging
  runs-on: ubuntu-latest
  needs: [build]
  if: github.ref == 'refs/heads/staging'
  environment: staging
  # ... deployment steps
```

2. **Create namespace** in Kubernetes:
```bash
kubectl create namespace staging
```

3. **Push to staging branch**:
```bash
git checkout -b staging
git push origin staging
```

### Model Versioning

The pipeline automatically versions models using:
- Git commit SHA
- Branch name
- Timestamp
- MLflow experiment tracking

### Blue-Green Deployment

The production deployment uses blue-green strategy:
1. Deploy new version as "green"
2. Test green deployment
3. Switch traffic from blue to green
4. Clean up old blue deployment

## 📋 Checklist for New Features

- [ ] Create feature branch from `develop`
- [ ] Write unit tests for new functionality
- [ ] Test locally with `pytest tests/`
- [ ] Update documentation if needed
- [ ] Push to feature branch
- [ ] Create PR to `develop`
- [ ] Verify development deployment
- [ ] Merge to `main` for production

## 🎉 Success Indicators

Your CI/CD pipeline is working correctly when you see:

✅ **Green checkmarks** in GitHub Actions  
✅ **All tests passing** in the test stage  
✅ **Docker image** successfully built and pushed  
✅ **Deployment** completes without errors  
✅ **Health checks** return successful responses  
✅ **Predictions** work correctly via API  

## 📞 Getting Help

If you encounter issues:

1. **Check GitHub Actions logs** for detailed error messages
2. **Review this documentation** for common solutions
3. **Test changes locally** before pushing
4. **Use kubectl** to debug Kubernetes deployments
5. **Check Google Cloud Console** for GKE cluster status

---

**Repository**: https://github.com/ALikesToCode/week6-iris-mlops  
**CI/CD Pipeline**: Fully automated and production-ready!  
**Status**: ✅ Ready for development and deployment