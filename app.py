#!/usr/bin/env python3
"""
Iris Classification API
A FastAPI-based microservice for iris flower classification with MLflow integration.

Author: Abhyudaya B Tharakan 22f3001492
"""

import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
import uvicorn
import mlflow
import mlflow.sklearn

# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import hyperparameter tuning module, make it optional for basic functionality
try:
    from hyperparameter_tuning import HyperparameterTuner, HyperparameterConfig
    HYPERPARAMETER_TUNING_AVAILABLE = True
    logger.info("Hyperparameter tuning module loaded successfully")
except ImportError as e:
    logger.warning(f"Hyperparameter tuning module not available: {e}")
    logger.warning("Tuning endpoints will be disabled")
    HYPERPARAMETER_TUNING_AVAILABLE = False
    
    # Create dummy classes to prevent errors
    class HyperparameterTuner:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("Hyperparameter tuning not available")
    
    class HyperparameterConfig:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("Hyperparameter tuning not available")

app = FastAPI(
    title="Iris Classification API",
    description="A machine learning API for iris flower classification with MLflow integration",
    version="1.0.0"
)

class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., description="Sepal length in cm", ge=0, le=10)
    sepal_width: float = Field(..., description="Sepal width in cm", ge=0, le=10)
    petal_length: float = Field(..., description="Petal length in cm", ge=0, le=10)
    petal_width: float = Field(..., description="Petal width in cm", ge=0, le=10)

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: Dict[str, float]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    api_version: str
    mlflow_tracking_uri: Optional[str] = None

class TuningRequest(BaseModel):
    models: List[str] = Field(default=["decision_tree", "random_forest"], description="Models to tune")
    n_trials: int = Field(default=20, description="Number of optimization trials", ge=1, le=100)
    cv_folds: int = Field(default=5, description="Cross-validation folds", ge=3, le=10)

class TuningResponse(BaseModel):
    job_id: str
    status: str
    message: str

class TuningStatusResponse(BaseModel):
    job_id: str
    status: str
    progress: Optional[Dict[str, Any]] = None
    results: Optional[Dict[str, Any]] = None
    best_model: Optional[str] = None

class IrisClassifier:
    def __init__(self):
        self.model = None
        self.model_path = "artifacts/model.joblib"
        self.classes = ['setosa', 'versicolor', 'virginica']
        self.mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        self.tuning_jobs = {}  # Store background tuning jobs
        self._setup_mlflow()
        self.load_model()
    
    def _setup_mlflow(self):
        """Setup MLflow tracking."""
        try:
            mlflow.set_tracking_uri(self.mlflow_tracking_uri)
            logger.info(f"MLflow tracking URI set to: {self.mlflow_tracking_uri}")
        except Exception as e:
            logger.warning(f"Could not connect to MLflow: {e}")
    
    def load_model(self):
        """Load the trained model from artifacts"""
        try:
            if Path(self.model_path).exists():
                self.model = joblib.load(self.model_path)
                logger.info(f"Model loaded successfully from {self.model_path}")
            else:
                logger.warning(f"Model file not found at {self.model_path}, training a simple model")
                self.train_simple_model()
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.train_simple_model()
    
    def train_simple_model(self):
        """Train a simple model if none is available"""
        try:
            from sklearn.tree import DecisionTreeClassifier
            from sklearn.model_selection import train_test_split
            
            # Load data
            data_path = "data/iris.csv"
            if Path(data_path).exists():
                data = pd.read_csv(data_path)
                X = data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
                y = data['species']
                
                # Train model
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                self.model = DecisionTreeClassifier(random_state=42)
                self.model.fit(X_train, y_train)
                
                # Save model
                Path("artifacts").mkdir(exist_ok=True)
                joblib.dump(self.model, self.model_path)
                logger.info("Simple model trained and saved successfully")
            else:
                logger.error(f"Data file not found at {data_path}")
                raise FileNotFoundError(f"Data file not found at {data_path}")
        except Exception as e:
            logger.error(f"Error training simple model: {e}")
            raise
    
    def predict(self, features: IrisFeatures) -> PredictionResponse:
        """Make prediction for given features"""
        if self.model is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        try:
            # Prepare features
            feature_array = np.array([[
                features.sepal_length,
                features.sepal_width,
                features.petal_length,
                features.petal_width
            ]])
            
            # Make prediction
            prediction = self.model.predict(feature_array)[0]
            probabilities = self.model.predict_proba(feature_array)[0]
            
            # Format response
            prob_dict = {class_name: float(prob) for class_name, prob in zip(self.classes, probabilities)}
            confidence = float(max(probabilities))
            
            return PredictionResponse(
                prediction=prediction,
                confidence=confidence,
                probabilities=prob_dict
            )
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
    
    def start_hyperparameter_tuning(self, models: List[str], n_trials: int, cv_folds: int) -> str:
        """Start hyperparameter tuning in the background."""
        job_id = f"tuning_{int(time.time())}"
        
        # Initialize tuning job status
        self.tuning_jobs[job_id] = {
            "status": "running",
            "start_time": time.time(),
            "progress": {},
            "results": None,
            "best_model": None
        }
        
        try:
            # Load data
            data_path = "data/iris.csv"
            if Path(data_path).exists():
                data = pd.read_csv(data_path)
                X = data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
                y = data['species']
                
                # Configure tuning
                config = HyperparameterConfig(
                    models_to_tune=models,
                    n_trials=n_trials,
                    cv_folds=cv_folds,
                    tracking_uri=self.mlflow_tracking_uri
                )
                
                # Create tuner
                tuner = HyperparameterTuner(config)
                
                # Prepare data
                X_train, X_test, y_train, y_test = tuner.prepare_data(X, y)
                
                # Tune models
                results = tuner.tune_all_models(X_train, y_train)
                
                # Get best model
                best_model_name, best_model, best_metrics = tuner.get_best_overall_model()
                
                # Save best model
                if best_model:
                    Path("artifacts").mkdir(exist_ok=True)
                    best_model_path = Path("artifacts") / "best_model.joblib"
                    joblib.dump(best_model, best_model_path)
                    
                    # Update current model
                    self.model = best_model
                    logger.info(f"Updated model to best performing: {best_model_name}")
                
                # Update job status
                self.tuning_jobs[job_id].update({
                    "status": "completed",
                    "end_time": time.time(),
                    "results": results,
                    "best_model": best_model_name,
                    "best_metrics": best_metrics
                })
                
            else:
                raise FileNotFoundError(f"Data file not found: {data_path}")
                
        except Exception as e:
            logger.error(f"Error in hyperparameter tuning: {e}")
            self.tuning_jobs[job_id].update({
                "status": "failed",
                "end_time": time.time(),
                "error": str(e)
            })
        
        return job_id
    
    def get_tuning_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a tuning job."""
        return self.tuning_jobs.get(job_id)

# Initialize classifier
classifier = IrisClassifier()

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {"message": "Iris Classification API", "version": "1.0.0"}

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model_loaded=classifier.model is not None,
        api_version="1.0.0",
        mlflow_tracking_uri=classifier.mlflow_tracking_uri
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(features: IrisFeatures):
    """Predict iris species from features"""
    try:
        return classifier.predict(features)
    except Exception as e:
        logger.error(f"Prediction endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=List[PredictionResponse])
async def predict_batch(features_list: List[IrisFeatures]):
    """Predict iris species for multiple samples"""
    try:
        results = []
        for features in features_list:
            result = classifier.predict(features)
            results.append(result)
        return results
    except Exception as e:
        logger.error(f"Batch prediction endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tune", response_model=TuningResponse)
async def start_tuning(request: TuningRequest, background_tasks: BackgroundTasks):
    """Start hyperparameter tuning in the background"""
    if not HYPERPARAMETER_TUNING_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Hyperparameter tuning service is not available. Please ensure hyperparameter_tuning module is installed."
        )
    
    try:
        # Validate models
        valid_models = ["decision_tree", "random_forest", "svm", "logistic_regression"]
        invalid_models = [m for m in request.models if m not in valid_models]
        if invalid_models:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid models: {invalid_models}. Valid models: {valid_models}"
            )
        
        # Start tuning in background
        job_id = classifier.start_hyperparameter_tuning(
            models=request.models,
            n_trials=request.n_trials,
            cv_folds=request.cv_folds
        )
        
        return TuningResponse(
            job_id=job_id,
            status="started",
            message=f"Hyperparameter tuning started with job ID: {job_id}"
        )
    except Exception as e:
        logger.error(f"Error starting tuning: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tune/{job_id}", response_model=TuningStatusResponse)
async def get_tuning_status(job_id: str):
    """Get status of hyperparameter tuning job"""
    if not HYPERPARAMETER_TUNING_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Hyperparameter tuning service is not available."
        )
    
    try:
        status = classifier.get_tuning_status(job_id)
        if status is None:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        
        return TuningStatusResponse(
            job_id=job_id,
            status=status["status"],
            progress=status.get("progress"),
            results=status.get("results"),
            best_model=status.get("best_model")
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting tuning status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/mlflow/experiments")
async def get_mlflow_experiments():
    """Get MLflow experiments"""
    try:
        client = mlflow.tracking.MlflowClient(tracking_uri=classifier.mlflow_tracking_uri)
        experiments = client.search_experiments()
        return {
            "experiments": [
                {
                    "experiment_id": exp.experiment_id,
                    "name": exp.name,
                    "lifecycle_stage": exp.lifecycle_stage,
                    "artifact_location": exp.artifact_location
                }
                for exp in experiments
            ]
        }
    except Exception as e:
        logger.error(f"Error getting MLflow experiments: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/mlflow/runs/{experiment_id}")
async def get_mlflow_runs(experiment_id: str, limit: int = 10):
    """Get MLflow runs for an experiment"""
    try:
        client = mlflow.tracking.MlflowClient(tracking_uri=classifier.mlflow_tracking_uri)
        runs = client.search_runs(
            experiment_ids=[experiment_id],
            max_results=limit,
            order_by=["start_time DESC"]
        )
        
        return {
            "runs": [
                {
                    "run_id": run.info.run_id,
                    "status": run.info.status,
                    "start_time": run.info.start_time,
                    "end_time": run.info.end_time,
                    "metrics": run.data.metrics,
                    "params": run.data.params
                }
                for run in runs
            ]
        }
    except Exception as e:
        logger.error(f"Error getting MLflow runs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)