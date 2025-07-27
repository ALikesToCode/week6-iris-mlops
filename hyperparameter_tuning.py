#!/usr/bin/env python3
"""
Hyperparameter Tuning Module for Iris Classification Pipeline

This module provides comprehensive hyperparameter tuning capabilities using MLflow
for experiment tracking and multiple optimization strategies including Grid Search,
Random Search, and Bayesian Optimization with Optuna.

Author: Abhyudaya B Tharakan 22f3001492
"""

import logging
import os
import json
import time
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import optuna
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV, 
    RandomizedSearchCV, StratifiedKFold
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import joblib

logger = logging.getLogger(__name__)


@dataclass
class HyperparameterConfig:
    """Configuration for hyperparameter tuning experiments."""
    
    # MLflow Configuration
    experiment_name: str = "iris-hyperparameter-tuning"
    tracking_uri: str = "http://localhost:5000"
    artifact_location: str = "mlruns"
    
    # Tuning Configuration
    optimization_strategy: str = "optuna"  # "grid", "random", "optuna"
    n_trials: int = 50
    cv_folds: int = 5
    scoring_metric: str = "accuracy"
    random_state: int = 42
    
    # Model Selection
    models_to_tune: List[str] = None
    
    # Early Stopping
    early_stopping: bool = True
    early_stopping_patience: int = 10
    
    def __post_init__(self):
        if self.models_to_tune is None:
            self.models_to_tune = ["decision_tree", "random_forest", "svm", "logistic_regression"]


class ModelRegistry:
    """Registry of models and their hyperparameter spaces."""
    
    @staticmethod
    def get_model_config(model_name: str) -> Dict[str, Any]:
        """Get model class and hyperparameter space for a given model."""
        
        configs = {
            "decision_tree": {
                "model_class": DecisionTreeClassifier,
                "base_params": {"random_state": 42},
                "param_space": {
                    "max_depth": [3, 5, 7, 10, 15, None],
                    "min_samples_split": [2, 5, 10, 15, 20],
                    "min_samples_leaf": [1, 2, 4, 6, 8],
                    "criterion": ["gini", "entropy"],
                    "max_features": ["sqrt", "log2", None]
                },
                "optuna_space": {
                    "max_depth": ("categorical", [3, 5, 7, 10, 15, None]),
                    "min_samples_split": ("int", 2, 20),
                    "min_samples_leaf": ("int", 1, 10),
                    "criterion": ("categorical", ["gini", "entropy"]),
                    "max_features": ("categorical", ["sqrt", "log2", None])
                }
            },
            "random_forest": {
                "model_class": RandomForestClassifier,
                "base_params": {"random_state": 42},
                "param_space": {
                    "n_estimators": [50, 100, 200, 300],
                    "max_depth": [5, 10, 15, None],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "max_features": ["sqrt", "log2"],
                    "bootstrap": [True, False]
                },
                "optuna_space": {
                    "n_estimators": ("int", 50, 300),
                    "max_depth": ("categorical", [5, 10, 15, None]),
                    "min_samples_split": ("int", 2, 15),
                    "min_samples_leaf": ("int", 1, 8),
                    "max_features": ("categorical", ["sqrt", "log2"]),
                    "bootstrap": ("categorical", [True, False])
                }
            },
            "svm": {
                "model_class": SVC,
                "base_params": {"random_state": 42},
                "param_space": {
                    "C": [0.1, 1, 10, 100],
                    "kernel": ["linear", "rbf", "poly"],
                    "gamma": ["scale", "auto", 0.001, 0.01, 0.1, 1]
                },
                "optuna_space": {
                    "C": ("loguniform", 0.01, 100),
                    "kernel": ("categorical", ["linear", "rbf", "poly"]),
                    "gamma": ("categorical", ["scale", "auto", 0.001, 0.01, 0.1, 1])
                }
            },
            "logistic_regression": {
                "model_class": LogisticRegression,
                "base_params": {"random_state": 42, "max_iter": 1000},
                "param_space": {
                    "C": [0.01, 0.1, 1, 10, 100],
                    "penalty": ["l1", "l2", "elasticnet"],
                    "solver": ["liblinear", "saga"],
                    "l1_ratio": [0.1, 0.5, 0.9]
                },
                "optuna_space": {
                    "C": ("loguniform", 0.01, 100),
                    "penalty": ("categorical", ["l1", "l2"]),
                    "solver": ("categorical", ["liblinear", "saga"])
                }
            }
        }
        
        if model_name not in configs:
            raise ValueError(f"Model {model_name} not supported. Available models: {list(configs.keys())}")
        
        return configs[model_name]


class MLflowTracker:
    """MLflow experiment tracking and management."""
    
    def __init__(self, config: HyperparameterConfig):
        self.config = config
        self.client = None
        self.experiment = None
        self._setup_mlflow()
    
    def _setup_mlflow(self):
        """Setup MLflow tracking server and experiment."""
        try:
            # Set tracking URI
            mlflow.set_tracking_uri(self.config.tracking_uri)
            
            # Create or get experiment
            try:
                self.experiment = mlflow.get_experiment_by_name(self.config.experiment_name)
                if self.experiment is None:
                    experiment_id = mlflow.create_experiment(
                        name=self.config.experiment_name,
                        artifact_location=self.config.artifact_location
                    )
                    self.experiment = mlflow.get_experiment(experiment_id)
            except Exception as e:
                logger.warning(f"Could not create MLflow experiment: {e}")
                # Fallback to default experiment
                self.experiment = mlflow.get_experiment_by_name("Default")
            
            # Set experiment
            mlflow.set_experiment(self.config.experiment_name)
            
            # Initialize client
            self.client = MlflowClient()
            
            logger.info(f"MLflow tracking initialized: {self.config.tracking_uri}")
            logger.info(f"Experiment: {self.config.experiment_name}")
            
        except Exception as e:
            logger.error(f"Failed to setup MLflow: {e}")
            raise
    
    def log_experiment_run(self, model_name: str, params: Dict[str, Any], 
                          metrics: Dict[str, float], model, 
                          additional_artifacts: Optional[Dict[str, str]] = None):
        """Log a single experiment run to MLflow."""
        
        with mlflow.start_run(run_name=f"{model_name}_{int(time.time())}"):
            # Log parameters
            for key, value in params.items():
                mlflow.log_param(key, value)
            
            # Log metrics
            for key, value in metrics.items():
                mlflow.log_metric(key, value)
            
            # Log model
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                registered_model_name=f"iris_{model_name}"
            )
            
            # Log additional artifacts
            if additional_artifacts:
                for name, content in additional_artifacts.items():
                    mlflow.log_text(content, f"{name}.txt")
            
            # Get run info
            run = mlflow.active_run()
            return run.info.run_id
    
    def get_best_run(self, model_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get the best run based on the scoring metric."""
        try:
            runs = self.client.search_runs(
                experiment_ids=[self.experiment.experiment_id],
                filter_string=f"params.model_name = '{model_name}'" if model_name else "",
                order_by=[f"metrics.{self.config.scoring_metric} DESC"]
            )
            
            if runs:
                best_run = runs[0]
                return {
                    "run_id": best_run.info.run_id,
                    "metrics": best_run.data.metrics,
                    "params": best_run.data.params,
                    "model_uri": f"runs:/{best_run.info.run_id}/model"
                }
            return None
            
        except Exception as e:
            logger.error(f"Error getting best run: {e}")
            return None


class HyperparameterTuner:
    """Main hyperparameter tuning class."""
    
    def __init__(self, config: HyperparameterConfig):
        self.config = config
        self.tracker = MLflowTracker(config)
        self.results = {}
        self.best_models = {}
    
    def prepare_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """Prepare data for hyperparameter tuning."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=0.2, 
            stratify=y, 
            random_state=self.config.random_state
        )
        
        logger.info(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
        return X_train, X_test, y_train, y_test
    
    def _optuna_tuning(self, model_name: str, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """Perform Optuna-based hyperparameter tuning."""
        model_config = ModelRegistry.get_model_config(model_name)
        
        def objective(trial):
            # Build hyperparameters
            params = {}
            for param_name, param_config in model_config["optuna_space"].items():
                if param_config[0] == "categorical":
                    params[param_name] = trial.suggest_categorical(param_name, param_config[1])
                elif param_config[0] == "int":
                    params[param_name] = trial.suggest_int(param_name, param_config[1], param_config[2])
                elif param_config[0] == "float":
                    params[param_name] = trial.suggest_float(param_name, param_config[1], param_config[2])
                elif param_config[0] == "loguniform":
                    params[param_name] = trial.suggest_loguniform(param_name, param_config[1], param_config[2])
            
            # Create and train model
            model_params = {**model_config["base_params"], **params}
            model = model_config["model_class"](**model_params)
            
            # Cross-validation
            cv = StratifiedKFold(n_splits=self.config.cv_folds, shuffle=True, random_state=self.config.random_state)
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=self.config.scoring_metric)
            score = scores.mean()
            
            # Log to MLflow
            run_params = {**model_params, "model_name": model_name, "trial_number": trial.number}
            run_metrics = {self.config.scoring_metric: score}
            
            # Fit model for logging
            model.fit(X_train, y_train)
            
            self.tracker.log_experiment_run(
                model_name=f"{model_name}_optuna_{trial.number}",
                params=run_params,
                metrics=run_metrics,
                model=model
            )
            
            return score
        
        # Create study
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.config.random_state)
        )
        
        # Optimize
        logger.info(f"Starting Optuna optimization for {model_name}...")
        study.optimize(objective, n_trials=self.config.n_trials)
        
        # Create best model
        best_params = {**model_config["base_params"], **study.best_params}
        best_model = model_config["model_class"](**best_params)
        best_model.fit(X_train, y_train)
        
        return {
            "best_params": study.best_params,
            "best_score": study.best_value,
            "best_model": best_model,
            "total_runs": self.config.n_trials,
            "study": study
        }
    
    def tune_single_model(self, model_name: str, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """Tune hyperparameters for a single model."""
        logger.info(f"Starting hyperparameter tuning for {model_name} using {self.config.optimization_strategy}")
        
        start_time = time.time()
        result = self._optuna_tuning(model_name, X_train, y_train)
        end_time = time.time()
        result["tuning_time"] = end_time - start_time
        
        logger.info(f"Completed {model_name} tuning in {result['tuning_time']:.2f} seconds")
        logger.info(f"Best {self.config.scoring_metric}: {result['best_score']:.4f}")
        
        return result
    
    def tune_all_models(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """Tune hyperparameters for all configured models."""
        logger.info("Starting hyperparameter tuning for all models...")
        
        results = {}
        for model_name in self.config.models_to_tune:
            try:
                results[model_name] = self.tune_single_model(model_name, X_train, y_train)
                self.results[model_name] = results[model_name]
                self.best_models[model_name] = results[model_name]["best_model"]
            except Exception as e:
                logger.error(f"Error tuning {model_name}: {e}")
                results[model_name] = {"error": str(e)}
        
        return results
    
    def get_best_overall_model(self) -> Tuple[str, Any, Dict[str, float]]:
        """Get the best overall model based on validation performance."""
        best_model_name = None
        best_score = -1
        best_model = None
        
        for model_name, result in self.results.items():
            if "error" not in result:
                score = result.get("best_score", 0)
                if score > best_score:
                    best_score = score
                    best_model_name = model_name
                    best_model = result["best_model"]
        
        if best_model_name:
            return best_model_name, best_model, {"best_score": best_score}
        else:
            raise ValueError("No valid models found in results")