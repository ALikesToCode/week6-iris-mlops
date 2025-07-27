#!/usr/bin/env python3
"""
MLflow Demo Script for Iris Classification Pipeline

This script demonstrates how to use MLflow for tracking experiments
and hyperparameter tuning.
"""

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time

def demo_basic_mlflow_tracking():
    """Demonstrate basic MLflow tracking."""
    print("🔬 Running Basic MLflow Tracking Demo...")
    
    # Set experiment
    mlflow.set_experiment("iris-demo")
    
    # Load data
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model with MLflow tracking
    with mlflow.start_run(run_name="demo_decision_tree"):
        # Parameters
        max_depth = 5
        random_state = 42
        
        # Log parameters
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("model_type", "DecisionTreeClassifier")
        
        # Train model
        model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        # Log artifact
        with open("demo_results.txt", "w") as f:
            f.write(f"Demo Results:\n")
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall: {recall:.4f}\n")
            f.write(f"F1 Score: {f1:.4f}\n")
        
        mlflow.log_artifact("demo_results.txt")
        
        print(f"✅ Demo completed! Accuracy: {accuracy:.4f}")
        print(f"📊 Check MLflow UI at: http://localhost:5000")

def demo_hyperparameter_comparison():
    """Demonstrate hyperparameter comparison."""
    print("\n🔄 Running Hyperparameter Comparison Demo...")
    
    # Set experiment
    mlflow.set_experiment("iris-hyperparameter-demo")
    
    # Load data
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Test different hyperparameters
    max_depths = [3, 5, 7, 10, None]
    min_samples_splits = [2, 5, 10]
    
    best_accuracy = 0
    best_run_id = None
    
    for max_depth in max_depths:
        for min_samples_split in min_samples_splits:
            with mlflow.start_run(run_name=f"dt_depth_{max_depth}_split_{min_samples_split}"):
                # Log parameters
                mlflow.log_param("max_depth", max_depth)
                mlflow.log_param("min_samples_split", min_samples_split)
                mlflow.log_param("model_type", "DecisionTreeClassifier")
                
                # Train model
                model = DecisionTreeClassifier(
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    random_state=42
                )
                model.fit(X_train, y_train)
                
                # Evaluate
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                # Log metrics
                mlflow.log_metric("accuracy", accuracy)
                
                # Track best model
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_run_id = mlflow.active_run().info.run_id
                
                print(f"  🎯 max_depth={max_depth}, min_samples_split={min_samples_split} -> accuracy={accuracy:.4f}")
    
    print(f"\n🏆 Best accuracy: {best_accuracy:.4f} (Run ID: {best_run_id})")
    print(f"📊 Check MLflow UI at: http://localhost:5000")

if __name__ == "__main__":
    print("🚀 MLflow Demo for Iris Classification Pipeline")
    print("=" * 50)
    
    # Check if MLflow server is running
    try:
        mlflow.set_tracking_uri("http://localhost:5000")
        
        # Run demos
        demo_basic_mlflow_tracking()
        demo_hyperparameter_comparison()
        
        print("\n✅ All demos completed successfully!")
        print("🔍 You can now explore the results in the MLflow UI at: http://localhost:5000")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("🔧 Make sure MLflow server is running: ./start_mlflow.sh")