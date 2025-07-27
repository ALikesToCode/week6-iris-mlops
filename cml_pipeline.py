#!/usr/bin/env python3
"""
CML Pipeline for Iris Classification MLOps
Author: Abhyudaya B Tharakan 22f3001492

This script implements a comprehensive CML pipeline with:
- Model training and hyperparameter tuning
- Performance evaluation and metrics
- Model validation and testing
- Automated reporting for CI/CD integration
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

# Configure matplotlib for headless environment
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix,
    precision_recall_fscore_support
)
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CMLPipeline:
    """CML Pipeline for comprehensive ML model management"""
    
    def __init__(self):
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.metrics = {}
        self.artifacts_dir = "artifacts"
        
        # Create artifacts directory
        os.makedirs(self.artifacts_dir, exist_ok=True)
        
    def load_data(self):
        """Load and split the Iris dataset"""
        logger.info("Loading Iris dataset...")
        
        # Load iris dataset
        iris = load_iris()
        X, y = iris.data, iris.target
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info(f"Dataset loaded: {len(X)} samples, {X.shape[1]} features")
        logger.info(f"Training set: {len(self.X_train)} samples")
        logger.info(f"Test set: {len(self.X_test)} samples")
        
        return iris
        
    def train_model(self):
        """Train the RandomForest model with hyperparameter tuning"""
        logger.info("Training RandomForest model...")
        
        # Use optimized hyperparameters from previous tuning
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42
        )
        
        # Train the model
        self.model.fit(self.X_train, self.y_train)
        
        # Save the model
        model_path = os.path.join(self.artifacts_dir, "model.joblib")
        joblib.dump(self.model, model_path)
        logger.info(f"Model saved to {model_path}")
        
    def evaluate_model(self):
        """Comprehensive model evaluation"""
        logger.info("Evaluating model performance...")
        
        # Make predictions
        y_pred = self.model.predict(self.X_test)
        y_prob = self.model.predict_proba(self.X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(self.y_test, y_pred, average='weighted')
        
        # Cross-validation score
        cv_scores = cross_val_score(self.model, self.X_train, self.y_train, cv=5)
        
        # Store metrics
        self.metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'cv_mean': float(cv_scores.mean()),
            'cv_std': float(cv_scores.std()),
            'test_samples': int(len(self.y_test)),
            'correct_predictions': int(sum(y_pred == self.y_test)),
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Model accuracy: {accuracy:.4f}")
        logger.info(f"Cross-validation score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Save metrics
        metrics_path = os.path.join(self.artifacts_dir, "metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        logger.info(f"Metrics saved to {metrics_path}")
        
        return self.metrics
    
    def generate_visualizations(self):
        """Generate performance visualizations"""
        logger.info("Generating model visualizations...")
        
        # Make predictions for visualization
        y_pred = self.model.predict(self.X_test)
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Iris Classification Model Performance', fontsize=16, fontweight='bold')
        
        # 1. Confusion Matrix
        cm = confusion_matrix(self.y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,0])
        axes[0,0].set_title('Confusion Matrix')
        axes[0,0].set_xlabel('Predicted')
        axes[0,0].set_ylabel('Actual')
        
        # 2. Feature Importance
        feature_names = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
        importances = self.model.feature_importances_
        axes[0,1].bar(feature_names, importances)
        axes[0,1].set_title('Feature Importance')
        axes[0,1].set_ylabel('Importance')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # 3. Model Metrics
        metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        metrics_values = [self.metrics['accuracy'], self.metrics['precision'], 
                         self.metrics['recall'], self.metrics['f1_score']]
        bars = axes[1,0].bar(metrics_names, metrics_values, color=['skyblue', 'lightgreen', 'lightcoral', 'lightyellow'])
        axes[1,0].set_title('Model Performance Metrics')
        axes[1,0].set_ylabel('Score')
        axes[1,0].set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, metrics_values):
            axes[1,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                          f'{value:.3f}', ha='center', va='bottom')
        
        # 4. Cross-validation scores
        cv_scores = cross_val_score(self.model, self.X_train, self.y_train, cv=5)
        axes[1,1].plot(range(1, 6), cv_scores, 'bo-', linewidth=2, markersize=8)
        axes[1,1].axhline(y=cv_scores.mean(), color='r', linestyle='--', 
                         label=f'Mean: {cv_scores.mean():.3f}')
        axes[1,1].set_title('Cross-Validation Scores')
        axes[1,1].set_xlabel('Fold')
        axes[1,1].set_ylabel('Accuracy')
        axes[1,1].set_ylim(0, 1)
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        plot_path = os.path.join(self.artifacts_dir, "model_performance.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Performance visualization saved to {plot_path}")
        
    def generate_cml_report(self):
        """Generate CML report with model performance"""
        logger.info("Generating CML report...")
        
        report_lines = [
            "# 🤖 ML Model Performance Report",
            "",
            "## 📊 Model Training Results",
            "",
            "### Model Information",
            f"- **Algorithm**: Random Forest Classifier",
            f"- **Training timestamp**: {self.metrics['timestamp']}",
            f"- **Model saved to**: `artifacts/model.joblib`",
            "",
            "### Performance Metrics",
            f"- **Accuracy**: {self.metrics['accuracy']:.4f}",
            f"- **Precision**: {self.metrics['precision']:.4f}",
            f"- **Recall**: {self.metrics['recall']:.4f}",
            f"- **F1-Score**: {self.metrics['f1_score']:.4f}",
            "",
            "### Cross-Validation Results",
            f"- **CV Mean Accuracy**: {self.metrics['cv_mean']:.4f}",
            f"- **CV Standard Deviation**: {self.metrics['cv_std']:.4f}",
            "",
            "### Test Set Performance",
            f"- **Test samples**: {self.metrics['test_samples']}",
            f"- **Correct predictions**: {self.metrics['correct_predictions']}",
            f"- **Test accuracy**: {self.metrics['accuracy']:.4f}",
            "",
            "## 📈 Model Performance Visualization",
            "",
            "![Model Performance](artifacts/model_performance.png)",
            "",
            "## ✅ Model Validation Status",
            ""
        ]
        
        # Add validation status
        if self.metrics['accuracy'] >= 0.95:
            report_lines.extend([
                "🎉 **EXCELLENT**: Model performance exceeds 95% accuracy!",
                "✅ Model is ready for production deployment."
            ])
        elif self.metrics['accuracy'] >= 0.90:
            report_lines.extend([
                "✅ **GOOD**: Model performance meets production standards.",
                "📦 Model approved for deployment."
            ])
        else:
            report_lines.extend([
                "⚠️  **WARNING**: Model accuracy below 90%.",
                "🔍 Consider additional hyperparameter tuning or feature engineering."
            ])
        
        # Save report
        report_content = "\n".join(report_lines)
        report_path = "cml_report.md"
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        logger.info(f"CML report generated: {report_path}")
        return report_path
    
    def run_pipeline(self):
        """Run the complete CML pipeline"""
        logger.info("Starting CML Pipeline...")
        
        try:
            # Load data
            iris_data = self.load_data()
            
            # Train model
            self.train_model()
            
            # Evaluate model
            metrics = self.evaluate_model()
            
            # Generate visualizations
            self.generate_visualizations()
            
            # Generate CML report
            report_path = self.generate_cml_report()
            
            logger.info("CML Pipeline completed successfully!")
            return True, metrics, report_path
            
        except Exception as e:
            logger.error(f"CML Pipeline failed: {str(e)}")
            return False, None, None

def main():
    """Main function to run the CML pipeline"""
    pipeline = CMLPipeline()
    success, metrics, report_path = pipeline.run_pipeline()
    
    if success:
        print(f"✅ CML Pipeline completed successfully!")
        print(f"📊 Model accuracy: {metrics['accuracy']:.4f}")
        print(f"📝 Report generated: {report_path}")
        return 0
    else:
        print("❌ CML Pipeline failed!")
        return 1

if __name__ == "__main__":
    exit(main())