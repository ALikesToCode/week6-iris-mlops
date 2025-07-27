#!/usr/bin/env python3
"""
Test suite for Iris Classification API

Author: Abhyudaya B Tharakan 22f3001492
"""

import pytest
import sys
import os
from pathlib import Path

# Add parent directory to path to import app
sys.path.append(str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient
from app import app, classifier

client = TestClient(app)

class TestIrisAPI:
    """Test suite for Iris Classification API"""
    
    def test_root_endpoint(self):
        """Test the root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
    
    def test_health_endpoint(self):
        """Test the health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "model_loaded" in data
        assert "api_version" in data
    
    def test_predict_endpoint_valid_input(self):
        """Test prediction with valid input"""
        payload = {
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert "confidence" in data
        assert "probabilities" in data
        assert data["prediction"] in ["setosa", "versicolor", "virginica"]
        assert 0 <= data["confidence"] <= 1
    
    def test_predict_endpoint_invalid_input(self):
        """Test prediction with invalid input"""
        payload = {
            "sepal_length": -1,  # Invalid negative value
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 422  # Validation error
    
    def test_predict_endpoint_missing_fields(self):
        """Test prediction with missing fields"""
        payload = {
            "sepal_length": 5.1,
            "sepal_width": 3.5
            # Missing petal_length and petal_width
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 422  # Validation error
    
    def test_batch_predict_endpoint(self):
        """Test batch prediction endpoint"""
        payload = [
            {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2
            },
            {
                "sepal_length": 7.0,
                "sepal_width": 3.2,
                "petal_length": 4.7,
                "petal_width": 1.4
            }
        ]
        response = client.post("/predict/batch", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        for prediction in data:
            assert "prediction" in prediction
            assert "confidence" in prediction
            assert "probabilities" in prediction
    
    def test_prediction_consistency(self):
        """Test that same input gives consistent predictions"""
        payload = {
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        }
        
        # Make multiple requests
        response1 = client.post("/predict", json=payload)
        response2 = client.post("/predict", json=payload)
        
        assert response1.status_code == 200
        assert response2.status_code == 200
        
        data1 = response1.json()
        data2 = response2.json()
        
        assert data1["prediction"] == data2["prediction"]
        assert data1["confidence"] == data2["confidence"]

class TestIrisClassifier:
    """Test suite for IrisClassifier class"""
    
    def test_classifier_initialization(self):
        """Test classifier initialization"""
        assert classifier.model is not None
        assert classifier.classes == ['setosa', 'versicolor', 'virginica']
    
    def test_model_prediction_shapes(self):
        """Test that model predictions have correct shapes"""
        from app import IrisFeatures
        
        features = IrisFeatures(
            sepal_length=5.1,
            sepal_width=3.5,
            petal_length=1.4,
            petal_width=0.2
        )
        
        result = classifier.predict(features)
        assert len(result.probabilities) == 3  # Three classes
        assert sum(result.probabilities.values()) == pytest.approx(1.0, rel=1e-5)  # Probabilities sum to 1