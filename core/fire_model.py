"""
Machine Learning model for forest fire prediction.
Handles model loading, prediction, and feature engineering.
"""

import logging
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Any

from models import WeatherData, VegetationData, Location, ModelFeatures
from config import MODEL_FILE, FEATURE_SCALER_FILE

logger = logging.getLogger(__name__)


class FireModel:
    """ML model for forest fire prediction."""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = [
            'temperature', 'humidity', 'wind_speed', 'rainfall',
            'ndvi', 'soil_moisture', 'drought_index',
            'latitude', 'longitude', 'month', 'day_of_year'
        ]
        
    def load_model(self) -> bool:
        """
        Load the trained ML model and scaler.
        
        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            if Path(MODEL_FILE).exists():
                self.model = joblib.load(MODEL_FILE)
                logger.info(f"Loaded ML model from {MODEL_FILE}")
            else:
                logger.warning(f"Model file not found at {MODEL_FILE}")
                return False
            
            if Path(FEATURE_SCALER_FILE).exists():
                self.scaler = joblib.load(FEATURE_SCALER_FILE)
                logger.info(f"Loaded feature scaler from {FEATURE_SCALER_FILE}")
            else:
                logger.warning(f"Scaler file not found at {FEATURE_SCALER_FILE}")
                # Continue without scaler if model exists
                return self.model is not None
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading ML model: {str(e)}")
            return False
    
    def predict_fire_probability(self, weather_data: WeatherData, 
                               vegetation_data: VegetationData) -> Optional[float]:
        """
        Predict fire probability using ML model.
        
        Args:
            weather_data: Current weather data
            vegetation_data: Vegetation and environmental data
            
        Returns:
            Fire probability (0-1 scale) or None if prediction fails
        """
        try:
            if self.model is None:
                logger.warning("ML model not loaded")
                return None
            
            # Extract features
            features = self._extract_features(weather_data, vegetation_data)
            
            # Convert to numpy array
            feature_array = np.array([features])
            
            # Apply scaling if scaler is available
            if self.scaler is not None:
                feature_array = self.scaler.transform(feature_array)
            
            # Make prediction
            probability = self.model.predict_proba(feature_array)[0, 1]
            
            logger.info(f"ML prediction: {probability:.3f}")
            return float(probability)
            
        except Exception as e:
            logger.error(f"Error in ML prediction: {str(e)}")
            return None
    
    def _extract_features(self, weather_data: WeatherData, 
                         vegetation_data: VegetationData) -> List[float]:
        """
        Extract features from weather and vegetation data.
        
        Args:
            weather_data: Current weather data
            vegetation_data: Vegetation and environmental data
            
        Returns:
            List of feature values
        """
        from datetime import datetime
        
        # Basic weather and vegetation features
        features = [
            weather_data.temperature,
            weather_data.humidity,
            weather_data.wind_speed,
            weather_data.rainfall,
            vegetation_data.ndvi,
            vegetation_data.soil_moisture,
            vegetation_data.drought_index,
            weather_data.location.latitude,
            weather_data.location.longitude
        ]
        
        # Temporal features
        now = datetime.now()
        features.extend([
            now.month,
            now.timetuple().tm_yday
        ])
        
        return features
    
    def create_model_features(self, weather_data: WeatherData, 
                            vegetation_data: VegetationData) -> ModelFeatures:
        """
        Create ModelFeatures object from weather and vegetation data.
        
        Args:
            weather_data: Current weather data
            vegetation_data: Vegetation and environmental data
            
        Returns:
            ModelFeatures object
        """
        from datetime import datetime
        
        now = datetime.now()
        
        return ModelFeatures(
            temperature=weather_data.temperature,
            humidity=weather_data.humidity,
            wind_speed=weather_data.wind_speed,
            rainfall=weather_data.rainfall,
            ndvi=vegetation_data.ndvi,
            soil_moisture=vegetation_data.soil_moisture,
            drought_index=vegetation_data.drought_index,
            latitude=weather_data.location.latitude,
            longitude=weather_data.location.longitude,
            month=now.month,
            day_of_year=now.timetuple().tm_yday
        )
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance from the trained model.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        try:
            if self.model is None:
                logger.warning("ML model not loaded")
                return None
            
            if hasattr(self.model, 'feature_importances_'):
                importance = self.model.feature_importances_
                return dict(zip(self.feature_names, importance))
            else:
                logger.warning("Model does not have feature_importances_ attribute")
                return None
                
        except Exception as e:
            logger.error(f"Error getting feature importance: {str(e)}")
            return None
    
    def validate_features(self, features: List[float]) -> bool:
        """
        Validate feature values for reasonableness.
        
        Args:
            features: List of feature values
            
        Returns:
            True if features are valid, False otherwise
        """
        try:
            if len(features) != len(self.feature_names):
                logger.error(f"Expected {len(self.feature_names)} features, got {len(features)}")
                return False
            
            # Temperature check (-50 to 60°C)
            if not (-50 <= features[0] <= 60):
                logger.warning(f"Unreasonable temperature: {features[0]}")
                return False
            
            # Humidity check (0-100%)
            if not (0 <= features[1] <= 100):
                logger.warning(f"Unreasonable humidity: {features[1]}")
                return False
            
            # Wind speed check (0-200 km/h)
            if not (0 <= features[2] <= 200):
                logger.warning(f"Unreasonable wind speed: {features[2]}")
                return False
            
            # Rainfall check (0-500 mm)
            if not (0 <= features[3] <= 500):
                logger.warning(f"Unreasonable rainfall: {features[3]}")
                return False
            
            # NDVI check (0-1)
            if not (0 <= features[4] <= 1):
                logger.warning(f"Unreasonable NDVI: {features[4]}")
                return False
            
            # Soil moisture check (0-100%)
            if not (0 <= features[5] <= 100):
                logger.warning(f"Unreasonable soil moisture: {features[5]}")
                return False
            
            # Drought index check (0-1)
            if not (0 <= features[6] <= 1):
                logger.warning(f"Unreasonable drought index: {features[6]}")
                return False
            
            # Latitude check (-90 to 90)
            if not (-90 <= features[7] <= 90):
                logger.warning(f"Unreasonable latitude: {features[7]}")
                return False
            
            # Longitude check (-180 to 180)
            if not (-180 <= features[8] <= 180):
                logger.warning(f"Unreasonable longitude: {features[8]}")
                return False
            
            # Month check (1-12)
            if not (1 <= features[9] <= 12):
                logger.warning(f"Unreasonable month: {features[9]}")
                return False
            
            # Day of year check (1-366)
            if not (1 <= features[10] <= 366):
                logger.warning(f"Unreasonable day of year: {features[10]}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating features: {str(e)}")
            return False
    
    def is_model_available(self) -> bool:
        """
        Check if the ML model is loaded and ready for predictions.
        
        Returns:
            True if model is available, False otherwise
        """
        return self.model is not None
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        try:
            if self.model is None:
                return {"status": "Model not loaded"}
            
            info = {
                "status": "Model loaded",
                "model_type": type(self.model).__name__,
                "feature_count": len(self.feature_names),
                "feature_names": self.feature_names,
                "scaler_loaded": self.scaler is not None
            }
            
            # Add model-specific information
            if hasattr(self.model, 'n_estimators'):
                info["n_estimators"] = self.model.n_estimators
            
            if hasattr(self.model, 'max_depth'):
                info["max_depth"] = self.model.max_depth
            
            if hasattr(self.model, 'random_state'):
                info["random_state"] = self.model.random_state
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting model info: {str(e)}")
            return {"status": "Error", "error": str(e)}
