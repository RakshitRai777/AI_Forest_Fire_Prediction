"""
Prediction wrapper for forest fire ML model.
Provides easy interface for making predictions with the trained model.
"""

import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from core.fire_model import FireModel
from models import WeatherData, VegetationData, Location
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FirePredictor:
    """Wrapper class for making fire risk predictions."""
    
    def __init__(self):
        self.fire_model = FireModel()
        self.model_loaded = False
        
    def initialize(self) -> bool:
        """
        Initialize the predictor by loading the ML model.
        
        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            self.model_loaded = self.fire_model.load_model()
            if self.model_loaded:
                logger.info("Fire predictor initialized successfully")
            else:
                logger.warning("Failed to load ML model - will use rule-based only")
            return self.model_loaded
        except Exception as e:
            logger.error(f"Error initializing predictor: {str(e)}")
            return False
    
    def predict_risk(self, weather_data: WeatherData, 
                    vegetation_data: VegetationData) -> dict:
        """
        Predict fire risk for given weather and vegetation data.
        
        Args:
            weather_data: Current weather data
            vegetation_data: Vegetation and environmental data
            
        Returns:
            Dictionary with prediction results
        """
        try:
            result = {
                'ml_probability': None,
                'ml_available': self.model_loaded,
                'prediction_time': datetime.now().isoformat(),
                'location': weather_data.location.city,
                'features': None
            }
            
            if self.model_loaded:
                # Extract features for logging
                features = self.fire_model.create_model_features(weather_data, vegetation_data)
                result['features'] = {
                    'temperature': features.temperature,
                    'humidity': features.humidity,
                    'wind_speed': features.wind_speed,
                    'rainfall': features.rainfall,
                    'ndvi': features.ndvi,
                    'soil_moisture': features.soil_moisture,
                    'drought_index': features.drought_index,
                    'latitude': features.latitude,
                    'longitude': features.longitude,
                    'month': features.month,
                    'day_of_year': features.day_of_year
                }
                
                # Make prediction
                ml_probability = self.fire_model.predict_fire_probability(
                    weather_data, vegetation_data
                )
                
                if ml_probability is not None:
                    result['ml_probability'] = ml_probability
                    logger.info(f"ML prediction for {weather_data.location.city}: {ml_probability:.3f}")
                else:
                    logger.warning("ML prediction failed")
            else:
                logger.info("ML model not available - using rule-based analysis only")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in risk prediction: {str(e)}")
            return {
                'ml_probability': None,
                'ml_available': False,
                'error': str(e),
                'prediction_time': datetime.now().isoformat()
            }
    
    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        return self.fire_model.get_model_info()
    
    def batch_predict(self, data_points: list) -> list:
        """
        Make predictions for multiple data points.
        
        Args:
            data_points: List of tuples (weather_data, vegetation_data)
            
        Returns:
            List of prediction results
        """
        results = []
        for i, (weather_data, vegetation_data) in enumerate(data_points):
            logger.info(f"Processing prediction {i+1}/{len(data_points)}")
            result = self.predict_risk(weather_data, vegetation_data)
            results.append(result)
        return results
    
    def validate_input_data(self, weather_data: WeatherData, 
                           vegetation_data: VegetationData) -> bool:
        """
        Validate input data before making predictions.
        
        Args:
            weather_data: Weather data to validate
            vegetation_data: Vegetation data to validate
            
        Returns:
            True if data is valid, False otherwise
        """
        try:
            # Validate weather data
            if not (-50 <= weather_data.temperature <= 60):
                logger.warning(f"Invalid temperature: {weather_data.temperature}")
                return False
            
            if not (0 <= weather_data.humidity <= 100):
                logger.warning(f"Invalid humidity: {weather_data.humidity}")
                return False
            
            if not (0 <= weather_data.wind_speed <= 200):
                logger.warning(f"Invalid wind speed: {weather_data.wind_speed}")
                return False
            
            if not (0 <= weather_data.rainfall <= 500):
                logger.warning(f"Invalid rainfall: {weather_data.rainfall}")
                return False
            
            # Validate vegetation data
            if not (0 <= vegetation_data.ndvi <= 1):
                logger.warning(f"Invalid NDVI: {vegetation_data.ndvi}")
                return False
            
            if not (0 <= vegetation_data.soil_moisture <= 100):
                logger.warning(f"Invalid soil moisture: {vegetation_data.soil_moisture}")
                return False
            
            if not (0 <= vegetation_data.drought_index <= 1):
                logger.warning(f"Invalid drought index: {vegetation_data.drought_index}")
                return False
            
            # Validate location
            if not (-90 <= weather_data.location.latitude <= 90):
                logger.warning(f"Invalid latitude: {weather_data.location.latitude}")
                return False
            
            if not (-180 <= weather_data.location.longitude <= 180):
                logger.warning(f"Invalid longitude: {weather_data.location.longitude}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating input data: {str(e)}")
            return False


def create_sample_data():
    """Create sample data for testing."""
    location = Location(
        city="Test City",
        latitude=40.7128,
        longitude=-74.0060,
        country="USA"
    )
    
    weather_data = WeatherData(
        temperature=35.0,
        humidity=25.0,
        wind_speed=20.0,
        rainfall=0.0,
        timestamp=datetime.now(),
        location=location
    )
    
    vegetation_data = VegetationData(
        ndvi=0.3,
        soil_moisture=15.0,
        drought_index=0.8,
        vegetation_type="Grassland"
    )
    
    return weather_data, vegetation_data


def main():
    """Main function for testing the predictor."""
    print("Forest Fire Prediction System - ML Predictor")
    print("=" * 50)
    
    # Initialize predictor
    predictor = FirePredictor()
    
    if not predictor.initialize():
        print("Failed to initialize predictor. Please train the model first.")
        return
    
    # Get model info
    model_info = predictor.get_model_info()
    print(f"Model Status: {model_info.get('status', 'Unknown')}")
    if model_info.get('model_type'):
        print(f"Model Type: {model_info['model_type']}")
    print()
    
    # Create sample data
    weather_data, vegetation_data = create_sample_data()
    
    # Validate input
    if not predictor.validate_input_data(weather_data, vegetation_data):
        print("Invalid input data!")
        return
    
    # Make prediction
    result = predictor.predict_risk(weather_data, vegetation_data)
    
    print("Prediction Results:")
    print("-" * 20)
    print(f"Location: {result['location']}")
    print(f"ML Available: {result['ml_available']}")
    if result['ml_probability'] is not None:
        print(f"ML Probability: {result['ml_probability']:.3f}")
    else:
        print("ML Probability: Not available")
    
    if result.get('features'):
        print("\nInput Features:")
        for feature, value in result['features'].items():
            print(f"  {feature}: {value}")


if __name__ == "__main__":
    main()
