"""
Vegetation and environmental dryness service.
Provides vegetation indices and drought indicators for fire risk assessment.
"""

import logging
import math
from typing import Optional, Dict, Any
from datetime import datetime, timedelta

from models import VegetationData, Location
from config import VEGETATION_THRESHOLDS

logger = logging.getLogger(__name__)


class VegetationService:
    """Service for vegetation and environmental data analysis."""
    
    def __init__(self):
        self.cache = {}
        
    def get_vegetation_data(self, location: Location, weather_data) -> Optional[VegetationData]:
        """
        Calculate vegetation data based on location and weather conditions.
        
        Args:
            location: Location object
            weather_data: Current weather data
            
        Returns:
            VegetationData object with calculated indices
        """
        cache_key = f"vegetation_{location.latitude}_{location.longitude}_{datetime.now().strftime('%Y-%m-%d')}"
        if cache_key in self.cache:
            logger.info(f"Returning cached vegetation data for {location.city}")
            return self.cache[cache_key]
        
        try:
            # Calculate NDVI (Normalized Difference Vegetation Index)
            # Simplified calculation based on weather patterns and location
            ndvi = self._calculate_ndvi(location, weather_data)
            
            # Calculate soil moisture based on recent rainfall and humidity
            soil_moisture = self._calculate_soil_moisture(weather_data)
            
            # Calculate drought index
            drought_index = self._calculate_drought_index(weather_data, soil_moisture)
            
            # Determine vegetation type based on location and NDVI
            vegetation_type = self._determine_vegetation_type(location, ndvi)
            
            vegetation_data = VegetationData(
                ndvi=ndvi,
                soil_moisture=soil_moisture,
                drought_index=drought_index,
                vegetation_type=vegetation_type
            )
            
            # Cache result
            self.cache[cache_key] = vegetation_data
            logger.info(f"Calculated vegetation data for {location.city}: "
                       f"NDVI={ndvi:.3f}, Soil Moisture={soil_moisture:.1f}%, "
                       f"Drought Index={drought_index:.3f}")
            
            return vegetation_data
            
        except Exception as e:
            logger.error(f"Error calculating vegetation data for {location.city}: {str(e)}")
            return None
    
    def _calculate_ndvi(self, location: Location, weather_data) -> float:
        """
        Calculate NDVI based on location and weather patterns.
        This is a simplified calculation for demonstration purposes.
        
        Args:
            location: Location object
            weather_data: Current weather data
            
        Returns:
            NDVI value (0-1 scale)
        """
        try:
            # Base NDVI calculation considering seasonal patterns
            day_of_year = datetime.now().timetuple().tm_yday
            
            # Seasonal factor (higher in summer, lower in winter for northern hemisphere)
            seasonal_factor = 0.5 + 0.3 * math.cos(2 * math.pi * (day_of_year - 172) / 365)
            
            # Temperature impact on vegetation
            temp_factor = max(0, min(1, (weather_data.temperature - 5) / 30))
            
            # Humidity impact on vegetation
            humidity_factor = weather_data.humidity / 100
            
            # Rainfall impact on vegetation
            rainfall_factor = min(1, weather_data.rainfall / 10)
            
            # Latitude impact (vegetation generally decreases towards poles)
            latitude_factor = max(0.3, 1 - abs(location.latitude) / 90)
            
            # Combine factors
            ndvi = (seasonal_factor * 0.3 + 
                   temp_factor * 0.2 + 
                   humidity_factor * 0.2 + 
                   rainfall_factor * 0.15 + 
                   latitude_factor * 0.15)
            
            # Add some randomness for realism
            import random
            ndvi += random.uniform(-0.05, 0.05)
            
            # Ensure NDVI is within valid range
            return max(0.0, min(1.0, ndvi))
            
        except Exception as e:
            logger.error(f"Error calculating NDVI: {str(e)}")
            return 0.5  # Default middle value
    
    def _calculate_soil_moisture(self, weather_data) -> float:
        """
        Calculate soil moisture based on weather conditions.
        
        Args:
            weather_data: Current weather data
            
        Returns:
            Soil moisture percentage (0-100%)
        """
        try:
            # Base soil moisture from humidity
            base_moisture = weather_data.humidity * 0.6
            
            # Recent rainfall contribution
            rainfall_contribution = min(40, weather_data.rainfall * 4)
            
            # Temperature impact (higher temperature reduces soil moisture)
            temp_impact = max(0, (35 - weather_data.temperature) * 0.5)
            
            # Wind impact (higher wind increases evaporation)
            wind_impact = max(0, 20 - weather_data.wind_speed * 0.2)
            
            # Calculate total soil moisture
            soil_moisture = base_moisture + rainfall_contribution + temp_impact + wind_impact
            
            # Add some randomness
            import random
            soil_moisture += random.uniform(-5, 5)
            
            # Ensure within valid range
            return max(0.0, min(100.0, soil_moisture))
            
        except Exception as e:
            logger.error(f"Error calculating soil moisture: {str(e)}")
            return 30.0  # Default moderate value
    
    def _calculate_drought_index(self, weather_data, soil_moisture: float) -> float:
        """
        Calculate drought index based on weather and soil conditions.
        
        Args:
            weather_data: Current weather data
            soil_moisture: Soil moisture percentage
            
        Returns:
            Drought index (0-1 scale, higher means more severe drought)
        """
        try:
            # Temperature factor (higher temperature increases drought)
            temp_factor = max(0, min(1, (weather_data.temperature - 15) / 30))
            
            # Humidity factor (lower humidity increases drought)
            humidity_factor = max(0, min(1, (60 - weather_data.humidity) / 60))
            
            # Rainfall factor (lower rainfall increases drought)
            rainfall_factor = max(0, min(1, (10 - weather_data.rainfall) / 10))
            
            # Soil moisture factor (lower soil moisture increases drought)
            soil_factor = max(0, min(1, (50 - soil_moisture) / 50))
            
            # Wind factor (higher wind increases drought)
            wind_factor = max(0, min(1, weather_data.wind_speed / 50))
            
            # Combine factors with weights
            drought_index = (temp_factor * 0.25 + 
                           humidity_factor * 0.20 + 
                           rainfall_factor * 0.25 + 
                           soil_factor * 0.20 + 
                           wind_factor * 0.10)
            
            # Add some randomness
            import random
            drought_index += random.uniform(-0.05, 0.05)
            
            # Ensure within valid range
            return max(0.0, min(1.0, drought_index))
            
        except Exception as e:
            logger.error(f"Error calculating drought index: {str(e)}")
            return 0.3  # Default moderate value
    
    def _determine_vegetation_type(self, location: Location, ndvi: float) -> str:
        """
        Determine vegetation type based on location and NDVI.
        
        Args:
            location: Location object
            ndvi: NDVI value
            
        Returns:
            Vegetation type string
        """
        try:
            # Simple classification based on NDVI and latitude
            if ndvi < 0.2:
                return "Barren/Urban"
            elif ndvi < 0.4:
                return "Grassland/Shrubland"
            elif ndvi < 0.6:
                return "Mixed Vegetation"
            elif ndvi < 0.8:
                return "Forest"
            else:
                return "Dense Forest"
                
        except Exception as e:
            logger.error(f"Error determining vegetation type: {str(e)}")
            return "Unknown"
    
    def get_fire_risk_from_vegetation(self, vegetation_data: VegetationData) -> float:
        """
        Calculate fire risk contribution from vegetation data.
        
        Args:
            vegetation_data: VegetationData object
            
        Returns:
            Fire risk factor (0-1 scale)
        """
        try:
            # NDVI risk (lower NDVI = higher fire risk)
            ndvi_risk = max(0, min(1, (0.6 - vegetation_data.ndvi) / 0.6))
            
            # Soil moisture risk (lower moisture = higher fire risk)
            soil_risk = max(0, min(1, (40 - vegetation_data.soil_moisture) / 40))
            
            # Drought risk (higher drought index = higher fire risk)
            drought_risk = vegetation_data.drought_index
            
            # Combine factors
            vegetation_risk = (ndvi_risk * 0.3 + 
                             soil_risk * 0.35 + 
                             drought_risk * 0.35)
            
            return vegetation_risk
            
        except Exception as e:
            logger.error(f"Error calculating vegetation fire risk: {str(e)}")
            return 0.5  # Default moderate risk
