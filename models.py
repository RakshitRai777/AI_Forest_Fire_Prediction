"""
Data models for AI Forest Fire Prediction System.
Defines dataclasses for consistent data structure throughout the application.
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class RiskLevel(Enum):
    """Fire risk levels."""
    LOW = "Low"
    MODERATE = "Moderate"
    HIGH = "High"
    EXTREME = "Extreme"


@dataclass
class Location:
    """Location information."""
    city: str
    latitude: float
    longitude: float
    country: Optional[str] = None
    state: Optional[str] = None


@dataclass
class WeatherData:
    """Weather information for fire risk analysis."""
    temperature: float  # Celsius
    humidity: float     # Percentage
    wind_speed: float   # km/h
    rainfall: float     # mm
    timestamp: datetime
    location: Location


@dataclass
class VegetationData:
    """Vegetation and environmental dryness indicators."""
    ndvi: float  # Normalized Difference Vegetation Index
    soil_moisture: float  # Percentage
    drought_index: float  # 0-1 scale
    vegetation_type: str


@dataclass
class FireRiskFactors:
    """Contributing factors for fire risk assessment."""
    temperature_factor: float
    humidity_factor: float
    wind_factor: float
    rainfall_factor: float
    vegetation_factor: float
    drought_factor: float


@dataclass
class FireRiskAssessment:
    """Complete fire risk assessment result."""
    location: Location
    risk_level: RiskLevel
    probability: float  # 0-100
    factors: FireRiskFactors
    weather_data: WeatherData
    vegetation_data: VegetationData
    timestamp: datetime
    recommendations: List[str]


@dataclass
class ChatMessage:
    """Chat message for AI assistant."""
    user_message: str
    bot_response: str
    timestamp: datetime
    context: Optional[Dict[str, Any]] = None


@dataclass
class ModelFeatures:
    """Features for ML model prediction."""
    temperature: float
    humidity: float
    wind_speed: float
    rainfall: float
    ndvi: float
    soil_moisture: float
    drought_index: float
    latitude: float
    longitude: float
    month: int  # Seasonal factor
    day_of_year: int  # Seasonal factor
