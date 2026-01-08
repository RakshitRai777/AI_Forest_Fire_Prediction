"""
Core fire risk analysis engine.
Combines rule-based logic with ML predictions for comprehensive fire risk assessment.
"""

import logging
from typing import List, Dict, Any, Tuple
from datetime import datetime

from models import (
    WeatherData, VegetationData, FireRiskAssessment, 
    FireRiskFactors, RiskLevel, Location
)
from config import WEATHER_THRESHOLDS, VEGETATION_THRESHOLDS, FACTOR_WEIGHTS

logger = logging.getLogger(__name__)


class FireAnalyzer:
    """Core fire risk analysis engine."""
    
    def __init__(self):
        self.risk_thresholds = {
            RiskLevel.LOW: (0.0, 0.25),
            RiskLevel.MODERATE: (0.25, 0.50),
            RiskLevel.HIGH: (0.50, 0.75),
            RiskLevel.EXTREME: (0.75, 1.0)
        }
        
    def analyze_fire_risk(self, location: Location, weather_data: WeatherData, 
                         vegetation_data: VegetationData, ml_probability: float = None) -> FireRiskAssessment:
        """
        Perform comprehensive fire risk analysis.
        
        Args:
            location: Location object
            weather_data: Current weather data
            vegetation_data: Vegetation and environmental data
            ml_probability: ML model probability (optional)
            
        Returns:
            Complete fire risk assessment
        """
        try:
            logger.info(f"Starting fire risk analysis for {location.city}")
            
            # Calculate individual risk factors
            factors = self._calculate_risk_factors(weather_data, vegetation_data)
            
            # Combine factors for overall risk score
            rule_based_probability = self._calculate_overall_probability(factors)
            
            # Combine with ML probability if available
            if ml_probability is not None:
                final_probability = self._combine_probabilities(rule_based_probability, ml_probability)
            else:
                final_probability = rule_based_probability
            
            # Determine risk level
            risk_level = self._determine_risk_level(final_probability)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(risk_level, factors)
            
            # Create assessment
            assessment = FireRiskAssessment(
                location=location,
                risk_level=risk_level,
                probability=final_probability,
                factors=factors,
                weather_data=weather_data,
                vegetation_data=vegetation_data,
                timestamp=datetime.now(),
                recommendations=recommendations
            )
            
            logger.info(f"Fire risk analysis completed for {location.city}: "
                       f"{risk_level.value} risk ({final_probability:.1%})")
            
            return assessment
            
        except Exception as e:
            logger.error(f"Error in fire risk analysis: {str(e)}")
            raise
    
    def _calculate_risk_factors(self, weather_data: WeatherData, 
                               vegetation_data: VegetationData) -> FireRiskFactors:
        """
        Calculate individual risk factors from weather and vegetation data.
        
        Args:
            weather_data: Current weather data
            vegetation_data: Vegetation and environmental data
            
        Returns:
            FireRiskFactors object with calculated factors
        """
        try:
            # Temperature risk factor
            temp_factor = self._calculate_temperature_factor(weather_data.temperature)
            
            # Humidity risk factor
            humidity_factor = self._calculate_humidity_factor(weather_data.humidity)
            
            # Wind speed risk factor
            wind_factor = self._calculate_wind_factor(weather_data.wind_speed)
            
            # Rainfall risk factor
            rainfall_factor = self._calculate_rainfall_factor(weather_data.rainfall)
            
            # Vegetation risk factor
            vegetation_factor = self._calculate_vegetation_factor(vegetation_data.ndvi)
            
            # Drought risk factor
            drought_factor = vegetation_data.drought_index
            
            return FireRiskFactors(
                temperature_factor=temp_factor,
                humidity_factor=humidity_factor,
                wind_factor=wind_factor,
                rainfall_factor=rainfall_factor,
                vegetation_factor=vegetation_factor,
                drought_factor=drought_factor
            )
            
        except Exception as e:
            logger.error(f"Error calculating risk factors: {str(e)}")
            raise
    
    def _calculate_temperature_factor(self, temperature: float) -> float:
        """Calculate temperature-based fire risk factor."""
        thresholds = WEATHER_THRESHOLDS["temperature"]
        
        if temperature >= thresholds["extreme"]:
            return 1.0
        elif temperature >= thresholds["high"]:
            return 0.75
        elif temperature >= thresholds["moderate"]:
            return 0.5
        elif temperature >= thresholds["low"]:
            return 0.25
        else:
            return 0.1
    
    def _calculate_humidity_factor(self, humidity: float) -> float:
        """Calculate humidity-based fire risk factor (inverted - lower humidity = higher risk)."""
        thresholds = WEATHER_THRESHOLDS["humidity"]
        
        if humidity <= thresholds["extreme"]:
            return 1.0
        elif humidity <= thresholds["high"]:
            return 0.75
        elif humidity <= thresholds["moderate"]:
            return 0.5
        elif humidity <= thresholds["low"]:
            return 0.25
        else:
            return 0.1
    
    def _calculate_wind_factor(self, wind_speed: float) -> float:
        """Calculate wind speed-based fire risk factor."""
        thresholds = WEATHER_THRESHOLDS["wind_speed"]
        
        if wind_speed >= thresholds["extreme"]:
            return 1.0
        elif wind_speed >= thresholds["high"]:
            return 0.75
        elif wind_speed >= thresholds["moderate"]:
            return 0.5
        elif wind_speed >= thresholds["low"]:
            return 0.25
        else:
            return 0.1
    
    def _calculate_rainfall_factor(self, rainfall: float) -> float:
        """Calculate rainfall-based fire risk factor (inverted - less rain = higher risk)."""
        thresholds = WEATHER_THRESHOLDS["rainfall"]
        
        if rainfall <= thresholds["extreme"]:
            return 1.0
        elif rainfall <= thresholds["high"]:
            return 0.75
        elif rainfall <= thresholds["moderate"]:
            return 0.5
        elif rainfall <= thresholds["low"]:
            return 0.25
        else:
            return 0.1
    
    def _calculate_vegetation_factor(self, ndvi: float) -> float:
        """Calculate vegetation-based fire risk factor (inverted - lower NDVI = higher risk)."""
        thresholds = VEGETATION_THRESHOLDS["ndvi"]
        
        if ndvi <= thresholds["extreme"]:
            return 1.0
        elif ndvi <= thresholds["high"]:
            return 0.75
        elif ndvi <= thresholds["moderate"]:
            return 0.5
        elif ndvi <= thresholds["low"]:
            return 0.25
        else:
            return 0.1
    
    def _calculate_overall_probability(self, factors: FireRiskFactors) -> float:
        """
        Calculate overall fire probability using weighted combination of factors.
        
        Args:
            factors: Individual risk factors
            
        Returns:
            Overall fire probability (0-1 scale)
        """
        try:
            weights = FACTOR_WEIGHTS
            
            probability = (
                factors.temperature_factor * weights["temperature"] +
                factors.humidity_factor * weights["humidity"] +
                factors.wind_factor * weights["wind_speed"] +
                factors.rainfall_factor * weights["rainfall"] +
                factors.vegetation_factor * weights["vegetation"] +
                factors.drought_factor * weights["drought"]
            )
            
            return min(1.0, max(0.0, probability))
            
        except Exception as e:
            logger.error(f"Error calculating overall probability: {str(e)}")
            return 0.5
    
    def _combine_probabilities(self, rule_based: float, ml_probability: float) -> float:
        """
        Combine rule-based and ML probabilities.
        
        Args:
            rule_based: Rule-based probability
            ml_probability: ML model probability
            
        Returns:
            Combined probability
        """
        # Weighted combination (can be adjusted based on model confidence)
        rule_weight = 0.6
        ml_weight = 0.4
        
        combined = rule_based * rule_weight + ml_probability * ml_weight
        return min(1.0, max(0.0, combined))
    
    def _determine_risk_level(self, probability: float) -> RiskLevel:
        """
        Determine risk level from probability.
        
        Args:
            probability: Fire probability (0-1 scale)
            
        Returns:
            RiskLevel enum value
        """
        for risk_level, (min_prob, max_prob) in self.risk_thresholds.items():
            if min_prob <= probability < max_prob:
                return risk_level
        
        # If probability is exactly 1.0, return EXTREME
        return RiskLevel.EXTREME
    
    def _generate_recommendations(self, risk_level: RiskLevel, 
                                factors: FireRiskFactors) -> List[str]:
        """
        Generate safety recommendations based on risk level and contributing factors.
        
        Args:
            risk_level: Current risk level
            factors: Risk factors contributing to assessment
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        # Base recommendations by risk level
        if risk_level == RiskLevel.EXTREME:
            recommendations.extend([
                "🔥 EXTREME FIRE DANGER - Avoid all outdoor activities",
                "Evacuation plans should be ready",
                "No outdoor burning or equipment use",
                "Monitor emergency services regularly"
            ])
        elif risk_level == RiskLevel.HIGH:
            recommendations.extend([
                "⚠️ HIGH FIRE RISK - Extreme caution required",
                "Avoid outdoor burning and fireworks",
                "Keep fire extinguishers accessible",
                "Monitor local fire conditions"
            ])
        elif risk_level == RiskLevel.MODERATE:
            recommendations.extend([
                "⚡ MODERATE FIRE RISK - Be cautious",
                "Supervise any outdoor fires",
                "Clear dry vegetation around property",
                "Have fire safety equipment ready"
            ])
        else:  # LOW
            recommendations.extend([
                "✅ LOW FIRE RISK - Normal precautions",
                "Standard fire safety practices recommended",
                "Stay informed about changing conditions"
            ])
        
        # Specific recommendations based on contributing factors
        if factors.temperature_factor > 0.7:
            recommendations.append("High temperatures increase fire risk - Stay hydrated and avoid heat exposure")
        
        if factors.humidity_factor > 0.7:
            recommendations.append("Very low humidity - Avoid activities that could create sparks")
        
        if factors.wind_factor > 0.7:
            recommendations.append("High winds can spread fire quickly - Secure loose materials")
        
        if factors.rainfall_factor > 0.7:
            recommendations.append("Dry conditions - Extra vigilance required")
        
        if factors.vegetation_factor > 0.7:
            recommendations.append("Dry vegetation present - Clear dead plants and leaves")
        
        if factors.drought_factor > 0.7:
            recommendations.append("Drought conditions - Water conservation and fire prevention critical")
        
        return recommendations
    
    def get_risk_explanation(self, assessment: FireRiskAssessment) -> Dict[str, Any]:
        """
        Generate detailed explanation of risk assessment.
        
        Args:
            assessment: Fire risk assessment
            
        Returns:
            Dictionary with risk explanation details
        """
        try:
            factors = assessment.factors
            
            # Sort factors by contribution
            factor_contributions = [
                ("Temperature", factors.temperature_factor, FACTOR_WEIGHTS["temperature"]),
                ("Humidity", factors.humidity_factor, FACTOR_WEIGHTS["humidity"]),
                ("Wind Speed", factors.wind_factor, FACTOR_WEIGHTS["wind_speed"]),
                ("Rainfall", factors.rainfall_factor, FACTOR_WEIGHTS["rainfall"]),
                ("Vegetation", factors.vegetation_factor, FACTOR_WEIGHTS["vegetation"]),
                ("Drought", factors.drought_factor, FACTOR_WEIGHTS["drought"])
            ]
            
            # Calculate weighted contributions
            weighted_contributions = [
                (name, factor * weight) 
                for name, factor, weight in factor_contributions
            ]
            
            # Sort by contribution
            weighted_contributions.sort(key=lambda x: x[1], reverse=True)
            
            return {
                "risk_level": assessment.risk_level.value,
                "probability": assessment.probability,
                "primary_factors": weighted_contributions[:3],
                "all_factors": weighted_contributions,
                "recommendations": assessment.recommendations,
                "weather_summary": {
                    "temperature": assessment.weather_data.temperature,
                    "humidity": assessment.weather_data.humidity,
                    "wind_speed": assessment.weather_data.wind_speed,
                    "rainfall": assessment.weather_data.rainfall
                },
                "vegetation_summary": {
                    "ndvi": assessment.vegetation_data.ndvi,
                    "soil_moisture": assessment.vegetation_data.soil_moisture,
                    "drought_index": assessment.vegetation_data.drought_index,
                    "vegetation_type": assessment.vegetation_data.vegetation_type
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating risk explanation: {str(e)}")
            return {}
