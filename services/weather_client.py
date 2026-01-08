"""
Weather data service for fetching real-time weather information.
Uses Open-Meteo API (no API key required).
"""

import logging
import requests
from datetime import datetime, timezone
from typing import Optional, Dict, Any
from urllib.parse import urlencode

from models import WeatherData, Location
from config import OPEN_METEO_BASE_URL, REQUEST_HEADERS, CACHE_TTL

logger = logging.getLogger(__name__)


class WeatherService:
    """Service for fetching weather data using Open-Meteo API."""
    
    def __init__(self):
        self.cache = {}
        self.base_url = OPEN_METEO_BASE_URL
        
    def get_weather_data(self, location: Location) -> Optional[WeatherData]:
        """
        Fetch current weather data for a location.
        
        Args:
            location: Location object with coordinates
            
        Returns:
            WeatherData object with current weather, or None if failed
        """
        cache_key = f"weather_{location.latitude}_{location.longitude}"
        if cache_key in self.cache:
            logger.info(f"Returning cached weather data for {location.city}")
            return self.cache[cache_key]
        
        try:
            # Prepare API request parameters
            params = {
                'latitude': location.latitude,
                'longitude': location.longitude,
                'current': 'temperature_2m,relative_humidity_2m,wind_speed_10m,rain',
                'timezone': 'auto'
            }
            
            url = f"{self.base_url}?{urlencode(params)}"
            logger.info(f"Fetching weather data for {location.city}")
            logger.info(f"Weather API URL: {url}")
            
            # Make API request
            response = requests.get(url, headers=REQUEST_HEADERS, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Extract current weather data
            current = data.get('current', {})
            current_units = data.get('current_units', {})
            
            # Get values with proper unit conversion
            temperature = current.get('temperature_2m')
            humidity = current.get('relative_humidity_2m')
            wind_speed = current.get('wind_speed_10m')
            rainfall = current.get('rain', 0.0)
            
            # Validate required data
            if None in [temperature, humidity, wind_speed]:
                logger.error(f"Missing required weather data for {location.city}")
                return None
            
            # Create weather data object
            weather_data = WeatherData(
                temperature=float(temperature),
                humidity=float(humidity),
                wind_speed=float(wind_speed),
                rainfall=float(rainfall),
                timestamp=datetime.now(timezone.utc),
                location=location
            )
            
            # Cache result
            self.cache[cache_key] = weather_data
            logger.info(f"Successfully fetched weather data for {location.city}: "
                       f"T={temperature}°C, H={humidity}%, W={wind_speed}km/h, R={rainfall}mm")
            
            return weather_data
            
        except requests.RequestException as e:
            logger.error(f"Error fetching weather data for {location.city}: {str(e)}")
            return None
        except (KeyError, ValueError) as e:
            logger.error(f"Error parsing weather data for {location.city}: {str(e)}")
            return None
    
    def get_weather_forecast(self, location: Location, days: int = 7) -> Optional[Dict[str, Any]]:
        """
        Fetch weather forecast for a location.
        
        Args:
            location: Location object with coordinates
            days: Number of forecast days (1-7)
            
        Returns:
            Dictionary with forecast data, or None if failed
        """
        try:
            # Prepare API request parameters
            params = {
                'latitude': location.latitude,
                'longitude': location.longitude,
                'forecast_days': min(days, 7),
                'hourly': 'temperature_2m,relative_humidity_2m,wind_speed_10m,rain',
                'timezone': 'auto'
            }
            
            url = f"{self.base_url}?{urlencode(params)}"
            logger.info(f"Fetching {days}-day weather forecast for {location.city}")
            
            response = requests.get(url, headers=REQUEST_HEADERS, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Process forecast data
            hourly = data.get('hourly', {})
            time_list = hourly.get('time', [])
            
            forecast_data = {
                'location': location,
                'forecast': []
            }
            
            # Group by day and calculate daily averages
            daily_data = {}
            for i, timestamp in enumerate(time_list):
                date_str = timestamp[:10]  # Extract date part
                
                if date_str not in daily_data:
                    daily_data[date_str] = {
                        'temperatures': [],
                        'humidities': [],
                        'wind_speeds': [],
                        'rainfall': 0.0
                    }
                
                # Add hourly data
                temp = hourly.get('temperature_2m', [])[i]
                humidity = hourly.get('relative_humidity_2m', [])[i]
                wind = hourly.get('wind_speed_10m', [])[i]
                rain = hourly.get('rain', [])[i]
                
                if None not in [temp, humidity, wind]:
                    daily_data[date_str]['temperatures'].append(float(temp))
                    daily_data[date_str]['humidities'].append(float(humidity))
                    daily_data[date_str]['wind_speeds'].append(float(wind))
                    daily_data[date_str]['rainfall'] += float(rain or 0.0)
            
            # Calculate daily averages
            for date_str, day_data in daily_data.items():
                if day_data['temperatures']:
                    avg_temp = sum(day_data['temperatures']) / len(day_data['temperatures'])
                    avg_humidity = sum(day_data['humidities']) / len(day_data['humidities'])
                    avg_wind = sum(day_data['wind_speeds']) / len(day_data['wind_speeds'])
                    
                    forecast_data['forecast'].append({
                        'date': date_str,
                        'temperature_avg': round(avg_temp, 1),
                        'humidity_avg': round(avg_humidity, 1),
                        'wind_speed_avg': round(avg_wind, 1),
                        'rainfall_total': round(day_data['rainfall'], 1)
                    })
            
            logger.info(f"Successfully fetched {len(forecast_data['forecast'])}-day forecast for {location.city}")
            return forecast_data
            
        except requests.RequestException as e:
            logger.error(f"Error fetching weather forecast for {location.city}: {str(e)}")
            return None
        except (KeyError, ValueError) as e:
            logger.error(f"Error parsing weather forecast for {location.city}: {str(e)}")
            return None
    
    def validate_weather_data(self, weather_data: WeatherData) -> bool:
        """
        Validate weather data for reasonableness.
        
        Args:
            weather_data: WeatherData object to validate
            
        Returns:
            True if data is reasonable, False otherwise
        """
        try:
            # Temperature range check (-50°C to 60°C)
            if not (-50 <= weather_data.temperature <= 60):
                logger.warning(f"Unreasonable temperature: {weather_data.temperature}°C")
                return False
            
            # Humidity range check (0-100%)
            if not (0 <= weather_data.humidity <= 100):
                logger.warning(f"Unreasonable humidity: {weather_data.humidity}%")
                return False
            
            # Wind speed range check (0-200 km/h)
            if not (0 <= weather_data.wind_speed <= 200):
                logger.warning(f"Unreasonable wind speed: {weather_data.wind_speed} km/h")
                return False
            
            # Rainfall range check (0-500 mm)
            if not (0 <= weather_data.rainfall <= 500):
                logger.warning(f"Unreasonable rainfall: {weather_data.rainfall} mm")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating weather data: {str(e)}")
            return False
