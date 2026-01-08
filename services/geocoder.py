"""
Geocoding service for converting city names to coordinates.
Uses Nominatim OpenStreetMap API (no API key required).
"""

import logging
import requests
from typing import Optional, Tuple
from urllib.parse import urlencode

from models import Location
from config import NOMINATIM_BASE_URL, REQUEST_HEADERS, CACHE_TTL

logger = logging.getLogger(__name__)


class GeocoderService:
    """Service for geocoding locations using Nominatim API."""
    
    def __init__(self):
        self.cache = {}
        self.base_url = NOMINATIM_BASE_URL
        
    def geocode_city(self, city: str) -> Optional[Location]:
        """
        Convert city name to coordinates.
        
        Args:
            city: City name to geocode
            
        Returns:
            Location object with coordinates, or None if not found
        """
        # Check cache first
        cache_key = f"city_{city.lower()}"
        if cache_key in self.cache:
            logger.info(f"Returning cached geocoding result for {city}")
            return self.cache[cache_key]
        
        try:
            # Prepare API request
            params = {
                'q': city,
                'format': 'json',
                'limit': 1,
                'addressdetails': 1
            }
            
            url = f"{self.base_url}?{urlencode(params)}"
            logger.info(f"Geocoding city: {city}")
            
            # Make API request
            response = requests.get(url, headers=REQUEST_HEADERS, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if not data:
                logger.warning(f"No results found for city: {city}")
                return None
            
            # Extract first result
            result = data[0]
            
            # Parse address components
            address = result.get('address', {})
            country = address.get('country')
            state = address.get('state') or address.get('county')
            
            location = Location(
                city=city.title(),
                latitude=float(result['lat']),
                longitude=float(result['lon']),
                country=country,
                state=state
            )
            
            # Cache result
            self.cache[cache_key] = location
            logger.info(f"Successfully geocoded {city}: {location.latitude}, {location.longitude}")
            
            return location
            
        except requests.RequestException as e:
            logger.error(f"Error geocoding {city}: {str(e)}")
            return None
        except (KeyError, ValueError) as e:
            logger.error(f"Error parsing geocoding response for {city}: {str(e)}")
            return None
    
    def geocode_coordinates(self, latitude: float, longitude: float) -> Optional[Location]:
        """
        Convert coordinates to location information (reverse geocoding).
        
        Args:
            latitude: Latitude
            longitude: Longitude
            
        Returns:
            Location object with address information, or None if not found
        """
        cache_key = f"coords_{latitude}_{longitude}"
        if cache_key in self.cache:
            logger.info(f"Returning cached reverse geocoding result for {latitude}, {longitude}")
            return self.cache[cache_key]
        
        try:
            # Prepare reverse geocoding request
            params = {
                'lat': latitude,
                'lon': longitude,
                'format': 'json',
                'addressdetails': 1,
                'zoom': 10
            }
            
            url = f"https://nominatim.openstreetmap.org/reverse?{urlencode(params)}"
            logger.info(f"Reverse geocoding coordinates: {latitude}, {longitude}")
            
            response = requests.get(url, headers=REQUEST_HEADERS, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if 'error' in data:
                logger.warning(f"No results found for coordinates: {latitude}, {longitude}")
                return None
            
            # Parse address components
            address = data.get('address', {})
            city = (address.get('city') or 
                   address.get('town') or 
                   address.get('village') or 
                   address.get('municipality') or 
                   "Unknown Location")
            
            country = address.get('country')
            state = address.get('state') or address.get('county')
            
            location = Location(
                city=city.title(),
                latitude=latitude,
                longitude=longitude,
                country=country,
                state=state
            )
            
            # Cache result
            self.cache[cache_key] = location
            logger.info(f"Successfully reverse geocoded {latitude}, {longitude}: {city}")
            
            return location
            
        except requests.RequestException as e:
            logger.error(f"Error reverse geocoding {latitude}, {longitude}: {str(e)}")
            return None
        except (KeyError, ValueError) as e:
            logger.error(f"Error parsing reverse geocoding response: {str(e)}")
            return None
    
    def parse_coordinates(self, location_input: str) -> Optional[Tuple[float, float]]:
        """
        Parse coordinates from string input.
        
        Args:
            location_input: String containing coordinates (e.g., "40.7128,-74.0060")
            
        Returns:
            Tuple of (latitude, longitude) or None if invalid
        """
        try:
            # Remove whitespace and split by comma
            coords = location_input.strip().split(',')
            if len(coords) != 2:
                return None
            
            lat = float(coords[0].strip())
            lon = float(coords[1].strip())
            
            # Validate coordinate ranges
            if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
                return None
            
            return lat, lon
            
        except ValueError:
            return None
