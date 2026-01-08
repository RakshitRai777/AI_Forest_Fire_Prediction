"""
Configuration settings for AI Forest Fire Prediction System.
Contains API endpoints, model parameters, and application settings.
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
ML_MODELS_DIR = BASE_DIR / "ml"
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"

# API Configuration
NOMINATIM_BASE_URL = "https://nominatim.openstreetmap.org/search"
OPEN_METEO_BASE_URL = "https://api.open-meteo.com/v1/forecast"

# Request headers for APIs
REQUEST_HEADERS = {
    "User-Agent": "Forest-Fire-Prediction-System/1.0"
}

# ML Model Configuration
MODEL_FILE = ML_MODELS_DIR / "fire_model.pkl"
FEATURE_SCALER_FILE = ML_MODELS_DIR / "feature_scaler.pkl"
RANDOM_STATE = 42
TEST_SIZE = 0.2
MODEL_N_ESTIMATORS = 100
MODEL_MAX_DEPTH = 10

# Fire Risk Thresholds
RISK_THRESHOLDS = {
    "low": (0.0, 0.25),
    "moderate": (0.25, 0.50),
    "high": (0.50, 0.75),
    "extreme": (0.75, 1.0)
}

# Risk Factor Weights
FACTOR_WEIGHTS = {
    "temperature": 0.25,
    "humidity": 0.20,
    "wind_speed": 0.20,
    "rainfall": 0.15,
    "vegetation": 0.10,
    "drought": 0.10
}

# Weather Risk Thresholds
WEATHER_THRESHOLDS = {
    "temperature": {
        "low": 15.0,
        "moderate": 25.0,
        "high": 35.0,
        "extreme": 45.0
    },
    "humidity": {
        "extreme": 20.0,
        "high": 40.0,
        "moderate": 60.0,
        "low": 80.0
    },
    "wind_speed": {
        "low": 10.0,
        "moderate": 25.0,
        "high": 40.0,
        "extreme": 60.0
    },
    "rainfall": {
        "extreme": 0.0,
        "high": 2.0,
        "moderate": 5.0,
        "low": 10.0
    }
}

# Vegetation Risk Thresholds
VEGETATION_THRESHOLDS = {
    "ndvi": {
        "extreme": 0.2,
        "high": 0.3,
        "moderate": 0.4,
        "low": 0.6
    },
    "soil_moisture": {
        "extreme": 10.0,
        "high": 20.0,
        "moderate": 30.0,
        "low": 50.0
    },
    "drought_index": {
        "low": 0.2,
        "moderate": 0.4,
        "high": 0.6,
        "extreme": 0.8
    }
}

# Flask Configuration
SECRET_KEY = os.environ.get("SECRET_KEY", "forest-fire-prediction-secret-key-2024")
DEBUG = os.environ.get("FLASK_DEBUG", "True").lower() == "true"

# Logging Configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Chat Configuration
CHAT_MAX_HISTORY = 10
CHAT_TEMPERATURE = 0.7

# Report Configuration
REPORT_TEMPLATE_DIR = TEMPLATES_DIR
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

# Cache Configuration (seconds)
CACHE_TTL = {
    "weather": 1800,  # 30 minutes
    "geocoding": 86400,  # 24 hours
    "vegetation": 3600  # 1 hour
}
