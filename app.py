"""
Flask web application for AI Forest Fire Prediction System.
Main entry point for the web interface.
"""

import logging
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file
from pathlib import Path

# Import application modules
from core.fire_analyzer import FireAnalyzer
from core.fire_model import FireModel
from core.report_generator import ReportGenerator
from services.geocoder import GeocoderService
from services.weather_client import WeatherService
from services.vegetation_service import VegetationService
from models import Location, WeatherData, VegetationData, FireRiskAssessment, ChatMessage
from config import SECRET_KEY, DEBUG, LOG_LEVEL, LOG_FORMAT

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = SECRET_KEY

# Initialize services
geocoder = GeocoderService()
weather_service = WeatherService()
vegetation_service = VegetationService()
fire_analyzer = FireAnalyzer()
fire_model = FireModel()
report_generator = ReportGenerator()

# Load ML model if available
try:
    fire_model.load_model()
    logger.info("ML model loaded successfully")
except Exception as e:
    logger.warning(f"Could not load ML model: {str(e)}")

# Global chat history
chat_history = []


@app.route('/')
def index():
    """Main page with fire risk input form."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handle fire risk prediction request."""
    try:
        # Get form data
        location_input = request.form.get('location', '').strip()
        
        if not location_input:
            return jsonify({
                'success': False,
                'error': 'Location is required'
            })
        
        logger.info(f"Processing prediction request for: {location_input}")
        
        # Parse location (city name or coordinates)
        location = None
        coords = geocoder.parse_coordinates(location_input)
        
        if coords:
            # Input is coordinates
            lat, lon = coords
            location = geocoder.geocode_coordinates(lat, lon)
            if not location:
                location = Location(
                    city=f"Coordinates {lat}, {lon}",
                    latitude=lat,
                    longitude=lon
                )
        else:
            # Input is city name
            location = geocoder.geocode_city(location_input)
            if not location:
                return jsonify({
                    'success': False,
                    'error': f'Location "{location_input}" not found'
                })
        
        # Get weather data
        weather_data = weather_service.get_weather_data(location)
        if not weather_data:
            return jsonify({
                'success': False,
                'error': 'Could not fetch weather data for this location'
            })
        
        # Get vegetation data
        vegetation_data = vegetation_service.get_vegetation_data(location, weather_data)
        if not vegetation_data:
            return jsonify({
                'success': False,
                'error': 'Could not process vegetation data'
            })
        
        # Get ML prediction if available
        ml_probability = None
        if fire_model.is_model_available():
            ml_probability = fire_model.predict_fire_probability(weather_data, vegetation_data)
        
        # Perform fire risk analysis
        assessment = fire_analyzer.analyze_fire_risk(
            location, weather_data, vegetation_data, ml_probability
        )
        
        # Generate reports
        text_report = report_generator.generate_text_report(assessment)
        html_report = report_generator.generate_html_report(assessment)
        
        # Save reports
        text_file_path = report_generator.save_report(assessment, "text")
        html_file_path = report_generator.save_report(assessment, "html")
        
        # Prepare response
        response_data = {
            'success': True,
            'assessment': {
                'location': {
                    'city': assessment.location.city,
                    'state': assessment.location.state,
                    'country': assessment.location.country,
                    'latitude': assessment.location.latitude,
                    'longitude': assessment.location.longitude
                },
                'risk_level': assessment.risk_level.value,
                'probability': f"{assessment.probability:.1%}",
                'weather': {
                    'temperature': f"{assessment.weather_data.temperature:.1f}°C",
                    'humidity': f"{assessment.weather_data.humidity:.1f}%",
                    'wind_speed': f"{assessment.weather_data.wind_speed:.1f} km/h",
                    'rainfall': f"{assessment.weather_data.rainfall:.1f} mm"
                },
                'vegetation': {
                    'type': assessment.vegetation_data.vegetation_type,
                    'ndvi': f"{assessment.vegetation_data.ndvi:.3f}",
                    'soil_moisture': f"{assessment.vegetation_data.soil_moisture:.1f}%",
                    'drought_index': f"{assessment.vegetation_data.drought_index:.3f}"
                },
                'factors': {
                    'temperature': assessment.factors.temperature_factor,
                    'humidity': assessment.factors.humidity_factor,
                    'wind': assessment.factors.wind_factor,
                    'rainfall': assessment.factors.rainfall_factor,
                    'vegetation': assessment.factors.vegetation_factor,
                    'drought': assessment.factors.drought_factor
                },
                'recommendations': assessment.recommendations,
                'timestamp': assessment.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC'),
                'ml_used': ml_probability is not None,
                'ml_probability': f"{ml_probability:.1%}" if ml_probability else None
            },
            'reports': {
                'text': text_report,
                'html': html_report,
                'text_file': text_file_path,
                'html_file': html_file_path
            }
        }
        
        logger.info(f"Prediction completed for {location.city}: {assessment.risk_level.value} risk")
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'An unexpected error occurred during prediction'
        })


@app.route('/report/<report_type>/<filename>')
def download_report(report_type, filename):
    """Download generated report."""
    try:
        reports_dir = Path(__file__).parent / "reports"
        file_path = reports_dir / filename
        
        if not file_path.exists():
            return jsonify({'error': 'Report not found'}), 404
        
        return send_file(file_path, as_attachment=True)
        
    except Exception as e:
        logger.error(f"Error downloading report: {str(e)}")
        return jsonify({'error': 'Failed to download report'}), 500


@app.route('/chat')
def chat_page():
    """Chat interface page."""
    return render_template('chat.html')


@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat messages."""
    try:
        user_message = request.form.get('message', '').strip()
        
        if not user_message:
            return jsonify({
                'success': False,
                'error': 'Message is required'
            })
        
        # Generate bot response
        bot_response = generate_chat_response(user_message)
        
        # Add to chat history
        chat_message = ChatMessage(
            user_message=user_message,
            bot_response=bot_response,
            timestamp=datetime.now()
        )
        chat_history.append(chat_message)
        
        # Keep only recent messages
        if len(chat_history) > 10:
            chat_history.pop(0)
        
        return jsonify({
            'success': True,
            'response': bot_response,
            'timestamp': chat_message.timestamp.strftime('%H:%M:%S')
        })
        
    except Exception as e:
        logger.error(f"Error in chat: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'An error occurred while processing your message'
        })


@app.route('/api/chat/history')
def chat_history_api():
    """Get chat history."""
    try:
        history_data = []
        for msg in chat_history:
            history_data.append({
                'user_message': msg.user_message,
                'bot_response': msg.bot_response,
                'timestamp': msg.timestamp.strftime('%H:%M:%S')
            })
        
        return jsonify({
            'success': True,
            'history': history_data
        })
        
    except Exception as e:
        logger.error(f"Error getting chat history: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Failed to get chat history'
        })


def generate_chat_response(user_message: str) -> str:
    """
    Generate chat bot response based on user message.
    Uses rule-based responses for fire safety questions.
    
    Args:
        user_message: User's message
        
    Returns:
        Bot response string
    """
    message_lower = user_message.lower()
    
    # Fire risk questions
    if any(word in message_lower for word in ['risk', 'danger', 'safe', 'dangerous']):
        if any(word in message_lower for word in ['high', 'extreme', 'severe']):
            return ("🔥 High fire risk conditions require extreme caution. Avoid all outdoor "
                   "burning, keep fire extinguishers ready, and monitor emergency services. "
                   "Consider evacuation if conditions worsen.")
        elif any(word in message_lower for word in ['low', 'minimal', 'safe']):
            return ("✅ Low fire risk conditions are present. Standard fire safety practices "
                   "are recommended, but the risk of fire is minimal. Always stay informed "
                   "about changing conditions.")
        else:
            return ("🔥 Fire risk varies based on weather conditions, vegetation dryness, "
                   "and drought levels. I can provide a detailed risk assessment for your "
                   "specific location if you tell me the city or coordinates.")
    
    # Weather-related questions
    elif any(word in message_lower for word in ['weather', 'temperature', 'humidity', 'wind', 'rain']):
        return ("🌤️ Weather conditions significantly impact fire risk. High temperatures, "
               "low humidity, strong winds, and lack of rainfall all increase fire danger. "
               "Get a detailed assessment for your location to see current conditions.")
    
    # Prevention questions
    elif any(word in message_lower for word in ['prevent', 'prevention', 'avoid', 'stop']):
        return ("🛡️ Fire prevention tips: Clear dry vegetation around property, keep "
               "fire extinguishers accessible, avoid outdoor burning during high risk, "
               "properly maintain equipment, and stay informed about local fire conditions.")
    
    # Emergency questions
    elif any(word in message_lower for word in ['emergency', 'evacuation', 'escape', 'help']):
        return ("🚨 In an emergency: Call emergency services immediately, follow evacuation "
               "orders, know multiple escape routes, keep emergency supplies ready, and "
               "stay informed through official channels. Always prioritize personal safety.")
    
    # General questions
    elif any(word in message_lower for word in ['what', 'how', 'explain', 'tell me']):
        return ("🔥 I'm your AI Fire Assistant. I can help you understand fire risk levels, "
               "provide safety recommendations, explain weather impacts, and offer guidance "
               "on fire prevention. Ask me about specific locations or fire safety topics!")
    
    # Greeting
    elif any(word in message_lower for word in ['hello', 'hi', 'hey', 'greetings']):
        return ("👋 Hello! I'm your AI Fire Assistant. I can help you with fire risk assessments, "
               "safety recommendations, and answer questions about fire prevention. "
               "How can I assist you today?")
    
    # Default response
    else:
        return ("🤔 I can help you with fire risk assessments, safety recommendations, and "
               "fire prevention guidance. Try asking about fire risk for a specific location, "
               "safety tips, or emergency procedures. What would you like to know?")


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return render_template('404.html'), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {str(error)}")
    return render_template('500.html'), 500


if __name__ == '__main__':
    logger.info("Starting AI Forest Fire Prediction System")
    logger.info(f"Debug mode: {DEBUG}")
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=DEBUG
    )
