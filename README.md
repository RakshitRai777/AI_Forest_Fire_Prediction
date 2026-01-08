# 🔥 AI Forest Fire Prediction System

An advanced AI-powered system for predicting forest fire risks using machine learning, real-time weather data, and environmental analysis. Built with Python and Flask, following clean architecture principles.

## 🌟 Features

- **🤖 AI-Powered Analysis**: Combines rule-based logic with machine learning predictions
- **🌍 Global Coverage**: Works with any location worldwide using geocoding services
- **🌤️ Real-time Weather**: Integrates with Open-Meteo API for current weather data
- **🌿 Environmental Analysis**: Calculates vegetation indices and drought indicators
- **📊 Comprehensive Reports**: Generates detailed text and HTML reports
- **💬 AI Assistant**: Interactive chat bot for fire safety guidance
- **📱 Responsive Design**: Modern, mobile-friendly web interface
- **🛡️ Safety Focus**: Provides actionable safety recommendations

## 🏗️ Architecture

```
AI-Forest-Fire-Prediction-System/
│
├── core/                          # Core business logic
│   ├── fire_analyzer.py          # Fire risk analysis engine
│   ├── fire_model.py             # ML model interface
│   └── report_generator.py       # Report generation
│
├── services/                      # External service integrations
│   ├── geocoder.py               # Location geocoding (Nominatim)
│   ├── weather_client.py         # Weather data (Open-Meteo)
│   └── vegetation_service.py     # Environmental analysis
│
├── ml/                           # Machine learning components
│   ├── train_model.py            # Model training script
│   ├── predict.py                # Prediction wrapper
│   └── fire_model.pkl            # Trained model (generated)
│
├── templates/                     # HTML templates
│   ├── base.html                 # Base template
│   ├── index.html                # Main prediction page
│   ├── chat.html                 # AI assistant interface
│   └── report.html               # Report display
│
├── static/                       # Static assets
│   └── style.css                 # Main stylesheet
│
├── app.py                        # Flask application entry point
├── models.py                     # Data models and types
├── config.py                     # Configuration settings
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- pip package manager
- Internet connection (for weather and geocoding APIs)

### 📥 Dataset Setup

**Important**: The dataset is not included in this repository due to its large size (88MB+). You need to download it first:

1. **Download the Forest Fire Dataset**:
   - The dataset contains fire and smoke images with annotations
   - [https://www.kaggle.com/datasets/kutaykutlu/forest-fire]
   - Extract and place the `DataSet/` folder in the project root

2. **Dataset Structure**:
   ```
   DataSet/
   ├── Datacluster Fire and Smoke Sample/
   │   ├── Datacluster Fire and Smoke Sample (1).jpg
   │   ├── Datacluster Fire and Smoke Sample (10).jpg
   │   └── ... (100 images total)
   └── Annotations/
       ├── Datacluster Fire and Smoke Sample (1).xml
       ├── Datacluster Fire and Smoke Sample (10).xml
       └── ... (100 annotation files total)
   ```

### Installation

1. **Clone or download the project**
   ```bash
   # If using git
   git clone https://github.com/RakshitRai777/AI_Forest_Fire_Prediction.git
   cd AI_Forest_Fire_Prediction
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Train the ML model**
   ```bash
   python main.py
   ```
   This will:
   - Load the dataset from the `DataSet/` folder
   - Train the forest fire detection model
   - Generate the trained model file `forest_fire_model.pkl`
   - Show training accuracy and metrics

5. **Run the application**
   ```bash
   python app.py
   ```

6. **Access the application**
   Open your web browser and go to `http://localhost:5000`

## 📖 Usage

### Getting a Fire Risk Prediction

1. **Enter Location**: Type a city name (e.g., "Los Angeles") or coordinates (e.g., "34.0522,-118.2437")
2. **Click "Predict Fire Risk"**: The system will analyze weather and environmental data
3. **View Results**: See risk level, probability, contributing factors, and safety recommendations
4. **Download Reports**: Get detailed text or HTML reports for further analysis

### Using the AI Assistant

1. **Navigate to Chat**: Click "AI Assistant" in the navigation
2. **Ask Questions**: Type questions about fire safety, prevention, or risk assessment
3. **Get Instant Answers**: Receive helpful guidance and recommendations

### Sample Questions for AI Assistant

- "What is fire risk and how is it calculated?"
- "How can I prevent fires around my property?"
- "What should I do during a wildfire emergency?"
- "How does weather affect fire risk?"
- "Is it safe to have a campfire today?"

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
FLASK_DEBUG=True
SECRET_KEY=your-secret-key-here
```

### API Configuration

The system uses free APIs that don't require API keys:

- **Nominatim**: OpenStreetMap geocoding (no API key required)
- **Open-Meteo**: Weather data (no API key required)

### Model Configuration

Edit `config.py` to adjust:

- ML model parameters
- Risk thresholds
- Factor weights
- API endpoints
- Cache settings

## 🧠 Machine Learning

### Model Training

The system uses a machine learning model trained on fire and smoke images:

1. **Dataset**: 100 images of fire and smoke scenes with XML annotations
2. **Training Script**: Run `python main.py` to train the model
3. **Model Output**: Generates `forest_fire_model.pkl` for predictions

### Training Process

```bash
python main.py
```

This will:
- Load images from `DataSet/Datacluster Fire and Smoke Sample/`
- Parse XML annotations from `DataSet/Annotations/`
- Extract features and train the classifier
- Save the trained model as `forest_fire_model.pkl`
- Display training accuracy and performance metrics

### Features Used

1. **Image Features**:
   - Color histograms (RGB channels)
   - Texture features (LBP - Local Binary Patterns)
   - Shape and contour features
   - Fire-specific color patterns

2. **Annotation Data**:
   - Bounding box coordinates
   - Object labels (fire, smoke)
   - Confidence scores

### Model Performance

- **Training Accuracy**: Displays during training
- **Model Type**: RandomForestClassifier (configurable)
- **Feature Extraction**: Computer vision techniques
- **Prediction Speed**: Real-time inference

### Retraining the Model

To retrain with new data:
1. Add new images to `DataSet/Datacluster Fire and Smoke Sample/`
2. Add corresponding XML annotations to `DataSet/Annotations/`
3. Run `python main.py` to retrain
4. The new model will overwrite `forest_fire_model.pkl`

## 🌐 API Endpoints

### Web Endpoints

- `GET /` - Main prediction interface
- `GET /chat` - AI assistant interface
- `GET /report/<type>/<filename>` - Download reports

### API Endpoints

- `POST /predict` - Get fire risk prediction
- `POST /api/chat` - Send message to AI assistant
- `GET /api/chat/history` - Get chat history

### Prediction Request Example

```python
import requests

response = requests.post('/predict', data={
    'location': 'Los Angeles'
})

data = response.json()
if data['success']:
    assessment = data['assessment']
    print(f"Risk Level: {assessment['risk_level']}")
    print(f"Probability: {assessment['probability']}")
```

## 🛠️ Development

### Project Structure

The project follows a clean, modular architecture:

- **Core Layer**: Business logic and analysis engines
- **Services Layer**: External API integrations
- **ML Layer**: Machine learning components
- **Web Layer**: Flask application and templates
- **Models Layer**: Data structures and validation

### Adding New Features

1. **New Data Sources**: Add to `services/` directory
2. **New Analysis Logic**: Add to `core/` directory
3. **New ML Models**: Extend `ml/` directory
4. **New Web Pages**: Add to `templates/` directory

### Testing

```bash
# Run tests (when implemented)
python -m pytest tests/

# Test individual components
python ml/predict.py
python services/geocoder.py
```

### Code Style

The project follows Python best practices:

- Type hints everywhere
- Comprehensive docstrings
- Logging instead of print statements
- Defensive programming
- Clean separation of concerns

## 🔒 Security

- Input validation and sanitization
- Rate limiting for API calls
- Secure session handling
- Environment-based configuration
- No hardcoded secrets

## 📊 Performance

- **Caching**: Weather and geocoding data cached to reduce API calls
- **Async Processing**: Non-blocking API requests
- **Optimized ML**: Efficient model loading and prediction
- **Responsive UI**: Fast loading and smooth interactions

## 🌍 Deployment

### Production Deployment

1. **Set Environment Variables**:
   ```env
   FLASK_DEBUG=False
   SECRET_KEY=production-secret-key
   ```

2. **Use Production Server**:
   ```bash
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:8000 app:app
   ```

3. **Behind Reverse Proxy**:
   Configure nginx or Apache to serve the application

### Docker Deployment

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 5000

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is open source and available under the MIT License.

## 🆘 Support

### Common Issues

1. **Model Not Found**: Run `python ml/train_model.py` first
2. **API Errors**: Check internet connection and API availability
3. **Location Not Found**: Try different location formats or coordinates

### Getting Help

- Check the logs for detailed error messages
- Verify all dependencies are installed
- Ensure the ML model is trained before running predictions

## 🔮 Future Enhancements

- [ ] Real-time satellite imagery integration
- [ ] Historical fire data integration
- [ ] Mobile app development
- [ ] Email/SMS alerts for high-risk areas
- [ ] Advanced ML models (deep learning)
- [ ] Multi-language support
- [ ] User accounts and saved locations
- [ ] API rate limiting and authentication
- [ ] Integration with emergency services

## 📈 System Requirements

- **Python**: 3.10 or higher
- **Memory**: Minimum 2GB RAM
- **Storage**: 500MB free space
- **Network**: Internet connection for API calls

## 🎯 Accuracy

The system provides risk assessments based on:

- **Weather Data**: Real-time meteorological conditions
- **Environmental Factors**: Vegetation dryness and drought conditions
- **Machine Learning**: Pattern recognition from historical data
- **Rule-Based Logic**: Expert fire safety knowledge

**Disclaimer**: This system is for informational purposes only and should not replace official emergency services or professional fire risk assessments.

---

**Built with ❤️ for community safety and wildfire prevention**
