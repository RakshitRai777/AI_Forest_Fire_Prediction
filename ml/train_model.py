"""
Machine Learning model training script for forest fire prediction.
Generates synthetic dataset and trains RandomForestClassifier.
"""

import logging
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from config import (
    MODEL_FILE, FEATURE_SCALER_FILE, RANDOM_STATE, 
    TEST_SIZE, MODEL_N_ESTIMATORS, MODEL_MAX_DEPTH
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FireModelTrainer:
    """Trainer for forest fire prediction ML model."""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = [
            'temperature', 'humidity', 'wind_speed', 'rainfall',
            'ndvi', 'soil_moisture', 'drought_index',
            'latitude', 'longitude', 'month', 'day_of_year'
        ]
    
    def generate_synthetic_dataset(self, n_samples: int = 5000) -> pd.DataFrame:
        """
        Generate synthetic dataset for training.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            DataFrame with features and target
        """
        logger.info(f"Generating synthetic dataset with {n_samples} samples...")
        
        np.random.seed(RANDOM_STATE)
        
        # Generate realistic weather data
        temperature = np.random.normal(25, 10, n_samples)  # Mean 25°C, std 10°C
        temperature = np.clip(temperature, -10, 50)  # Realistic range
        
        humidity = np.random.normal(60, 20, n_samples)  # Mean 60%, std 20%
        humidity = np.clip(humidity, 10, 100)  # Realistic range
        
        wind_speed = np.random.exponential(15, n_samples)  # Exponential distribution
        wind_speed = np.clip(wind_speed, 0, 80)  # Realistic range
        
        rainfall = np.random.exponential(2, n_samples)  # Exponential distribution
        rainfall = np.clip(rainfall, 0, 50)  # Realistic range
        
        # Generate vegetation data
        ndvi = np.random.beta(2, 2, n_samples)  # Beta distribution for NDVI
        ndvi = np.clip(ndvi, 0, 1)
        
        soil_moisture = np.random.normal(40, 15, n_samples)
        soil_moisture = np.clip(soil_moisture, 5, 95)
        
        # Drought index correlated with weather
        drought_index = (
            0.3 * (temperature / 50) +  # Higher temp = higher drought
            0.3 * (1 - humidity / 100) +  # Lower humidity = higher drought
            0.2 * (1 - rainfall / 20) +  # Less rain = higher drought
            0.2 * (1 - soil_moisture / 100)  # Less soil moisture = higher drought
        )
        drought_index = np.clip(drought_index, 0, 1)
        
        # Generate location data
        latitude = np.random.uniform(-60, 60, n_samples)  # Most populated areas
        longitude = np.random.uniform(-180, 180, n_samples)
        
        # Temporal features
        dates = [datetime(2020, 1, 1) + timedelta(days=np.random.randint(0, 365)) 
                for _ in range(n_samples)]
        months = np.array([d.month for d in dates])
        day_of_year = np.array([d.timetuple().tm_yday for d in dates])
        
        # Create DataFrame
        df = pd.DataFrame({
            'temperature': temperature,
            'humidity': humidity,
            'wind_speed': wind_speed,
            'rainfall': rainfall,
            'ndvi': ndvi,
            'soil_moisture': soil_moisture,
            'drought_index': drought_index,
            'latitude': latitude,
            'longitude': longitude,
            'month': months,
            'day_of_year': day_of_year
        })
        
        # Generate target variable (fire occurrence)
        # Fire risk increases with high temperature, low humidity, high wind, low rain
        fire_probability = (
            0.25 * np.clip(temperature / 40, 0, 1) +  # Temperature factor
            0.20 * (1 - np.clip(humidity / 100, 0, 1)) +  # Humidity factor (inverted)
            0.20 * np.clip(wind_speed / 40, 0, 1) +  # Wind factor
            0.15 * (1 - np.clip(rainfall / 10, 0, 1)) +  # Rainfall factor (inverted)
            0.10 * (1 - np.clip(ndvi, 0, 1)) +  # NDVI factor (inverted)
            0.10 * drought_index  # Drought factor
        )
        
        # Add seasonal variation (higher fire risk in summer)
        seasonal_factor = np.sin(2 * np.pi * (day_of_year - 80) / 365)  # Peak around day 80 (spring)
        fire_probability += 0.1 * np.clip(seasonal_factor, 0, 1)
        
        # Generate binary target with some noise
        fire_probability = np.clip(fire_probability, 0, 1)
        noise = np.random.normal(0, 0.1, n_samples)
        fire_probability_with_noise = np.clip(fire_probability + noise, 0, 1)
        
        # Convert to binary target (0 = no fire, 1 = fire)
        target = (fire_probability_with_noise > 0.5).astype(int)
        
        df['fire_occurred'] = target
        
        logger.info(f"Dataset generated: {target.sum()} fire cases out of {n_samples} samples "
                   f"({target.sum()/n_samples:.1%})")
        
        return df
    
    def train_model(self, df: pd.DataFrame) -> dict:
        """
        Train the RandomForest model.
        
        Args:
            df: Training dataset
            
        Returns:
            Dictionary with training results
        """
        logger.info("Starting model training...")
        
        # Separate features and target
        X = df[self.feature_names]
        y = df['fire_occurred']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model = RandomForestClassifier(
            n_estimators=MODEL_N_ESTIMATORS,
            max_depth=MODEL_MAX_DEPTH,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        
        # Feature importance
        feature_importance = dict(zip(self.feature_names, self.model.feature_importances_))
        
        results = {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'feature_importance': feature_importance,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'n_features': len(self.feature_names)
        }
        
        logger.info(f"Model training completed. Accuracy: {accuracy:.3f}")
        logger.info(f"Feature importance: {feature_importance}")
        
        return results
    
    def save_model(self) -> bool:
        """
        Save the trained model and scaler.
        
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            # Create directories if they don't exist
            Path(MODEL_FILE).parent.mkdir(parents=True, exist_ok=True)
            Path(FEATURE_SCALER_FILE).parent.mkdir(parents=True, exist_ok=True)
            
            # Save model
            joblib.dump(self.model, MODEL_FILE)
            logger.info(f"Model saved to {MODEL_FILE}")
            
            # Save scaler
            joblib.dump(self.scaler, FEATURE_SCALER_FILE)
            logger.info(f"Scaler saved to {FEATURE_SCALER_FILE}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return False
    
    def evaluate_model(self, df: pd.DataFrame) -> dict:
        """
        Evaluate the trained model with additional metrics.
        
        Args:
            df: Test dataset
            
        Returns:
            Dictionary with evaluation results
        """
        if self.model is None or self.scaler is None:
            raise ValueError("Model not trained yet")
        
        # Separate features and target
        X = df[self.feature_names]
        y = df['fire_occurred']
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        y_pred = self.model.predict(X_scaled)
        y_pred_proba = self.model.predict_proba(X_scaled)[:, 1]
        
        # Calculate additional metrics
        from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
        
        roc_auc = roc_auc_score(y, y_pred_proba)
        precision, recall, _ = precision_recall_curve(y, y_pred_proba)
        pr_auc = auc(recall, precision)
        
        # Risk level distribution
        risk_levels = []
        for prob in y_pred_proba:
            if prob < 0.25:
                risk_levels.append("Low")
            elif prob < 0.50:
                risk_levels.append("Moderate")
            elif prob < 0.75:
                risk_levels.append("High")
            else:
                risk_levels.append("Extreme")
        
        risk_distribution = pd.Series(risk_levels).value_counts().to_dict()
        
        evaluation = {
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'risk_distribution': risk_distribution,
            'avg_prediction_probability': float(np.mean(y_pred_proba)),
            'std_prediction_probability': float(np.std(y_pred_proba))
        }
        
        logger.info(f"Model evaluation completed. ROC AUC: {roc_auc:.3f}, PR AUC: {pr_auc:.3f}")
        
        return evaluation
    
    def run_training_pipeline(self, n_samples: int = 5000) -> dict:
        """
        Run the complete training pipeline.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            Dictionary with complete training results
        """
        logger.info("Starting ML training pipeline...")
        
        # Generate dataset
        df = self.generate_synthetic_dataset(n_samples)
        
        # Train model
        training_results = self.train_model(df)
        
        # Evaluate model
        evaluation_results = self.evaluate_model(df)
        
        # Save model
        save_success = self.save_model()
        
        # Combine results
        complete_results = {
            'training': training_results,
            'evaluation': evaluation_results,
            'model_saved': save_success,
            'dataset_size': n_samples,
            'feature_names': self.feature_names
        }
        
        logger.info("ML training pipeline completed successfully!")
        
        return complete_results


def main():
    """Main function to run the training."""
    trainer = FireModelTrainer()
    results = trainer.run_training_pipeline(n_samples=5000)
    
    print("\n" + "="*50)
    print("TRAINING RESULTS SUMMARY")
    print("="*50)
    print(f"Dataset Size: {results['dataset_size']}")
    print(f"Model Accuracy: {results['training']['accuracy']:.3f}")
    print(f"ROC AUC: {results['evaluation']['roc_auc']:.3f}")
    print(f"PR AUC: {results['evaluation']['pr_auc']:.3f}")
    print(f"Model Saved: {results['model_saved']}")
    print("\nFeature Importance:")
    for feature, importance in results['training']['feature_importance'].items():
        print(f"  {feature}: {importance:.3f}")
    print("\nRisk Distribution:")
    for risk_level, count in results['evaluation']['risk_distribution'].items():
        print(f"  {risk_level}: {count}")
    print("="*50)


if __name__ == "__main__":
    main()
