"""AI model training pipeline for intelligent dimensioning using TensorFlow Keras.

This module trains deep neural networks on professional DWG data to learn
dimensioning patterns and create an intelligent dimensioning system with
automatic hyperparameter tuning and state-of-the-art techniques.
"""

import logging
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Callable
from pathlib import Path
import sqlite3
from datetime import datetime
from dataclasses import dataclass
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, callbacks
from tensorflow.keras.utils import to_categorical
import keras_tuner as kt
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import joblib
import warnings
warnings.filterwarnings('ignore')

from .dimension_ai import DrawingType, EntityImportance, DWGAnalyzer

logger = logging.getLogger(__name__)

# Set TensorFlow to use GPU if available
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logger.info(f"Using GPU: {len(gpus)} devices found")
    except RuntimeError as e:
        logger.warning(f"GPU setup failed: {e}")
else:
    logger.info("Using CPU for training")


@dataclass
class ModelPerformance:
    """Enhanced model performance metrics for neural networks."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    val_accuracy: float
    val_loss: float
    confusion_matrix: List[List[int]]
    classification_report: str
    training_history: Dict[str, List[float]]
    best_hyperparameters: Dict[str, Any]
    ensemble_accuracy: Optional[float] = None


class AdvancedFeatureEngineer:
    """Advanced feature engineering for neural networks."""
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = RobustScaler()  # More robust than StandardScaler
        self.feature_names = []
        self.categorical_feature_dims = {}
        self.is_fitted = False
    
    def create_entity_features(self, geometry_features: List[Dict]) -> np.ndarray:
        """Create advanced feature vectors for entities."""
        features = []
        
        for entity in geometry_features:
            feature_vector = []
            
            # Enhanced geometric features
            length = entity.get('length', 0) or 0
            area = entity.get('area', 0) or 0
            angle = entity.get('angle', 0) or 0
            lineweight = entity.get('lineweight', 0) or 0
            color = entity.get('color', 0) or 0
            
            # Normalize and add geometric features
            feature_vector.extend([
                length,
                area,
                np.sin(angle) if angle else 0,  # Trigonometric encoding
                np.cos(angle) if angle else 0,
                lineweight,
                color,
                np.log1p(length) if length > 0 else 0,  # Log transformation
                np.log1p(area) if area > 0 else 0,
            ])
            
            # Enhanced bounding box features
            bbox = entity.get('bounding_box', [0, 0, 0, 0])
            if bbox and len(bbox) >= 4:
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                area_bbox = width * height
                aspect_ratio = width / height if height > 0 else 0
                perimeter = 2 * (width + height)
                
                feature_vector.extend([
                    width, height, area_bbox, aspect_ratio, perimeter,
                    np.log1p(width) if width > 0 else 0,
                    np.log1p(height) if height > 0 else 0,
                    width * height,  # Area calculation
                    width / (width + height) if (width + height) > 0 else 0,  # Width ratio
                    height / (width + height) if (width + height) > 0 else 0,  # Height ratio
                ])
            else:
                feature_vector.extend([0] * 10)
            
            # Advanced categorical encoding
            entity_type = entity.get('entity_type', 'Unknown')
            layer_name = entity.get('layer_name', 'Unknown')
            linetype = entity.get('linetype', 'Unknown')
            
            # Use label encoding for categorical features
            if not self.is_fitted:
                # During training, we'll fit these properly
                feature_vector.extend([
                    hash(entity_type) % 50,
                    hash(layer_name) % 50,
                    hash(linetype) % 20,
                ])
            else:
                # Use fitted encoders
                feature_vector.extend([
                    self._encode_categorical('entity_type', entity_type),
                    self._encode_categorical('layer_name', layer_name),
                    self._encode_categorical('linetype', linetype),
                ])
            
            # Entity complexity features
            complexity_score = self._calculate_entity_complexity(entity)
            feature_vector.append(complexity_score)
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def _encode_categorical(self, feature_name: str, value: str) -> int:
        """Encode categorical feature using fitted encoder."""
        if feature_name in self.label_encoders:
            try:
                return self.label_encoders[feature_name].transform([value])[0]
            except ValueError:
                # Handle unseen categories
                return 0
        return hash(value) % 50
    
    def _calculate_entity_complexity(self, entity: Dict) -> float:
        """Calculate entity complexity score."""
        complexity = 0.0
        
        # Length-based complexity
        length = entity.get('length', 0) or 0
        if length > 0:
            complexity += min(length / 1000.0, 10.0)  # Normalize to 0-10
        
        # Area-based complexity
        area = entity.get('area', 0) or 0
        if area > 0:
            complexity += min(area / 100000.0, 5.0)  # Normalize to 0-5
        
        # Layer-based complexity
        layer_name = entity.get('layer_name', '').lower()
        if any(keyword in layer_name for keyword in ['wall', 'structure', 'beam']):
            complexity += 3.0
        elif any(keyword in layer_name for keyword in ['door', 'window']):
            complexity += 2.0
        elif any(keyword in layer_name for keyword in ['text', 'dimension']):
            complexity -= 1.0
        
        return complexity
    
    def fit_categorical_encoders(self, geometry_features: List[Dict]):
        """Fit categorical encoders on training data."""
        entity_types = []
        layer_names = []
        linetypes = []
        
        for entity in geometry_features:
            entity_types.append(entity.get('entity_type', 'Unknown'))
            layer_names.append(entity.get('layer_name', 'Unknown'))
            linetypes.append(entity.get('linetype', 'Unknown'))
        
        # Fit encoders
        self.label_encoders['entity_type'] = LabelEncoder()
        self.label_encoders['layer_name'] = LabelEncoder()
        self.label_encoders['linetype'] = LabelEncoder()
        
        self.label_encoders['entity_type'].fit(entity_types)
        self.label_encoders['layer_name'].fit(layer_names)
        self.label_encoders['linetype'].fit(linetypes)
        
        self.is_fitted = True
        logger.info("Categorical encoders fitted successfully")
    
# Import the advanced neural network trainer
from .ai_trainer_nn import AdvancedDimensionAITrainer, AdvancedFeatureEngineer

# Create alias for backward compatibility
DimensionAITrainer = AdvancedDimensionAITrainer
FeatureEngineer = AdvancedFeatureEngineer

if __name__ == "__main__":
    # Example usage
    trainer = AdvancedDimensionAITrainer()
    print("Advanced Neural Network AI Trainer initialized")
    print("Use trainer.train_all_models() to train AI models with hyperparameter tuning")