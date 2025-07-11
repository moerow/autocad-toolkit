"""Advanced TensorFlow Keras AI trainer with hyperparameter tuning."""

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


class EntityImportanceHyperModel(kt.HyperModel):
    """Hyperparameter tuning model for entity importance classification."""
    
    def __init__(self, input_shape: int, num_classes: int):
        self.input_shape = input_shape
        self.num_classes = num_classes
    
    def build(self, hp):
        model = keras.Sequential()
        
        # Input layer
        model.add(layers.Input(shape=(self.input_shape,)))
        
        # Batch normalization
        model.add(layers.BatchNormalization())
        
        # Hidden layers with hyperparameter tuning
        for i in range(hp.Int('num_layers', 2, 6)):
            model.add(layers.Dense(
                units=hp.Int(f'units_{i}', min_value=32, max_value=512, step=32),
                activation=hp.Choice('activation', ['relu', 'elu', 'selu', 'swish']),
                kernel_regularizer=keras.regularizers.l2(hp.Float('l2_reg', 1e-5, 1e-1, sampling='LOG'))
            ))
            
            # Batch normalization
            model.add(layers.BatchNormalization())
            
            # Dropout
            model.add(layers.Dropout(hp.Float(f'dropout_{i}', 0.1, 0.5, step=0.1)))
        
        # Output layer
        model.add(layers.Dense(self.num_classes, activation='softmax'))
        
        # Compile with hyperparameter tuning
        model.compile(
            optimizer=hp.Choice('optimizer', ['adam', 'adamw', 'rmsprop']),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        # Set learning rate
        if hp.get('optimizer') == 'adam':
            model.optimizer.learning_rate = hp.Float('learning_rate', 1e-4, 1e-1, sampling='LOG')
        
        return model


class DimensionPlacementHyperModel(kt.HyperModel):
    """Hyperparameter tuning model for dimension placement prediction."""
    
    def __init__(self, input_shape: int, num_classes: int):
        self.input_shape = input_shape
        self.num_classes = num_classes
    
    def build(self, hp):
        # Input layers
        entity_input = layers.Input(shape=(self.input_shape,), name='entity_input')
        
        # Entity processing branch
        x = layers.BatchNormalization()(entity_input)
        x = layers.Dense(
            hp.Int('entity_dense_1', 64, 256, step=32),
            activation=hp.Choice('activation', ['relu', 'elu', 'selu'])
        )(x)
        x = layers.Dropout(hp.Float('dropout_1', 0.1, 0.4))(x)
        
        x = layers.Dense(
            hp.Int('entity_dense_2', 32, 128, step=16),
            activation=hp.Choice('activation', ['relu', 'elu', 'selu'])
        )(x)
        x = layers.Dropout(hp.Float('dropout_2', 0.1, 0.4))(x)
        
        # Spatial reasoning layers
        x = layers.Dense(
            hp.Int('spatial_dense', 16, 64, step=8),
            activation='relu'
        )(x)
        
        # Output layer
        output = layers.Dense(self.num_classes, activation='softmax', name='placement_output')(x)
        
        model = keras.Model(inputs=entity_input, outputs=output)
        
        model.compile(
            optimizer=hp.Choice('optimizer', ['adam', 'adamw']),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model


class AdvancedDimensionAITrainer:
    """Advanced AI trainer using TensorFlow Keras with hyperparameter tuning."""
    
    def __init__(self, storage_path: str = "ai_training_data"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        self.models_path = self.storage_path / "models"
        self.models_path.mkdir(exist_ok=True)
        
        self.tuner_path = self.storage_path / "tuner"
        self.tuner_path.mkdir(exist_ok=True)
        
        self.db_path = self.storage_path / "training_data.db"
        
        # Initialize components
        self.feature_engineer = AdvancedFeatureEngineer()
        self.entity_importance_model = None
        self.dimension_placement_model = None
        self.ensemble_models = []
        
        # Training configuration
        self.config = {
            'batch_size': 32,
            'epochs': 100,
            'patience': 15,
            'validation_split': 0.2,
            'test_split': 0.15,
            'cv_folds': 5,
            'ensemble_size': 3,
            'tuner_max_trials': 50,
            'tuner_executions_per_trial': 2
        }
        
        logger.info(f"Advanced AI Trainer initialized with storage at: {self.storage_path}")
    
    def load_training_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load training data from database."""
        conn = sqlite3.connect(self.db_path)
        
        # Load training examples
        examples_df = pd.read_sql_query("""
            SELECT * FROM training_examples
        """, conn)
        
        # Load geometry features
        geometry_df = pd.read_sql_query("""
            SELECT * FROM geometry_features
        """, conn)
        
        # Load dimension features
        dimensions_df = pd.read_sql_query("""
            SELECT * FROM dimension_features
        """, conn)
        
        # Load relationships
        relationships_df = pd.read_sql_query("""
            SELECT * FROM entity_dimension_relationships
        """, conn)
        
        conn.close()
        
        logger.info(f"Loaded {len(examples_df)} training examples, {len(geometry_df)} entities, {len(dimensions_df)} dimensions")
        return examples_df, geometry_df, dimensions_df
    
    def prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare training data for neural networks."""
        logger.info("Preparing training data for neural networks...")
        
        examples_df, geometry_df, dimensions_df = self.load_training_data()
        
        # Load relationships
        conn = sqlite3.connect(self.db_path)
        relationships_df = pd.read_sql_query("""
            SELECT * FROM entity_dimension_relationships
        """, conn)
        conn.close()
        
        # Prepare features and labels
        X = []
        y = []
        
        # Group by example
        for example_id in examples_df['id'].unique():
            example_data = examples_df[examples_df['id'] == example_id].iloc[0]
            entities = geometry_df[geometry_df['example_id'] == example_id]
            
            # Get context features for this drawing
            context_features = self._create_context_features(example_data)
            
            # Process each entity
            for _, entity in entities.iterrows():
                # Create entity features
                entity_dict = {
                    'length': entity['length'],
                    'area': entity['area'],
                    'angle': entity['angle'],
                    'lineweight': entity['lineweight'],
                    'color': entity['color'],
                    'bounding_box': json.loads(entity['bounding_box']) if entity['bounding_box'] else [0, 0, 0, 0],
                    'entity_type': entity['entity_type'],
                    'layer_name': entity['layer_name'],
                    'linetype': entity['linetype']
                }
                
                # Create entity features
                entity_features = self.feature_engineer.create_entity_features([entity_dict])[0]
                
                # Combine with context features
                combined_features = np.concatenate([entity_features, context_features])
                X.append(combined_features)
                
                # Determine importance label
                is_dimensioned = len(relationships_df[
                    (relationships_df['example_id'] == example_id) & 
                    (relationships_df['entity_id'] == entity['entity_id'])
                ]) > 0
                
                importance = self._determine_entity_importance(entity, is_dimensioned)
                y.append(importance.value)
        
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        # Encode labels
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        # Save label encoder
        joblib.dump(label_encoder, self.models_path / "label_encoder.pkl")
        
        # Convert to categorical
        y_categorical = to_categorical(y_encoded)
        
        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y_categorical, test_size=self.config['test_split'] + self.config['validation_split'], 
            random_state=42, stratify=y_categorical.argmax(axis=1)
        )
        
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=self.config['test_split'] / (self.config['test_split'] + self.config['validation_split']), 
            random_state=42, stratify=y_temp.argmax(axis=1)
        )
        
        # Fit feature engineer
        self.feature_engineer.fit_categorical_encoders([
            {'entity_type': 'AcDbLine', 'layer_name': 'WALLS', 'linetype': 'Continuous'}
        ])
        
        # Scale features
        X_train_scaled = self.feature_engineer.scaler.fit_transform(X_train)
        X_val_scaled = self.feature_engineer.scaler.transform(X_val)
        X_test_scaled = self.feature_engineer.scaler.transform(X_test)
        
        # Save scaler
        joblib.dump(self.feature_engineer.scaler, self.models_path / "feature_scaler.pkl")
        
        logger.info(f"Training data prepared: {X_train.shape}, Validation: {X_val.shape}, Test: {X_test.shape}")
        return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test
    
    def _create_context_features(self, example_data: pd.Series) -> np.ndarray:
        """Create context features for the drawing."""
        context_features = []
        
        # Drawing type features
        drawing_type = example_data['drawing_type']
        drawing_type_encoded = hash(drawing_type) % 10
        context_features.append(drawing_type_encoded)
        
        # Scale features
        drawing_scale = example_data['drawing_scale'] or 1.0
        context_features.append(drawing_scale)
        
        # Complexity features
        total_entities = example_data['total_entities']
        dimensioned_entities = example_data['dimensioned_entities']
        dimensioning_ratio = dimensioned_entities / total_entities if total_entities > 0 else 0
        
        context_features.extend([
            total_entities,
            dimensioned_entities,
            dimensioning_ratio
        ])
        
        # Drawing bounds features
        drawing_bounds = json.loads(example_data['drawing_bounds'])
        if drawing_bounds and len(drawing_bounds) >= 4:
            width = drawing_bounds[2] - drawing_bounds[0]
            height = drawing_bounds[3] - drawing_bounds[1]
            area = width * height
            context_features.extend([width, height, area])
        else:
            context_features.extend([0, 0, 0])
        
        return np.array(context_features)
    
    def _determine_entity_importance(self, entity: pd.Series, is_dimensioned: bool) -> EntityImportance:
        """Determine entity importance based on characteristics."""
        entity_type = entity['entity_type']
        layer_name = entity['layer_name'].lower()
        length = entity['length'] or 0
        
        # Enhanced rule-based importance classification
        if is_dimensioned:
            if any(keyword in layer_name for keyword in ['wall', 'structure', 'beam', 'column']):
                return EntityImportance.CRITICAL
            elif any(keyword in layer_name for keyword in ['door', 'window', 'opening']):
                return EntityImportance.CRITICAL
            elif length > 1000:
                return EntityImportance.IMPORTANT
            else:
                return EntityImportance.DETAIL
        else:
            if any(keyword in layer_name for keyword in ['text', 'dimension', 'hatch', 'symbol']):
                return EntityImportance.IGNORE
            elif any(keyword in layer_name for keyword in ['construction', 'hidden', 'temp']):
                return EntityImportance.IGNORE
            elif entity_type in ['AcDbText', 'AcDbMText', 'AcDbHatch']:
                return EntityImportance.IGNORE
            else:
                return EntityImportance.DETAIL
    
    def train_entity_importance_model(self) -> ModelPerformance:
        """Train entity importance model with hyperparameter tuning."""
        logger.info("Training entity importance model with hyperparameter tuning...")
        
        # Prepare data
        X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_training_data()
        
        # Initialize hyperparameter tuner
        tuner = kt.RandomSearch(
            EntityImportanceHyperModel(input_shape=X_train.shape[1], num_classes=y_train.shape[1]),
            objective='val_accuracy',
            max_trials=self.config['tuner_max_trials'],
            executions_per_trial=self.config['tuner_executions_per_trial'],
            directory=self.tuner_path,
            project_name='entity_importance_tuning'
        )
        
        # Callbacks
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=self.config['patience'],
            restore_best_weights=True
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
        
        # Search for best hyperparameters
        logger.info("Starting hyperparameter search...")
        tuner.search(
            X_train, y_train,
            epochs=self.config['epochs'],
            validation_data=(X_val, y_val),
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        # Get best model
        best_model = tuner.get_best_models(num_models=1)[0]
        best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]
        
        logger.info(f"Best hyperparameters: {best_hyperparameters.values}")
        
        # Train ensemble of models
        ensemble_models = []
        ensemble_histories = []
        
        for i in range(self.config['ensemble_size']):
            logger.info(f"Training ensemble model {i+1}/{self.config['ensemble_size']}")
            
            # Create model with best hyperparameters
            model = EntityImportanceHyperModel(
                input_shape=X_train.shape[1], 
                num_classes=y_train.shape[1]
            ).build(best_hyperparameters)
            
            # Train model
            history = model.fit(
                X_train, y_train,
                epochs=self.config['epochs'],
                batch_size=self.config['batch_size'],
                validation_data=(X_val, y_val),
                callbacks=[early_stopping, reduce_lr],
                verbose=0
            )
            
            ensemble_models.append(model)
            ensemble_histories.append(history.history)
        
        # Save ensemble
        for i, model in enumerate(ensemble_models):
            model.save(self.models_path / f"entity_importance_model_{i}.keras")
        
        # Evaluate ensemble
        ensemble_predictions = []
        for model in ensemble_models:
            pred = model.predict(X_test)
            ensemble_predictions.append(pred)
        
        # Average predictions
        ensemble_pred = np.mean(ensemble_predictions, axis=0)
        ensemble_pred_classes = np.argmax(ensemble_pred, axis=1)
        y_test_classes = np.argmax(y_test, axis=1)
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        ensemble_accuracy = accuracy_score(y_test_classes, ensemble_pred_classes)
        precision = precision_score(y_test_classes, ensemble_pred_classes, average='weighted')
        recall = recall_score(y_test_classes, ensemble_pred_classes, average='weighted')
        f1 = f1_score(y_test_classes, ensemble_pred_classes, average='weighted')
        
        # Get best individual model performance
        best_pred = best_model.predict(X_test)
        best_pred_classes = np.argmax(best_pred, axis=1)
        best_accuracy = accuracy_score(y_test_classes, best_pred_classes)
        
        # Create performance object
        performance = ModelPerformance(
            accuracy=best_accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            val_accuracy=max([max(h['val_accuracy']) for h in ensemble_histories]),
            val_loss=min([min(h['val_loss']) for h in ensemble_histories]),
            confusion_matrix=confusion_matrix(y_test_classes, ensemble_pred_classes).tolist(),
            classification_report=classification_report(y_test_classes, ensemble_pred_classes),
            training_history=ensemble_histories[0],  # Use first model's history
            best_hyperparameters=best_hyperparameters.values,
            ensemble_accuracy=ensemble_accuracy
        )
        
        # Save best model as primary
        best_model.save(self.models_path / "entity_importance_model.keras")
        self.entity_importance_model = best_model
        
        logger.info(f"Entity importance model trained - Accuracy: {best_accuracy:.3f}, Ensemble: {ensemble_accuracy:.3f}")
        return performance
    
    def train_all_models(self) -> Dict[str, ModelPerformance]:
        """Train all AI models with advanced techniques."""
        logger.info("Training all AI models with advanced techniques...")
        
        performances = {}
        
        # Train entity importance model
        performances['entity_importance'] = self.train_entity_importance_model()
        
        # Save training metadata
        metadata = {
            'training_date': datetime.now().isoformat(),
            'model_versions': {
                'entity_importance': 'v2.0_neural_network',
                'dimension_placement': 'v2.0_neural_network'
            },
            'performance_metrics': {
                'entity_importance': {
                    'accuracy': performances['entity_importance'].accuracy,
                    'ensemble_accuracy': performances['entity_importance'].ensemble_accuracy,
                    'val_accuracy': performances['entity_importance'].val_accuracy
                }
            },
            'hyperparameters': performances['entity_importance'].best_hyperparameters,
            'training_config': self.config
        }
        
        metadata_path = self.models_path / "training_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info("All models trained successfully with neural networks")
        return performances
    
    def load_trained_models(self):
        """Load previously trained models."""
        try:
            # Load entity importance model
            model_path = self.models_path / "entity_importance_model.keras"
            if model_path.exists():
                self.entity_importance_model = keras.models.load_model(model_path)
                logger.info("Entity importance model loaded")
            
            # Load feature scaler
            scaler_path = self.models_path / "feature_scaler.pkl"
            if scaler_path.exists():
                self.feature_engineer.scaler = joblib.load(scaler_path)
                logger.info("Feature scaler loaded")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
    
    def predict_entity_importance(self, entity_features: np.ndarray, 
                                context_features: np.ndarray) -> EntityImportance:
        """Predict the importance of an entity for dimensioning."""
        if self.entity_importance_model is None:
            return EntityImportance.DETAIL
        
        try:
            # Combine features
            combined_features = np.concatenate([entity_features, context_features])
            combined_features = combined_features.reshape(1, -1)
            
            # Scale features
            scaled_features = self.feature_engineer.scaler.transform(combined_features)
            
            # Predict
            prediction = self.entity_importance_model.predict(scaled_features, verbose=0)[0]
            predicted_class = np.argmax(prediction)
            
            # Load label encoder
            label_encoder = joblib.load(self.models_path / "label_encoder.pkl")
            importance_label = label_encoder.inverse_transform([predicted_class])[0]
            
            # Convert to EntityImportance
            importance_map = {
                'critical': EntityImportance.CRITICAL,
                'important': EntityImportance.IMPORTANT,
                'detail': EntityImportance.DETAIL,
                'ignore': EntityImportance.IGNORE
            }
            
            return importance_map.get(importance_label, EntityImportance.DETAIL)
            
        except Exception as e:
            logger.error(f"Error predicting entity importance: {e}")
            return EntityImportance.DETAIL
    
    def incremental_training(self, new_dwg_files: List[str], cad_connection) -> ModelPerformance:
        """Perform incremental training with new DWG files."""
        logger.info(f"Starting incremental training with {len(new_dwg_files)} new files...")
        
        # Analyze new files
        analyzer = DWGAnalyzer(storage_path=str(self.storage_path))
        for dwg_file in new_dwg_files:
            analyzer.analyze_dwg_file(dwg_file, cad_connection)
        
        # Retrain models
        return self.train_all_models()
    
    def get_model_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about trained models."""
        stats = {}
        
        # Check which models exist
        stats['models_available'] = []
        
        if (self.models_path / "entity_importance_model.keras").exists():
            stats['models_available'].append('entity_importance_neural_network')
        
        # Load training metadata if available
        metadata_path = self.models_path / "training_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                stats['training_metadata'] = metadata
        
        return stats


class AdvancedFeatureEngineer:
    """Advanced feature engineering for neural networks."""
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = RobustScaler()
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
                np.sin(angle) if angle else 0,
                np.cos(angle) if angle else 0,
                lineweight,
                color,
                np.log1p(length) if length > 0 else 0,
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
                    width * height,
                    width / (width + height) if (width + height) > 0 else 0,
                    height / (width + height) if (width + height) > 0 else 0,
                ])
            else:
                feature_vector.extend([0] * 10)
            
            # Advanced categorical encoding
            entity_type = entity.get('entity_type', 'Unknown')
            layer_name = entity.get('layer_name', 'Unknown')
            linetype = entity.get('linetype', 'Unknown')
            
            # Use label encoding for categorical features
            if not self.is_fitted:
                feature_vector.extend([
                    hash(entity_type) % 50,
                    hash(layer_name) % 50,
                    hash(linetype) % 20,
                ])
            else:
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
                return 0
        return hash(value) % 50
    
    def _calculate_entity_complexity(self, entity: Dict) -> float:
        """Calculate entity complexity score."""
        complexity = 0.0
        
        # Length-based complexity
        length = entity.get('length', 0) or 0
        if length > 0:
            complexity += min(length / 1000.0, 10.0)
        
        # Area-based complexity
        area = entity.get('area', 0) or 0
        if area > 0:
            complexity += min(area / 100000.0, 5.0)
        
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


if __name__ == "__main__":
    # Example usage
    trainer = AdvancedDimensionAITrainer()
    print("Advanced Neural Network AI Trainer initialized")
    print("Use trainer.train_all_models() to train AI models with hyperparameter tuning")