"""
Main training interface for AI dimensioning models.

This is the primary module you'll use to train and manage your AI models.
Run this script to train neural networks on your professional DWG files.
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import argparse
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.infrastructure.ai.dimension_ai import DWGAnalyzer
from src.infrastructure.ai.ai_trainer_nn import AdvancedDimensionAITrainer
from src.infrastructure.autocad.connection import AutoCADConnection

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ai_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class AIModelTrainer:
    """Main interface for training AI dimensioning models."""
    
    def __init__(self, storage_path: str = "ai_training_data"):
        """Initialize the AI model trainer.
        
        Args:
            storage_path: Directory to store training data and models
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        # Initialize components
        self.analyzer = DWGAnalyzer(storage_path=str(self.storage_path))
        self.trainer = AdvancedDimensionAITrainer(storage_path=str(self.storage_path))
        self.cad_connection = None
        
        logger.info(f"AI Model Trainer initialized with storage: {self.storage_path}")
    
    def setup_autocad_connection(self) -> bool:
        """Setup AutoCAD connection for analyzing DWG files."""
        try:
            self.cad_connection = AutoCADConnection()
            if self.cad_connection.connect():
                logger.info("AutoCAD connection established")
                return True
            else:
                logger.error("Failed to connect to AutoCAD")
                return False
        except Exception as e:
            logger.error(f"Error setting up AutoCAD connection: {e}")
            return False
    
    def analyze_dwg_directory(self, dwg_directory: str) -> int:
        """Analyze all DWG files in a directory and extract training data.
        
        Args:
            dwg_directory: Path to directory containing DWG files
            
        Returns:
            Number of successfully analyzed files
        """
        logger.info(f"Starting analysis of DWG directory: {dwg_directory}")
        
        if not self.cad_connection:
            if not self.setup_autocad_connection():
                return 0
        
        # Find all DWG files
        dwg_files = list(Path(dwg_directory).glob("**/*.dwg"))
        logger.info(f"Found {len(dwg_files)} DWG files to analyze")
        
        # Analyze each file
        successful_analyses = 0
        failed_analyses = []
        
        for i, dwg_file in enumerate(dwg_files, 1):
            try:
                logger.info(f"Analyzing file {i}/{len(dwg_files)}: {dwg_file.name}")
                
                example = self.analyzer.analyze_dwg_file(str(dwg_file), self.cad_connection)
                if example:
                    successful_analyses += 1
                    logger.info(f"✅ Successfully analyzed: {dwg_file.name}")
                else:
                    failed_analyses.append(dwg_file.name)
                    logger.warning(f"❌ Failed to analyze: {dwg_file.name}")
                    
            except Exception as e:
                failed_analyses.append(dwg_file.name)
                logger.error(f"❌ Error analyzing {dwg_file.name}: {e}")
        
        logger.info(f"Analysis complete: {successful_analyses}/{len(dwg_files)} files successful")
        if failed_analyses:
            logger.warning(f"Failed files: {', '.join(failed_analyses)}")
        
        return successful_analyses
    
    def train_models(self, force_retrain: bool = False) -> Dict[str, Any]:
        """Train AI models on the extracted data.
        
        Args:
            force_retrain: Force retraining even if models exist
            
        Returns:
            Training performance metrics
        """
        logger.info("Starting AI model training...")
        
        # Check if models already exist
        if not force_retrain and self._models_exist():
            logger.info("Models already exist. Use force_retrain=True to retrain.")
            return self.get_training_statistics()
        
        # Get training data statistics
        stats = self.analyzer.get_training_statistics()
        if stats['total_examples'] == 0:
            logger.error("No training data available. Please analyze DWG files first.")
            return {}
        
        logger.info(f"Training on {stats['total_examples']} examples with {stats['total_entities']} entities")
        
        # Train models
        try:
            performance = self.trainer.train_all_models()
            
            # Log results
            entity_perf = performance['entity_importance']
            logger.info(f"🎯 Training completed successfully!")
            logger.info(f"📊 Entity Classification Accuracy: {entity_perf.accuracy:.2%}")
            logger.info(f"📊 Ensemble Accuracy: {entity_perf.ensemble_accuracy:.2%}")
            logger.info(f"📊 Validation Accuracy: {entity_perf.val_accuracy:.2%}")
            
            return performance
            
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            return {}
    
    def incremental_training(self, new_dwg_files: List[str]) -> Dict[str, Any]:
        """Add new DWG files and retrain models.
        
        Args:
            new_dwg_files: List of paths to new DWG files
            
        Returns:
            Updated training performance metrics
        """
        logger.info(f"Starting incremental training with {len(new_dwg_files)} new files")
        
        if not self.cad_connection:
            if not self.setup_autocad_connection():
                return {}
        
        # Analyze new files
        successful_analyses = 0
        for dwg_file in new_dwg_files:
            try:
                example = self.analyzer.analyze_dwg_file(dwg_file, self.cad_connection)
                if example:
                    successful_analyses += 1
                    logger.info(f"✅ Analyzed new file: {Path(dwg_file).name}")
            except Exception as e:
                logger.error(f"❌ Error analyzing {dwg_file}: {e}")
        
        logger.info(f"Analyzed {successful_analyses}/{len(new_dwg_files)} new files")
        
        # Retrain models
        return self.train_models(force_retrain=True)
    
    def _models_exist(self) -> bool:
        """Check if trained models exist."""
        models_path = self.storage_path / "models"
        return (models_path / "entity_importance_model.keras").exists()
    
    def get_training_statistics(self) -> Dict[str, Any]:
        """Get comprehensive training statistics."""
        stats = {}
        
        # Data statistics
        data_stats = self.analyzer.get_training_statistics()
        stats['data'] = data_stats
        
        # Model statistics
        model_stats = self.trainer.get_model_statistics()
        stats['models'] = model_stats
        
        return stats
    
    def validate_model_performance(self) -> Dict[str, Any]:
        """Validate model performance on test data."""
        logger.info("Validating model performance...")
        
        # Load models
        self.trainer.load_trained_models()
        
        # Get model statistics
        stats = self.get_training_statistics()
        
        # Check performance thresholds
        performance_report = {
            'model_loaded': self.trainer.entity_importance_model is not None,
            'meets_accuracy_threshold': False,
            'recommendations': []
        }
        
        if 'training_metadata' in stats['models']:
            metadata = stats['models']['training_metadata']
            entity_accuracy = metadata['performance_metrics']['entity_importance']['accuracy']
            
            performance_report['accuracy'] = entity_accuracy
            performance_report['meets_accuracy_threshold'] = entity_accuracy >= 0.90
            
            if entity_accuracy < 0.90:
                performance_report['recommendations'].append(
                    "Accuracy below 90% - Consider adding more training data"
                )
            elif entity_accuracy < 0.95:
                performance_report['recommendations'].append(
                    "Good accuracy - Consider fine-tuning with more diverse examples"
                )
            else:
                performance_report['recommendations'].append(
                    "Excellent accuracy - Model is ready for production use"
                )
        
        return performance_report
    
    def create_model_backup(self) -> str:
        """Create a backup of current models."""
        backup_path = self.storage_path / "model_backups" / datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path.mkdir(parents=True, exist_ok=True)
        
        import shutil
        models_path = self.storage_path / "models"
        if models_path.exists():
            shutil.copytree(models_path, backup_path / "models")
            logger.info(f"Model backup created at: {backup_path}")
            return str(backup_path)
        else:
            logger.warning("No models to backup")
            return ""


def main():
    """Main entry point for training AI models."""
    parser = argparse.ArgumentParser(description="Train AI dimensioning models")
    parser.add_argument("--dwg-dir", type=str, help="Directory containing DWG files for training")
    parser.add_argument("--storage", type=str, default="ai_training_data", 
                       help="Directory to store training data and models")
    parser.add_argument("--force-retrain", action="store_true", 
                       help="Force retraining even if models exist")
    parser.add_argument("--validate", action="store_true", 
                       help="Validate existing model performance")
    parser.add_argument("--stats", action="store_true", 
                       help="Show training statistics")
    parser.add_argument("--backup", action="store_true", 
                       help="Create model backup before training")
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = AIModelTrainer(storage_path=args.storage)
    
    try:
        # Show statistics
        if args.stats:
            stats = trainer.get_training_statistics()
            print("\n📊 Training Statistics:")
            print(f"Total Examples: {stats['data']['total_examples']}")
            print(f"Total Entities: {stats['data']['total_entities']}")
            print(f"Dimensioned Entities: {stats['data']['total_dimensioned_entities']}")
            print(f"Common Entity Types: {stats['data']['common_entity_types']}")
            
            if 'training_metadata' in stats['models']:
                metadata = stats['models']['training_metadata']
                print(f"\nModel Performance:")
                print(f"Training Date: {metadata['training_date']}")
                print(f"Accuracy: {metadata['performance_metrics']['entity_importance']['accuracy']:.2%}")
            
            return
        
        # Validate model
        if args.validate:
            performance = trainer.validate_model_performance()
            print(f"\n🔍 Model Validation Results:")
            print(f"Model Loaded: {performance['model_loaded']}")
            print(f"Meets Accuracy Threshold: {performance['meets_accuracy_threshold']}")
            if 'accuracy' in performance:
                print(f"Current Accuracy: {performance['accuracy']:.2%}")
            print(f"Recommendations: {'; '.join(performance['recommendations'])}")
            return
        
        # Create backup if requested
        if args.backup:
            backup_path = trainer.create_model_backup()
            if backup_path:
                print(f"📦 Model backup created: {backup_path}")
        
        # Analyze DWG files
        if args.dwg_dir:
            print(f"🔍 Analyzing DWG files in: {args.dwg_dir}")
            successful_analyses = trainer.analyze_dwg_directory(args.dwg_dir)
            print(f"✅ Successfully analyzed {successful_analyses} DWG files")
        
        # Train models
        print("🧠 Training AI models...")
        performance = trainer.train_models(force_retrain=args.force_retrain)
        
        if performance:
            entity_perf = performance['entity_importance']
            print(f"\n🎯 Training Results:")
            print(f"📊 Entity Classification Accuracy: {entity_perf.accuracy:.2%}")
            print(f"📊 Ensemble Accuracy: {entity_perf.ensemble_accuracy:.2%}")
            print(f"📊 Validation Accuracy: {entity_perf.val_accuracy:.2%}")
            print(f"📊 F1 Score: {entity_perf.f1_score:.2%}")
            
            # Validate performance
            validation = trainer.validate_model_performance()
            print(f"\n✅ Model validation: {'; '.join(validation['recommendations'])}")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        print(f"❌ Error: {e}")
        return 1
    
    print("\n🎉 Training completed successfully!")
    print("You can now use the AI models for intelligent dimensioning.")
    return 0


if __name__ == "__main__":
    exit(main())