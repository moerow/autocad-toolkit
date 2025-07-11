"""Intelligent dimensioning engine using AI models.

This module implements the intelligent dimensioning system that uses trained AI models
to automatically select and dimension entities following professional standards.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import math

from src.infrastructure.autocad.connection import AutoCADConnection
from src.application.services.dimension_service import DimensionService
from .dimension_ai import DrawingType, EntityImportance, GeometryFeature, DimensionFeature
from .ai_trainer import AdvancedDimensionAITrainer, AdvancedFeatureEngineer

logger = logging.getLogger(__name__)


@dataclass
class IntelligentDimensionPlan:
    """A plan for intelligent dimensioning of a drawing."""
    critical_entities: List[Dict[str, Any]]
    important_entities: List[Dict[str, Any]]
    detail_entities: List[Dict[str, Any]]
    dimension_groups: List[List[Dict[str, Any]]]
    placement_strategy: str
    estimated_dimensions: int
    confidence_score: float


@dataclass
class DimensionPlacement:
    """Information about where to place a dimension."""
    entity_1_id: str
    entity_2_id: str
    dimension_type: str
    placement_position: Tuple[float, float]
    text_position: Tuple[float, float]
    dimension_value: float
    confidence: float
    priority: int


class IntelligentDimensionService:
    """AI-powered intelligent dimensioning service."""
    
    def __init__(self, cad_connection: AutoCADConnection):
        self.cad_connection = cad_connection
        self.traditional_service = DimensionService(cad_connection)
        self.ai_trainer = AdvancedDimensionAITrainer()
        self.feature_engineer = AdvancedFeatureEngineer()
        
        # Load trained models
        self.ai_trainer.load_trained_models()
        
        # Configuration
        self.config = {
            'min_critical_length': 1.0,     # Minimum length for critical dimensions
            'min_important_length': 0.5,    # Minimum length for important dimensions
            'min_detail_length': 0.1,       # Minimum length for detail dimensions
            'max_dimensions_per_entity': 2,  # Maximum dimensions per entity
            'dimension_spacing': 0.5,        # Spacing between dimension lines
            'text_height': 0.08,            # Default text height
            'arrow_size': 0.05,             # Default arrow size
            'layer_name': 'AI_DIMENSIONS',   # Layer for AI dimensions
        }
        
        logger.info("Intelligent Dimension Service initialized")
    
    def analyze_drawing_for_intelligent_dimensioning(self, layer_filter: Optional[str] = None) -> IntelligentDimensionPlan:
        """Analyze drawing and create intelligent dimensioning plan."""
        logger.info("Analyzing drawing for intelligent dimensioning...")
        
        # Get all entities from the drawing
        entities = self._extract_entities_from_drawing(layer_filter)
        
        if not entities:
            logger.warning("No entities found in drawing")
            return IntelligentDimensionPlan([], [], [], [], "none", 0, 0.0)
        
        # Classify drawing type
        drawing_type = self._classify_current_drawing(entities)
        
        # Create context features
        context_features = self._create_drawing_context_features(entities, drawing_type)
        
        # Classify entity importance using AI
        classified_entities = self._classify_entity_importance(entities, context_features)
        
        # Group entities for logical dimensioning
        dimension_groups = self._create_dimension_groups(classified_entities)
        
        # Determine placement strategy
        placement_strategy = self._determine_placement_strategy(drawing_type, classified_entities)
        
        # Estimate number of dimensions
        estimated_dimensions = self._estimate_dimension_count(classified_entities)
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence_score(classified_entities)
        
        plan = IntelligentDimensionPlan(
            critical_entities=classified_entities['critical'],
            important_entities=classified_entities['important'],
            detail_entities=classified_entities['detail'],
            dimension_groups=dimension_groups,
            placement_strategy=placement_strategy,
            estimated_dimensions=estimated_dimensions,
            confidence_score=confidence_score
        )
        
        logger.info(f"Intelligent dimensioning plan created: {estimated_dimensions} dimensions planned")
        return plan
    
    def execute_intelligent_dimensioning(self, plan: IntelligentDimensionPlan) -> Dict[str, int]:
        """Execute the intelligent dimensioning plan."""
        logger.info("Executing intelligent dimensioning plan...")
        
        results = {
            'critical_dimensions': 0,
            'important_dimensions': 0,
            'detail_dimensions': 0,
            'total_dimensions': 0,
            'skipped_entities': 0
        }
        
        # Set up layer for AI dimensions
        self._ensure_ai_dimension_layer()
        
        # Execute dimensioning by priority
        if plan.placement_strategy == "hierarchical":
            # Dimension critical entities first
            results['critical_dimensions'] = self._dimension_entity_group(
                plan.critical_entities, priority=1
            )
            
            # Then important entities
            results['important_dimensions'] = self._dimension_entity_group(
                plan.important_entities, priority=2
            )
            
            # Finally detail entities (if space permits)
            if results['critical_dimensions'] + results['important_dimensions'] < 50:  # Avoid clutter
                results['detail_dimensions'] = self._dimension_entity_group(
                    plan.detail_entities, priority=3
                )
        
        elif plan.placement_strategy == "grouped":
            # Dimension by logical groups
            for group in plan.dimension_groups:
                group_results = self._dimension_entity_group(group, priority=1)
                results['total_dimensions'] += group_results
        
        else:
            # Default strategy - dimension by importance
            results['critical_dimensions'] = self._dimension_entity_group(
                plan.critical_entities, priority=1
            )
            results['important_dimensions'] = self._dimension_entity_group(
                plan.important_entities, priority=2
            )
        
        # Calculate totals
        results['total_dimensions'] = (
            results['critical_dimensions'] + 
            results['important_dimensions'] + 
            results['detail_dimensions']
        )
        
        logger.info(f"Intelligent dimensioning completed: {results['total_dimensions']} dimensions added")
        return results
    
    def intelligent_dimension_all(self, layer_filter: Optional[str] = None) -> Dict[str, Any]:
        """Complete intelligent dimensioning workflow."""
        logger.info("Starting intelligent dimensioning workflow...")
        
        # Analyze drawing
        plan = self.analyze_drawing_for_intelligent_dimensioning(layer_filter)
        
        # Execute plan
        results = self.execute_intelligent_dimensioning(plan)
        
        # Add plan information to results
        results['plan'] = {
            'placement_strategy': plan.placement_strategy,
            'estimated_dimensions': plan.estimated_dimensions,
            'confidence_score': plan.confidence_score,
            'critical_entities': len(plan.critical_entities),
            'important_entities': len(plan.important_entities),
            'detail_entities': len(plan.detail_entities)
        }
        
        return results
    
    def _extract_entities_from_drawing(self, layer_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Extract entities from the current drawing."""
        entities = []
        
        try:
            for i, entity in enumerate(self.cad_connection.model):
                # Apply layer filter if specified
                if layer_filter and hasattr(entity, 'Layer') and entity.Layer != layer_filter:
                    continue
                
                # Extract entity information
                entity_info = {
                    'id': f"entity_{i}",
                    'autocad_entity': entity,
                    'type': entity.ObjectName,
                    'layer': getattr(entity, 'Layer', 'Unknown'),
                    'color': getattr(entity, 'Color', 0),
                    'linetype': getattr(entity, 'Linetype', 'Continuous'),
                    'lineweight': getattr(entity, 'LineWeight', 0),
                }
                
                # Get geometric properties
                if entity.ObjectName == 'AcDbLine':
                    entity_info.update({
                        'start_point': (entity.StartPoint[0], entity.StartPoint[1]),
                        'end_point': (entity.EndPoint[0], entity.EndPoint[1]),
                        'length': entity.Length,
                        'angle': entity.Angle,
                    })
                elif entity.ObjectName in ['AcDbPolyline', 'AcDb2dPolyline']:
                    entity_info.update({
                        'length': entity.Length,
                        'area': getattr(entity, 'Area', 0),
                        'coordinates': entity.Coordinates,
                    })
                elif entity.ObjectName == 'AcDbCircle':
                    entity_info.update({
                        'center': (entity.Center[0], entity.Center[1]),
                        'radius': entity.Radius,
                        'area': entity.Area,
                    })
                elif entity.ObjectName == 'AcDbArc':
                    entity_info.update({
                        'center': (entity.Center[0], entity.Center[1]),
                        'radius': entity.Radius,
                        'start_angle': entity.StartAngle,
                        'end_angle': entity.EndAngle,
                        'length': entity.ArcLength,
                    })
                
                # Get bounding box
                try:
                    bounds = entity.GetBoundingBox()
                    entity_info['bounding_box'] = (
                        bounds[0][0], bounds[0][1], bounds[1][0], bounds[1][1]
                    )
                except:
                    entity_info['bounding_box'] = (0, 0, 0, 0)
                
                entities.append(entity_info)
        
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
        
        return entities
    
    def _classify_current_drawing(self, entities: List[Dict[str, Any]]) -> DrawingType:
        """Classify the current drawing type."""
        # Simple heuristic-based classification
        # In a real implementation, this could use AI
        
        # Check for typical architectural elements
        has_walls = any('wall' in entity['layer'].lower() for entity in entities)
        has_doors = any('door' in entity['layer'].lower() for entity in entities)
        has_windows = any('window' in entity['layer'].lower() for entity in entities)
        
        if has_walls and (has_doors or has_windows):
            return DrawingType.FLOOR_PLAN
        
        # Check for typical section elements
        has_structure = any('beam' in entity['layer'].lower() or 'column' in entity['layer'].lower() 
                          for entity in entities)
        if has_structure:
            return DrawingType.SECTION
        
        # Check for site plan elements
        has_site_elements = any('site' in entity['layer'].lower() or 'boundary' in entity['layer'].lower()
                              for entity in entities)
        if has_site_elements:
            return DrawingType.SITE_PLAN
        
        return DrawingType.UNKNOWN
    
    def _create_drawing_context_features(self, entities: List[Dict[str, Any]], 
                                       drawing_type: DrawingType) -> np.ndarray:
        """Create context features for the drawing."""
        context_data = {
            'drawing_type': drawing_type.value,
            'drawing_scale': 1.0,  # Would need to determine this
            'total_entities': len(entities),
            'dimensioned_entities': 0,  # Unknown for current drawing
            'drawing_bounds': self._calculate_drawing_bounds(entities)
        }
        
        return self.feature_engineer.create_context_features(context_data)
    
    def _calculate_drawing_bounds(self, entities: List[Dict[str, Any]]) -> List[float]:
        """Calculate the overall bounds of the drawing."""
        if not entities:
            return [0, 0, 0, 0]
        
        bboxes = [entity['bounding_box'] for entity in entities if 'bounding_box' in entity]
        if not bboxes:
            return [0, 0, 0, 0]
        
        min_x = min(bbox[0] for bbox in bboxes)
        min_y = min(bbox[1] for bbox in bboxes)
        max_x = max(bbox[2] for bbox in bboxes)
        max_y = max(bbox[3] for bbox in bboxes)
        
        return [min_x, min_y, max_x, max_y]
    
    def _classify_entity_importance(self, entities: List[Dict[str, Any]], 
                                  context_features: np.ndarray) -> Dict[str, List[Dict[str, Any]]]:
        """Classify entity importance using AI models."""
        classified = {
            'critical': [],
            'important': [],
            'detail': [],
            'ignore': []
        }
        
        for entity in entities:
            # Create entity features
            entity_features = self._create_entity_features(entity)
            
            # Predict importance using AI model
            if self.ai_trainer.entity_importance_model:
                importance = self.ai_trainer.predict_entity_importance(entity_features, context_features)
            else:
                # Fallback to rule-based classification
                importance = self._rule_based_importance_classification(entity)
            
            # Add to appropriate group
            entity['importance'] = importance
            classified[importance.value].append(entity)
        
        logger.info(f"Entity classification: {len(classified['critical'])} critical, "
                   f"{len(classified['important'])} important, "
                   f"{len(classified['detail'])} detail, "
                   f"{len(classified['ignore'])} ignore")
        
        return classified
    
    def _create_entity_features(self, entity: Dict[str, Any]) -> np.ndarray:
        """Create feature vector for an entity."""
        # Convert entity to format expected by feature engineer
        entity_data = {
            'length': entity.get('length', 0),
            'area': entity.get('area', 0),
            'angle': entity.get('angle', 0),
            'lineweight': entity.get('lineweight', 0),
            'color': entity.get('color', 0),
            'bounding_box': entity.get('bounding_box', [0, 0, 0, 0]),
            'entity_type': entity.get('type', 'Unknown'),
            'layer_name': entity.get('layer', 'Unknown'),
            'linetype': entity.get('linetype', 'Continuous'),
        }
        
        return self.feature_engineer.create_entity_features([entity_data])[0]
    
    def _rule_based_importance_classification(self, entity: Dict[str, Any]) -> EntityImportance:
        """Fallback rule-based entity importance classification."""
        entity_type = entity.get('type', '')
        layer_name = entity.get('layer', '').lower()
        length = entity.get('length', 0)
        
        # Critical entities
        if any(keyword in layer_name for keyword in ['wall', 'structure', 'beam', 'column']):
            return EntityImportance.CRITICAL
        
        if any(keyword in layer_name for keyword in ['door', 'window', 'opening']):
            return EntityImportance.CRITICAL
        
        # Important entities
        if entity_type == 'AcDbLine' and length > 1.0:
            return EntityImportance.IMPORTANT
        
        if entity_type in ['AcDbPolyline', 'AcDb2dPolyline'] and length > 2.0:
            return EntityImportance.IMPORTANT
        
        # Ignore entities
        if entity_type in ['AcDbText', 'AcDbMText', 'AcDbHatch']:
            return EntityImportance.IGNORE
        
        if any(keyword in layer_name for keyword in ['text', 'dimension', 'hatch', 'symbol']):
            return EntityImportance.IGNORE
        
        # Default to detail
        return EntityImportance.DETAIL
    
    def _create_dimension_groups(self, classified_entities: Dict[str, List[Dict[str, Any]]]) -> List[List[Dict[str, Any]]]:
        """Create logical groups of entities for dimensioning."""
        groups = []
        
        # Group by layer
        layer_groups = {}
        for importance_level in ['critical', 'important']:
            for entity in classified_entities[importance_level]:
                layer = entity['layer']
                if layer not in layer_groups:
                    layer_groups[layer] = []
                layer_groups[layer].append(entity)
        
        # Create groups from layers
        for layer, entities in layer_groups.items():
            if len(entities) > 1:
                groups.append(entities)
        
        return groups
    
    def _determine_placement_strategy(self, drawing_type: DrawingType, 
                                    classified_entities: Dict[str, List[Dict[str, Any]]]) -> str:
        """Determine the best placement strategy for this drawing."""
        total_entities = (len(classified_entities['critical']) + 
                         len(classified_entities['important']) + 
                         len(classified_entities['detail']))
        
        if total_entities < 10:
            return "simple"
        elif drawing_type == DrawingType.FLOOR_PLAN:
            return "hierarchical"
        elif len(classified_entities['critical']) > 20:
            return "grouped"
        else:
            return "hierarchical"
    
    def _estimate_dimension_count(self, classified_entities: Dict[str, List[Dict[str, Any]]]) -> int:
        """Estimate the number of dimensions to be created."""
        # Conservative estimate based on entity importance
        estimated = 0
        
        # Critical entities - likely to be dimensioned
        estimated += len(classified_entities['critical']) * 0.8
        
        # Important entities - some will be dimensioned
        estimated += len(classified_entities['important']) * 0.4
        
        # Detail entities - few will be dimensioned
        estimated += len(classified_entities['detail']) * 0.1
        
        return int(estimated)
    
    def _calculate_confidence_score(self, classified_entities: Dict[str, List[Dict[str, Any]]]) -> float:
        """Calculate confidence score for the dimensioning plan."""
        total_entities = sum(len(entities) for entities in classified_entities.values())
        
        if total_entities == 0:
            return 0.0
        
        # Higher confidence with more critical entities
        critical_ratio = len(classified_entities['critical']) / total_entities
        important_ratio = len(classified_entities['important']) / total_entities
        
        # Base confidence on entity classification quality
        confidence = (critical_ratio * 0.9) + (important_ratio * 0.7) + 0.3
        
        # Cap at 1.0
        return min(confidence, 1.0)
    
    def _ensure_ai_dimension_layer(self):
        """Ensure the AI dimension layer exists."""
        try:
            layer = self.cad_connection.doc.Layers.Add(self.config['layer_name'])
            layer.Color = 3  # Green color for AI dimensions
            self.cad_connection.doc.ActiveLayer = layer
        except:
            # Layer might already exist
            pass
    
    def _dimension_entity_group(self, entities: List[Dict[str, Any]], priority: int) -> int:
        """Dimension a group of entities."""
        dimensions_added = 0
        
        for entity in entities:
            try:
                # Only dimension lines and polylines for now
                if entity['type'] in ['AcDbLine', 'AcDbPolyline', 'AcDb2dPolyline']:
                    if self._should_dimension_entity(entity, priority):
                        if self._add_dimension_to_entity(entity):
                            dimensions_added += 1
            except Exception as e:
                logger.error(f"Error dimensioning entity {entity['id']}: {e}")
        
        return dimensions_added
    
    def _should_dimension_entity(self, entity: Dict[str, Any], priority: int) -> bool:
        """Determine if an entity should be dimensioned."""
        length = entity.get('length', 0)
        
        # Check minimum length based on priority
        if priority == 1:  # Critical
            return length >= self.config['min_critical_length']
        elif priority == 2:  # Important
            return length >= self.config['min_important_length']
        elif priority == 3:  # Detail
            return length >= self.config['min_detail_length']
        
        return False
    
    def _add_dimension_to_entity(self, entity: Dict[str, Any]) -> bool:
        """Add a dimension to an entity using the traditional service."""
        try:
            autocad_entity = entity['autocad_entity']
            
            # Use the traditional dimension service to add the dimension
            # This delegates to the proven dimensioning logic
            return self.traditional_service._add_dimension_to_line(autocad_entity)
            
        except Exception as e:
            logger.error(f"Error adding dimension to entity: {e}")
            return False


if __name__ == "__main__":
    # Example usage
    print("Intelligent Dimensioning Service")
    print("This service uses AI models to intelligently select and dimension entities")
    print("Usage: service.intelligent_dimension_all(layer_filter='WALLS')")