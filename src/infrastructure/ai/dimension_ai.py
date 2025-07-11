"""AI-powered intelligent dimensioning system.

This module provides AI-driven dimensioning capabilities that learn from
professional DWG files to automatically place dimensions following industry
standards and professional practices.
"""

import logging
import json
import pickle
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum
import sqlite3
from datetime import datetime

logger = logging.getLogger(__name__)


class DrawingType(Enum):
    """Types of architectural/engineering drawings."""
    FLOOR_PLAN = "floor_plan"
    SECTION = "section"
    ELEVATION = "elevation"
    DETAIL = "detail"
    SITE_PLAN = "site_plan"
    STRUCTURAL_PLAN = "structural_plan"
    ELECTRICAL_PLAN = "electrical_plan"
    PLUMBING_PLAN = "plumbing_plan"
    UNKNOWN = "unknown"


class EntityImportance(Enum):
    """Importance levels for entities in dimensioning."""
    CRITICAL = "critical"      # Must be dimensioned (walls, openings)
    IMPORTANT = "important"    # Should be dimensioned (rooms, clearances)
    DETAIL = "detail"         # Optional dimensioning (minor elements)
    IGNORE = "ignore"         # Never dimension (text, construction lines)


@dataclass
class GeometryFeature:
    """Geometric features of an entity."""
    entity_type: str           # Line, Arc, Circle, Polyline, etc.
    layer_name: str           # AutoCAD layer
    length: Optional[float]    # For linear entities
    area: Optional[float]     # For closed entities
    start_point: Optional[Tuple[float, float]]
    end_point: Optional[Tuple[float, float]]
    center_point: Optional[Tuple[float, float]]
    bounding_box: Tuple[float, float, float, float]  # min_x, min_y, max_x, max_y
    angle: Optional[float]     # Orientation angle
    color: Optional[int]       # Entity color
    linetype: Optional[str]    # Line type
    lineweight: Optional[float] # Line weight


@dataclass
class DimensionFeature:
    """Features of a dimension in the drawing."""
    dimension_type: str        # Linear, Angular, Radial, etc.
    measurement_value: float   # Actual measured value
    dimension_text: str        # Displayed text
    text_position: Tuple[float, float]
    dimension_line_position: Tuple[float, float]
    extension_line_1: Tuple[float, float, float, float]  # start_x, start_y, end_x, end_y
    extension_line_2: Tuple[float, float, float, float]
    associated_entities: List[str]  # IDs of entities being dimensioned
    layer_name: str
    text_height: float
    arrow_size: float
    style_name: str


@dataclass
class TrainingExample:
    """A single training example from a professional DWG."""
    file_path: str
    drawing_type: DrawingType
    drawing_scale: Optional[float]
    total_entities: int
    dimensioned_entities: int
    geometry_features: List[GeometryFeature]
    dimension_features: List[DimensionFeature]
    entity_dimension_relationships: List[Tuple[str, str]]  # (entity_id, dimension_id)
    drawing_bounds: Tuple[float, float, float, float]  # Drawing extents
    timestamp: datetime
    metadata: Dict[str, Any]   # Additional drawing information


class DWGAnalyzer:
    """Analyzes DWG files to extract training data for AI dimensioning."""
    
    def __init__(self, storage_path: str = "ai_training_data"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        # Initialize database for training data
        self.db_path = self.storage_path / "training_data.db"
        self._init_database()
        
        logger.info(f"DWG Analyzer initialized with storage at: {self.storage_path}")
    
    def _init_database(self):
        """Initialize SQLite database for storing training data."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables for training data
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS training_examples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT UNIQUE,
                drawing_type TEXT,
                drawing_scale REAL,
                total_entities INTEGER,
                dimensioned_entities INTEGER,
                drawing_bounds TEXT,
                timestamp TEXT,
                metadata TEXT,
                processed BOOLEAN DEFAULT FALSE
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS geometry_features (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                example_id INTEGER,
                entity_id TEXT,
                entity_type TEXT,
                layer_name TEXT,
                length REAL,
                area REAL,
                start_point TEXT,
                end_point TEXT,
                center_point TEXT,
                bounding_box TEXT,
                angle REAL,
                color INTEGER,
                linetype TEXT,
                lineweight REAL,
                importance TEXT,
                FOREIGN KEY (example_id) REFERENCES training_examples (id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS dimension_features (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                example_id INTEGER,
                dimension_id TEXT,
                dimension_type TEXT,
                measurement_value REAL,
                dimension_text TEXT,
                text_position TEXT,
                dimension_line_position TEXT,
                extension_line_1 TEXT,
                extension_line_2 TEXT,
                associated_entities TEXT,
                layer_name TEXT,
                text_height REAL,
                arrow_size REAL,
                style_name TEXT,
                FOREIGN KEY (example_id) REFERENCES training_examples (id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS entity_dimension_relationships (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                example_id INTEGER,
                entity_id TEXT,
                dimension_id TEXT,
                relationship_type TEXT,
                FOREIGN KEY (example_id) REFERENCES training_examples (id)
            )
        ''')
        
        conn.commit()
        conn.close()
        
        logger.info("Training data database initialized")
    
    def analyze_dwg_file(self, dwg_file_path: str, cad_connection) -> Optional[TrainingExample]:
        """Analyze a single DWG file and extract training data."""
        try:
            logger.info(f"Analyzing DWG file: {dwg_file_path}")
            
            # Open the DWG file in AutoCAD
            doc = cad_connection.open_document(dwg_file_path)
            if not doc:
                logger.error(f"Failed to open DWG file: {dwg_file_path}")
                return None
            
            # Extract all entities
            geometry_features = self._extract_geometry_features(cad_connection)
            
            # Extract all dimensions
            dimension_features = self._extract_dimension_features(cad_connection)
            
            # Analyze relationships between entities and dimensions
            relationships = self._analyze_entity_dimension_relationships(
                geometry_features, dimension_features
            )
            
            # Classify drawing type
            drawing_type = self._classify_drawing_type(dwg_file_path, geometry_features)
            
            # Get drawing bounds
            drawing_bounds = self._get_drawing_bounds(geometry_features)
            
            # Create training example
            training_example = TrainingExample(
                file_path=dwg_file_path,
                drawing_type=drawing_type,
                drawing_scale=self._estimate_drawing_scale(geometry_features),
                total_entities=len(geometry_features),
                dimensioned_entities=len([g for g in geometry_features if self._is_dimensioned(g, relationships)]),
                geometry_features=geometry_features,
                dimension_features=dimension_features,
                entity_dimension_relationships=relationships,
                drawing_bounds=drawing_bounds,
                timestamp=datetime.now(),
                metadata=self._extract_metadata(cad_connection)
            )
            
            # Store in database
            self._store_training_example(training_example)
            
            logger.info(f"Successfully analyzed {dwg_file_path}: {len(geometry_features)} entities, {len(dimension_features)} dimensions")
            return training_example
            
        except Exception as e:
            logger.error(f"Error analyzing DWG file {dwg_file_path}: {e}")
            return None
    
    def _extract_geometry_features(self, cad_connection) -> List[GeometryFeature]:
        """Extract geometric features from all entities in the drawing."""
        features = []
        
        try:
            for entity in cad_connection.model:
                feature = self._analyze_entity(entity)
                if feature:
                    features.append(feature)
        except Exception as e:
            logger.error(f"Error extracting geometry features: {e}")
        
        return features
    
    def _extract_dimension_features(self, cad_connection) -> List[DimensionFeature]:
        """Extract features from all dimensions in the drawing."""
        features = []
        
        dimension_types = [
            'AcDbAlignedDimension', 'AcDbRotatedDimension', 'AcDbRadialDimension',
            'AcDbDiametricDimension', 'AcDbLinearDimension', 'AcDbOrdinateDimension',
            'AcDbAngularDimension', 'AcDb3PointAngularDimension', 'AcDbArcDimension'
        ]
        
        try:
            for entity in cad_connection.model:
                if entity.ObjectName in dimension_types:
                    feature = self._analyze_dimension(entity)
                    if feature:
                        features.append(feature)
        except Exception as e:
            logger.error(f"Error extracting dimension features: {e}")
        
        return features
    
    def _analyze_entity(self, entity) -> Optional[GeometryFeature]:
        """Analyze a single entity and extract its features."""
        try:
            # Get basic properties
            entity_type = entity.ObjectName
            layer_name = getattr(entity, 'Layer', 'Unknown')
            color = getattr(entity, 'Color', None)
            linetype = getattr(entity, 'Linetype', None)
            lineweight = getattr(entity, 'LineWeight', None)
            
            # Get geometric properties based on entity type
            length = None
            area = None
            start_point = None
            end_point = None
            center_point = None
            angle = None
            
            if entity_type == 'AcDbLine':
                start_point = (entity.StartPoint[0], entity.StartPoint[1])
                end_point = (entity.EndPoint[0], entity.EndPoint[1])
                length = entity.Length
                angle = entity.Angle
            elif entity_type in ['AcDbPolyline', 'AcDb2dPolyline']:
                length = entity.Length
                area = getattr(entity, 'Area', None) if hasattr(entity, 'Area') else None
                # Get first and last points
                coords = entity.Coordinates
                if len(coords) >= 4:
                    start_point = (coords[0], coords[1])
                    end_point = (coords[-2], coords[-1])
            elif entity_type == 'AcDbCircle':
                center_point = (entity.Center[0], entity.Center[1])
                length = entity.Circumference
                area = entity.Area
            elif entity_type == 'AcDbArc':
                center_point = (entity.Center[0], entity.Center[1])
                length = entity.ArcLength
                angle = entity.StartAngle
            
            # Get bounding box
            try:
                bounds = entity.GetBoundingBox()
                bounding_box = (bounds[0][0], bounds[0][1], bounds[1][0], bounds[1][1])
            except:
                bounding_box = (0, 0, 0, 0)
            
            return GeometryFeature(
                entity_type=entity_type,
                layer_name=layer_name,
                length=length,
                area=area,
                start_point=start_point,
                end_point=end_point,
                center_point=center_point,
                bounding_box=bounding_box,
                angle=angle,
                color=color,
                linetype=linetype,
                lineweight=lineweight
            )
            
        except Exception as e:
            logger.error(f"Error analyzing entity: {e}")
            return None
    
    def _analyze_dimension(self, dimension) -> Optional[DimensionFeature]:
        """Analyze a single dimension and extract its features."""
        try:
            dimension_type = dimension.ObjectName
            measurement_value = getattr(dimension, 'Measurement', 0)
            dimension_text = getattr(dimension, 'TextOverride', str(measurement_value))
            text_height = getattr(dimension, 'TextHeight', 0)
            arrow_size = getattr(dimension, 'ArrowheadSize', 0)
            style_name = getattr(dimension, 'StyleName', 'Standard')
            layer_name = getattr(dimension, 'Layer', 'Unknown')
            
            # Get dimension positions
            text_position = (0, 0)
            dimension_line_position = (0, 0)
            extension_line_1 = (0, 0, 0, 0)
            extension_line_2 = (0, 0, 0, 0)
            
            try:
                text_position = (dimension.TextPosition[0], dimension.TextPosition[1])
            except:
                pass
            
            try:
                dimension_line_position = (dimension.DimLinePoint[0], dimension.DimLinePoint[1])
            except:
                pass
            
            # For linear dimensions, get extension lines
            if hasattr(dimension, 'ExtLine1Point') and hasattr(dimension, 'ExtLine2Point'):
                try:
                    ext1_start = dimension.ExtLine1StartPoint
                    ext1_end = dimension.ExtLine1Point
                    ext2_start = dimension.ExtLine2StartPoint
                    ext2_end = dimension.ExtLine2Point
                    
                    extension_line_1 = (ext1_start[0], ext1_start[1], ext1_end[0], ext1_end[1])
                    extension_line_2 = (ext2_start[0], ext2_start[1], ext2_end[0], ext2_end[1])
                except:
                    pass
            
            return DimensionFeature(
                dimension_type=dimension_type,
                measurement_value=measurement_value,
                dimension_text=dimension_text,
                text_position=text_position,
                dimension_line_position=dimension_line_position,
                extension_line_1=extension_line_1,
                extension_line_2=extension_line_2,
                associated_entities=[],  # Will be populated by relationship analysis
                layer_name=layer_name,
                text_height=text_height,
                arrow_size=arrow_size,
                style_name=style_name
            )
            
        except Exception as e:
            logger.error(f"Error analyzing dimension: {e}")
            return None
    
    def _analyze_entity_dimension_relationships(self, geometry_features, dimension_features) -> List[Tuple[str, str]]:
        """Analyze relationships between entities and dimensions."""
        relationships = []
        
        # For each dimension, find which entities it's measuring
        for i, dim in enumerate(dimension_features):
            dim_id = f"dim_{i}"
            
            # Find entities that are close to the dimension's extension lines
            for j, entity in enumerate(geometry_features):
                entity_id = f"entity_{j}"
                
                if self._is_entity_dimensioned_by(entity, dim):
                    relationships.append((entity_id, dim_id))
        
        return relationships
    
    def _is_entity_dimensioned_by(self, entity: GeometryFeature, dimension: DimensionFeature) -> bool:
        """Check if an entity is being dimensioned by a specific dimension."""
        # Simplified check based on proximity
        # In a real implementation, this would be more sophisticated
        
        if not entity.start_point or not entity.end_point:
            return False
        
        # Check if entity endpoints are close to dimension extension lines
        tolerance = 0.1  # Tolerance for proximity check
        
        ext1_start = dimension.extension_line_1[:2]
        ext1_end = dimension.extension_line_1[2:]
        ext2_start = dimension.extension_line_2[:2]
        ext2_end = dimension.extension_line_2[2:]
        
        # Check if entity points are close to extension line endpoints
        entity_points = [entity.start_point, entity.end_point]
        ext_points = [ext1_start, ext1_end, ext2_start, ext2_end]
        
        for entity_point in entity_points:
            for ext_point in ext_points:
                if ext_point and entity_point:
                    distance = ((entity_point[0] - ext_point[0])**2 + (entity_point[1] - ext_point[1])**2)**0.5
                    if distance < tolerance:
                        return True
        
        return False
    
    def _classify_drawing_type(self, file_path: str, geometry_features: List[GeometryFeature]) -> DrawingType:
        """Classify the type of drawing based on filename and content."""
        file_name = Path(file_path).name.lower()
        
        # Simple classification based on filename
        if any(keyword in file_name for keyword in ['floor', 'plan', 'fp']):
            return DrawingType.FLOOR_PLAN
        elif any(keyword in file_name for keyword in ['section', 'sect']):
            return DrawingType.SECTION
        elif any(keyword in file_name for keyword in ['elevation', 'elev']):
            return DrawingType.ELEVATION
        elif any(keyword in file_name for keyword in ['detail', 'det']):
            return DrawingType.DETAIL
        elif any(keyword in file_name for keyword in ['site', 'plot']):
            return DrawingType.SITE_PLAN
        elif any(keyword in file_name for keyword in ['struct', 'beam', 'column']):
            return DrawingType.STRUCTURAL_PLAN
        elif any(keyword in file_name for keyword in ['elect', 'power']):
            return DrawingType.ELECTRICAL_PLAN
        elif any(keyword in file_name for keyword in ['plumb', 'water']):
            return DrawingType.PLUMBING_PLAN
        
        return DrawingType.UNKNOWN
    
    def _estimate_drawing_scale(self, geometry_features: List[GeometryFeature]) -> Optional[float]:
        """Estimate the drawing scale based on geometry."""
        # Simple heuristic - could be improved with more sophisticated analysis
        if not geometry_features:
            return None
        
        # Get typical dimensions
        lengths = [f.length for f in geometry_features if f.length and f.length > 0]
        if not lengths:
            return None
        
        avg_length = sum(lengths) / len(lengths)
        
        # Estimate scale based on typical architectural dimensions
        if avg_length > 10000:  # Likely in mm, building scale
            return 1.0  # 1:1 scale
        elif avg_length > 1000:  # Likely in mm, room scale
            return 0.1  # 1:10 scale
        elif avg_length > 100:   # Likely in mm, detail scale
            return 0.01  # 1:100 scale
        else:
            return None
    
    def _get_drawing_bounds(self, geometry_features: List[GeometryFeature]) -> Tuple[float, float, float, float]:
        """Get the overall bounds of the drawing."""
        if not geometry_features:
            return (0, 0, 0, 0)
        
        min_x = min(f.bounding_box[0] for f in geometry_features)
        min_y = min(f.bounding_box[1] for f in geometry_features)
        max_x = max(f.bounding_box[2] for f in geometry_features)
        max_y = max(f.bounding_box[3] for f in geometry_features)
        
        return (min_x, min_y, max_x, max_y)
    
    def _is_dimensioned(self, geometry_feature: GeometryFeature, relationships: List[Tuple[str, str]]) -> bool:
        """Check if a geometry feature is dimensioned."""
        # This would check if the entity appears in any relationships
        # Simplified for now
        return len(relationships) > 0
    
    def _extract_metadata(self, cad_connection) -> Dict[str, Any]:
        """Extract metadata from the drawing."""
        metadata = {}
        
        try:
            # Get drawing properties
            doc = cad_connection.doc
            metadata['author'] = getattr(doc, 'Author', 'Unknown')
            metadata['title'] = getattr(doc, 'Title', 'Unknown')
            metadata['subject'] = getattr(doc, 'Subject', 'Unknown')
            metadata['keywords'] = getattr(doc, 'Keywords', 'Unknown')
            
            # Get system variables
            metadata['units'] = 'Unknown'  # Would need to get from system variables
            metadata['scale'] = 'Unknown'
            
        except Exception as e:
            logger.error(f"Error extracting metadata: {e}")
        
        return metadata
    
    def _store_training_example(self, example: TrainingExample):
        """Store a training example in the database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Insert training example
            cursor.execute('''
                INSERT OR REPLACE INTO training_examples 
                (file_path, drawing_type, drawing_scale, total_entities, dimensioned_entities, 
                 drawing_bounds, timestamp, metadata, processed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                example.file_path,
                example.drawing_type.value,
                example.drawing_scale,
                example.total_entities,
                example.dimensioned_entities,
                json.dumps(example.drawing_bounds),
                example.timestamp.isoformat(),
                json.dumps(example.metadata),
                False
            ))
            
            example_id = cursor.lastrowid
            
            # Insert geometry features
            for i, feature in enumerate(example.geometry_features):
                cursor.execute('''
                    INSERT INTO geometry_features 
                    (example_id, entity_id, entity_type, layer_name, length, area, 
                     start_point, end_point, center_point, bounding_box, angle, color, linetype, lineweight)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    example_id, f"entity_{i}", feature.entity_type, feature.layer_name,
                    feature.length, feature.area,
                    json.dumps(feature.start_point), json.dumps(feature.end_point),
                    json.dumps(feature.center_point), json.dumps(feature.bounding_box),
                    feature.angle, feature.color, feature.linetype, feature.lineweight
                ))
            
            # Insert dimension features
            for i, feature in enumerate(example.dimension_features):
                cursor.execute('''
                    INSERT INTO dimension_features 
                    (example_id, dimension_id, dimension_type, measurement_value, dimension_text,
                     text_position, dimension_line_position, extension_line_1, extension_line_2,
                     associated_entities, layer_name, text_height, arrow_size, style_name)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    example_id, f"dim_{i}", feature.dimension_type, feature.measurement_value,
                    feature.dimension_text, json.dumps(feature.text_position),
                    json.dumps(feature.dimension_line_position),
                    json.dumps(feature.extension_line_1), json.dumps(feature.extension_line_2),
                    json.dumps(feature.associated_entities), feature.layer_name,
                    feature.text_height, feature.arrow_size, feature.style_name
                ))
            
            # Insert relationships
            for entity_id, dim_id in example.entity_dimension_relationships:
                cursor.execute('''
                    INSERT INTO entity_dimension_relationships 
                    (example_id, entity_id, dimension_id, relationship_type)
                    VALUES (?, ?, ?, ?)
                ''', (example_id, entity_id, dim_id, 'dimensioned_by'))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Stored training example: {example.file_path}")
            
        except Exception as e:
            logger.error(f"Error storing training example: {e}")
    
    def batch_analyze_dwgs(self, dwg_directory: str, cad_connection) -> List[TrainingExample]:
        """Analyze all DWG files in a directory."""
        dwg_files = list(Path(dwg_directory).glob("*.dwg"))
        training_examples = []
        
        logger.info(f"Found {len(dwg_files)} DWG files to analyze")
        
        for dwg_file in dwg_files:
            try:
                example = self.analyze_dwg_file(str(dwg_file), cad_connection)
                if example:
                    training_examples.append(example)
                    logger.info(f"Processed: {dwg_file.name}")
                else:
                    logger.warning(f"Failed to process: {dwg_file.name}")
            except Exception as e:
                logger.error(f"Error processing {dwg_file.name}: {e}")
        
        logger.info(f"Successfully analyzed {len(training_examples)} DWG files")
        return training_examples
    
    def get_training_statistics(self) -> Dict[str, Any]:
        """Get statistics about the training data."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        stats = {}
        
        # Get total number of examples
        cursor.execute("SELECT COUNT(*) FROM training_examples")
        stats['total_examples'] = cursor.fetchone()[0]
        
        # Get examples by drawing type
        cursor.execute("""
            SELECT drawing_type, COUNT(*) 
            FROM training_examples 
            GROUP BY drawing_type
        """)
        stats['examples_by_type'] = dict(cursor.fetchall())
        
        # Get total entities and dimensions
        cursor.execute("SELECT SUM(total_entities), SUM(dimensioned_entities) FROM training_examples")
        result = cursor.fetchone()
        stats['total_entities'] = result[0] or 0
        stats['total_dimensioned_entities'] = result[1] or 0
        
        # Get most common entity types
        cursor.execute("""
            SELECT entity_type, COUNT(*) 
            FROM geometry_features 
            GROUP BY entity_type 
            ORDER BY COUNT(*) DESC 
            LIMIT 10
        """)
        stats['common_entity_types'] = dict(cursor.fetchall())
        
        # Get most common dimension types
        cursor.execute("""
            SELECT dimension_type, COUNT(*) 
            FROM dimension_features 
            GROUP BY dimension_type 
            ORDER BY COUNT(*) DESC
        """)
        stats['common_dimension_types'] = dict(cursor.fetchall())
        
        conn.close()
        return stats


if __name__ == "__main__":
    # Example usage
    analyzer = DWGAnalyzer()
    print("DWG Analyzer initialized")
    print("Use analyzer.batch_analyze_dwgs('/path/to/dwg/files', cad_connection) to start training data extraction")