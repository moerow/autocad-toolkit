"""AutoCAD service for entity detection and manipulation."""
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
import math
import win32com.client
from enum import Enum

from src.infrastructure.autocad.connection import AutoCADConnection
from src.core.entities.geometry import Point, Line, Circle, Arc, Polyline

logger = logging.getLogger(__name__)


class EntityType(Enum):
    """AutoCAD entity types"""
    LINE = "AcDbLine"
    CIRCLE = "AcDbCircle"
    ARC = "AcDbArc"
    POLYLINE = "AcDbPolyline"
    LWPOLYLINE = "AcDb2dPolyline"
    SPLINE = "AcDbSpline"
    ELLIPSE = "AcDbEllipse"
    BLOCK = "AcDbBlockReference"
    TEXT = "AcDbText"
    MTEXT = "AcDbMText"
    DIMENSION = "AcDbDimension"
    HATCH = "AcDbHatch"


@dataclass
class EntityInfo:
    """Information about an AutoCAD entity"""
    handle: str
    entity_type: str
    layer: str
    color: int
    linetype: str
    lineweight: float
    properties: Dict[str, Any]


class AutoCADService:
    """Service for AutoCAD operations and entity detection"""
    
    def __init__(self, connection: AutoCADConnection):
        self.connection = connection
        self._acad = connection.acad
        self._model = connection.model
        self._doc = connection.doc
        
    def get_all_entities(self, entity_types: Optional[List[EntityType]] = None,
                        layers: Optional[List[str]] = None) -> List[EntityInfo]:
        """Get all entities from model space with optional filtering"""
        entities = []
        
        try:
            for entity in self._model:
                try:
                    entity_type = entity.ObjectName
                    
                    # Filter by entity type if specified
                    if entity_types:
                        if not any(entity_type == et.value for et in entity_types):
                            continue
                    
                    # Filter by layer if specified
                    if layers and entity.Layer not in layers:
                        continue
                    
                    # Extract entity information
                    info = self._extract_entity_info(entity)
                    if info:
                        entities.append(info)
                        
                except Exception as e:
                    logger.warning(f"Error processing entity: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error getting entities: {e}")
            
        return entities
    
    def _extract_entity_info(self, entity) -> Optional[EntityInfo]:
        """Extract detailed information from an entity"""
        try:
            base_info = EntityInfo(
                handle=entity.Handle,
                entity_type=entity.ObjectName,
                layer=entity.Layer,
                color=entity.color,
                linetype=entity.Linetype,
                lineweight=entity.Lineweight,
                properties={}
            )
            
            # Extract type-specific properties
            if entity.ObjectName == EntityType.LINE.value:
                base_info.properties = {
                    'start_point': self._point_to_tuple(entity.StartPoint),
                    'end_point': self._point_to_tuple(entity.EndPoint),
                    'length': entity.Length,
                    'angle': entity.Angle
                }
            
            elif entity.ObjectName == EntityType.CIRCLE.value:
                base_info.properties = {
                    'center': self._point_to_tuple(entity.Center),
                    'radius': entity.Radius,
                    'diameter': entity.Diameter,
                    'area': entity.Area,
                    'circumference': entity.Circumference
                }
            
            elif entity.ObjectName == EntityType.ARC.value:
                base_info.properties = {
                    'center': self._point_to_tuple(entity.Center),
                    'radius': entity.Radius,
                    'start_angle': entity.StartAngle,
                    'end_angle': entity.EndAngle,
                    'total_angle': entity.TotalAngle,
                    'start_point': self._point_to_tuple(entity.StartPoint),
                    'end_point': self._point_to_tuple(entity.EndPoint),
                    'arc_length': entity.ArcLength
                }
            
            elif entity.ObjectName in [EntityType.POLYLINE.value, EntityType.LWPOLYLINE.value]:
                vertices = []
                coordinates = entity.Coordinates
                # Convert flat coordinate array to list of points
                for i in range(0, len(coordinates), 2):
                    vertices.append((coordinates[i], coordinates[i+1], 0.0))
                
                base_info.properties = {
                    'vertices': vertices,
                    'is_closed': entity.Closed,
                    'length': entity.Length,
                    'area': entity.Area if entity.Closed else 0
                }
            
            elif entity.ObjectName == EntityType.BLOCK.value:
                base_info.properties = {
                    'name': entity.Name,
                    'position': self._point_to_tuple(entity.InsertionPoint),
                    'scale_x': entity.XScaleFactor,
                    'scale_y': entity.YScaleFactor,
                    'scale_z': entity.ZScaleFactor,
                    'rotation': entity.Rotation
                }
            
            elif entity.ObjectName == EntityType.TEXT.value:
                base_info.properties = {
                    'text': entity.TextString,
                    'position': self._point_to_tuple(entity.InsertionPoint),
                    'height': entity.Height,
                    'rotation': entity.Rotation,
                    'style': entity.StyleName
                }
            
            elif entity.ObjectName == EntityType.MTEXT.value:
                base_info.properties = {
                    'text': entity.TextString,
                    'position': self._point_to_tuple(entity.InsertionPoint),
                    'width': entity.Width,
                    'height': entity.Height,
                    'rotation': entity.Rotation
                }
            
            return base_info
            
        except Exception as e:
            logger.warning(f"Error extracting entity info: {e}")
            return None
    
    def get_lines(self, layer: Optional[str] = None) -> List[Line]:
        """Get all lines from the drawing"""
        lines = []
        entities = self.get_all_entities(
            entity_types=[EntityType.LINE],
            layers=[layer] if layer else None
        )
        
        for entity in entities:
            props = entity.properties
            line = Line(
                start=Point(*props['start_point']),
                end=Point(*props['end_point'])
            )
            lines.append(line)
            
        return lines
    
    def get_circles(self, layer: Optional[str] = None) -> List[Circle]:
        """Get all circles from the drawing"""
        circles = []
        entities = self.get_all_entities(
            entity_types=[EntityType.CIRCLE],
            layers=[layer] if layer else None
        )
        
        for entity in entities:
            props = entity.properties
            circle = Circle(
                center=Point(*props['center']),
                radius=props['radius']
            )
            circles.append(circle)
            
        return circles
    
    def get_arcs(self, layer: Optional[str] = None) -> List[Arc]:
        """Get all arcs from the drawing"""
        arcs = []
        entities = self.get_all_entities(
            entity_types=[EntityType.ARC],
            layers=[layer] if layer else None
        )
        
        for entity in entities:
            props = entity.properties
            arc = Arc(
                center=Point(*props['center']),
                radius=props['radius'],
                start_angle=props['start_angle'],
                end_angle=props['end_angle']
            )
            arcs.append(arc)
            
        return arcs
    
    def get_polylines(self, layer: Optional[str] = None) -> List[Polyline]:
        """Get all polylines from the drawing"""
        polylines = []
        entities = self.get_all_entities(
            entity_types=[EntityType.POLYLINE, EntityType.LWPOLYLINE],
            layers=[layer] if layer else None
        )
        
        for entity in entities:
            props = entity.properties
            vertices = [Point(*v) for v in props['vertices']]
            polyline = Polyline(
                vertices=vertices,
                is_closed=props['is_closed']
            )
            polylines.append(polyline)
            
        return polylines
    
    def get_blocks(self, layer: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all block references from the drawing"""
        blocks = []
        entities = self.get_all_entities(
            entity_types=[EntityType.BLOCK],
            layers=[layer] if layer else None
        )
        
        for entity in entities:
            blocks.append({
                'handle': entity.handle,
                'layer': entity.layer,
                **entity.properties
            })
            
        return blocks
    
    def get_texts(self, layer: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all text entities from the drawing"""
        texts = []
        entities = self.get_all_entities(
            entity_types=[EntityType.TEXT, EntityType.MTEXT],
            layers=[layer] if layer else None
        )
        
        for entity in entities:
            texts.append({
                'handle': entity.handle,
                'layer': entity.layer,
                'type': entity.entity_type,
                **entity.properties
            })
            
        return texts
    
    def find_parallel_lines(self, tolerance_angle: float = 1.0,
                           min_distance: float = 50.0,
                           max_distance: float = 500.0) -> List[Tuple[Line, Line, float]]:
        """Find pairs of parallel lines (potential walls)
        
        Args:
            tolerance_angle: Maximum angle difference in degrees
            min_distance: Minimum distance between lines
            max_distance: Maximum distance between lines
            
        Returns:
            List of tuples (line1, line2, distance)
        """
        lines = self.get_lines()
        parallel_pairs = []
        
        # Convert tolerance to radians
        tolerance_rad = math.radians(tolerance_angle)
        
        for i, line1 in enumerate(lines):
            for j, line2 in enumerate(lines[i+1:], i+1):
                # Calculate angles
                angle1 = math.atan2(
                    line1.end.y - line1.start.y,
                    line1.end.x - line1.start.x
                )
                angle2 = math.atan2(
                    line2.end.y - line2.start.y,
                    line2.end.x - line2.start.x
                )
                
                # Check if parallel (considering both directions)
                angle_diff = abs(angle1 - angle2)
                angle_diff = min(angle_diff, abs(angle_diff - math.pi))
                
                if angle_diff <= tolerance_rad:
                    # Calculate distance between lines
                    distance = self._distance_between_lines(line1, line2)
                    
                    if min_distance <= distance <= max_distance:
                        parallel_pairs.append((line1, line2, distance))
        
        return parallel_pairs
    
    def _distance_between_lines(self, line1: Line, line2: Line) -> float:
        """Calculate the perpendicular distance between two parallel lines"""
        # Vector from line1 start to line2 start
        v = Point(
            line2.start.x - line1.start.x,
            line2.start.y - line1.start.y,
            0
        )
        
        # Direction vector of line1
        d = Point(
            line1.end.x - line1.start.x,
            line1.end.y - line1.start.y,
            0
        )
        
        # Normalize direction vector
        length = math.sqrt(d.x**2 + d.y**2)
        if length > 0:
            d = Point(d.x/length, d.y/length, 0)
        
        # Perpendicular vector
        perp = Point(-d.y, d.x, 0)
        
        # Distance is the dot product of v and perpendicular
        distance = abs(v.x * perp.x + v.y * perp.y)
        
        return distance
    
    def get_entity_by_handle(self, handle: str) -> Optional[Any]:
        """Get an entity by its handle"""
        try:
            return self._doc.HandleToObject(handle)
        except Exception as e:
            logger.error(f"Error getting entity by handle {handle}: {e}")
            return None
    
    def get_layers(self) -> List[str]:
        """Get all layer names in the drawing"""
        layers = []
        try:
            for layer in self._doc.Layers:
                layers.append(layer.Name)
        except Exception as e:
            logger.error(f"Error getting layers: {e}")
        return layers
    
    def create_layer(self, name: str, color: int = 7) -> bool:
        """Create a new layer"""
        try:
            layer = self._doc.Layers.Add(name)
            layer.color = color
            return True
        except Exception as e:
            logger.error(f"Error creating layer {name}: {e}")
            return False
    
    def _point_to_tuple(self, point) -> Tuple[float, float, float]:
        """Convert AutoCAD point to tuple"""
        return (float(point[0]), float(point[1]), float(point[2]) if len(point) > 2 else 0.0)
    
    def get_drawing_extents(self) -> Tuple[Point, Point]:
        """Get the extents of the drawing"""
        try:
            db = self._doc.Database
            ext_min = db.Extmin
            ext_max = db.Extmax
            
            return (
                Point(ext_min[0], ext_min[1], ext_min[2]),
                Point(ext_max[0], ext_max[1], ext_max[2])
            )
        except Exception as e:
            logger.error(f"Error getting drawing extents: {e}")
            # Return default extents
            return (Point(0, 0, 0), Point(1000, 1000, 0))