"""Enhanced AutoCAD entity detection."""
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import math

logger = logging.getLogger(__name__)


@dataclass
class EntityInfo:
    """Information about an AutoCAD entity."""
    entity_type: str
    handle: str
    layer: str
    color: int
    properties: Dict[str, Any]


class EntityDetector:
    """Detects and analyzes AutoCAD entities."""
    
    def __init__(self, connection):
        self.connection = connection
        self.model = connection.model
        
    def get_all_entities(self, entity_types: Optional[List[str]] = None) -> List[EntityInfo]:
        """Get all entities, optionally filtered by type."""
        entities = []
        
        try:
            for entity in self.model:
                entity_type = entity.ObjectName
                
                if entity_types and entity_type not in entity_types:
                    continue
                    
                info = self._extract_entity_info(entity)
                if info:
                    entities.append(info)
                    
        except Exception as e:
            logger.error(f"Error getting entities: {e}")
            
        return entities
    
    def get_lines(self) -> List[EntityInfo]:
        """Get all line entities."""
        return self.get_all_entities(['AcDbLine'])
    
    def get_circles(self) -> List[EntityInfo]:
        """Get all circle entities."""
        return self.get_all_entities(['AcDbCircle'])
    
    def get_arcs(self) -> List[EntityInfo]:
        """Get all arc entities."""
        return self.get_all_entities(['AcDbArc'])
    
    def get_polylines(self) -> List[EntityInfo]:
        """Get all polyline entities."""
        return self.get_all_entities(['AcDbPolyline', 'AcDb2dPolyline', 'AcDb3dPolyline'])
    
    def get_blocks(self) -> List[EntityInfo]:
        """Get all block reference entities."""
        return self.get_all_entities(['AcDbBlockReference'])
    
    def _extract_entity_info(self, entity) -> Optional[EntityInfo]:
        """Extract information from an entity."""
        try:
            entity_type = entity.ObjectName
            properties = {}
            
            # Common properties
            handle = entity.Handle
            layer = entity.Layer
            color = entity.Color
            
            # Type-specific properties
            if entity_type == 'AcDbLine':
                properties.update({
                    'start': (entity.StartPoint[0], entity.StartPoint[1]),
                    'end': (entity.EndPoint[0], entity.EndPoint[1]),
                    'length': entity.Length,
                    'angle': entity.Angle
                })
            
            elif entity_type == 'AcDbCircle':
                properties.update({
                    'center': (entity.Center[0], entity.Center[1]),
                    'radius': entity.Radius,
                    'diameter': entity.Diameter,
                    'area': entity.Area,
                    'circumference': entity.Circumference
                })
            
            elif entity_type == 'AcDbArc':
                properties.update({
                    'center': (entity.Center[0], entity.Center[1]),
                    'radius': entity.Radius,
                    'start_angle': math.degrees(entity.StartAngle),
                    'end_angle': math.degrees(entity.EndAngle),
                    'total_angle': math.degrees(entity.TotalAngle),
                    'arc_length': entity.ArcLength
                })
            
            elif entity_type in ['AcDbPolyline', 'AcDb2dPolyline']:
                coords = []
                for i in range(entity.NumberOfVertices):
                    coord = entity.Coordinate(i)
                    coords.append((coord[0], coord[1]))
                
                properties.update({
                    'vertices': coords,
                    'closed': entity.Closed,
                    'length': entity.Length,
                    'area': entity.Area if entity.Closed else 0
                })
            
            elif entity_type == 'AcDbBlockReference':
                properties.update({
                    'name': entity.Name,
                    'position': (entity.InsertionPoint[0], entity.InsertionPoint[1]),
                    'scale': (entity.XScaleFactor, entity.YScaleFactor, entity.ZScaleFactor),
                    'rotation': math.degrees(entity.Rotation)
                })
            
            return EntityInfo(
                entity_type=entity_type,
                handle=handle,
                layer=layer,
                color=color,
                properties=properties
            )
            
        except Exception as e:
            logger.error(f"Error extracting entity info: {e}")
            return None
    
    def find_entities_by_layer(self, layer_name: str) -> List[EntityInfo]:
        """Find all entities on a specific layer."""
        entities = []
        
        for entity in self.get_all_entities():
            if entity.layer == layer_name:
                entities.append(entity)
                
        return entities
    
    def find_entities_in_area(self, min_point: Tuple[float, float], 
                             max_point: Tuple[float, float]) -> List[EntityInfo]:
        """Find entities within a rectangular area."""
        entities = []
        
        for entity in self.get_all_entities():
            if self._is_entity_in_area(entity, min_point, max_point):
                entities.append(entity)
                
        return entities
    
    def _is_entity_in_area(self, entity: EntityInfo, min_pt: Tuple[float, float], 
                          max_pt: Tuple[float, float]) -> bool:
        """Check if entity is within area bounds."""
        props = entity.properties
        
        if entity.entity_type == 'AcDbLine':
            start = props['start']
            end = props['end']
            return (self._point_in_area(start, min_pt, max_pt) or 
                    self._point_in_area(end, min_pt, max_pt))
        
        elif entity.entity_type in ['AcDbCircle', 'AcDbArc']:
            center = props['center']
            radius = props['radius']
            return (center[0] - radius >= min_pt[0] and 
                    center[0] + radius <= max_pt[0] and
                    center[1] - radius >= min_pt[1] and 
                    center[1] + radius <= max_pt[1])
        
        elif 'AcDbPolyline' in entity.entity_type:
            vertices = props.get('vertices', [])
            return any(self._point_in_area(v, min_pt, max_pt) for v in vertices)
        
        elif entity.entity_type == 'AcDbBlockReference':
            pos = props['position']
            return self._point_in_area(pos, min_pt, max_pt)
        
        return False
    
    def _point_in_area(self, point: Tuple[float, float], min_pt: Tuple[float, float], 
                       max_pt: Tuple[float, float]) -> bool:
        """Check if point is within area."""
        return (min_pt[0] <= point[0] <= max_pt[0] and 
                min_pt[1] <= point[1] <= max_pt[1])