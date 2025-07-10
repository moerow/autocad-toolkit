"""Service for detecting walls from parallel lines in AutoCAD drawings."""
import logging
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
import math

from src.infrastructure.autocad.autocad_service import AutoCADService
from src.core.entities.geometry import Point, Line

logger = logging.getLogger(__name__)


@dataclass
class Wall:
    """Represents a detected wall"""
    id: int
    line1: Line
    line2: Line
    thickness: float
    center_line: Line
    layer: str = "WALL"
    
    def length(self) -> float:
        """Get wall length"""
        return self.center_line.length()
    
    def area(self) -> float:
        """Get wall area (2D)"""
        return self.length() * self.thickness


@dataclass
class WallGroup:
    """Group of connected walls"""
    id: int
    walls: List[Wall]
    is_closed: bool = False
    
    def total_length(self) -> float:
        """Get total length of all walls"""
        return sum(wall.length() for wall in self.walls)
    
    def total_area(self) -> float:
        """Get total area of all walls"""
        return sum(wall.area() for wall in self.walls)


class WallDetectionService:
    """Service for detecting walls in AutoCAD drawings"""
    
    def __init__(self, autocad_service: AutoCADService):
        self.autocad_service = autocad_service
        self.detected_walls: List[Wall] = []
        self.wall_groups: List[WallGroup] = []
        
        # Default parameters
        self.tolerance_angle = 1.0  # degrees
        self.min_wall_thickness = 50.0  # mm
        self.max_wall_thickness = 500.0  # mm
        self.min_wall_length = 500.0  # mm
        self.connection_tolerance = 10.0  # mm
        
    def detect_walls(self, layer: Optional[str] = None) -> List[Wall]:
        """Detect walls from parallel lines
        
        Args:
            layer: Optional layer to search in
            
        Returns:
            List of detected walls
        """
        logger.info("Starting wall detection...")
        
        # Find parallel line pairs
        parallel_pairs = self.autocad_service.find_parallel_lines(
            tolerance_angle=self.tolerance_angle,
            min_distance=self.min_wall_thickness,
            max_distance=self.max_wall_thickness
        )
        
        # Convert to walls
        self.detected_walls = []
        wall_id = 1
        
        for line1, line2, distance in parallel_pairs:
            # Check minimum length
            if line1.length() < self.min_wall_length and line2.length() < self.min_wall_length:
                continue
            
            # Check if lines overlap sufficiently
            if not self._lines_overlap(line1, line2):
                continue
            
            # Create wall
            center_line = self._calculate_center_line(line1, line2)
            wall = Wall(
                id=wall_id,
                line1=line1,
                line2=line2,
                thickness=distance,
                center_line=center_line,
                layer=layer or "0"
            )
            
            self.detected_walls.append(wall)
            wall_id += 1
        
        logger.info(f"Detected {len(self.detected_walls)} walls")
        
        # Group connected walls
        self._group_walls()
        
        return self.detected_walls
    
    def _lines_overlap(self, line1: Line, line2: Line, min_overlap: float = 0.5) -> bool:
        """Check if two parallel lines overlap sufficiently
        
        Args:
            line1: First line
            line2: Second line
            min_overlap: Minimum overlap ratio (0-1)
            
        Returns:
            True if lines overlap sufficiently
        """
        # Project lines onto their common direction
        angle = line1.angle()
        
        # Project all points onto the direction vector
        projections = []
        for point in [line1.start, line1.end, line2.start, line2.end]:
            proj = point.x * math.cos(angle) + point.y * math.sin(angle)
            projections.append(proj)
        
        # Find overlap
        line1_min, line1_max = min(projections[0:2]), max(projections[0:2])
        line2_min, line2_max = min(projections[2:4]), max(projections[2:4])
        
        overlap_start = max(line1_min, line2_min)
        overlap_end = min(line1_max, line2_max)
        
        if overlap_end <= overlap_start:
            return False
        
        overlap_length = overlap_end - overlap_start
        min_length = min(line1_max - line1_min, line2_max - line2_min)
        
        return overlap_length >= min_length * min_overlap
    
    def _calculate_center_line(self, line1: Line, line2: Line) -> Line:
        """Calculate center line between two parallel lines"""
        # Find the overlap region
        angle = line1.angle()
        
        # Project all points
        points_data = []
        for point in [line1.start, line1.end, line2.start, line2.end]:
            proj = point.x * math.cos(angle) + point.y * math.sin(angle)
            points_data.append((proj, point))
        
        # Sort by projection
        points_data.sort(key=lambda x: x[0])
        
        # The middle two points define the overlap
        start_point = Point(
            (points_data[1][1].x + points_data[2][1].x) / 2,
            (points_data[1][1].y + points_data[2][1].y) / 2,
            (points_data[1][1].z + points_data[2][1].z) / 2
        )
        
        # Calculate end point based on overlap
        overlap_length = points_data[2][0] - points_data[1][0]
        end_point = Point(
            start_point.x + overlap_length * math.cos(angle),
            start_point.y + overlap_length * math.sin(angle),
            start_point.z
        )
        
        return Line(start_point, end_point)
    
    def _group_walls(self) -> List[WallGroup]:
        """Group connected walls together"""
        if not self.detected_walls:
            return []
        
        # Initialize groups
        self.wall_groups = []
        used_walls = set()
        group_id = 1
        
        for wall in self.detected_walls:
            if wall.id in used_walls:
                continue
            
            # Start new group
            group = WallGroup(id=group_id, walls=[wall])
            used_walls.add(wall.id)
            
            # Find connected walls
            self._find_connected_walls(wall, group, used_walls)
            
            # Check if group is closed
            if len(group.walls) >= 3:
                first_wall = group.walls[0]
                last_wall = group.walls[-1]
                if self._walls_connected(first_wall, last_wall):
                    group.is_closed = True
            
            self.wall_groups.append(group)
            group_id += 1
        
        logger.info(f"Created {len(self.wall_groups)} wall groups")
        return self.wall_groups
    
    def _find_connected_walls(self, current_wall: Wall, group: WallGroup, used_walls: set):
        """Recursively find walls connected to current wall"""
        for wall in self.detected_walls:
            if wall.id in used_walls:
                continue
            
            if self._walls_connected(current_wall, wall):
                group.walls.append(wall)
                used_walls.add(wall.id)
                self._find_connected_walls(wall, group, used_walls)
    
    def _walls_connected(self, wall1: Wall, wall2: Wall) -> bool:
        """Check if two walls are connected at their endpoints"""
        tolerance = self.connection_tolerance
        
        # Check all endpoint combinations
        endpoints1 = [wall1.center_line.start, wall1.center_line.end]
        endpoints2 = [wall2.center_line.start, wall2.center_line.end]
        
        for ep1 in endpoints1:
            for ep2 in endpoints2:
                if ep1.distance_to(ep2) <= tolerance:
                    return True
        
        return False
    
    def get_wall_statistics(self) -> Dict[str, Any]:
        """Get statistics about detected walls"""
        if not self.detected_walls:
            return {
                'total_walls': 0,
                'total_groups': 0,
                'total_length': 0,
                'total_area': 0,
                'avg_thickness': 0,
                'closed_groups': 0
            }
        
        total_length = sum(wall.length() for wall in self.detected_walls)
        total_area = sum(wall.area() for wall in self.detected_walls)
        avg_thickness = sum(wall.thickness for wall in self.detected_walls) / len(self.detected_walls)
        closed_groups = sum(1 for group in self.wall_groups if group.is_closed)
        
        return {
            'total_walls': len(self.detected_walls),
            'total_groups': len(self.wall_groups),
            'total_length': total_length,
            'total_area': total_area,
            'avg_thickness': avg_thickness,
            'closed_groups': closed_groups,
            'thickness_range': (
                min(wall.thickness for wall in self.detected_walls),
                max(wall.thickness for wall in self.detected_walls)
            )
        }
    
    def mark_walls_in_autocad(self, color: int = 1):
        """Mark detected walls in AutoCAD with a specific color"""
        if not self.autocad_service.connection.is_connected():
            logger.error("Not connected to AutoCAD")
            return
        
        try:
            # Create layer for walls if it doesn't exist
            self.autocad_service.create_layer("DETECTED_WALLS", color)
            
            # Draw center lines for all walls
            for wall in self.detected_walls:
                # Add center line to AutoCAD
                acad_line = self.autocad_service._model.AddLine(
                    list(wall.center_line.start),
                    list(wall.center_line.end)
                )
                acad_line.Layer = "DETECTED_WALLS"
                acad_line.color = color
                
                # Add thickness text at midpoint
                midpoint = wall.center_line.midpoint()
                text = self.autocad_service._model.AddText(
                    f"t={wall.thickness:.0f}",
                    list(midpoint),
                    50  # text height
                )
                text.Layer = "DETECTED_WALLS"
                text.color = color
            
            logger.info(f"Marked {len(self.detected_walls)} walls in AutoCAD")
            
        except Exception as e:
            logger.error(f"Error marking walls: {e}")
    
    def export_walls(self) -> List[Dict[str, Any]]:
        """Export walls data as dictionaries"""
        walls_data = []
        
        for wall in self.detected_walls:
            wall_data = {
                'id': wall.id,
                'thickness': wall.thickness,
                'length': wall.length(),
                'area': wall.area(),
                'center_line': {
                    'start': [wall.center_line.start.x, wall.center_line.start.y],
                    'end': [wall.center_line.end.x, wall.center_line.end.y]
                },
                'layer': wall.layer
            }
            walls_data.append(wall_data)
        
        return walls_data