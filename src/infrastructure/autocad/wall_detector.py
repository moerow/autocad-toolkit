"""Wall detection algorithm for finding parallel line pairs."""
import logging
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import math

logger = logging.getLogger(__name__)


@dataclass
class Wall:
    """Represents a detected wall."""
    line1_handle: str
    line2_handle: str
    thickness: float
    length: float
    start_point: Tuple[float, float]
    end_point: Tuple[float, float]
    angle: float
    midline: List[Tuple[float, float]]


class WallDetector:
    """Detects walls from parallel line pairs."""
    
    def __init__(self, tolerance: float = 0.1, max_wall_thickness: float = 500.0):
        self.tolerance = tolerance
        self.max_wall_thickness = max_wall_thickness
        
    def detect_walls(self, lines: List[Dict]) -> List[Wall]:
        """Detect walls from a list of lines."""
        walls = []
        used_lines = set()
        
        for i, line1 in enumerate(lines):
            if line1['handle'] in used_lines:
                continue
                
            for j, line2 in enumerate(lines[i + 1:], i + 1):
                if line2['handle'] in used_lines:
                    continue
                    
                wall = self._check_parallel_lines(line1, line2)
                if wall:
                    walls.append(wall)
                    used_lines.add(line1['handle'])
                    used_lines.add(line2['handle'])
                    break
                    
        return walls
    
    def _check_parallel_lines(self, line1: Dict, line2: Dict) -> Optional[Wall]:
        """Check if two lines form a wall."""
        # Extract line endpoints
        start1 = line1['properties']['start']
        end1 = line1['properties']['end']
        start2 = line2['properties']['start']
        end2 = line2['properties']['end']
        
        # Calculate line vectors and angles
        vec1 = (end1[0] - start1[0], end1[1] - start1[1])
        vec2 = (end2[0] - start2[0], end2[1] - start2[1])
        
        angle1 = math.atan2(vec1[1], vec1[0])
        angle2 = math.atan2(vec2[1], vec2[0])
        
        # Check if lines are parallel (within tolerance)
        angle_diff = abs(angle1 - angle2)
        if angle_diff > math.pi:
            angle_diff = 2 * math.pi - angle_diff
            
        if angle_diff > self.tolerance and (math.pi - angle_diff) > self.tolerance:
            return None
            
        # Calculate perpendicular distances
        dist1 = self._point_to_line_distance(start1, start2, end2)
        dist2 = self._point_to_line_distance(end1, start2, end2)
        dist3 = self._point_to_line_distance(start2, start1, end1)
        dist4 = self._point_to_line_distance(end2, start1, end1)
        
        # Check if distances are consistent (wall thickness)
        distances = [dist1, dist2, dist3, dist4]
        avg_dist = sum(distances) / len(distances)
        
        if avg_dist > self.max_wall_thickness:
            return None
            
        # Check if variance is within tolerance
        variance = sum((d - avg_dist) ** 2 for d in distances) / len(distances)
        if math.sqrt(variance) > self.tolerance * avg_dist:
            return None
            
        # Check overlap
        overlap = self._calculate_overlap(line1, line2)
        if overlap < 0.5:  # Less than 50% overlap
            return None
            
        # Create wall object
        wall_start, wall_end = self._calculate_wall_endpoints(line1, line2)
        wall_length = math.sqrt((wall_end[0] - wall_start[0])**2 + 
                               (wall_end[1] - wall_start[1])**2)
        
        midline = [
            ((wall_start[0] + wall_end[0]) / 2, (wall_start[1] + wall_end[1]) / 2)
        ]
        
        return Wall(
            line1_handle=line1['handle'],
            line2_handle=line2['handle'],
            thickness=avg_dist,
            length=wall_length,
            start_point=wall_start,
            end_point=wall_end,
            angle=angle1,
            midline=midline
        )
    
    def _point_to_line_distance(self, point: Tuple[float, float], 
                               line_start: Tuple[float, float], 
                               line_end: Tuple[float, float]) -> float:
        """Calculate perpendicular distance from point to line."""
        x0, y0 = point
        x1, y1 = line_start
        x2, y2 = line_end
        
        # Line length squared
        len_sq = (x2 - x1)**2 + (y2 - y1)**2
        
        if len_sq == 0:
            return math.sqrt((x0 - x1)**2 + (y0 - y1)**2)
        
        # Parameter t for closest point on line
        t = max(0, min(1, ((x0 - x1) * (x2 - x1) + (y0 - y1) * (y2 - y1)) / len_sq))
        
        # Closest point on line
        closest_x = x1 + t * (x2 - x1)
        closest_y = y1 + t * (y2 - y1)
        
        # Distance to closest point
        return math.sqrt((x0 - closest_x)**2 + (y0 - closest_y)**2)
    
    def _calculate_overlap(self, line1: Dict, line2: Dict) -> float:
        """Calculate overlap percentage between two parallel lines."""
        # Project lines onto common axis
        start1 = line1['properties']['start']
        end1 = line1['properties']['end']
        start2 = line2['properties']['start']
        end2 = line2['properties']['end']
        
        # Use line1's direction as reference
        vec = (end1[0] - start1[0], end1[1] - start1[1])
        length = math.sqrt(vec[0]**2 + vec[1]**2)
        
        if length == 0:
            return 0
            
        unit_vec = (vec[0] / length, vec[1] / length)
        
        # Project all points onto line1
        proj1_start = 0
        proj1_end = length
        
        proj2_start = ((start2[0] - start1[0]) * unit_vec[0] + 
                      (start2[1] - start1[1]) * unit_vec[1])
        proj2_end = ((end2[0] - start1[0]) * unit_vec[0] + 
                    (end2[1] - start1[1]) * unit_vec[1])
        
        # Ensure correct ordering
        if proj2_start > proj2_end:
            proj2_start, proj2_end = proj2_end, proj2_start
            
        # Calculate overlap
        overlap_start = max(proj1_start, proj2_start)
        overlap_end = min(proj1_end, proj2_end)
        
        if overlap_start >= overlap_end:
            return 0
            
        overlap_length = overlap_end - overlap_start
        min_length = min(proj1_end - proj1_start, proj2_end - proj2_start)
        
        return overlap_length / min_length if min_length > 0 else 0
    
    def _calculate_wall_endpoints(self, line1: Dict, line2: Dict) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Calculate the actual wall endpoints from two parallel lines."""
        # Get all endpoints
        points = [
            line1['properties']['start'],
            line1['properties']['end'],
            line2['properties']['start'],
            line2['properties']['end']
        ]
        
        # Find extreme points along line direction
        vec = (line1['properties']['end'][0] - line1['properties']['start'][0],
               line1['properties']['end'][1] - line1['properties']['start'][1])
        
        length = math.sqrt(vec[0]**2 + vec[1]**2)
        if length == 0:
            return points[0], points[1]
            
        unit_vec = (vec[0] / length, vec[1] / length)
        
        # Project all points and find extremes
        projections = []
        for point in points:
            proj = ((point[0] - points[0][0]) * unit_vec[0] + 
                   (point[1] - points[0][1]) * unit_vec[1])
            projections.append((proj, point))
            
        projections.sort(key=lambda x: x[0])
        
        # Average the positions perpendicular to line direction
        start_points = [p[1] for p in projections[:2]]
        end_points = [p[1] for p in projections[2:]]
        
        wall_start = (
            sum(p[0] for p in start_points) / 2,
            sum(p[1] for p in start_points) / 2
        )
        wall_end = (
            sum(p[0] for p in end_points) / 2,
            sum(p[1] for p in end_points) / 2
        )
        
        return wall_start, wall_end
    
    def group_walls_by_room(self, walls: List[Wall]) -> Dict[int, List[Wall]]:
        """Group walls that form enclosed spaces."""
        # This is a simplified version - full implementation would use
        # graph algorithms to find closed loops
        rooms = {}
        room_id = 0
        
        # For now, just group walls by proximity
        used_walls = set()
        
        for wall in walls:
            if wall in used_walls:
                continue
                
            room_walls = [wall]
            used_walls.add(wall)
            
            # Find connected walls
            for other_wall in walls:
                if other_wall in used_walls:
                    continue
                    
                if self._walls_connected(wall, other_wall):
                    room_walls.append(other_wall)
                    used_walls.add(other_wall)
                    
            if len(room_walls) >= 3:  # Minimum for enclosed space
                rooms[room_id] = room_walls
                room_id += 1
                
        return rooms
    
    def _walls_connected(self, wall1: Wall, wall2: Wall, tolerance: float = 50.0) -> bool:
        """Check if two walls are connected at endpoints."""
        endpoints1 = [wall1.start_point, wall1.end_point]
        endpoints2 = [wall2.start_point, wall2.end_point]
        
        for p1 in endpoints1:
            for p2 in endpoints2:
                dist = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
                if dist < tolerance:
                    return True
                    
        return False