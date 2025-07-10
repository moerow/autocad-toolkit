"""Simple geometry classes for AutoCAD entities."""
from dataclasses import dataclass
from typing import List, Optional, Tuple
import math


@dataclass
class Point:
    """3D Point"""
    x: float
    y: float
    z: float = 0.0
    
    def distance_to(self, other: 'Point') -> float:
        """Calculate distance to another point"""
        return math.sqrt(
            (self.x - other.x) ** 2 +
            (self.y - other.y) ** 2 +
            (self.z - other.z) ** 2
        )
    
    def __iter__(self):
        """Make Point iterable for easy conversion"""
        return iter([self.x, self.y, self.z])


@dataclass
class Line:
    """Line entity"""
    start: Point
    end: Point
    
    def length(self) -> float:
        """Get line length"""
        return self.start.distance_to(self.end)
    
    def angle(self) -> float:
        """Get line angle in radians"""
        return math.atan2(
            self.end.y - self.start.y,
            self.end.x - self.start.x
        )
    
    def midpoint(self) -> Point:
        """Get midpoint of line"""
        return Point(
            (self.start.x + self.end.x) / 2,
            (self.start.y + self.end.y) / 2,
            (self.start.z + self.end.z) / 2
        )


@dataclass
class Circle:
    """Circle entity"""
    center: Point
    radius: float
    
    def area(self) -> float:
        """Get circle area"""
        return math.pi * self.radius ** 2
    
    def circumference(self) -> float:
        """Get circle circumference"""
        return 2 * math.pi * self.radius


@dataclass
class Arc:
    """Arc entity"""
    center: Point
    radius: float
    start_angle: float  # in radians
    end_angle: float    # in radians
    
    def arc_length(self) -> float:
        """Get arc length"""
        angle_span = self.end_angle - self.start_angle
        if angle_span < 0:
            angle_span += 2 * math.pi
        return self.radius * angle_span


@dataclass
class Polyline:
    """Polyline entity"""
    vertices: List[Point]
    is_closed: bool = False
    
    def length(self) -> float:
        """Get total length of polyline"""
        total = 0.0
        for i in range(len(self.vertices) - 1):
            total += self.vertices[i].distance_to(self.vertices[i + 1])
        
        if self.is_closed and len(self.vertices) > 2:
            total += self.vertices[-1].distance_to(self.vertices[0])
        
        return total
    
    def segments(self) -> List[Line]:
        """Get line segments"""
        segs = []
        for i in range(len(self.vertices) - 1):
            segs.append(Line(self.vertices[i], self.vertices[i + 1]))
        
        if self.is_closed and len(self.vertices) > 2:
            segs.append(Line(self.vertices[-1], self.vertices[0]))
        
        return segs