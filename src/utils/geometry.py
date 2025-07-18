"""Geometry utilities."""
import math
from typing import Tuple

def calculate_distance(p1: Tuple[float, float, float], 
                      p2: Tuple[float, float, float]) -> float:
    """Calculate distance between two 3D points."""
    return math.sqrt(
        (p2[0] - p1[0])**2 + 
        (p2[1] - p1[1])**2 + 
        (p2[2] - p1[2])**2
    )

def calculate_midpoint(p1: Tuple[float, float, float], 
                      p2: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """Calculate midpoint between two points."""
    return (
        (p1[0] + p2[0]) / 2,
        (p1[1] + p2[1]) / 2,
        (p1[2] + p2[2]) / 2
    )
