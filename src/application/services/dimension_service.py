"""Dimension service implementation."""
import logging
import math
from typing import List, Dict, Any, Optional, Tuple
from uuid import uuid4
from pyautocad import APoint

logger = logging.getLogger(__name__)


class DimensionService:
    """Service for managing dimensions in drawings."""

    def __init__(self, cad_connection):
        self.cad = cad_connection
        self.config = {
            'offset_distance': 500,
            'text_height': 100,
            'layer': 'DIMENSIONS',
            'color': 3,
            'min_length': 100
        }

    def dimension_all_lines(self, layer_filter: Optional[str] = None) -> Dict[str, int]:
        """Add dimensions to all lines in the drawing."""
        if not self.cad.is_connected():
            raise Exception("Not connected to AutoCAD")

        results = {'lines': 0, 'total': 0}

        try:
            # Set dimension layer
            self._ensure_layer(self.config['layer'])

            # Process all lines
            for entity in self.cad.model:
                if entity.ObjectName != 'AcDbLine':
                    continue
                    
                if layer_filter and entity.Layer != layer_filter:
                    continue

                if self._add_dimension_to_line(entity):
                    results['lines'] += 1
                    results['total'] += 1

        except Exception as e:
            logger.error(f"Error dimensioning lines: {e}")

        return results

    def _add_dimension_to_line(self, line) -> bool:
        """Add dimension to a single line."""
        try:
            # Get endpoints
            start = APoint(line.StartPoint)
            end = APoint(line.EndPoint)

            # Calculate length
            length = math.sqrt((end.x - start.x) ** 2 + (end.y - start.y) ** 2)
            if length < self.config['min_length']:
                return False

            # Calculate dimension position
            mid_x = (start.x + end.x) / 2
            mid_y = (start.y + end.y) / 2

            # Calculate perpendicular offset
            angle = math.atan2(end.y - start.y, end.x - start.x)
            offset_angle = angle + math.pi / 2

            dim_x = mid_x + self.config['offset_distance'] * math.cos(offset_angle)
            dim_y = mid_y + self.config['offset_distance'] * math.sin(offset_angle)
            dim_location = APoint(dim_x, dim_y)

            # Add dimension
            dim = self.cad.model.AddDimAligned(start, end, dim_location)
            dim.TextHeight = self.config['text_height']
            dim.Layer = self.config['layer']

            return True

        except Exception as e:
            logger.error(f"Error adding dimension: {e}")
            return False

    def _ensure_layer(self, layer_name: str):
        """Ensure layer exists and set it as current."""
        try:
            layer = self.cad.doc.Layers.Add(layer_name)
            self.cad.doc.ActiveLayer = layer
        except:
            # Layer might already exist
            pass

    def clear_all_dimensions(self) -> int:
        """Remove all dimensions from the drawing."""
        count = 0
        dim_types = ['AcDbAlignedDimension', 'AcDbRotatedDimension', 'AcDbRadialDimension', 'AcDbDiametricDimension']

        for entity in self.cad.model:
            if entity.ObjectName in dim_types:
                try:
                    entity.Delete()
                    count += 1
                except:
                    pass

        return count