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
            'offset_distance': 0.2,   # Very close to lines (0.2mm from drawing)
            'text_height': 0.08,      # Tiny text for clean architectural look (0.8mm text)
            'arrow_size': 0.05,       # Minimal arrow size
            'layer': 'DIMENSIONS',
            'color': 3,
            'min_length': 0.5,  # Much smaller for architectural drawings (0.5 units = 50cm in meters, or 0.5mm in mm)
            'duplicate_tolerance': 0.1  # Tolerance for detecting duplicate dimensions
        }
        self.existing_dimensions = []  # Track placed dimensions to avoid duplicates

    def dimension_all_lines(self, layer_filter: Optional[str] = None) -> Dict[str, int]:
        """Add dimensions to all lines in the drawing."""
        logger.info("Starting dimensioning process...")
        
        # COM already initialized in main thread
        
        # Simple checks - just verify we have the basic objects
        if not self.cad:
            raise Exception("No AutoCAD connection object")
            
        if not hasattr(self.cad, 'model') or not self.cad.model:
            raise Exception("No AutoCAD model space available")

        results = {'lines': 0, 'total': 0}
        self.existing_dimensions = []  # Reset tracking for this session

        try:
            # Set dimension layer
            self._ensure_layer(self.config['layer'])

            # Process lines and polylines (for architectural drawings)
            entity_count = 0
            line_count = 0
            polyline_count = 0
            
            for entity in self.cad.model:
                entity_count += 1
                
                # Handle both Lines and Polylines (common in architectural drawings)
                if entity.ObjectName == 'AcDbLine':
                    line_count += 1
                    logger.debug(f"Processing line {line_count} on layer '{entity.Layer}'")
                    if layer_filter and entity.Layer != layer_filter:
                        logger.debug(f"Skipping line - wrong layer (need '{layer_filter}', got '{entity.Layer}')")
                        continue
                    logger.debug(f"Attempting to add dimension to line {line_count}")
                    if self._add_dimension_to_line(entity):
                        results['lines'] += 1
                        results['total'] += 1
                        logger.debug(f"Successfully dimensioned line {line_count}")
                    else:
                        logger.debug(f"Failed to dimension line {line_count}")
                        
                elif entity.ObjectName in ['AcDbPolyline', 'AcDb2dPolyline']:
                    polyline_count += 1
                    if layer_filter and entity.Layer != layer_filter:
                        continue
                    # Process polyline segments
                    segments_added = self._add_dimensions_to_polyline(entity)
                    results['lines'] += segments_added
                    results['total'] += segments_added
                    
            logger.info(f"Processed {entity_count} entities: {line_count} lines, {polyline_count} polylines")
            logger.info(f"Successfully dimensioned {results['total']} entities")

        except Exception as e:
            logger.error(f"Error dimensioning lines: {e}")

        return results

    def _add_dimension_to_line(self, line) -> bool:
        """Add dimension to a single line."""
        try:
            # Get endpoints
            start = APoint(line.StartPoint)
            end = APoint(line.EndPoint)

            # Use shared segment dimensioning method
            return self._add_dimension_to_segment(start, end)

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

    def analyze_existing_dimensions(self) -> Dict[str, Any]:
        """Analyze existing dimensions in the drawing."""
        analysis = {
            'total_count': 0,
            'by_type': {},
            'by_layer': {},
            'dimension_styles': set(),
            'text_heights': [],
            'dimension_list': []
        }
        
        dim_types = ['AcDbAlignedDimension', 'AcDbRotatedDimension', 'AcDbRadialDimension', 
                    'AcDbDiametricDimension', 'AcDbLinearDimension', 'AcDbOrdinateDimension',
                    'AcDbAngularDimension', 'AcDb3PointAngularDimension', 'AcDbArcDimension']
        
        try:
            for entity in self.cad.model:
                if entity.ObjectName in dim_types:
                    analysis['total_count'] += 1
                    
                    # Count by type
                    dim_type = entity.ObjectName.replace('AcDb', '').replace('Dimension', '')
                    analysis['by_type'][dim_type] = analysis['by_type'].get(dim_type, 0) + 1
                    
                    # Count by layer
                    layer = entity.Layer
                    analysis['by_layer'][layer] = analysis['by_layer'].get(layer, 0) + 1
                    
                    # Collect dimension info
                    try:
                        dim_info = {
                            'type': dim_type,
                            'layer': layer,
                            'text_height': entity.TextHeight if hasattr(entity, 'TextHeight') else None,
                            'measurement': entity.Measurement if hasattr(entity, 'Measurement') else None
                        }
                        
                        # Get dimension style
                        if hasattr(entity, 'StyleName'):
                            analysis['dimension_styles'].add(entity.StyleName)
                            dim_info['style'] = entity.StyleName
                            
                        # Collect text heights
                        if hasattr(entity, 'TextHeight'):
                            analysis['text_heights'].append(entity.TextHeight)
                            
                        analysis['dimension_list'].append(dim_info)
                    except:
                        pass
                        
        except Exception as e:
            logger.error(f"Error analyzing dimensions: {e}")
            
        # Convert set to list for JSON serialization
        analysis['dimension_styles'] = list(analysis['dimension_styles'])
        
        # Calculate text height statistics
        if analysis['text_heights']:
            analysis['text_height_stats'] = {
                'min': min(analysis['text_heights']),
                'max': max(analysis['text_heights']),
                'avg': sum(analysis['text_heights']) / len(analysis['text_heights'])
            }
        
        return analysis

    def clear_all_dimensions(self) -> int:
        """Remove all dimensions from the drawing."""
        count = 0
        dim_types = ['AcDbAlignedDimension', 'AcDbRotatedDimension', 'AcDbRadialDimension', 
                    'AcDbDiametricDimension', 'AcDbLinearDimension', 'AcDbOrdinateDimension',
                    'AcDbAngularDimension', 'AcDb3PointAngularDimension', 'AcDbArcDimension']

        for entity in self.cad.model:
            if entity.ObjectName in dim_types:
                try:
                    entity.Delete()
                    count += 1
                except:
                    pass

        return count
        
    def clear_dimensions_by_layer(self, layer_name: str) -> int:
        """Remove dimensions from a specific layer."""
        count = 0
        dim_types = ['AcDbAlignedDimension', 'AcDbRotatedDimension', 'AcDbRadialDimension', 
                    'AcDbDiametricDimension', 'AcDbLinearDimension', 'AcDbOrdinateDimension',
                    'AcDbAngularDimension', 'AcDb3PointAngularDimension', 'AcDbArcDimension']
        
        for entity in self.cad.model:
            if entity.ObjectName in dim_types and entity.Layer == layer_name:
                try:
                    entity.Delete()
                    count += 1
                except:
                    pass
                    
        return count

    def _add_dimensions_to_polyline(self, polyline) -> int:
        """Add dimensions to polyline segments (for architectural walls)."""
        try:
            dimensions_added = 0
            
            # Get polyline coordinates
            coordinates = polyline.Coordinates
            
            # Convert flat coordinate array to list of points (X,Y pairs)
            points = []
            for i in range(0, len(coordinates), 2):
                points.append((coordinates[i], coordinates[i+1]))
            
            # Process each segment of the polyline
            for i in range(len(points) - 1):
                start_point = APoint(points[i][0], points[i][1])
                end_point = APoint(points[i+1][0], points[i+1][1])
                
                # Calculate segment length
                length = math.sqrt((end_point.x - start_point.x) ** 2 + (end_point.y - start_point.y) ** 2)
                
                # Skip very short segments (like wall thickness lines)
                if length < self.config['min_length']:
                    continue
                
                # Add dimension for this segment
                if self._add_dimension_to_segment(start_point, end_point):
                    dimensions_added += 1
                    
            return dimensions_added
            
        except Exception as e:
            logger.error(f"Error adding dimensions to polyline: {e}")
            return 0

    def _add_dimension_to_segment(self, start_point, end_point) -> bool:
        """Add dimension to a line segment (shared by lines and polyline segments)."""
        try:
            # Calculate length
            length = math.sqrt((end_point.x - start_point.x) ** 2 + (end_point.y - start_point.y) ** 2)
            logger.debug(f"Segment length: {length:.2f}mm, min_length: {self.config['min_length']}mm")
            
            if length < self.config['min_length']:
                logger.debug(f"Skipping segment - too short ({length:.2f} < {self.config['min_length']})")
                return False
                
            # Check if a similar dimension already exists
            if self._is_duplicate_dimension(start_point, end_point, length):
                logger.debug(f"Skipping duplicate dimension for segment {length:.2f}mm")
                return False

            # Calculate dimension position
            mid_x = (start_point.x + end_point.x) / 2
            mid_y = (start_point.y + end_point.y) / 2

            # Calculate perpendicular offset
            angle = math.atan2(end_point.y - start_point.y, end_point.x - start_point.x)
            offset_angle = angle + math.pi / 2

            dim_x = mid_x + self.config['offset_distance'] * math.cos(offset_angle)
            dim_y = mid_y + self.config['offset_distance'] * math.sin(offset_angle)
            dim_location = APoint(dim_x, dim_y)

            logger.debug(f"Adding dimension: start={start_point.x:.1f},{start_point.y:.1f} end={end_point.x:.1f},{end_point.y:.1f}")

            # Determine if line is more horizontal or vertical
            dx = abs(end_point.x - start_point.x)
            dy = abs(end_point.y - start_point.y)
            
            # Add LINEAR dimension (DIMLINEAR style) - 0 for horizontal, pi/2 for vertical
            if dx > dy:
                # Horizontal dimension
                dim = self.cad.model.AddDimRotated(start_point, end_point, dim_location, 0)
            else:
                # Vertical dimension
                dim = self.cad.model.AddDimRotated(start_point, end_point, dim_location, 1.5708)  # 90 degrees in radians
            dim.TextHeight = self.config['text_height']
            dim.Layer = self.config['layer']
            
            # Make dimensions minimal and clean
            try:
                dim.ArrowheadSize = self.config['arrow_size']  # Tiny arrows
                dim.ExtensionLineExtend = 0.02  # Very short extension lines
                dim.ExtensionLineOffset = 0.01  # Minimal offset from object
                dim.DimensionLineWeight = 25    # Thin dimension lines (0.25mm)
                dim.TextGap = 0.01             # Minimal gap between text and line
            except:
                pass  # Some properties might not be available

            logger.debug(f"Successfully added dimension with length {length:.2f}mm")
            
            # Track this dimension to avoid duplicates
            self.existing_dimensions.append({
                'start': (start_point.x, start_point.y),
                'end': (end_point.x, end_point.y),
                'length': length,
                'location': (dim_x, dim_y)
            })
            
            return True

        except Exception as e:
            logger.error(f"Error adding dimension to segment: {e}")
            return False
            
    def _is_duplicate_dimension(self, start_point, end_point, length) -> bool:
        """Check if a similar dimension already exists."""
        tolerance = self.config['duplicate_tolerance']
        
        for existing in self.existing_dimensions:
            # Check if lengths are similar
            if abs(existing['length'] - length) > tolerance:
                continue
                
            # Check if it's the same line segment (in either direction)
            start1 = (start_point.x, start_point.y)
            end1 = (end_point.x, end_point.y)
            start2 = existing['start']
            end2 = existing['end']
            
            # Check both directions
            same_segment = (
                (abs(start1[0] - start2[0]) < tolerance and abs(start1[1] - start2[1]) < tolerance and
                 abs(end1[0] - end2[0]) < tolerance and abs(end1[1] - end2[1]) < tolerance) or
                (abs(start1[0] - end2[0]) < tolerance and abs(start1[1] - end2[1]) < tolerance and
                 abs(end1[0] - start2[0]) < tolerance and abs(end1[1] - start2[1]) < tolerance)
            )
            
            # Check if parallel and close (likely wall thickness)
            if not same_segment:
                # Calculate if lines are parallel and close
                dx1 = end_point.x - start_point.x
                dy1 = end_point.y - start_point.y
                dx2 = existing['end'][0] - existing['start'][0]
                dy2 = existing['end'][1] - existing['start'][1]
                
                # Check if parallel (cross product near zero)
                cross = abs(dx1 * dy2 - dy1 * dx2)
                if cross < tolerance * length:  # Parallel lines
                    # Check distance between lines
                    # Distance from start_point to the existing line
                    dist = abs((existing['end'][1] - existing['start'][1]) * start_point.x - 
                              (existing['end'][0] - existing['start'][0]) * start_point.y + 
                              existing['end'][0] * existing['start'][1] - 
                              existing['end'][1] * existing['start'][0]) / existing['length']
                    
                    if dist < 0.5:  # If lines are very close (wall thickness typically < 0.5)
                        logger.debug(f"Found parallel line at distance {dist:.2f}, treating as duplicate")
                        return True
            else:
                return True
                
        return False