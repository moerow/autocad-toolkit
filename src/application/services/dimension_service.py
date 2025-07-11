"""Dimension service implementation with AI integration."""
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
            'text_height': 0.15,      # Bigger text for better readability (1.5mm text)
            'arrow_size': 0.08,       # Slightly bigger arrow size
            'layer': 'DIMENSIONS',
            'color': 3,
            'min_length': 0.5,  # Much smaller for architectural drawings (0.5 units = 50cm in meters, or 0.5mm in mm)
            'duplicate_tolerance': 0.1,  # Tolerance for detecting duplicate dimensions
            # Professional 3-layer dimensioning configuration
            'professional_config': {
                'layer_1_offset': 3.0,   # Total length dimension (outermost) - 3cm from drawing
                'layer_2_offset': 2.0,   # Projections dimension (middle) - 2cm from drawing
                'layer_3_offset': 1.0,   # Openings dimension (closest to drawing, 1cm gap as specified)
                'inner_offset': 0.1      # Inner dimensions offset
            }
        }
        self.existing_dimensions = []  # Track placed dimensions to avoid duplicates
        
        # AI integration (lazy-loaded)
        self._ai_service = None

    def dimension_all_lines(self, layer_filter: Optional[str] = None) -> Dict[str, int]:
        """Add dimensions to all lines in the drawing (traditional method)."""
        logger.info("Starting traditional dimensioning process...")
        
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
    
    def dimension_all_lines_ai(self, layer_filter: Optional[str] = None) -> Dict[str, int]:
        """Add dimensions to all lines using AI intelligence."""
        logger.info("Starting AI-powered dimensioning process...")
        
        # Initialize AI service if not loaded
        if not self._ai_service:
            try:
                from src.infrastructure.ai.intelligent_dimensioning import IntelligentDimensionService
                self._ai_service = IntelligentDimensionService(self.cad)
                logger.info("AI dimensioning service initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize AI service: {e}")
                logger.info("Falling back to traditional dimensioning")
                return self.dimension_all_lines(layer_filter)
        
        # Use AI to intelligently dimension the drawing
        try:
            ai_results = self._ai_service.intelligent_dimension_all(layer_filter)
            
            # Convert AI results to traditional format
            results = {
                'lines': ai_results.get('total_dimensions', 0),
                'total': ai_results.get('total_dimensions', 0),
                'ai_plan': ai_results.get('plan', {}),
                'ai_breakdown': {
                    'critical': ai_results.get('critical_dimensions', 0),
                    'important': ai_results.get('important_dimensions', 0),
                    'detail': ai_results.get('detail_dimensions', 0)
                }
            }
            
            logger.info(f"AI dimensioning completed: {results['total']} dimensions added")
            logger.info(f"AI breakdown - Critical: {results['ai_breakdown']['critical']}, "
                       f"Important: {results['ai_breakdown']['important']}, "
                       f"Detail: {results['ai_breakdown']['detail']}")
            
            return results
            
        except Exception as e:
            logger.error(f"AI dimensioning failed: {e}")
            logger.info("Falling back to traditional dimensioning")
            return self.dimension_all_lines(layer_filter)
    
    def get_ai_service(self):
        """Get the AI service instance (for external access)."""
        if not self._ai_service:
            try:
                from src.infrastructure.ai.intelligent_dimensioning import IntelligentDimensionService
                self._ai_service = IntelligentDimensionService(self.cad)
            except Exception as e:
                logger.error(f"Failed to initialize AI service: {e}")
                return None
        return self._ai_service
    
    def dimension_professional_3_layer(self, layer_filter: Optional[str] = None, 
                                     excluded_layers: Optional[List[str]] = None) -> Dict[str, Any]:
        """Professional 3-layer dimensioning system as per father's specifications.
        
        Creates 4 sets of dimensions (one per side: bottom, top, left, right).
        Each set has 3 layers:
        1. Outer layer: Total length end-to-end (3cm from drawing)
        2. Middle layer: Major segments (2cm from drawing)  
        3. Inner layer: Detailed segments (1cm from drawing)
        
        SAFE IMPLEMENTATION: Only adds dimensions, never modifies existing entities.
        """
        logger.info("Starting professional 3-layer dimensioning system...")
        
        if not self.cad:
            raise Exception("No AutoCAD connection object")
            
        if not hasattr(self.cad, 'model') or not self.cad.model:
            raise Exception("No AutoCAD model space available")

        results = {
            'total_dimensions': 0,
            'layer_1_total': 0,      # Total length dimensions (4 total - one per side)
            'layer_2_major': 0,      # Major segment dimensions
            'layer_3_detailed': 0,   # Detailed segment dimensions
            'sides_processed': {'bottom': 0, 'top': 0, 'left': 0, 'right': 0}
        }
        
        try:
            # Set dimension layer (safe operation)
            self._ensure_layer(self.config['layer'])
            
            # Get drawing bounds safely (READ-ONLY)
            bounds = self._get_drawing_bounds_safe()
            logger.info(f"Drawing bounds: {bounds}")
            
            if bounds is None:
                logger.error("Could not determine drawing bounds")
                return results
            
            # Get offsets from config
            config = self.config['professional_config']
            
            # Create 4 sets of dimensions (one per side)
            sides = ['bottom', 'top', 'left', 'right']
            
            for side in sides:
                logger.info(f"Processing {side} side...")
                
                # Layer 1: Total length (outermost)
                total_dims = self._add_total_length_dimension(side, bounds, config['layer_1_offset'])
                results['layer_1_total'] += total_dims
                
                # Layer 2: Major segments (middle)
                major_dims = self._add_major_segment_dimensions(side, bounds, config['layer_2_offset'], layer_filter, excluded_layers)
                results['layer_2_major'] += major_dims
                
                # Layer 3: Detailed segments (closest to drawing)
                detail_dims = self._add_detailed_segment_dimensions(side, bounds, config['layer_3_offset'], layer_filter, excluded_layers)
                results['layer_3_detailed'] += detail_dims
                
                side_total = total_dims + major_dims + detail_dims
                results['sides_processed'][side] = side_total
                logger.info(f"{side} side completed: {side_total} dimensions")
            
            # Calculate total
            results['total_dimensions'] = (
                results['layer_1_total'] + 
                results['layer_2_major'] + 
                results['layer_3_detailed']
            )
            
            logger.info(f"Professional dimensioning completed!")
            logger.info(f"Total dimensions: {results['total_dimensions']}")
            logger.info(f"Layer 1 (Total): {results['layer_1_total']}")
            logger.info(f"Layer 2 (Major): {results['layer_2_major']}")
            logger.info(f"Layer 3 (Detailed): {results['layer_3_detailed']}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in professional dimensioning: {e}")
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
    
    def _get_filtered_entities(self, layer_filter: Optional[str] = None, 
                              excluded_layers: Optional[List[str]] = None) -> List[Any]:
        """Get entities based on layer filtering."""
        filtered_entities = []
        
        try:
            for entity in self.cad.model:
                # Skip non-geometric entities
                if entity.ObjectName not in ['AcDbLine', 'AcDbPolyline', 'AcDb2dPolyline', 
                                            'AcDbCircle', 'AcDbArc', 'AcDbBlockReference']:
                    continue
                    
                # Apply layer filtering
                entity_layer = entity.Layer
                
                # Exclude specific layers
                if excluded_layers and entity_layer in excluded_layers:
                    continue
                    
                # Include only specific layer
                if layer_filter and entity_layer != layer_filter:
                    continue
                    
                filtered_entities.append(entity)
                
        except Exception as e:
            logger.error(f"Error filtering entities: {e}")
            
        return filtered_entities
    
    def _analyze_drawing_for_professional_dimensioning(self, entities: List[Any]) -> Dict[str, Any]:
        """Analyze drawing to prepare for professional dimensioning."""
        analysis = {
            'bounds': {'min_x': float('inf'), 'max_x': float('-inf'),
                      'min_y': float('inf'), 'max_y': float('-inf')},
            'walls': [],
            'openings': [],
            'projections': [],
            'other_components': []
        }
        
        try:
            for entity in entities:
                # Update bounds
                if hasattr(entity, 'StartPoint') and hasattr(entity, 'EndPoint'):
                    # Lines
                    start = entity.StartPoint
                    end = entity.EndPoint
                    analysis['bounds']['min_x'] = min(analysis['bounds']['min_x'], start[0], end[0])
                    analysis['bounds']['max_x'] = max(analysis['bounds']['max_x'], start[0], end[0])
                    analysis['bounds']['min_y'] = min(analysis['bounds']['min_y'], start[1], end[1])
                    analysis['bounds']['max_y'] = max(analysis['bounds']['max_y'], start[1], end[1])
                    
                    # Classify entity
                    length = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
                    if length >= self.config['min_length']:
                        analysis['walls'].append(entity)
                        
                elif hasattr(entity, 'Center') and hasattr(entity, 'Radius'):
                    # Circles/Arcs
                    center = entity.Center
                    radius = entity.Radius
                    analysis['bounds']['min_x'] = min(analysis['bounds']['min_x'], center[0] - radius)
                    analysis['bounds']['max_x'] = max(analysis['bounds']['max_x'], center[0] + radius)
                    analysis['bounds']['min_y'] = min(analysis['bounds']['min_y'], center[1] - radius)
                    analysis['bounds']['max_y'] = max(analysis['bounds']['max_y'], center[1] + radius)
                    
                elif entity.ObjectName == 'AcDbBlockReference':
                    # Block references (doors, windows)
                    insertion_point = entity.InsertionPoint
                    analysis['bounds']['min_x'] = min(analysis['bounds']['min_x'], insertion_point[0])
                    analysis['bounds']['max_x'] = max(analysis['bounds']['max_x'], insertion_point[0])
                    analysis['bounds']['min_y'] = min(analysis['bounds']['min_y'], insertion_point[1])
                    analysis['bounds']['max_y'] = max(analysis['bounds']['max_y'], insertion_point[1])
                    
                    # Classify as opening based on block name
                    block_name = entity.Name.lower()
                    if any(keyword in block_name for keyword in ['door', 'window', 'opening']):
                        analysis['openings'].append(entity)
                    else:
                        analysis['other_components'].append(entity)
                        
        except Exception as e:
            logger.error(f"Error analyzing drawing: {e}")
            
        return analysis
    
    def _create_3_layer_dimensions_for_side(self, side: str, drawing_analysis: Dict[str, Any], 
                                           entities: List[Any]) -> Dict[str, int]:
        """Create 3-layer dimensions for one side of the drawing."""
        results = {'layer_1': 0, 'layer_2': 0, 'layer_3': 0, 'total': 0}
        
        try:
            bounds = drawing_analysis['bounds']
            config = self.config['professional_config']
            
            # Determine side orientation and base coordinates
            if side == 'bottom':
                base_y = bounds['min_y']
                is_horizontal = True
                offsets = {
                    'layer_1': base_y - config['layer_1_offset'],
                    'layer_2': base_y - config['layer_2_offset'],
                    'layer_3': base_y - config['layer_3_offset']
                }
            elif side == 'top':
                base_y = bounds['max_y']
                is_horizontal = True
                offsets = {
                    'layer_1': base_y + config['layer_1_offset'],
                    'layer_2': base_y + config['layer_2_offset'],
                    'layer_3': base_y + config['layer_3_offset']
                }
            elif side == 'left':
                base_x = bounds['min_x']
                is_horizontal = False
                offsets = {
                    'layer_1': base_x - config['layer_1_offset'],
                    'layer_2': base_x - config['layer_2_offset'],
                    'layer_3': base_x - config['layer_3_offset']
                }
            else:  # right
                base_x = bounds['max_x']
                is_horizontal = False
                offsets = {
                    'layer_1': base_x + config['layer_1_offset'],
                    'layer_2': base_x + config['layer_2_offset'],
                    'layer_3': base_x + config['layer_3_offset']
                }
            
            # Layer 1: Total length end-to-end
            if is_horizontal:
                start_pt = APoint(bounds['min_x'], offsets['layer_1'])
                end_pt = APoint(bounds['max_x'], offsets['layer_1'])
                dim_loc = APoint((bounds['min_x'] + bounds['max_x']) / 2, offsets['layer_1'])
                dim = self.cad.model.AddDimRotated(start_pt, end_pt, dim_loc, 0)
            else:
                start_pt = APoint(offsets['layer_1'], bounds['min_y'])
                end_pt = APoint(offsets['layer_1'], bounds['max_y'])
                dim_loc = APoint(offsets['layer_1'], (bounds['min_y'] + bounds['max_y']) / 2)
                dim = self.cad.model.AddDimRotated(start_pt, end_pt, dim_loc, 1.5708)
            
            self._format_dimension(dim)
            results['layer_1'] += 1
            
            # Layer 2: All projections detailed
            projections = self._identify_projections(entities, side, bounds)
            for projection in projections:
                if is_horizontal:
                    start_pt = APoint(projection['start_x'], offsets['layer_2'])
                    end_pt = APoint(projection['end_x'], offsets['layer_2'])
                    dim_loc = APoint((projection['start_x'] + projection['end_x']) / 2, offsets['layer_2'])
                    dim = self.cad.model.AddDimRotated(start_pt, end_pt, dim_loc, 0)
                else:
                    start_pt = APoint(offsets['layer_2'], projection['start_y'])
                    end_pt = APoint(offsets['layer_2'], projection['end_y'])
                    dim_loc = APoint(offsets['layer_2'], (projection['start_y'] + projection['end_y']) / 2)
                    dim = self.cad.model.AddDimRotated(start_pt, end_pt, dim_loc, 1.5708)
                
                self._format_dimension(dim)
                results['layer_2'] += 1
            
            # Layer 3: Openings (doors, windows)
            openings = self._identify_openings_for_side(drawing_analysis['openings'], side, bounds)
            for opening in openings:
                if is_horizontal:
                    start_pt = APoint(opening['start_x'], offsets['layer_3'])
                    end_pt = APoint(opening['end_x'], offsets['layer_3'])
                    dim_loc = APoint((opening['start_x'] + opening['end_x']) / 2, offsets['layer_3'])
                    dim = self.cad.model.AddDimRotated(start_pt, end_pt, dim_loc, 0)
                else:
                    start_pt = APoint(offsets['layer_3'], opening['start_y'])
                    end_pt = APoint(offsets['layer_3'], opening['end_y'])
                    dim_loc = APoint(offsets['layer_3'], (opening['start_y'] + opening['end_y']) / 2)
                    dim = self.cad.model.AddDimRotated(start_pt, end_pt, dim_loc, 1.5708)
                
                self._format_dimension(dim)
                results['layer_3'] += 1
            
            results['total'] = results['layer_1'] + results['layer_2'] + results['layer_3']
            
        except Exception as e:
            logger.error(f"Error creating 3-layer dimensions for {side}: {e}")
            
        return results
    
    def _identify_projections(self, entities: List[Any], side: str, bounds: Dict[str, float]) -> List[Dict[str, float]]:
        """Identify projections for Layer 2 dimensioning."""
        projections = []
        
        try:
            # For now, create basic projections based on major wall segments
            # This is a simplified implementation - could be enhanced with more sophisticated analysis
            
            relevant_entities = []
            for entity in entities:
                if hasattr(entity, 'StartPoint') and hasattr(entity, 'EndPoint'):
                    start = entity.StartPoint
                    end = entity.EndPoint
                    
                    # Filter entities relevant to this side
                    if side in ['bottom', 'top']:
                        # For horizontal sides, look for vertical or near-vertical lines
                        if abs(end[0] - start[0]) > self.config['min_length']:
                            relevant_entities.append({
                                'start_x': min(start[0], end[0]),
                                'end_x': max(start[0], end[0]),
                                'y': (start[1] + end[1]) / 2
                            })
                    else:
                        # For vertical sides, look for horizontal or near-horizontal lines
                        if abs(end[1] - start[1]) > self.config['min_length']:
                            relevant_entities.append({
                                'start_y': min(start[1], end[1]),
                                'end_y': max(start[1], end[1]),
                                'x': (start[0] + end[0]) / 2
                            })
            
            # Group and merge overlapping projections
            if side in ['bottom', 'top']:
                # Sort by start_x and merge overlapping segments
                relevant_entities.sort(key=lambda x: x['start_x'])
                current_start = bounds['min_x']
                
                for entity in relevant_entities:
                    if entity['start_x'] > current_start:
                        projections.append({
                            'start_x': current_start,
                            'end_x': entity['start_x']
                        })
                    current_start = max(current_start, entity['end_x'])
                
                # Add final projection
                if current_start < bounds['max_x']:
                    projections.append({
                        'start_x': current_start,
                        'end_x': bounds['max_x']
                    })
            else:
                # Sort by start_y and merge overlapping segments
                relevant_entities.sort(key=lambda x: x['start_y'])
                current_start = bounds['min_y']
                
                for entity in relevant_entities:
                    if entity['start_y'] > current_start:
                        projections.append({
                            'start_y': current_start,
                            'end_y': entity['start_y']
                        })
                    current_start = max(current_start, entity['end_y'])
                
                # Add final projection
                if current_start < bounds['max_y']:
                    projections.append({
                        'start_y': current_start,
                        'end_y': bounds['max_y']
                    })
            
        except Exception as e:
            logger.error(f"Error identifying projections: {e}")
            
        return projections
    
    def _identify_openings_for_side(self, openings: List[Any], side: str, bounds: Dict[str, float]) -> List[Dict[str, float]]:
        """Identify openings for Layer 3 dimensioning."""
        side_openings = []
        
        try:
            for opening in openings:
                if hasattr(opening, 'InsertionPoint'):
                    point = opening.InsertionPoint
                    
                    # Get opening dimensions (simplified)
                    width = 0.9  # Default door width
                    height = 2.1  # Default door height
                    
                    # Try to get actual dimensions if available
                    if hasattr(opening, 'XScaleFactor'):
                        width = opening.XScaleFactor
                    if hasattr(opening, 'YScaleFactor'):
                        height = opening.YScaleFactor
                    
                    # Create opening dimensions based on side
                    if side in ['bottom', 'top']:
                        side_openings.append({
                            'start_x': point[0] - width/2,
                            'end_x': point[0] + width/2
                        })
                    else:
                        side_openings.append({
                            'start_y': point[1] - height/2,
                            'end_y': point[1] + height/2
                        })
                        
        except Exception as e:
            logger.error(f"Error identifying openings: {e}")
            
        return side_openings
    
    def _add_inner_dimensions(self, entities: List[Any]) -> int:
        """Add inner dimensions for other components."""
        count = 0
        
        try:
            # This is similar to the traditional dimensioning but with smaller offset
            inner_offset = self.config['professional_config']['inner_offset']
            original_offset = self.config['offset_distance']
            
            # Temporarily use smaller offset for inner dimensions
            self.config['offset_distance'] = inner_offset
            
            for entity in entities:
                if entity.ObjectName == 'AcDbLine':
                    if self._add_dimension_to_line(entity):
                        count += 1
                elif entity.ObjectName in ['AcDbPolyline', 'AcDb2dPolyline']:
                    count += self._add_dimensions_to_polyline(entity)
            
            # Restore original offset
            self.config['offset_distance'] = original_offset
            
        except Exception as e:
            logger.error(f"Error adding inner dimensions: {e}")
            
        return count
    
    def _format_dimension(self, dim):
        """Apply consistent formatting to dimensions."""
        try:
            dim.TextHeight = self.config['text_height']
            dim.Layer = self.config['layer']
            dim.ArrowheadSize = self.config['arrow_size']
            dim.ExtensionLineExtend = 0.02
            dim.ExtensionLineOffset = 0.01
            dim.DimensionLineWeight = 25
            dim.TextGap = 0.01
        except Exception as e:
            logger.debug(f"Could not set all dimension properties: {e}")
    
    def get_available_layers(self) -> List[str]:
        """Get all available layers in the drawing."""
        layers = []
        
        try:
            for layer in self.cad.doc.Layers:
                layers.append(layer.Name)
        except Exception as e:
            logger.error(f"Error getting layers: {e}")
            
        return layers
    
    def get_layer_statistics(self) -> Dict[str, Dict[str, int]]:
        """Get statistics about entities on each layer."""
        stats = {}
        
        try:
            for entity in self.cad.model:
                layer_name = entity.Layer
                if layer_name not in stats:
                    stats[layer_name] = {
                        'lines': 0,
                        'polylines': 0,
                        'circles': 0,
                        'arcs': 0,
                        'blocks': 0,
                        'other': 0,
                        'total': 0
                    }
                
                if entity.ObjectName == 'AcDbLine':
                    stats[layer_name]['lines'] += 1
                elif entity.ObjectName in ['AcDbPolyline', 'AcDb2dPolyline']:
                    stats[layer_name]['polylines'] += 1
                elif entity.ObjectName == 'AcDbCircle':
                    stats[layer_name]['circles'] += 1
                elif entity.ObjectName == 'AcDbArc':
                    stats[layer_name]['arcs'] += 1
                elif entity.ObjectName == 'AcDbBlockReference':
                    stats[layer_name]['blocks'] += 1
                else:
                    stats[layer_name]['other'] += 1
                    
                stats[layer_name]['total'] += 1
                
        except Exception as e:
            logger.error(f"Error getting layer statistics: {e}")
            
        return stats
    
    def _get_drawing_bounds_safe(self) -> Optional[Dict[str, float]]:
        """Safely get drawing bounds by reading entities (READ-ONLY)."""
        try:
            min_x = float('inf')
            max_x = float('-inf')
            min_y = float('inf')
            max_y = float('-inf')
            
            entity_count = 0
            
            # READ-ONLY: Just get coordinates, don't modify anything
            for entity in self.cad.model:
                entity_count += 1
                
                try:
                    if hasattr(entity, 'StartPoint') and hasattr(entity, 'EndPoint'):
                        # Lines
                        start = entity.StartPoint
                        end = entity.EndPoint
                        min_x = min(min_x, start[0], end[0])
                        max_x = max(max_x, start[0], end[0])
                        min_y = min(min_y, start[1], end[1])
                        max_y = max(max_y, start[1], end[1])
                        
                    elif hasattr(entity, 'Center') and hasattr(entity, 'Radius'):
                        # Circles/Arcs
                        center = entity.Center
                        radius = entity.Radius
                        min_x = min(min_x, center[0] - radius)
                        max_x = max(max_x, center[0] + radius)
                        min_y = min(min_y, center[1] - radius)
                        max_y = max(max_y, center[1] + radius)
                        
                    elif hasattr(entity, 'InsertionPoint'):
                        # Blocks
                        point = entity.InsertionPoint
                        min_x = min(min_x, point[0])
                        max_x = max(max_x, point[0])
                        min_y = min(min_y, point[1])
                        max_y = max(max_y, point[1])
                        
                except Exception:
                    # Skip entities that can't be read
                    continue
            
            if entity_count == 0 or min_x == float('inf'):
                logger.error("No entities found or could not determine bounds")
                return None
                
            bounds = {
                'min_x': min_x,
                'max_x': max_x,
                'min_y': min_y,
                'max_y': max_y
            }
            
            logger.info(f"Found {entity_count} entities, bounds: {bounds}")
            return bounds
            
        except Exception as e:
            logger.error(f"Error getting drawing bounds: {e}")
            return None
    
    def _add_total_length_dimension(self, side: str, bounds: Dict[str, float], offset: float) -> int:
        """Add total length dimension for one side (Layer 1)."""
        try:
            if side == 'bottom':
                # Horizontal dimension at bottom
                start_pt = APoint(bounds['min_x'], bounds['min_y'] - offset)
                end_pt = APoint(bounds['max_x'], bounds['min_y'] - offset)
                dim_loc = APoint((bounds['min_x'] + bounds['max_x']) / 2, bounds['min_y'] - offset)
                dim = self.cad.model.AddDimRotated(start_pt, end_pt, dim_loc, 0)
                
            elif side == 'top':
                # Horizontal dimension at top
                start_pt = APoint(bounds['min_x'], bounds['max_y'] + offset)
                end_pt = APoint(bounds['max_x'], bounds['max_y'] + offset)
                dim_loc = APoint((bounds['min_x'] + bounds['max_x']) / 2, bounds['max_y'] + offset)
                dim = self.cad.model.AddDimRotated(start_pt, end_pt, dim_loc, 0)
                
            elif side == 'left':
                # Vertical dimension at left
                start_pt = APoint(bounds['min_x'] - offset, bounds['min_y'])
                end_pt = APoint(bounds['min_x'] - offset, bounds['max_y'])
                dim_loc = APoint(bounds['min_x'] - offset, (bounds['min_y'] + bounds['max_y']) / 2)
                dim = self.cad.model.AddDimRotated(start_pt, end_pt, dim_loc, 1.5708)  # 90 degrees
                
            else:  # right
                # Vertical dimension at right
                start_pt = APoint(bounds['max_x'] + offset, bounds['min_y'])
                end_pt = APoint(bounds['max_x'] + offset, bounds['max_y'])
                dim_loc = APoint(bounds['max_x'] + offset, (bounds['min_y'] + bounds['max_y']) / 2)
                dim = self.cad.model.AddDimRotated(start_pt, end_pt, dim_loc, 1.5708)  # 90 degrees
            
            # Format dimension safely
            self._format_dimension(dim)
            logger.info(f"Added total length dimension for {side} side")
            return 1
            
        except Exception as e:
            logger.error(f"Error adding total length dimension for {side}: {e}")
            return 0
    
    def _add_major_segment_dimensions(self, side: str, bounds: Dict[str, float], offset: float, 
                                    layer_filter: Optional[str], excluded_layers: Optional[List[str]]) -> int:
        """Add major segment dimensions for one side (Layer 2)."""
        try:
            # Get major segments by analyzing existing lines (READ-ONLY)
            major_points = self._get_major_segment_points(side, bounds, layer_filter, excluded_layers)
            
            if len(major_points) < 2:
                logger.info(f"No major segments found for {side} side")
                return 0
                
            dimensions_added = 0
            
            # Create dimensions between consecutive major points
            for i in range(len(major_points) - 1):
                start_coord = major_points[i]
                end_coord = major_points[i + 1]
                
                if side in ['bottom', 'top']:
                    # Horizontal segments
                    y_pos = bounds['min_y'] - offset if side == 'bottom' else bounds['max_y'] + offset
                    start_pt = APoint(start_coord, y_pos)
                    end_pt = APoint(end_coord, y_pos)
                    dim_loc = APoint((start_coord + end_coord) / 2, y_pos)
                    dim = self.cad.model.AddDimRotated(start_pt, end_pt, dim_loc, 0)
                else:
                    # Vertical segments
                    x_pos = bounds['min_x'] - offset if side == 'left' else bounds['max_x'] + offset
                    start_pt = APoint(x_pos, start_coord)
                    end_pt = APoint(x_pos, end_coord)
                    dim_loc = APoint(x_pos, (start_coord + end_coord) / 2)
                    dim = self.cad.model.AddDimRotated(start_pt, end_pt, dim_loc, 1.5708)
                
                self._format_dimension(dim)
                dimensions_added += 1
                
            logger.info(f"Added {dimensions_added} major segment dimensions for {side} side")
            return dimensions_added
            
        except Exception as e:
            logger.error(f"Error adding major segment dimensions for {side}: {e}")
            return 0
    
    def _add_detailed_segment_dimensions(self, side: str, bounds: Dict[str, float], offset: float,
                                       layer_filter: Optional[str], excluded_layers: Optional[List[str]]) -> int:
        """Add detailed segment dimensions for one side (Layer 3)."""
        try:
            # Get detailed segments by analyzing existing lines (READ-ONLY)
            detail_points = self._get_detailed_segment_points(side, bounds, layer_filter, excluded_layers)
            
            if len(detail_points) < 2:
                logger.info(f"No detailed segments found for {side} side")
                return 0
                
            dimensions_added = 0
            
            # Create dimensions between consecutive detail points
            for i in range(len(detail_points) - 1):
                start_coord = detail_points[i]
                end_coord = detail_points[i + 1]
                
                # Skip very small segments
                if abs(end_coord - start_coord) < self.config['min_length']:
                    continue
                
                if side in ['bottom', 'top']:
                    # Horizontal segments
                    y_pos = bounds['min_y'] - offset if side == 'bottom' else bounds['max_y'] + offset
                    start_pt = APoint(start_coord, y_pos)
                    end_pt = APoint(end_coord, y_pos)
                    dim_loc = APoint((start_coord + end_coord) / 2, y_pos)
                    dim = self.cad.model.AddDimRotated(start_pt, end_pt, dim_loc, 0)
                else:
                    # Vertical segments
                    x_pos = bounds['min_x'] - offset if side == 'left' else bounds['max_x'] + offset
                    start_pt = APoint(x_pos, start_coord)
                    end_pt = APoint(x_pos, end_coord)
                    dim_loc = APoint(x_pos, (start_coord + end_coord) / 2)
                    dim = self.cad.model.AddDimRotated(start_pt, end_pt, dim_loc, 1.5708)
                
                self._format_dimension(dim)
                dimensions_added += 1
                
            logger.info(f"Added {dimensions_added} detailed segment dimensions for {side} side")
            return dimensions_added
            
        except Exception as e:
            logger.error(f"Error adding detailed segment dimensions for {side}: {e}")
            return 0
    
    def _get_major_segment_points(self, side: str, bounds: Dict[str, float], 
                                layer_filter: Optional[str], excluded_layers: Optional[List[str]]) -> List[float]:
        """Get major segment points for a side (READ-ONLY analysis)."""
        try:
            points = set()
            
            # Always include the boundary points
            if side in ['bottom', 'top']:
                points.add(bounds['min_x'])
                points.add(bounds['max_x'])
            else:
                points.add(bounds['min_y'])
                points.add(bounds['max_y'])
            
            # Analyze existing entities to find major segment points
            for entity in self.cad.model:
                try:
                    # Apply layer filtering
                    if layer_filter and hasattr(entity, 'Layer') and entity.Layer != layer_filter:
                        continue
                    if excluded_layers and hasattr(entity, 'Layer') and entity.Layer in excluded_layers:
                        continue
                        
                    # Get major intersection points
                    if hasattr(entity, 'StartPoint') and hasattr(entity, 'EndPoint'):
                        start = entity.StartPoint
                        end = entity.EndPoint
                        
                        if side in ['bottom', 'top']:
                            # For horizontal sides, collect x-coordinates of vertical or significant lines
                            if abs(end[0] - start[0]) > self.config['min_length']:  # Horizontal line
                                points.add(start[0])
                                points.add(end[0])
                        else:
                            # For vertical sides, collect y-coordinates of horizontal or significant lines
                            if abs(end[1] - start[1]) > self.config['min_length']:  # Vertical line
                                points.add(start[1])
                                points.add(end[1])
                                
                except Exception:
                    continue
            
            # Convert to sorted list
            return sorted(list(points))
            
        except Exception as e:
            logger.error(f"Error getting major segment points for {side}: {e}")
            return []
    
    def _get_detailed_segment_points(self, side: str, bounds: Dict[str, float],
                                   layer_filter: Optional[str], excluded_layers: Optional[List[str]]) -> List[float]:
        """Get detailed segment points for a side (READ-ONLY analysis)."""
        try:
            points = set()
            
            # Always include the boundary points
            if side in ['bottom', 'top']:
                points.add(bounds['min_x'])
                points.add(bounds['max_x'])
            else:
                points.add(bounds['min_y'])
                points.add(bounds['max_y'])
            
            # Analyze existing entities to find detailed segment points
            for entity in self.cad.model:
                try:
                    # Apply layer filtering
                    if layer_filter and hasattr(entity, 'Layer') and entity.Layer != layer_filter:
                        continue
                    if excluded_layers and hasattr(entity, 'Layer') and entity.Layer in excluded_layers:
                        continue
                        
                    # Get all intersection points including small details
                    if hasattr(entity, 'StartPoint') and hasattr(entity, 'EndPoint'):
                        start = entity.StartPoint
                        end = entity.EndPoint
                        
                        if side in ['bottom', 'top']:
                            # For horizontal sides, collect all x-coordinates
                            points.add(start[0])
                            points.add(end[0])
                        else:
                            # For vertical sides, collect all y-coordinates
                            points.add(start[1])
                            points.add(end[1])
                            
                    elif hasattr(entity, 'InsertionPoint'):
                        # Include block insertion points (doors, windows, etc.)
                        point = entity.InsertionPoint
                        if side in ['bottom', 'top']:
                            points.add(point[0])
                        else:
                            points.add(point[1])
                            
                except Exception:
                    continue
            
            # Convert to sorted list
            return sorted(list(points))
            
        except Exception as e:
            logger.error(f"Error getting detailed segment points for {side}: {e}")
            return []