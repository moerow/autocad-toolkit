"""Comprehensive interfaces for AutoCAD operations."""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple, Union
from enum import Enum


class EntityType(Enum):
    """Supported AutoCAD entity types."""
    LINE = "Line"
    POLYLINE = "Polyline"
    LWPOLYLINE = "LWPolyline"
    CIRCLE = "Circle"
    ARC = "Arc"
    ELLIPSE = "Ellipse"
    SPLINE = "Spline"
    TEXT = "Text"
    MTEXT = "MText"
    BLOCK = "Block"
    INSERT = "Insert"
    DIMENSION = "Dimension"
    HATCH = "Hatch"
    LEADER = "Leader"
    VIEWPORT = "Viewport"


class DrawingUnits(Enum):
    """Drawing units."""
    UNITLESS = 0
    INCHES = 1
    FEET = 2
    MILES = 3
    MILLIMETERS = 4
    CENTIMETERS = 5
    METERS = 6
    KILOMETERS = 7
    MICROINCHES = 8
    MILS = 9
    YARDS = 10
    ANGSTROMS = 11
    NANOMETERS = 12
    MICRONS = 13
    DECIMETERS = 14
    DECAMETERS = 15
    HECTOMETERS = 16
    GIGAMETERS = 17
    ASTRONOMICAL_UNITS = 18
    LIGHT_YEARS = 19
    PARSECS = 20


class IAutoCADConnection(ABC):
    """Interface for AutoCAD connection management."""

    @abstractmethod
    def connect(self, create_if_not_exists: bool = True) -> bool:
        """Establish connection to AutoCAD."""
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from AutoCAD."""
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """Check if connected to AutoCAD."""
        pass

    @abstractmethod
    def get_autocad_version(self) -> str:
        """Get AutoCAD version information."""
        pass

    @abstractmethod
    def get_active_document(self) -> 'IAutoCADDocument':
        """Get the active document."""
        pass

    @abstractmethod
    def create_new_document(self, template_path: Optional[str] = None) -> 'IAutoCADDocument':
        """Create a new document."""
        pass

    @abstractmethod
    def open_document(self, file_path: str) -> 'IAutoCADDocument':
        """Open an existing document."""
        pass

    @abstractmethod
    def save_document(self, file_path: Optional[str] = None) -> bool:
        """Save the active document."""
        pass

    @abstractmethod
    def close_document(self, save_changes: bool = True) -> bool:
        """Close the active document."""
        pass


class IAutoCADDocument(ABC):
    """Interface for AutoCAD document operations."""

    @abstractmethod
    def get_name(self) -> str:
        """Get document name."""
        pass

    @abstractmethod
    def get_path(self) -> str:
        """Get document path."""
        pass

    @abstractmethod
    def get_units(self) -> DrawingUnits:
        """Get drawing units."""
        pass

    @abstractmethod
    def set_units(self, units: DrawingUnits) -> bool:
        """Set drawing units."""
        pass

    @abstractmethod
    def get_limits(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Get drawing limits (min_point, max_point)."""
        pass

    @abstractmethod
    def set_limits(self, min_point: Tuple[float, float], max_point: Tuple[float, float]) -> bool:
        """Set drawing limits."""
        pass

    @abstractmethod
    def zoom_extents(self) -> bool:
        """Zoom to extents."""
        pass

    @abstractmethod
    def zoom_window(self, corner1: Tuple[float, float], corner2: Tuple[float, float]) -> bool:
        """Zoom to window."""
        pass

    @abstractmethod
    def regenerate(self) -> bool:
        """Regenerate the drawing."""
        pass

    @abstractmethod
    def get_modelspace(self) -> 'IAutoCADModelSpace':
        """Get model space."""
        pass

    @abstractmethod
    def get_paperspace(self) -> 'IAutoCADPaperSpace':
        """Get paper space."""
        pass

    @abstractmethod
    def get_layer_manager(self) -> 'IAutoCADLayerManager':
        """Get layer manager."""
        pass

    @abstractmethod
    def get_block_manager(self) -> 'IAutoCADBlockManager':
        """Get block manager."""
        pass

    @abstractmethod
    def get_text_style_manager(self) -> 'IAutoCADTextStyleManager':
        """Get text style manager."""
        pass

    @abstractmethod
    def get_dimension_style_manager(self) -> 'IAutoCADDimensionStyleManager':
        """Get dimension style manager."""
        pass


class IAutoCADModelSpace(ABC):
    """Interface for AutoCAD model space operations."""

    @abstractmethod
    def get_all_entities(self, entity_type: Optional[EntityType] = None, 
                        layer_filter: Optional[str] = None) -> List[Any]:
        """Get all entities in model space."""
        pass

    @abstractmethod
    def add_line(self, start_point: Tuple[float, float, float], 
                end_point: Tuple[float, float, float], layer: Optional[str] = None) -> Any:
        """Add a line entity."""
        pass

    @abstractmethod
    def add_polyline(self, points: List[Tuple[float, float]], 
                    closed: bool = False, layer: Optional[str] = None) -> Any:
        """Add a polyline entity."""
        pass

    @abstractmethod
    def add_circle(self, center: Tuple[float, float, float], 
                  radius: float, layer: Optional[str] = None) -> Any:
        """Add a circle entity."""
        pass

    @abstractmethod
    def add_arc(self, center: Tuple[float, float, float], radius: float, 
               start_angle: float, end_angle: float, layer: Optional[str] = None) -> Any:
        """Add an arc entity."""
        pass

    @abstractmethod
    def add_text(self, text: str, insertion_point: Tuple[float, float, float], 
                height: float, layer: Optional[str] = None) -> Any:
        """Add a text entity."""
        pass

    @abstractmethod
    def add_mtext(self, text: str, insertion_point: Tuple[float, float, float], 
                 width: float, height: float, layer: Optional[str] = None) -> Any:
        """Add an mtext entity."""
        pass

    @abstractmethod
    def add_dimension_aligned(self, ext_line1_point: Tuple[float, float, float],
                            ext_line2_point: Tuple[float, float, float],
                            dim_line_point: Tuple[float, float, float],
                            layer: Optional[str] = None) -> Any:
        """Add an aligned dimension."""
        pass

    @abstractmethod
    def add_dimension_linear(self, ext_line1_point: Tuple[float, float, float],
                           ext_line2_point: Tuple[float, float, float],
                           dim_line_point: Tuple[float, float, float],
                           angle: float = 0.0,
                           layer: Optional[str] = None) -> Any:
        """Add a linear dimension."""
        pass

    @abstractmethod
    def add_dimension_radial(self, center: Tuple[float, float, float],
                           chord_point: Tuple[float, float, float],
                           leader_length: float,
                           layer: Optional[str] = None) -> Any:
        """Add a radial dimension."""
        pass

    @abstractmethod
    def add_dimension_diametric(self, chord_point1: Tuple[float, float, float],
                              chord_point2: Tuple[float, float, float],
                              leader_length: float,
                              layer: Optional[str] = None) -> Any:
        """Add a diametric dimension."""
        pass

    @abstractmethod
    def add_block_reference(self, block_name: str, 
                          insertion_point: Tuple[float, float, float],
                          scale_factor: float = 1.0,
                          rotation_angle: float = 0.0,
                          layer: Optional[str] = None) -> Any:
        """Add a block reference."""
        pass

    @abstractmethod
    def add_hatch(self, boundary_entities: List[Any], 
                 pattern_name: str = "SOLID",
                 layer: Optional[str] = None) -> Any:
        """Add a hatch entity."""
        pass

    @abstractmethod
    def delete_entity(self, entity: Any) -> bool:
        """Delete an entity."""
        pass

    @abstractmethod
    def select_entities(self, point1: Optional[Tuple[float, float]] = None,
                       point2: Optional[Tuple[float, float]] = None,
                       selection_mode: str = "window") -> List[Any]:
        """Select entities by window/crossing."""
        pass

    @abstractmethod
    def get_entity_properties(self, entity: Any) -> Dict[str, Any]:
        """Get entity properties."""
        pass

    @abstractmethod
    def set_entity_properties(self, entity: Any, properties: Dict[str, Any]) -> bool:
        """Set entity properties."""
        pass


class IAutoCADPaperSpace(ABC):
    """Interface for AutoCAD paper space operations."""

    @abstractmethod
    def get_all_entities(self, entity_type: Optional[EntityType] = None) -> List[Any]:
        """Get all entities in paper space."""
        pass

    @abstractmethod
    def add_viewport(self, center: Tuple[float, float], 
                   width: float, height: float) -> Any:
        """Add a viewport."""
        pass

    @abstractmethod
    def set_viewport_scale(self, viewport: Any, scale: float) -> bool:
        """Set viewport scale."""
        pass


class IAutoCADLayerManager(ABC):
    """Interface for AutoCAD layer management."""

    @abstractmethod
    def get_all_layers(self) -> List[str]:
        """Get all layer names."""
        pass

    @abstractmethod
    def create_layer(self, name: str, color: int = 7, 
                    linetype: str = "Continuous") -> bool:
        """Create a new layer."""
        pass

    @abstractmethod
    def delete_layer(self, name: str) -> bool:
        """Delete a layer."""
        pass

    @abstractmethod
    def set_active_layer(self, name: str) -> bool:
        """Set active layer."""
        pass

    @abstractmethod
    def get_active_layer(self) -> str:
        """Get active layer name."""
        pass

    @abstractmethod
    def layer_exists(self, name: str) -> bool:
        """Check if layer exists."""
        pass

    @abstractmethod
    def freeze_layer(self, name: str) -> bool:
        """Freeze a layer."""
        pass

    @abstractmethod
    def thaw_layer(self, name: str) -> bool:
        """Thaw a layer."""
        pass

    @abstractmethod
    def lock_layer(self, name: str) -> bool:
        """Lock a layer."""
        pass

    @abstractmethod
    def unlock_layer(self, name: str) -> bool:
        """Unlock a layer."""
        pass

    @abstractmethod
    def turn_layer_on(self, name: str) -> bool:
        """Turn layer on."""
        pass

    @abstractmethod
    def turn_layer_off(self, name: str) -> bool:
        """Turn layer off."""
        pass

    @abstractmethod
    def set_layer_color(self, name: str, color: int) -> bool:
        """Set layer color."""
        pass

    @abstractmethod
    def set_layer_linetype(self, name: str, linetype: str) -> bool:
        """Set layer linetype."""
        pass

    @abstractmethod
    def set_layer_lineweight(self, name: str, lineweight: float) -> bool:
        """Set layer lineweight."""
        pass

    @abstractmethod
    def get_layer_properties(self, name: str) -> Dict[str, Any]:
        """Get layer properties."""
        pass


class IAutoCADBlockManager(ABC):
    """Interface for AutoCAD block management."""

    @abstractmethod
    def get_all_blocks(self) -> List[str]:
        """Get all block names."""
        pass

    @abstractmethod
    def create_block(self, name: str, base_point: Tuple[float, float, float],
                    entities: List[Any]) -> bool:
        """Create a new block."""
        pass

    @abstractmethod
    def delete_block(self, name: str) -> bool:
        """Delete a block."""
        pass

    @abstractmethod
    def block_exists(self, name: str) -> bool:
        """Check if block exists."""
        pass

    @abstractmethod
    def explode_block(self, block_reference: Any) -> List[Any]:
        """Explode a block reference."""
        pass

    @abstractmethod
    def get_block_references(self, block_name: str) -> List[Any]:
        """Get all references to a block."""
        pass


class IAutoCADTextStyleManager(ABC):
    """Interface for AutoCAD text style management."""

    @abstractmethod
    def get_all_text_styles(self) -> List[str]:
        """Get all text style names."""
        pass

    @abstractmethod
    def create_text_style(self, name: str, font_name: str, height: float = 0.0) -> bool:
        """Create a new text style."""
        pass

    @abstractmethod
    def delete_text_style(self, name: str) -> bool:
        """Delete a text style."""
        pass

    @abstractmethod
    def set_active_text_style(self, name: str) -> bool:
        """Set active text style."""
        pass

    @abstractmethod
    def get_active_text_style(self) -> str:
        """Get active text style name."""
        pass


class IAutoCADDimensionStyleManager(ABC):
    """Interface for AutoCAD dimension style management."""

    @abstractmethod
    def get_all_dimension_styles(self) -> List[str]:
        """Get all dimension style names."""
        pass

    @abstractmethod
    def create_dimension_style(self, name: str, properties: Dict[str, Any]) -> bool:
        """Create a new dimension style."""
        pass

    @abstractmethod
    def delete_dimension_style(self, name: str) -> bool:
        """Delete a dimension style."""
        pass

    @abstractmethod
    def set_active_dimension_style(self, name: str) -> bool:
        """Set active dimension style."""
        pass

    @abstractmethod
    def get_active_dimension_style(self) -> str:
        """Get active dimension style name."""
        pass

    @abstractmethod
    def get_dimension_style_properties(self, name: str) -> Dict[str, Any]:
        """Get dimension style properties."""
        pass

    @abstractmethod
    def set_dimension_style_properties(self, name: str, properties: Dict[str, Any]) -> bool:
        """Set dimension style properties."""
        pass