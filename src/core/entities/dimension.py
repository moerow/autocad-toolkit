"""Domain entities for dimensions."""
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional, Tuple, List
from uuid import uuid4
import math


class DimensionType(Enum):
    """Types of dimensions."""
    LINEAR = "linear"
    ALIGNED = "aligned"
    ANGULAR = "angular"
    RADIAL = "radial"
    DIAMETRIC = "diametric"
    ORDINATE = "ordinate"
    BASELINE = "baseline"
    CONTINUE = "continue"
    LEADER = "leader"
    TOLERANCE = "tolerance"


class DimensionStyle(Enum):
    """Dimension styles."""
    STANDARD = "Standard"
    ARCHITECTURAL = "Architectural"
    ENGINEERING = "Engineering"
    METRIC = "Metric"
    CUSTOM = "Custom"


class DimensionPlacement(Enum):
    """Dimension placement options."""
    ABOVE = "above"
    BELOW = "below"
    LEFT = "left"
    RIGHT = "right"
    CENTER = "center"
    AUTO = "auto"


class DimensionUnits(Enum):
    """Dimension units."""
    MILLIMETERS = "mm"
    CENTIMETERS = "cm"
    METERS = "m"
    INCHES = "in"
    FEET = "ft"
    UNITLESS = ""


@dataclass
class DimensionStyle:
    """Dimension style configuration."""
    name: str
    text_height: float = 100.0
    arrow_size: float = 100.0
    extension_line_offset: float = 50.0
    extension_line_extension: float = 50.0
    dimension_line_offset: float = 500.0
    text_gap: float = 50.0
    text_color: int = 3
    line_color: int = 3
    arrow_color: int = 3
    precision: int = 2
    units: DimensionUnits = DimensionUnits.MILLIMETERS
    suppress_leading_zeros: bool = False
    suppress_trailing_zeros: bool = True
    text_style: str = "Standard"
    arrow_type: str = "ClosedFilled"
    text_placement: DimensionPlacement = DimensionPlacement.AUTO
    text_rotation: float = 0.0
    scale_factor: float = 1.0
    tolerances_enabled: bool = False
    upper_tolerance: float = 0.0
    lower_tolerance: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DimensionPoint:
    """Represents a point in 3D space for dimensions."""
    x: float
    y: float
    z: float = 0.0

    def distance_to(self, other: 'DimensionPoint') -> float:
        """Calculate distance to another point."""
        return math.sqrt(
            (self.x - other.x) ** 2 +
            (self.y - other.y) ** 2 +
            (self.z - other.z) ** 2
        )

    def midpoint_to(self, other: 'DimensionPoint') -> 'DimensionPoint':
        """Get midpoint to another point."""
        return DimensionPoint(
            (self.x + other.x) / 2,
            (self.y + other.y) / 2,
            (self.z + other.z) / 2
        )

    def to_tuple(self) -> Tuple[float, float, float]:
        """Convert to tuple."""
        return (self.x, self.y, self.z)

    @classmethod
    def from_tuple(cls, point: Tuple[float, float, float]) -> 'DimensionPoint':
        """Create from tuple."""
        return cls(point[0], point[1], point[2] if len(point) > 2 else 0.0)


@dataclass
class DimensionConstraints:
    """Constraints for dimension placement."""
    min_distance: float = 100.0
    max_distance: float = 10000.0
    avoid_overlaps: bool = True
    preferred_side: Optional[str] = None
    collision_buffer: float = 50.0
    angle_tolerance: float = 0.017453  # 1 degree in radians
    parallel_tolerance: float = 0.001
    perpendicular_tolerance: float = 0.001


@dataclass
class Dimension:
    """Represents a dimension in AutoCAD."""
    id: str = field(default_factory=lambda: str(uuid4()))
    dimension_type: DimensionType = DimensionType.LINEAR
    start_point: DimensionPoint = field(default_factory=lambda: DimensionPoint(0, 0, 0))
    end_point: DimensionPoint = field(default_factory=lambda: DimensionPoint(0, 0, 0))
    text_position: DimensionPoint = field(default_factory=lambda: DimensionPoint(0, 0, 0))
    measurement: float = 0.0
    layer: str = "DIMENSIONS"
    style: DimensionStyle = field(default_factory=lambda: DimensionStyle("Standard"))
    constraints: DimensionConstraints = field(default_factory=DimensionConstraints)
    autocad_entity: Optional[Any] = None
    is_associative: bool = True
    is_locked: bool = False
    custom_text: Optional[str] = None
    prefix: str = ""
    suffix: str = ""
    text_override: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.measurement == 0.0:
            self.measurement = self.calculate_measurement()

    def calculate_measurement(self) -> float:
        """Calculate the measurement based on dimension type."""
        if self.dimension_type == DimensionType.LINEAR or self.dimension_type == DimensionType.ALIGNED:
            return self.start_point.distance_to(self.end_point)
        elif self.dimension_type == DimensionType.RADIAL:
            return self.start_point.distance_to(self.end_point)
        elif self.dimension_type == DimensionType.DIAMETRIC:
            return self.start_point.distance_to(self.end_point)
        elif self.dimension_type == DimensionType.ANGULAR:
            return self.calculate_angle()
        else:
            return 0.0

    def calculate_angle(self) -> float:
        """Calculate angle for angular dimensions."""
        dx = self.end_point.x - self.start_point.x
        dy = self.end_point.y - self.start_point.y
        return math.atan2(dy, dx)

    def get_display_text(self) -> str:
        """Get the dimension display text."""
        if self.text_override:
            return self.text_override
        
        if self.custom_text:
            return self.custom_text
        
        if self.dimension_type == DimensionType.ANGULAR:
            angle_degrees = math.degrees(self.measurement)
            return f"{self.prefix}{angle_degrees:.{self.style.precision}f}°{self.suffix}"
        else:
            unit_symbol = self.style.units.value
            return f"{self.prefix}{self.measurement:.{self.style.precision}f}{unit_symbol}{self.suffix}"

    def is_valid(self) -> bool:
        """Check if dimension is valid."""
        if self.measurement < self.constraints.min_distance:
            return False
        if self.measurement > self.constraints.max_distance:
            return False
        if self.start_point.distance_to(self.end_point) < 0.001:
            return False
        return True

    def update_measurement(self) -> None:
        """Update measurement based on current points."""
        self.measurement = self.calculate_measurement()

    def move_text_position(self, new_position: DimensionPoint) -> None:
        """Move text position."""
        self.text_position = new_position

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'type': self.dimension_type.value,
            'start_point': self.start_point.to_tuple(),
            'end_point': self.end_point.to_tuple(),
            'text_position': self.text_position.to_tuple(),
            'measurement': self.measurement,
            'layer': self.layer,
            'display_text': self.get_display_text(),
            'is_valid': self.is_valid(),
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Dimension':
        """Create from dictionary."""
        return cls(
            id=data.get('id', str(uuid4())),
            dimension_type=DimensionType(data.get('type', 'linear')),
            start_point=DimensionPoint.from_tuple(data.get('start_point', (0, 0, 0))),
            end_point=DimensionPoint.from_tuple(data.get('end_point', (0, 0, 0))),
            text_position=DimensionPoint.from_tuple(data.get('text_position', (0, 0, 0))),
            measurement=data.get('measurement', 0.0),
            layer=data.get('layer', 'DIMENSIONS'),
            metadata=data.get('metadata', {})
        )


@dataclass
class DimensionChain:
    """Represents a chain of continuous dimensions."""
    id: str = field(default_factory=lambda: str(uuid4()))
    dimensions: List[Dimension] = field(default_factory=list)
    base_dimension: Optional[Dimension] = None
    chain_type: str = "continue"  # continue, baseline, or ordinate
    layer: str = "DIMENSIONS"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_dimension(self, dimension: Dimension) -> None:
        """Add dimension to chain."""
        self.dimensions.append(dimension)
        if not self.base_dimension:
            self.base_dimension = dimension

    def get_total_length(self) -> float:
        """Get total length of all dimensions in chain."""
        return sum(dim.measurement for dim in self.dimensions)

    def get_chain_bounds(self) -> Tuple[DimensionPoint, DimensionPoint]:
        """Get bounding box of the chain."""
        if not self.dimensions:
            return DimensionPoint(0, 0, 0), DimensionPoint(0, 0, 0)
        
        all_points = []
        for dim in self.dimensions:
            all_points.extend([dim.start_point, dim.end_point, dim.text_position])
        
        min_x = min(p.x for p in all_points)
        max_x = max(p.x for p in all_points)
        min_y = min(p.y for p in all_points)
        max_y = max(p.y for p in all_points)
        min_z = min(p.z for p in all_points)
        max_z = max(p.z for p in all_points)
        
        return DimensionPoint(min_x, min_y, min_z), DimensionPoint(max_x, max_y, max_z)

    def validate_chain(self) -> bool:
        """Validate that dimensions form a valid chain."""
        if len(self.dimensions) < 2:
            return False
        
        for i in range(len(self.dimensions) - 1):
            current = self.dimensions[i]
            next_dim = self.dimensions[i + 1]
            
            # Check if dimensions are connected
            if current.end_point.distance_to(next_dim.start_point) > 0.001:
                return False
        
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'chain_type': self.chain_type,
            'layer': self.layer,
            'total_length': self.get_total_length(),
            'dimension_count': len(self.dimensions),
            'dimensions': [dim.to_dict() for dim in self.dimensions],
            'metadata': self.metadata
        }


@dataclass
class DimensionGroup:
    """Represents a group of related dimensions."""
    id: str = field(default_factory=lambda: str(uuid4()))
    name: str = "Dimension Group"
    dimensions: List[Dimension] = field(default_factory=list)
    chains: List[DimensionChain] = field(default_factory=list)
    layer: str = "DIMENSIONS"
    is_locked: bool = False
    created_at: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_dimension(self, dimension: Dimension) -> None:
        """Add dimension to group."""
        self.dimensions.append(dimension)

    def add_chain(self, chain: DimensionChain) -> None:
        """Add dimension chain to group."""
        self.chains.append(chain)

    def get_all_dimensions(self) -> List[Dimension]:
        """Get all dimensions including those in chains."""
        all_dims = list(self.dimensions)
        for chain in self.chains:
            all_dims.extend(chain.dimensions)
        return all_dims

    def get_bounds(self) -> Tuple[DimensionPoint, DimensionPoint]:
        """Get bounding box of all dimensions in group."""
        all_dims = self.get_all_dimensions()
        if not all_dims:
            return DimensionPoint(0, 0, 0), DimensionPoint(0, 0, 0)
        
        all_points = []
        for dim in all_dims:
            all_points.extend([dim.start_point, dim.end_point, dim.text_position])
        
        min_x = min(p.x for p in all_points)
        max_x = max(p.x for p in all_points)
        min_y = min(p.y for p in all_points)
        max_y = max(p.y for p in all_points)
        min_z = min(p.z for p in all_points)
        max_z = max(p.z for p in all_points)
        
        return DimensionPoint(min_x, min_y, min_z), DimensionPoint(max_x, max_y, max_z)

    def clear_all(self) -> None:
        """Clear all dimensions and chains."""
        self.dimensions.clear()
        self.chains.clear()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'name': self.name,
            'layer': self.layer,
            'is_locked': self.is_locked,
            'dimension_count': len(self.get_all_dimensions()),
            'chain_count': len(self.chains),
            'dimensions': [dim.to_dict() for dim in self.dimensions],
            'chains': [chain.to_dict() for chain in self.chains],
            'metadata': self.metadata
        }