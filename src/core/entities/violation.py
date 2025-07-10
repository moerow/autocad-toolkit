"""Domain entities for compliance violations."""
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from uuid import uuid4

from .rule import ComplianceRule, RuleSeverity
from .geometry import Point3D, GeometryEntity


class ViolationStatus(Enum):
    """Status of a compliance violation."""
    DETECTED = "detected"
    CONFIRMED = "confirmed"
    RESOLVED = "resolved"
    IGNORED = "ignored"
    REVIEWING = "reviewing"
    ESCALATED = "escalated"


class ViolationPriority(Enum):
    """Priority levels for violations."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class VisualizationStyle(Enum):
    """Visualization styles for violations."""
    CIRCLE = "circle"
    RECTANGLE = "rectangle"
    HIGHLIGHT = "highlight"
    ARROW = "arrow"
    TEXT = "text"
    HATCH = "hatch"
    DIMENSION = "dimension"


@dataclass
class ViolationLocation:
    """Represents the location of a violation."""
    point: Point3D = field(default_factory=lambda: Point3D(0, 0, 0))
    entity_id: Optional[str] = None
    entity_type: Optional[str] = None
    layer: Optional[str] = None
    bounding_box: Optional[Tuple[Point3D, Point3D]] = None
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'point': self.point.to_tuple(),
            'entity_id': self.entity_id,
            'entity_type': self.entity_type,
            'layer': self.layer,
            'bounding_box': [bb.to_tuple() for bb in self.bounding_box] if self.bounding_box else None,
            'description': self.description
        }


@dataclass
class ViolationEvidence:
    """Evidence supporting a violation."""
    id: str = field(default_factory=lambda: str(uuid4()))
    type: str = ""  # measurement, calculation, visual, reference
    description: str = ""
    value: Any = None
    expected_value: Any = None
    tolerance: float = 0.0
    measurement_method: str = ""
    screenshot_path: Optional[str] = None
    drawing_markup: Optional[str] = None  # Path to marked up drawing
    reference_documents: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'type': self.type,
            'description': self.description,
            'value': self.value,
            'expected_value': self.expected_value,
            'tolerance': self.tolerance,
            'measurement_method': self.measurement_method,
            'screenshot_path': self.screenshot_path,
            'drawing_markup': self.drawing_markup,
            'reference_documents': self.reference_documents,
            'created_at': self.created_at.isoformat(),
            'metadata': self.metadata
        }


@dataclass
class ViolationRecommendation:
    """Recommendation for fixing a violation."""
    id: str = field(default_factory=lambda: str(uuid4()))
    title: str = ""
    description: str = ""
    action_type: str = ""  # modify, add, remove, relocate
    priority: ViolationPriority = ViolationPriority.MEDIUM
    estimated_effort: str = ""  # time estimate
    cost_impact: str = ""  # cost estimate
    risk_level: str = ""  # low, medium, high
    steps: List[str] = field(default_factory=list)
    tools_required: List[str] = field(default_factory=list)
    skills_required: List[str] = field(default_factory=list)
    references: List[str] = field(default_factory=list)
    automated_fix: bool = False
    fix_script: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'title': self.title,
            'description': self.description,
            'action_type': self.action_type,
            'priority': self.priority.value,
            'estimated_effort': self.estimated_effort,
            'cost_impact': self.cost_impact,
            'risk_level': self.risk_level,
            'steps': self.steps,
            'tools_required': self.tools_required,
            'skills_required': self.skills_required,
            'references': self.references,
            'automated_fix': self.automated_fix,
            'fix_script': self.fix_script,
            'created_at': self.created_at.isoformat(),
            'metadata': self.metadata
        }


@dataclass
class ViolationVisualization:
    """Visualization settings for a violation."""
    style: VisualizationStyle = VisualizationStyle.CIRCLE
    color: str = "#FF0000"  # Red
    size: float = 100.0
    opacity: float = 0.7
    layer: str = "VIOLATIONS"
    text_height: float = 50.0
    line_width: float = 2.0
    show_text: bool = True
    show_leader: bool = True
    show_dimensions: bool = False
    animation_enabled: bool = False
    blink_interval: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'style': self.style.value,
            'color': self.color,
            'size': self.size,
            'opacity': self.opacity,
            'layer': self.layer,
            'text_height': self.text_height,
            'line_width': self.line_width,
            'show_text': self.show_text,
            'show_leader': self.show_leader,
            'show_dimensions': self.show_dimensions,
            'animation_enabled': self.animation_enabled,
            'blink_interval': self.blink_interval,
            'metadata': self.metadata
        }


@dataclass
class ComplianceViolation:
    """Represents a compliance rule violation."""
    id: str = field(default_factory=lambda: str(uuid4()))
    rule: Optional[ComplianceRule] = None
    rule_id: Optional[str] = None
    
    # Violation details
    title: str = ""
    description: str = ""
    severity: RuleSeverity = RuleSeverity.MEDIUM
    priority: ViolationPriority = ViolationPriority.MEDIUM
    status: ViolationStatus = ViolationStatus.DETECTED
    
    # Entity information
    entity_id: Optional[str] = None
    entity_type: Optional[str] = None
    entity: Optional[GeometryEntity] = None
    
    # Values
    actual_value: Any = None
    expected_value: Any = None
    tolerance: float = 0.0
    deviation: float = 0.0
    deviation_percentage: float = 0.0
    
    # Location and visualization
    location: Optional[ViolationLocation] = None
    visualization: ViolationVisualization = field(default_factory=ViolationVisualization)
    
    # Evidence and recommendations
    evidence: List[ViolationEvidence] = field(default_factory=list)
    recommendations: List[ViolationRecommendation] = field(default_factory=list)
    
    # Workflow
    assigned_to: Optional[str] = None
    due_date: Optional[datetime] = None
    estimated_fix_time: Optional[str] = None
    
    # Tracking
    detected_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    resolved_at: Optional[datetime] = None
    detected_by: str = "system"
    resolved_by: Optional[str] = None
    
    # Comments and notes
    comments: List[str] = field(default_factory=list)
    internal_notes: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    
    # References
    related_violations: List[str] = field(default_factory=list)
    parent_violation: Optional[str] = None
    child_violations: List[str] = field(default_factory=list)
    
    # Context
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.rule:
            self.rule_id = self.rule.id
            if not self.title:
                self.title = f"Violation: {self.rule.name}"
            if not self.description:
                self.description = self.rule.description
            self.severity = self.rule.severity
            self.expected_value = self.rule.value
        
        self.calculate_deviation()
        self.update_priority()

    def calculate_deviation(self) -> None:
        """Calculate deviation from expected value."""
        if self.expected_value is None or self.actual_value is None:
            return
        
        try:
            expected = float(self.expected_value)
            actual = float(self.actual_value)
            self.deviation = abs(actual - expected)
            
            if expected != 0:
                self.deviation_percentage = (self.deviation / expected) * 100
            else:
                self.deviation_percentage = 0.0
                
        except (ValueError, TypeError):
            self.deviation = 0.0
            self.deviation_percentage = 0.0

    def update_priority(self) -> None:
        """Update priority based on severity and deviation."""
        if self.severity == RuleSeverity.CRITICAL:
            self.priority = ViolationPriority.CRITICAL
        elif self.severity == RuleSeverity.HIGH:
            self.priority = ViolationPriority.HIGH
        elif self.deviation_percentage > 50:
            self.priority = ViolationPriority.HIGH
        elif self.deviation_percentage > 20:
            self.priority = ViolationPriority.MEDIUM
        else:
            self.priority = ViolationPriority.LOW

    def add_evidence(self, evidence: ViolationEvidence) -> None:
        """Add evidence to violation."""
        self.evidence.append(evidence)
        self.last_updated = datetime.now()

    def add_recommendation(self, recommendation: ViolationRecommendation) -> None:
        """Add recommendation to violation."""
        self.recommendations.append(recommendation)
        self.last_updated = datetime.now()

    def add_comment(self, comment: str, is_internal: bool = False) -> None:
        """Add comment to violation."""
        if is_internal:
            self.internal_notes.append(f"[{datetime.now().isoformat()}] {comment}")
        else:
            self.comments.append(f"[{datetime.now().isoformat()}] {comment}")
        self.last_updated = datetime.now()

    def resolve(self, resolved_by: str, notes: str = "") -> None:
        """Mark violation as resolved."""
        self.status = ViolationStatus.RESOLVED
        self.resolved_at = datetime.now()
        self.resolved_by = resolved_by
        self.last_updated = datetime.now()
        
        if notes:
            self.add_comment(f"Resolved: {notes}")

    def ignore(self, reason: str = "") -> None:
        """Mark violation as ignored."""
        self.status = ViolationStatus.IGNORED
        self.last_updated = datetime.now()
        
        if reason:
            self.add_comment(f"Ignored: {reason}")

    def escalate(self, reason: str = "") -> None:
        """Escalate violation."""
        self.status = ViolationStatus.ESCALATED
        self.priority = ViolationPriority.CRITICAL
        self.last_updated = datetime.now()
        
        if reason:
            self.add_comment(f"Escalated: {reason}")

    def assign_to(self, user: str) -> None:
        """Assign violation to user."""
        self.assigned_to = user
        self.last_updated = datetime.now()

    def set_due_date(self, due_date: datetime) -> None:
        """Set due date for resolution."""
        self.due_date = due_date
        self.last_updated = datetime.now()

    def is_overdue(self) -> bool:
        """Check if violation is overdue."""
        if not self.due_date:
            return False
        return datetime.now() > self.due_date and self.status != ViolationStatus.RESOLVED

    def get_age_days(self) -> int:
        """Get age of violation in days."""
        return (datetime.now() - self.detected_at).days

    def get_resolution_time(self) -> Optional[int]:
        """Get resolution time in hours."""
        if not self.resolved_at:
            return None
        return int((self.resolved_at - self.detected_at).total_seconds() / 3600)

    def get_display_text(self) -> str:
        """Get display text for violation."""
        if self.rule:
            return f"{self.rule.name}: {self.actual_value} (expected: {self.expected_value})"
        return self.title or self.description

    def get_severity_color(self) -> str:
        """Get color for severity level."""
        severity_colors = {
            RuleSeverity.CRITICAL: "#FF0000",  # Red
            RuleSeverity.HIGH: "#FF8000",      # Orange
            RuleSeverity.MEDIUM: "#FFFF00",    # Yellow
            RuleSeverity.LOW: "#00FF00",       # Green
            RuleSeverity.INFO: "#0080FF"       # Blue
        }
        return severity_colors.get(self.severity, "#808080")

    def get_status_color(self) -> str:
        """Get color for status."""
        status_colors = {
            ViolationStatus.DETECTED: "#FF0000",     # Red
            ViolationStatus.CONFIRMED: "#FF8000",    # Orange
            ViolationStatus.REVIEWING: "#FFFF00",    # Yellow
            ViolationStatus.RESOLVED: "#00FF00",     # Green
            ViolationStatus.IGNORED: "#808080",      # Gray
            ViolationStatus.ESCALATED: "#FF0080"     # Magenta
        }
        return status_colors.get(self.status, "#808080")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'rule_id': self.rule_id,
            'rule_name': self.rule.name if self.rule else None,
            'title': self.title,
            'description': self.description,
            'severity': self.severity.value,
            'priority': self.priority.value,
            'status': self.status.value,
            'entity_id': self.entity_id,
            'entity_type': self.entity_type,
            'actual_value': self.actual_value,
            'expected_value': self.expected_value,
            'tolerance': self.tolerance,
            'deviation': self.deviation,
            'deviation_percentage': self.deviation_percentage,
            'location': self.location.to_dict() if self.location else None,
            'visualization': self.visualization.to_dict(),
            'evidence': [e.to_dict() for e in self.evidence],
            'recommendations': [r.to_dict() for r in self.recommendations],
            'assigned_to': self.assigned_to,
            'due_date': self.due_date.isoformat() if self.due_date else None,
            'estimated_fix_time': self.estimated_fix_time,
            'detected_at': self.detected_at.isoformat(),
            'last_updated': self.last_updated.isoformat(),
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None,
            'detected_by': self.detected_by,
            'resolved_by': self.resolved_by,
            'comments': self.comments,
            'internal_notes': self.internal_notes,
            'tags': self.tags,
            'related_violations': self.related_violations,
            'parent_violation': self.parent_violation,
            'child_violations': self.child_violations,
            'context': self.context,
            'metadata': self.metadata,
            'display_text': self.get_display_text(),
            'severity_color': self.get_severity_color(),
            'status_color': self.get_status_color(),
            'is_overdue': self.is_overdue(),
            'age_days': self.get_age_days(),
            'resolution_time': self.get_resolution_time()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], rule: Optional[ComplianceRule] = None) -> 'ComplianceViolation':
        """Create from dictionary."""
        location = None
        if data.get('location'):
            loc_data = data['location']
            location = ViolationLocation(
                point=Point3D.from_tuple(loc_data['point']),
                entity_id=loc_data.get('entity_id'),
                entity_type=loc_data.get('entity_type'),
                layer=loc_data.get('layer'),
                description=loc_data.get('description', '')
            )
        
        visualization = ViolationVisualization()
        if data.get('visualization'):
            viz_data = data['visualization']
            visualization = ViolationVisualization(
                style=VisualizationStyle(viz_data.get('style', 'circle')),
                color=viz_data.get('color', '#FF0000'),
                size=viz_data.get('size', 100.0),
                opacity=viz_data.get('opacity', 0.7),
                layer=viz_data.get('layer', 'VIOLATIONS'),
                text_height=viz_data.get('text_height', 50.0),
                line_width=viz_data.get('line_width', 2.0),
                show_text=viz_data.get('show_text', True),
                show_leader=viz_data.get('show_leader', True),
                show_dimensions=viz_data.get('show_dimensions', False),
                animation_enabled=viz_data.get('animation_enabled', False),
                blink_interval=viz_data.get('blink_interval', 1.0)
            )
        
        return cls(
            id=data.get('id', str(uuid4())),
            rule=rule,
            rule_id=data.get('rule_id'),
            title=data.get('title', ''),
            description=data.get('description', ''),
            severity=RuleSeverity(data.get('severity', 'medium')),
            priority=ViolationPriority(data.get('priority', 'medium')),
            status=ViolationStatus(data.get('status', 'detected')),
            entity_id=data.get('entity_id'),
            entity_type=data.get('entity_type'),
            actual_value=data.get('actual_value'),
            expected_value=data.get('expected_value'),
            tolerance=data.get('tolerance', 0.0),
            deviation=data.get('deviation', 0.0),
            deviation_percentage=data.get('deviation_percentage', 0.0),
            location=location,
            visualization=visualization,
            assigned_to=data.get('assigned_to'),
            estimated_fix_time=data.get('estimated_fix_time'),
            detected_by=data.get('detected_by', 'system'),
            resolved_by=data.get('resolved_by'),
            comments=data.get('comments', []),
            internal_notes=data.get('internal_notes', []),
            tags=data.get('tags', []),
            related_violations=data.get('related_violations', []),
            parent_violation=data.get('parent_violation'),
            child_violations=data.get('child_violations', []),
            context=data.get('context', {}),
            metadata=data.get('metadata', {})
        )


@dataclass
class ViolationGroup:
    """Represents a group of related violations."""
    id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    description: str = ""
    violations: List[ComplianceViolation] = field(default_factory=list)
    group_type: str = "proximity"  # proximity, rule, entity, severity
    center_point: Optional[Point3D] = None
    radius: float = 1000.0  # grouping radius in mm
    priority: ViolationPriority = ViolationPriority.MEDIUM
    status: ViolationStatus = ViolationStatus.DETECTED
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_violation(self, violation: ComplianceViolation) -> None:
        """Add violation to group."""
        self.violations.append(violation)
        self.update_properties()

    def remove_violation(self, violation_id: str) -> None:
        """Remove violation from group."""
        self.violations = [v for v in self.violations if v.id != violation_id]
        self.update_properties()

    def update_properties(self) -> None:
        """Update group properties based on violations."""
        if not self.violations:
            return
        
        # Update priority to highest in group
        priorities = [v.priority for v in self.violations]
        if ViolationPriority.CRITICAL in priorities:
            self.priority = ViolationPriority.CRITICAL
        elif ViolationPriority.HIGH in priorities:
            self.priority = ViolationPriority.HIGH
        elif ViolationPriority.MEDIUM in priorities:
            self.priority = ViolationPriority.MEDIUM
        else:
            self.priority = ViolationPriority.LOW
        
        # Update status
        statuses = [v.status for v in self.violations]
        if all(s == ViolationStatus.RESOLVED for s in statuses):
            self.status = ViolationStatus.RESOLVED
        elif any(s == ViolationStatus.ESCALATED for s in statuses):
            self.status = ViolationStatus.ESCALATED
        else:
            self.status = ViolationStatus.DETECTED
        
        # Update center point for proximity groups
        if self.group_type == "proximity" and self.violations:
            locations = [v.location.point for v in self.violations if v.location]
            if locations:
                x_avg = sum(p.x for p in locations) / len(locations)
                y_avg = sum(p.y for p in locations) / len(locations)
                z_avg = sum(p.z for p in locations) / len(locations)
                self.center_point = Point3D(x_avg, y_avg, z_avg)
        
        self.updated_at = datetime.now()

    def get_severity_summary(self) -> Dict[str, int]:
        """Get summary of violations by severity."""
        summary = {}
        for violation in self.violations:
            severity = violation.severity.value
            summary[severity] = summary.get(severity, 0) + 1
        return summary

    def get_status_summary(self) -> Dict[str, int]:
        """Get summary of violations by status."""
        summary = {}
        for violation in self.violations:
            status = violation.status.value
            summary[status] = summary.get(status, 0) + 1
        return summary

    def get_resolution_rate(self) -> float:
        """Get percentage of resolved violations."""
        if not self.violations:
            return 0.0
        
        resolved_count = sum(1 for v in self.violations if v.status == ViolationStatus.RESOLVED)
        return (resolved_count / len(self.violations)) * 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'group_type': self.group_type,
            'center_point': self.center_point.to_tuple() if self.center_point else None,
            'radius': self.radius,
            'priority': self.priority.value,
            'status': self.status.value,
            'violation_count': len(self.violations),
            'severity_summary': self.get_severity_summary(),
            'status_summary': self.get_status_summary(),
            'resolution_rate': self.get_resolution_rate(),
            'violations': [v.to_dict() for v in self.violations],
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ViolationGroup':
        """Create from dictionary."""
        center_point = None
        if data.get('center_point'):
            center_point = Point3D.from_tuple(data['center_point'])
        
        return cls(
            id=data.get('id', str(uuid4())),
            name=data.get('name', ''),
            description=data.get('description', ''),
            group_type=data.get('group_type', 'proximity'),
            center_point=center_point,
            radius=data.get('radius', 1000.0),
            priority=ViolationPriority(data.get('priority', 'medium')),
            status=ViolationStatus(data.get('status', 'detected')),
            violations=[ComplianceViolation.from_dict(v) for v in data.get('violations', [])],
            metadata=data.get('metadata', {})
        )