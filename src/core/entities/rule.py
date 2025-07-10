"""Domain entities for compliance rules."""
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
from uuid import uuid4
import re


class RuleType(Enum):
    """Types of compliance rules."""
    MINIMUM = "minimum"
    MAXIMUM = "maximum"
    EXACT = "exact"
    RANGE = "range"
    BOOLEAN = "boolean"
    CONDITIONAL = "conditional"
    CALCULATION = "calculation"
    PATTERN = "pattern"
    COUNT = "count"
    PERCENTAGE = "percentage"


class RuleCategory(Enum):
    """Categories of compliance rules."""
    WALL = "wall"
    DOOR = "door"
    WINDOW = "window"
    ROOM = "room"
    STRUCTURAL = "structural"
    SAFETY = "safety"
    ACCESSIBILITY = "accessibility"
    FIRE_SAFETY = "fire_safety"
    ELECTRICAL = "electrical"
    PLUMBING = "plumbing"
    HVAC = "hvac"
    FOUNDATION = "foundation"
    ROOF = "roof"
    STAIRS = "stairs"
    PARKING = "parking"
    ZONING = "zoning"
    ENERGY = "energy"
    GENERAL = "general"


class RuleSeverity(Enum):
    """Severity levels for rule violations."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class RuleUnit(Enum):
    """Units for rule values."""
    MILLIMETERS = "mm"
    CENTIMETERS = "cm"
    METERS = "m"
    INCHES = "in"
    FEET = "ft"
    SQUARE_METERS = "m²"
    SQUARE_FEET = "ft²"
    CUBIC_METERS = "m³"
    CUBIC_FEET = "ft³"
    DEGREES = "°"
    PERCENT = "%"
    COUNT = "count"
    RATIO = "ratio"
    UNITLESS = ""


class RuleScope(Enum):
    """Scope of rule application."""
    GLOBAL = "global"
    BUILDING = "building"
    FLOOR = "floor"
    ROOM = "room"
    ELEMENT = "element"
    COMPONENT = "component"


class RuleSource(Enum):
    """Source of the rule."""
    BUILDING_CODE = "building_code"
    ZONING_CODE = "zoning_code"
    ACCESSIBILITY_CODE = "accessibility_code"
    FIRE_CODE = "fire_code"
    COMPANY_STANDARD = "company_standard"
    CUSTOM = "custom"
    AI_EXTRACTED = "ai_extracted"
    MANUAL = "manual"


@dataclass
class RuleCondition:
    """Represents a condition for conditional rules."""
    property_name: str
    operator: str  # ==, !=, <, >, <=, >=, in, not_in, contains, regex
    value: Any
    logical_operator: str = "and"  # and, or, not

    def evaluate(self, context: Dict[str, Any]) -> bool:
        """Evaluate the condition against a context."""
        if self.property_name not in context:
            return False
        
        actual_value = context[self.property_name]
        
        if self.operator == "==":
            return actual_value == self.value
        elif self.operator == "!=":
            return actual_value != self.value
        elif self.operator == "<":
            return actual_value < self.value
        elif self.operator == ">":
            return actual_value > self.value
        elif self.operator == "<=":
            return actual_value <= self.value
        elif self.operator == ">=":
            return actual_value >= self.value
        elif self.operator == "in":
            return actual_value in self.value
        elif self.operator == "not_in":
            return actual_value not in self.value
        elif self.operator == "contains":
            return self.value in str(actual_value)
        elif self.operator == "regex":
            return bool(re.match(self.value, str(actual_value)))
        
        return False


@dataclass
class RuleFormula:
    """Represents a calculation formula for rules."""
    expression: str
    variables: List[str] = field(default_factory=list)
    description: str = ""
    
    def evaluate(self, context: Dict[str, Any]) -> float:
        """Evaluate the formula with given context."""
        try:
            # Basic formula evaluation (extend with proper parser for production)
            safe_dict = {"__builtins__": {}}
            safe_dict.update(context)
            return eval(self.expression, safe_dict)
        except Exception:
            return 0.0


@dataclass
class ComplianceRule:
    """Represents a construction compliance rule."""
    id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    description: str = ""
    rule_type: RuleType = RuleType.MINIMUM
    category: RuleCategory = RuleCategory.GENERAL
    severity: RuleSeverity = RuleSeverity.MEDIUM
    scope: RuleScope = RuleScope.ELEMENT
    source: RuleSource = RuleSource.CUSTOM
    
    # Rule values
    value: Optional[float] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    unit: RuleUnit = RuleUnit.MILLIMETERS
    tolerance: float = 0.001
    
    # Conditional rules
    conditions: List[RuleCondition] = field(default_factory=list)
    formula: Optional[RuleFormula] = None
    
    # Rule properties
    is_active: bool = True
    is_mandatory: bool = True
    priority: int = 100
    tags: List[str] = field(default_factory=list)
    
    # Documentation
    code_reference: str = ""
    section_reference: str = ""
    url: str = ""
    examples: List[str] = field(default_factory=list)
    
    # Metadata
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    created_by: str = ""
    version: str = "1.0"
    language: str = "en"
    region: str = ""
    jurisdiction: str = ""
    
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()

    def check_compliance(self, test_value: Union[float, str, bool], 
                        context: Optional[Dict[str, Any]] = None) -> bool:
        """Check if a value complies with this rule."""
        if not self.is_active:
            return True
        
        # Check conditions first
        if self.conditions and context:
            for condition in self.conditions:
                if not condition.evaluate(context):
                    return True  # Rule doesn't apply
        
        # Handle different rule types
        if self.rule_type == RuleType.MINIMUM:
            return float(test_value) >= self.value
        elif self.rule_type == RuleType.MAXIMUM:
            return float(test_value) <= self.value
        elif self.rule_type == RuleType.EXACT:
            return abs(float(test_value) - self.value) <= self.tolerance
        elif self.rule_type == RuleType.RANGE:
            return self.min_value <= float(test_value) <= self.max_value
        elif self.rule_type == RuleType.BOOLEAN:
            return bool(test_value) == bool(self.value)
        elif self.rule_type == RuleType.PATTERN:
            return bool(re.match(str(self.value), str(test_value)))
        elif self.rule_type == RuleType.COUNT:
            return int(test_value) == int(self.value)
        elif self.rule_type == RuleType.PERCENTAGE:
            return 0 <= float(test_value) <= 100
        elif self.rule_type == RuleType.CALCULATION and self.formula:
            expected_value = self.formula.evaluate(context or {})
            return abs(float(test_value) - expected_value) <= self.tolerance
        
        return True

    def get_expected_value(self, context: Optional[Dict[str, Any]] = None) -> Optional[float]:
        """Get the expected value for this rule."""
        if self.rule_type == RuleType.CALCULATION and self.formula:
            return self.formula.evaluate(context or {})
        return self.value

    def get_display_text(self) -> str:
        """Get human-readable rule description."""
        if self.rule_type == RuleType.MINIMUM:
            return f"{self.name} must be at least {self.value}{self.unit.value}"
        elif self.rule_type == RuleType.MAXIMUM:
            return f"{self.name} must be at most {self.value}{self.unit.value}"
        elif self.rule_type == RuleType.EXACT:
            return f"{self.name} must be exactly {self.value}{self.unit.value}"
        elif self.rule_type == RuleType.RANGE:
            return f"{self.name} must be between {self.min_value}{self.unit.value} and {self.max_value}{self.unit.value}"
        elif self.rule_type == RuleType.BOOLEAN:
            return f"{self.name} must be {'true' if self.value else 'false'}"
        else:
            return self.description or self.name

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'rule_type': self.rule_type.value,
            'category': self.category.value,
            'severity': self.severity.value,
            'scope': self.scope.value,
            'source': self.source.value,
            'value': self.value,
            'min_value': self.min_value,
            'max_value': self.max_value,
            'unit': self.unit.value,
            'tolerance': self.tolerance,
            'is_active': self.is_active,
            'is_mandatory': self.is_mandatory,
            'priority': self.priority,
            'tags': self.tags,
            'code_reference': self.code_reference,
            'section_reference': self.section_reference,
            'url': self.url,
            'examples': self.examples,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'created_by': self.created_by,
            'version': self.version,
            'language': self.language,
            'region': self.region,
            'jurisdiction': self.jurisdiction,
            'display_text': self.get_display_text(),
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ComplianceRule':
        """Create from dictionary."""
        return cls(
            id=data.get('id', str(uuid4())),
            name=data.get('name', ''),
            description=data.get('description', ''),
            rule_type=RuleType(data.get('rule_type', 'minimum')),
            category=RuleCategory(data.get('category', 'general')),
            severity=RuleSeverity(data.get('severity', 'medium')),
            scope=RuleScope(data.get('scope', 'element')),
            source=RuleSource(data.get('source', 'custom')),
            value=data.get('value'),
            min_value=data.get('min_value'),
            max_value=data.get('max_value'),
            unit=RuleUnit(data.get('unit', 'mm')),
            tolerance=data.get('tolerance', 0.001),
            is_active=data.get('is_active', True),
            is_mandatory=data.get('is_mandatory', True),
            priority=data.get('priority', 100),
            tags=data.get('tags', []),
            code_reference=data.get('code_reference', ''),
            section_reference=data.get('section_reference', ''),
            url=data.get('url', ''),
            examples=data.get('examples', []),
            created_by=data.get('created_by', ''),
            version=data.get('version', '1.0'),
            language=data.get('language', 'en'),
            region=data.get('region', ''),
            jurisdiction=data.get('jurisdiction', ''),
            metadata=data.get('metadata', {})
        )


@dataclass
class ComplianceViolation:
    """Represents a violation of a compliance rule."""
    id: str = field(default_factory=lambda: str(uuid4()))
    rule: ComplianceRule = None
    entity_id: str = ""
    entity_type: str = ""
    actual_value: Any = None
    expected_value: Any = None
    deviation: float = 0.0
    location: Optional[Dict[str, float]] = None
    description: str = ""
    severity: RuleSeverity = RuleSeverity.MEDIUM
    is_resolved: bool = False
    resolution_notes: str = ""
    detected_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.detected_at is None:
            self.detected_at = datetime.now()
        if self.rule:
            self.severity = self.rule.severity

    def get_display_text(self) -> str:
        """Get human-readable violation description."""
        if self.description:
            return self.description
        
        if self.rule:
            return f"{self.rule.name}: Expected {self.expected_value}, got {self.actual_value}"
        
        return f"Violation in {self.entity_type} {self.entity_id}"

    def calculate_deviation(self) -> float:
        """Calculate deviation from expected value."""
        if self.expected_value is None or self.actual_value is None:
            return 0.0
        
        try:
            expected = float(self.expected_value)
            actual = float(self.actual_value)
            self.deviation = abs(actual - expected)
            return self.deviation
        except (ValueError, TypeError):
            return 0.0

    def resolve(self, notes: str = "") -> None:
        """Mark violation as resolved."""
        self.is_resolved = True
        self.resolved_at = datetime.now()
        self.resolution_notes = notes

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'rule_id': self.rule.id if self.rule else None,
            'rule_name': self.rule.name if self.rule else None,
            'entity_id': self.entity_id,
            'entity_type': self.entity_type,
            'actual_value': self.actual_value,
            'expected_value': self.expected_value,
            'deviation': self.deviation,
            'location': self.location,
            'description': self.get_display_text(),
            'severity': self.severity.value,
            'is_resolved': self.is_resolved,
            'resolution_notes': self.resolution_notes,
            'detected_at': self.detected_at.isoformat() if self.detected_at else None,
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None,
            'context': self.context,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], rule: Optional[ComplianceRule] = None) -> 'ComplianceViolation':
        """Create from dictionary."""
        return cls(
            id=data.get('id', str(uuid4())),
            rule=rule,
            entity_id=data.get('entity_id', ''),
            entity_type=data.get('entity_type', ''),
            actual_value=data.get('actual_value'),
            expected_value=data.get('expected_value'),
            deviation=data.get('deviation', 0.0),
            location=data.get('location'),
            description=data.get('description', ''),
            severity=RuleSeverity(data.get('severity', 'medium')),
            is_resolved=data.get('is_resolved', False),
            resolution_notes=data.get('resolution_notes', ''),
            context=data.get('context', {}),
            metadata=data.get('metadata', {})
        )


@dataclass
class ComplianceReport:
    """Represents a compliance check report."""
    id: str = field(default_factory=lambda: str(uuid4()))
    name: str = "Compliance Report"
    drawing_path: str = ""
    drawing_name: str = ""
    total_rules: int = 0
    passed_rules: int = 0
    failed_rules: int = 0
    violations: List[ComplianceViolation] = field(default_factory=list)
    rules_checked: List[ComplianceRule] = field(default_factory=list)
    check_duration: float = 0.0
    created_at: Optional[datetime] = None
    summary: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        self.update_summary()

    def add_violation(self, violation: ComplianceViolation) -> None:
        """Add a violation to the report."""
        self.violations.append(violation)
        self.failed_rules += 1
        self.update_summary()

    def add_passed_rule(self, rule: ComplianceRule) -> None:
        """Add a passed rule to the report."""
        self.rules_checked.append(rule)
        self.passed_rules += 1
        self.total_rules += 1
        self.update_summary()

    def update_summary(self) -> None:
        """Update report summary."""
        self.total_rules = len(self.rules_checked) + len(self.violations)
        
        # Group violations by severity
        severity_counts = {}
        for violation in self.violations:
            severity = violation.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        # Group violations by category
        category_counts = {}
        for violation in self.violations:
            if violation.rule:
                category = violation.rule.category.value
                category_counts[category] = category_counts.get(category, 0) + 1
        
        self.summary = {
            'total_rules': self.total_rules,
            'passed_rules': self.passed_rules,
            'failed_rules': self.failed_rules,
            'compliance_rate': (self.passed_rules / self.total_rules * 100) if self.total_rules > 0 else 0,
            'severity_breakdown': severity_counts,
            'category_breakdown': category_counts,
            'critical_violations': len([v for v in self.violations if v.severity == RuleSeverity.CRITICAL]),
            'high_violations': len([v for v in self.violations if v.severity == RuleSeverity.HIGH]),
            'medium_violations': len([v for v in self.violations if v.severity == RuleSeverity.MEDIUM]),
            'low_violations': len([v for v in self.violations if v.severity == RuleSeverity.LOW]),
            'resolved_violations': len([v for v in self.violations if v.is_resolved]),
            'unresolved_violations': len([v for v in self.violations if not v.is_resolved])
        }

    def get_violations_by_severity(self, severity: RuleSeverity) -> List[ComplianceViolation]:
        """Get violations by severity level."""
        return [v for v in self.violations if v.severity == severity]

    def get_violations_by_category(self, category: RuleCategory) -> List[ComplianceViolation]:
        """Get violations by rule category."""
        return [v for v in self.violations if v.rule and v.rule.category == category]

    def is_compliant(self) -> bool:
        """Check if drawing is compliant (no critical or high violations)."""
        critical_violations = self.get_violations_by_severity(RuleSeverity.CRITICAL)
        high_violations = self.get_violations_by_severity(RuleSeverity.HIGH)
        return len(critical_violations) == 0 and len(high_violations) == 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'name': self.name,
            'drawing_path': self.drawing_path,
            'drawing_name': self.drawing_name,
            'total_rules': self.total_rules,
            'passed_rules': self.passed_rules,
            'failed_rules': self.failed_rules,
            'check_duration': self.check_duration,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'summary': self.summary,
            'violations': [v.to_dict() for v in self.violations],
            'is_compliant': self.is_compliant(),
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ComplianceReport':
        """Create from dictionary."""
        return cls(
            id=data.get('id', str(uuid4())),
            name=data.get('name', 'Compliance Report'),
            drawing_path=data.get('drawing_path', ''),
            drawing_name=data.get('drawing_name', ''),
            total_rules=data.get('total_rules', 0),
            passed_rules=data.get('passed_rules', 0),
            failed_rules=data.get('failed_rules', 0),
            check_duration=data.get('check_duration', 0.0),
            violations=[ComplianceViolation.from_dict(v) for v in data.get('violations', [])],
            metadata=data.get('metadata', {})
        )