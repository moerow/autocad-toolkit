"""Simple compliance violation entities for the compliance service."""
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ViolationType(Enum):
    """Types of compliance violations."""
    DIMENSION_TOO_SMALL = "dimension_too_small"
    DIMENSION_TOO_LARGE = "dimension_too_large"
    MISSING_ELEMENT = "missing_element"
    INCORRECT_PLACEMENT = "incorrect_placement"
    CODE_VIOLATION = "code_violation"


class Severity(Enum):
    """Severity levels for violations."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ComplianceViolation:
    """Represents a building code compliance violation."""
    rule_id: str
    rule_title: str
    violation_type: ViolationType
    severity: Severity
    description: str
    location: str
    measured_value: Optional[float] = None
    required_value: Optional[float] = None
    code_reference: Optional[str] = None
    
    def to_dict(self):
        """Convert to dictionary for serialization."""
        return {
            'rule_id': self.rule_id,
            'rule_title': self.rule_title,
            'violation_type': self.violation_type.value,
            'severity': self.severity.value,
            'description': self.description,
            'location': self.location,
            'measured_value': self.measured_value,
            'required_value': self.required_value,
            'code_reference': self.code_reference
        }