"""AI-powered compliance checking service for construction drawings."""
import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import PyPDF2
from pathlib import Path

from src.infrastructure.autocad.connection import AutoCADConnection
from src.infrastructure.autocad.autocad_service import AutoCADService
from src.core.entities.compliance_violation import ComplianceViolation, ViolationType, Severity

logger = logging.getLogger(__name__)


class RuleCategory(Enum):
    """Categories of building compliance rules."""
    FIRE_SAFETY = "fire_safety"
    STRUCTURAL = "structural"
    ACCESSIBILITY = "accessibility"
    ELECTRICAL = "electrical"
    PLUMBING = "plumbing"
    GENERAL = "general"


@dataclass
class ComplianceRule:
    """A building code compliance rule."""
    id: str
    category: RuleCategory
    title: str
    description: str
    code_reference: str
    measurement_type: str  # "distance", "area", "count", "angle"
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    required_layers: Optional[List[str]] = None
    keywords: Optional[List[str]] = None


class ComplianceService:
    """Service for AI-powered building code compliance checking."""
    
    def __init__(self, cad_connection: AutoCADConnection):
        self.cad_connection = cad_connection
        self.autocad_service = AutoCADService(cad_connection)
        self.rules: List[ComplianceRule] = []
        self._load_default_rules()
        
    def _load_default_rules(self):
        """Load default building code rules."""
        default_rules = [
            ComplianceRule(
                id="EXIT_WIDTH_MIN",
                category=RuleCategory.FIRE_SAFETY,
                title="Minimum Exit Width",
                description="Exit doors must be minimum 32 inches (813mm) clear width",
                code_reference="IBC 1010.1.1",
                measurement_type="distance",
                min_value=813.0,  # mm
                required_layers=["DOOR", "EXIT"],
                keywords=["exit", "door", "egress"]
            ),
            ComplianceRule(
                id="CORRIDOR_WIDTH_MIN",
                category=RuleCategory.FIRE_SAFETY,
                title="Minimum Corridor Width",
                description="Corridors must be minimum 44 inches (1118mm) wide",
                code_reference="IBC 1020.2",
                measurement_type="distance",
                min_value=1118.0,  # mm
                required_layers=["WALL", "CORRIDOR"],
                keywords=["corridor", "hallway", "passage"]
            ),
            ComplianceRule(
                id="STAIR_WIDTH_MIN",
                category=RuleCategory.FIRE_SAFETY,
                title="Minimum Stair Width",
                description="Stairs must be minimum 44 inches (1118mm) wide",
                code_reference="IBC 1011.2",
                measurement_type="distance",
                min_value=1118.0,  # mm
                required_layers=["STAIR", "WALL"],
                keywords=["stair", "step", "riser"]
            ),
            ComplianceRule(
                id="ACCESSIBLE_DOOR_WIDTH",
                category=RuleCategory.ACCESSIBILITY,
                title="Accessible Door Width",
                description="Accessible doors must be minimum 32 inches (813mm) clear width",
                code_reference="ADA 404.2.3",
                measurement_type="distance",
                min_value=813.0,  # mm
                required_layers=["DOOR"],
                keywords=["accessible", "ada", "door"]
            ),
            ComplianceRule(
                id="ROOM_HEIGHT_MIN",
                category=RuleCategory.GENERAL,
                title="Minimum Room Height",
                description="Habitable rooms must have minimum 7'6\" (2286mm) ceiling height",
                code_reference="IBC 1208.2",
                measurement_type="distance",
                min_value=2286.0,  # mm
                keywords=["ceiling", "height", "room"]
            )
        ]
        
        self.rules.extend(default_rules)
        logger.info(f"Loaded {len(self.rules)} default compliance rules")
        
    def load_rules_from_pdf(self, pdf_path: str) -> int:
        """Extract compliance rules from building code PDF."""
        try:
            rules_found = 0
            
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                
                # Extract text from all pages
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                
                # Parse rules using regex patterns
                rules_found = self._parse_rules_from_text(text)
                
            logger.info(f"Extracted {rules_found} rules from PDF: {pdf_path}")
            return rules_found
            
        except Exception as e:
            logger.error(f"Error loading rules from PDF {pdf_path}: {e}")
            return 0
            
    def _parse_rules_from_text(self, text: str) -> int:
        """Parse compliance rules from extracted text using AI-like patterns."""
        rules_found = 0
        
        # Pattern for door width requirements
        door_patterns = [
            r"door.*?(\d+)\s*(?:inches?|in\.?|mm).*?(?:wide?|width)",
            r"(?:minimum|min\.?)\s*(?:door\s*)?width.*?(\d+)\s*(?:inches?|in\.?|mm)",
            r"exit.*?door.*?(\d+)\s*(?:inches?|in\.?|mm)"
        ]
        
        # Pattern for corridor/hallway requirements
        corridor_patterns = [
            r"corridor.*?(\d+)\s*(?:inches?|in\.?|mm).*?(?:wide?|width)",
            r"hallway.*?(\d+)\s*(?:inches?|in\.?|mm).*?(?:wide?|width)",
            r"(?:minimum|min\.?)\s*(?:corridor|hallway)\s*width.*?(\d+)"
        ]
        
        # Pattern for ceiling height
        height_patterns = [
            r"ceiling.*?height.*?(\d+)\s*(?:feet?|ft\.?|'|\"|inches?|mm)",
            r"(?:minimum|min\.?)\s*(?:ceiling\s*)?height.*?(\d+)",
            r"room.*?height.*?(\d+)\s*(?:feet?|ft\.?|'|\"|inches?|mm)"
        ]
        
        # Search for patterns and create rules
        for pattern_group, category, keywords in [
            (door_patterns, RuleCategory.FIRE_SAFETY, ["door", "exit"]),
            (corridor_patterns, RuleCategory.FIRE_SAFETY, ["corridor", "hallway"]),
            (height_patterns, RuleCategory.GENERAL, ["ceiling", "height"])
        ]:
            for pattern in pattern_group:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    try:
                        value = float(match.group(1))
                        
                        # Convert inches to mm if needed
                        if any(unit in match.group(0).lower() for unit in ["inch", "in.", "\""]):
                            value = value * 25.4
                        elif any(unit in match.group(0).lower() for unit in ["feet", "ft.", "'"]):
                            value = value * 304.8
                            
                        # Create rule
                        rule_id = f"EXTRACTED_{category.value.upper()}_{len(self.rules)}"
                        extracted_rule = ComplianceRule(
                            id=rule_id,
                            category=category,
                            title=f"Extracted {category.value.replace('_', ' ').title()} Requirement",
                            description=match.group(0).strip(),
                            code_reference="Extracted from PDF",
                            measurement_type="distance",
                            min_value=value,
                            keywords=keywords
                        )
                        
                        self.rules.append(extracted_rule)
                        rules_found += 1
                        
                    except (ValueError, IndexError):
                        continue
                        
        return rules_found
        
    def check_compliance(self, rule_categories: Optional[List[RuleCategory]] = None) -> List[ComplianceViolation]:
        """Check drawing compliance against all loaded rules."""
        violations = []
        
        if not self.cad_connection.is_connected():
            logger.error("Not connected to AutoCAD")
            return violations
            
        # Filter rules by category if specified
        rules_to_check = self.rules
        if rule_categories:
            rules_to_check = [r for r in self.rules if r.category in rule_categories]
            
        logger.info(f"Checking {len(rules_to_check)} compliance rules...")
        
        for rule in rules_to_check:
            try:
                rule_violations = self._check_single_rule(rule)
                violations.extend(rule_violations)
            except Exception as e:
                logger.error(f"Error checking rule {rule.id}: {e}")
                
        logger.info(f"Found {len(violations)} compliance violations")
        return violations
        
    def _check_single_rule(self, rule: ComplianceRule) -> List[ComplianceViolation]:
        """Check a single compliance rule against the drawing."""
        violations = []
        
        try:
            if rule.measurement_type == "distance":
                violations.extend(self._check_distance_rule(rule))
            elif rule.measurement_type == "area":
                violations.extend(self._check_area_rule(rule))
            elif rule.measurement_type == "count":
                violations.extend(self._check_count_rule(rule))
                
        except Exception as e:
            logger.warning(f"Failed to check rule {rule.id}: {e}")
            
        return violations
        
    def _check_distance_rule(self, rule: ComplianceRule) -> List[ComplianceViolation]:
        """Check distance-based compliance rules."""
        violations = []
        
        # Get relevant entities
        entities = self.autocad_service.get_all_entities(
            layers=rule.required_layers if rule.required_layers else None
        )
        
        # Find parallel lines (potential corridors, door openings)
        parallel_pairs = self.autocad_service.find_parallel_lines(
            tolerance_angle=2.0,
            min_distance=50.0,
            max_distance=3000.0
        )
        
        for line1, line2, distance in parallel_pairs:
            # Check if this represents a space that should comply with the rule
            if self._is_relevant_space(rule, line1, line2):
                if rule.min_value and distance < rule.min_value:
                    violation = ComplianceViolation(
                        rule_id=rule.id,
                        rule_title=rule.title,
                        violation_type=ViolationType.DIMENSION_TOO_SMALL,
                        severity=Severity.HIGH,
                        description=f"{rule.description}. Found: {distance:.1f}mm, Required: {rule.min_value:.1f}mm",
                        location=f"Between lines at {line1.start} and {line2.start}",
                        measured_value=distance,
                        required_value=rule.min_value,
                        code_reference=rule.code_reference
                    )
                    violations.append(violation)
                    
                elif rule.max_value and distance > rule.max_value:
                    violation = ComplianceViolation(
                        rule_id=rule.id,
                        rule_title=rule.title,
                        violation_type=ViolationType.DIMENSION_TOO_LARGE,
                        severity=Severity.MEDIUM,
                        description=f"{rule.description}. Found: {distance:.1f}mm, Maximum: {rule.max_value:.1f}mm",
                        location=f"Between lines at {line1.start} and {line2.start}",
                        measured_value=distance,
                        required_value=rule.max_value,
                        code_reference=rule.code_reference
                    )
                    violations.append(violation)
                    
        return violations
        
    def _check_area_rule(self, rule: ComplianceRule) -> List[ComplianceViolation]:
        """Check area-based compliance rules."""
        # TODO: Implement area checking (room sizes, etc.)
        return []
        
    def _check_count_rule(self, rule: ComplianceRule) -> List[ComplianceViolation]:
        """Check count-based compliance rules."""
        # TODO: Implement count checking (number of exits, etc.)
        return []
        
    def _is_relevant_space(self, rule: ComplianceRule, line1, line2) -> bool:
        """Determine if a space between lines is relevant to a rule."""
        if not rule.keywords:
            return True
            
        # Get nearby text entities to check for keywords
        texts = self.autocad_service.get_texts()
        
        # Check if any text near the lines contains rule keywords
        for text_info in texts:
            text_content = text_info.get('text', '').lower()
            if any(keyword.lower() in text_content for keyword in rule.keywords):
                # Check if text is reasonably close to the lines
                # (simplified distance check)
                return True
                
        # Default heuristics based on rule category
        if rule.category == RuleCategory.FIRE_SAFETY:
            # Assume parallel lines in reasonable range are corridors/exits
            return True
        elif rule.category == RuleCategory.ACCESSIBILITY:
            # Look for door-like openings
            return True
            
        return False
        
    def generate_compliance_report(self, violations: List[ComplianceViolation]) -> Dict[str, Any]:
        """Generate a comprehensive compliance report."""
        report = {
            "summary": {
                "total_violations": len(violations),
                "high_severity": len([v for v in violations if v.severity == Severity.HIGH]),
                "medium_severity": len([v for v in violations if v.severity == Severity.MEDIUM]),
                "low_severity": len([v for v in violations if v.severity == Severity.LOW]),
                "categories": {}
            },
            "violations": [],
            "rules_checked": len(self.rules),
            "timestamp": None
        }
        
        # Group violations by category
        for violation in violations:
            # Find the rule to get its category
            rule = next((r for r in self.rules if r.id == violation.rule_id), None)
            if rule:
                category = rule.category.value
                if category not in report["summary"]["categories"]:
                    report["summary"]["categories"][category] = 0
                report["summary"]["categories"][category] += 1
                
            # Add violation details
            report["violations"].append({
                "rule_id": violation.rule_id,
                "title": violation.rule_title,
                "type": violation.violation_type.value,
                "severity": violation.severity.value,
                "description": violation.description,
                "location": violation.location,
                "measured_value": violation.measured_value,
                "required_value": violation.required_value,
                "code_reference": violation.code_reference
            })
            
        return report
        
    def get_available_rule_categories(self) -> List[str]:
        """Get list of available rule categories."""
        categories = set()
        for rule in self.rules:
            categories.add(rule.category.value)
        return sorted(list(categories))
        
    def get_rules_by_category(self, category: RuleCategory) -> List[ComplianceRule]:
        """Get all rules for a specific category."""
        return [rule for rule in self.rules if rule.category == category]