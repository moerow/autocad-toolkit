"""Custom exceptions for AutoCAD operations."""


class AutoCADException(Exception):
    """Base exception for AutoCAD operations."""
    
    def __init__(self, message: str, error_code: str = None, details: dict = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}


class AutoCADConnectionException(AutoCADException):
    """Exception raised when AutoCAD connection fails."""
    pass


class AutoCADNotRunningException(AutoCADConnectionException):
    """Exception raised when AutoCAD is not running."""
    
    def __init__(self, message: str = "AutoCAD is not running"):
        super().__init__(message, "AUTOCAD_NOT_RUNNING")


class AutoCADDocumentException(AutoCADException):
    """Exception raised for document-related errors."""
    pass


class AutoCADNoActiveDocumentException(AutoCADDocumentException):
    """Exception raised when no active document is available."""
    
    def __init__(self, message: str = "No active AutoCAD document"):
        super().__init__(message, "NO_ACTIVE_DOCUMENT")


class AutoCADEntityException(AutoCADException):
    """Exception raised for entity-related errors."""
    pass


class AutoCADEntityNotFoundException(AutoCADEntityException):
    """Exception raised when entity is not found."""
    
    def __init__(self, entity_id: str = None):
        message = f"Entity not found: {entity_id}" if entity_id else "Entity not found"
        super().__init__(message, "ENTITY_NOT_FOUND", {"entity_id": entity_id})


class AutoCADLayerException(AutoCADException):
    """Exception raised for layer-related errors."""
    pass


class AutoCADLayerNotFoundException(AutoCADLayerException):
    """Exception raised when layer is not found."""
    
    def __init__(self, layer_name: str):
        super().__init__(f"Layer not found: {layer_name}", "LAYER_NOT_FOUND", {"layer_name": layer_name})


class AutoCADDimensionException(AutoCADException):
    """Exception raised for dimension-related errors."""
    pass


class AutoCADInvalidDimensionException(AutoCADDimensionException):
    """Exception raised for invalid dimension operations."""
    pass


class AutoCADDrawingException(AutoCADException):
    """Exception raised for drawing-related errors."""
    pass


class AutoCADFileException(AutoCADException):
    """Exception raised for file-related errors."""
    pass


class AutoCADFileNotFoundError(AutoCADFileException):
    """Exception raised when AutoCAD file is not found."""
    
    def __init__(self, file_path: str):
        super().__init__(f"AutoCAD file not found: {file_path}", "FILE_NOT_FOUND", {"file_path": file_path})


class AutoCADPermissionException(AutoCADException):
    """Exception raised for permission-related errors."""
    pass


class AutoCADVersionException(AutoCADException):
    """Exception raised for version compatibility issues."""
    pass


class AutoCADCOMException(AutoCADException):
    """Exception raised for COM interface errors."""
    pass


class AutoCADTimeoutException(AutoCADException):
    """Exception raised when AutoCAD operation times out."""
    
    def __init__(self, operation: str, timeout: float):
        super().__init__(
            f"AutoCAD operation timed out: {operation} (timeout: {timeout}s)",
            "OPERATION_TIMEOUT",
            {"operation": operation, "timeout": timeout}
        )


class AutoCADGeometryException(AutoCADException):
    """Exception raised for geometry-related errors."""
    pass


class AutoCADInvalidGeometryException(AutoCADGeometryException):
    """Exception raised for invalid geometry operations."""
    pass


class AutoCADSelectionException(AutoCADException):
    """Exception raised for selection-related errors."""
    pass


class AutoCADBlockException(AutoCADException):
    """Exception raised for block-related errors."""
    pass


class AutoCADBlockNotFoundError(AutoCADBlockException):
    """Exception raised when block is not found."""
    
    def __init__(self, block_name: str):
        super().__init__(f"Block not found: {block_name}", "BLOCK_NOT_FOUND", {"block_name": block_name})


class AutoCADStyleException(AutoCADException):
    """Exception raised for style-related errors."""
    pass


class AutoCADTextStyleNotFoundError(AutoCADStyleException):
    """Exception raised when text style is not found."""
    
    def __init__(self, style_name: str):
        super().__init__(f"Text style not found: {style_name}", "TEXT_STYLE_NOT_FOUND", {"style_name": style_name})


class AutoCADDimensionStyleNotFoundError(AutoCADStyleException):
    """Exception raised when dimension style is not found."""
    
    def __init__(self, style_name: str):
        super().__init__(f"Dimension style not found: {style_name}", "DIMENSION_STYLE_NOT_FOUND", {"style_name": style_name})


class AutoCADLinetypeNotFoundError(AutoCADStyleException):
    """Exception raised when linetype is not found."""
    
    def __init__(self, linetype_name: str):
        super().__init__(f"Linetype not found: {linetype_name}", "LINETYPE_NOT_FOUND", {"linetype_name": linetype_name})


class ComplianceException(Exception):
    """Base exception for compliance operations."""
    
    def __init__(self, message: str, error_code: str = None, details: dict = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}


class ComplianceRuleException(ComplianceException):
    """Exception raised for rule-related errors."""
    pass


class ComplianceRuleNotFoundError(ComplianceRuleException):
    """Exception raised when compliance rule is not found."""
    
    def __init__(self, rule_id: str):
        super().__init__(f"Compliance rule not found: {rule_id}", "RULE_NOT_FOUND", {"rule_id": rule_id})


class ComplianceInvalidRuleException(ComplianceRuleException):
    """Exception raised for invalid rule definitions."""
    pass


class ComplianceViolationException(ComplianceException):
    """Exception raised for violation-related errors."""
    pass


class ComplianceCheckException(ComplianceException):
    """Exception raised during compliance checking."""
    pass


class ComplianceReportException(ComplianceException):
    """Exception raised for report-related errors."""
    pass


class CompliancePDFException(ComplianceException):
    """Exception raised for PDF processing errors."""
    pass


class ComplianceAIException(ComplianceException):
    """Exception raised for AI processing errors."""
    pass


class DimensionException(Exception):
    """Base exception for dimension operations."""
    
    def __init__(self, message: str, error_code: str = None, details: dict = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}


class DimensionPlacementException(DimensionException):
    """Exception raised for dimension placement errors."""
    pass


class DimensionCollisionException(DimensionPlacementException):
    """Exception raised when dimensions would collide."""
    pass


class DimensionInvalidEntityException(DimensionException):
    """Exception raised for invalid entity dimensioning."""
    pass


class DimensionStyleException(DimensionException):
    """Exception raised for dimension style errors."""
    pass


class GeometryException(Exception):
    """Base exception for geometry operations."""
    
    def __init__(self, message: str, error_code: str = None, details: dict = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}


class GeometryInvalidException(GeometryException):
    """Exception raised for invalid geometry."""
    pass


class GeometryCalculationException(GeometryException):
    """Exception raised for geometry calculation errors."""
    pass


class ConfigurationException(Exception):
    """Base exception for configuration errors."""
    
    def __init__(self, message: str, error_code: str = None, details: dict = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}


class ConfigurationFileNotFoundError(ConfigurationException):
    """Exception raised when configuration file is not found."""
    
    def __init__(self, config_path: str):
        super().__init__(f"Configuration file not found: {config_path}", "CONFIG_FILE_NOT_FOUND", {"config_path": config_path})


class ConfigurationInvalidException(ConfigurationException):
    """Exception raised for invalid configuration."""
    pass


class ValidationException(Exception):
    """Base exception for validation errors."""
    
    def __init__(self, message: str, error_code: str = None, details: dict = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}


class ValidationFieldException(ValidationException):
    """Exception raised for field validation errors."""
    
    def __init__(self, field_name: str, value: any, validation_rule: str):
        super().__init__(
            f"Validation failed for field '{field_name}': {validation_rule}",
            "FIELD_VALIDATION_FAILED",
            {"field_name": field_name, "value": value, "validation_rule": validation_rule}
        )


class ServiceException(Exception):
    """Base exception for service layer operations."""
    
    def __init__(self, message: str, error_code: str = None, details: dict = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}


class ServiceNotAvailableException(ServiceException):
    """Exception raised when service is not available."""
    pass


class ServiceTimeoutException(ServiceException):
    """Exception raised when service operation times out."""
    
    def __init__(self, service_name: str, timeout: float):
        super().__init__(
            f"Service operation timed out: {service_name} (timeout: {timeout}s)",
            "SERVICE_TIMEOUT",
            {"service_name": service_name, "timeout": timeout}
        )


class InfrastructureException(Exception):
    """Base exception for infrastructure layer operations."""
    
    def __init__(self, message: str, error_code: str = None, details: dict = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}


class DatabaseException(InfrastructureException):
    """Exception raised for database errors."""
    pass


class AIServiceException(InfrastructureException):
    """Exception raised for AI service errors."""
    pass


class FileSystemException(InfrastructureException):
    """Exception raised for file system errors."""
    pass


class NetworkException(InfrastructureException):
    """Exception raised for network errors."""
    pass