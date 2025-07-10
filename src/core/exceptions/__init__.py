"""Core exceptions package."""

from .autocad_exceptions import (
    # Base exceptions
    AutoCADException,
    ComplianceException,
    DimensionException,
    GeometryException,
    ConfigurationException,
    ValidationException,
    ServiceException,
    InfrastructureException,
    
    # AutoCAD specific exceptions
    AutoCADConnectionException,
    AutoCADNotRunningException,
    AutoCADDocumentException,
    AutoCADNoActiveDocumentException,
    AutoCADEntityException,
    AutoCADEntityNotFoundException,
    AutoCADLayerException,
    AutoCADLayerNotFoundException,
    AutoCADDimensionException,
    AutoCADInvalidDimensionException,
    AutoCADDrawingException,
    AutoCADFileException,
    AutoCADFileNotFoundError,
    AutoCADPermissionException,
    AutoCADVersionException,
    AutoCADCOMException,
    AutoCADTimeoutException,
    AutoCADGeometryException,
    AutoCADInvalidGeometryException,
    AutoCADSelectionException,
    AutoCADBlockException,
    AutoCADBlockNotFoundError,
    AutoCADStyleException,
    AutoCADTextStyleNotFoundError,
    AutoCADDimensionStyleNotFoundError,
    AutoCADLinetypeNotFoundError,
    
    # Compliance exceptions
    ComplianceRuleException,
    ComplianceRuleNotFoundError,
    ComplianceInvalidRuleException,
    ComplianceViolationException,
    ComplianceCheckException,
    ComplianceReportException,
    CompliancePDFException,
    ComplianceAIException,
    
    # Dimension exceptions
    DimensionPlacementException,
    DimensionCollisionException,
    DimensionInvalidEntityException,
    DimensionStyleException,
    
    # Geometry exceptions
    GeometryInvalidException,
    GeometryCalculationException,
    
    # Configuration exceptions
    ConfigurationFileNotFoundError,
    ConfigurationInvalidException,
    
    # Validation exceptions
    ValidationFieldException,
    
    # Service exceptions
    ServiceNotAvailableException,
    ServiceTimeoutException,
    
    # Infrastructure exceptions
    DatabaseException,
    AIServiceException,
    FileSystemException,
    NetworkException,
)

__all__ = [
    # Base exceptions
    "AutoCADException",
    "ComplianceException",
    "DimensionException",
    "GeometryException",
    "ConfigurationException",
    "ValidationException",
    "ServiceException",
    "InfrastructureException",
    
    # AutoCAD specific exceptions
    "AutoCADConnectionException",
    "AutoCADNotRunningException",
    "AutoCADDocumentException",
    "AutoCADNoActiveDocumentException",
    "AutoCADEntityException",
    "AutoCADEntityNotFoundException",
    "AutoCADLayerException",
    "AutoCADLayerNotFoundException",
    "AutoCADDimensionException",
    "AutoCADInvalidDimensionException",
    "AutoCADDrawingException",
    "AutoCADFileException",
    "AutoCADFileNotFoundError",
    "AutoCADPermissionException",
    "AutoCADVersionException",
    "AutoCADCOMException",
    "AutoCADTimeoutException",
    "AutoCADGeometryException",
    "AutoCADInvalidGeometryException",
    "AutoCADSelectionException",
    "AutoCADBlockException",
    "AutoCADBlockNotFoundError",
    "AutoCADStyleException",
    "AutoCADTextStyleNotFoundError",
    "AutoCADDimensionStyleNotFoundError",
    "AutoCADLinetypeNotFoundError",
    
    # Compliance exceptions
    "ComplianceRuleException",
    "ComplianceRuleNotFoundError",
    "ComplianceInvalidRuleException",
    "ComplianceViolationException",
    "ComplianceCheckException",
    "ComplianceReportException",
    "CompliancePDFException",
    "ComplianceAIException",
    
    # Dimension exceptions
    "DimensionPlacementException",
    "DimensionCollisionException",
    "DimensionInvalidEntityException",
    "DimensionStyleException",
    
    # Geometry exceptions
    "GeometryInvalidException",
    "GeometryCalculationException",
    
    # Configuration exceptions
    "ConfigurationFileNotFoundError",
    "ConfigurationInvalidException",
    
    # Validation exceptions
    "ValidationFieldException",
    
    # Service exceptions
    "ServiceNotAvailableException",
    "ServiceTimeoutException",
    
    # Infrastructure exceptions
    "DatabaseException",
    "AIServiceException",
    "FileSystemException",
    "NetworkException",
]
