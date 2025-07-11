"""AutoCAD connection implementation."""
import logging
from typing import Any, Optional
import pythoncom
from pyautocad import Autocad

from src.core.interfaces.autocad_interface import IAutoCADConnection

logger = logging.getLogger(__name__)


class AutoCADConnection(IAutoCADConnection):
    """Manages connection to AutoCAD application."""

    def __init__(self):
        self._acad = None
        self._app = None
        self._doc = None
        self._model = None

    def connect(self) -> bool:
        """Establish connection to AutoCAD."""
        try:
            # Initialize COM for thread
            pythoncom.CoInitialize()

            # Try to connect to existing AutoCAD instance
            self._acad = Autocad(create_if_not_exists=True)
            self._app = self._acad.app
            self._doc = self._acad.doc
            self._model = self._acad.model

            logger.info(f"Connected to AutoCAD - Document: {self._doc.Name}")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to AutoCAD: {e}")
            return False

    def get_open_documents(self) -> list:
        """Get list of all open documents."""
        try:
            if not self._app:
                return []
            
            documents = []
            for doc in self._app.Documents:
                documents.append({
                    'name': doc.Name,
                    'path': doc.FullName if hasattr(doc, 'FullName') else '',
                    'is_active': doc == self._app.ActiveDocument,
                    'document': doc
                })
            return documents
        except Exception as e:
            logger.error(f"Error getting open documents: {e}")
            return []

    def switch_to_document(self, document_name: str) -> bool:
        """Switch to a specific document by name."""
        try:
            if not self._app:
                logger.error("No AutoCAD application connection")
                return False
                
            # Initialize COM for this thread
            pythoncom.CoInitialize()
            
            # Find the document
            target_doc = None
            for doc in self._app.Documents:
                if doc.Name == document_name:
                    target_doc = doc
                    break
                    
            if not target_doc:
                logger.warning(f"Document not found: {document_name}")
                return False
                
            # Switch to the document
            self._app.ActiveDocument = target_doc
            self._doc = target_doc
            self._model = target_doc.ModelSpace
            
            logger.info(f"Switched to document: {document_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error switching to document {document_name}: {e}")
            return False

    def disconnect(self) -> None:
        """Disconnect from AutoCAD."""
        try:
            if self._acad:
                self._model = None
                self._doc = None
                self._app = None
                self._acad = None
                pythoncom.CoUninitialize()
                logger.info("Disconnected from AutoCAD")
        except Exception as e:
            logger.error(f"Error during disconnect: {e}")

    def is_connected(self) -> bool:
        """Check if connected to AutoCAD."""
        try:
            if self._acad and self._doc and self._model:
                # Try to access document name to verify connection
                _ = self._doc.Name
                return True
        except:
            pass
        return False

    @property
    def acad(self):
        return self._acad

    @property
    def model(self):
        return self._model

    @property
    def doc(self):
        return self._doc

    def get_autocad_version(self) -> str:
        """Get AutoCAD version information."""
        try:
            if self._app:
                return self._app.Version
            return "Unknown"
        except:
            return "Unknown"

    def get_active_document(self):
        """Get the active document."""
        return self._doc

    def create_new_document(self, template_path: str = None):
        """Create a new document."""
        try:
            if self._app:
                return self._app.Documents.Add(template_path)
            return None
        except:
            return None

    def open_document(self, file_path: str):
        """Open an existing document."""
        try:
            if self._app:
                return self._app.Documents.Open(file_path)
            return None
        except:
            return None

    def save_document(self, file_path: str = None) -> bool:
        """Save the active document."""
        try:
            if self._doc:
                if file_path:
                    self._doc.SaveAs(file_path)
                else:
                    self._doc.Save()
                return True
            return False
        except:
            return False

    def close_document(self, save_changes: bool = True) -> bool:
        """Close the active document."""
        try:
            if self._doc:
                self._doc.Close(save_changes)
                return True
            return False
        except:
            return False