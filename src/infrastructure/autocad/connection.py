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
            if self._doc:
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