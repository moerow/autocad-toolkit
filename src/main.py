"""Main entry point for the AutoCAD Construction Engineering Toolkit."""
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.infrastructure.autocad.connection import AutoCADConnection
from src.application.services.dimension_service import DimensionService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_dimension_service():
    """Test the dimension service."""
    print("AutoCAD Construction Toolkit - Dimension Test")
    print("=" * 50)

    # Connect to AutoCAD
    connection = AutoCADConnection()
    if not connection.connect():
        print("Failed to connect to AutoCAD!")
        return

    print(f"Connected to: {connection.doc.Name}")

    # Create dimension service
    dim_service = DimensionService(connection)

    # Test dimensioning
    print("\nDimensioning all lines on WALL layer...")
    results = dim_service.dimension_all_lines(layer_filter="WALL")

    print(f"\nResults:")
    print(f"  Lines dimensioned: {results['lines']}")
    print(f"  Total dimensions: {results['total']}")

    # Zoom to see results
    connection.acad.app.ZoomExtents()

    print("\nDimension test completed!")

def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        if sys.argv[1] == "test":
            test_dimension_service()
        elif sys.argv[1] == "gui":
            from src.presentation.gui.main_window import MainWindow
            app = MainWindow()
            app.run()
        else:
            print("Invalid argument. Use 'test' or 'gui'")
    else:
        print("AutoCAD Construction Engineering Toolkit")
        print("\nUsage:")
        print("  python -m src.main test    # Run dimension test")
        print("  python -m src.main gui     # Launch GUI")
        print("\nMake sure AutoCAD is running with a drawing open!")

if __name__ == "__main__":
    main()