"""
Minimal working example for testing AutoCAD dimension automation
Run this to verify your setup works!
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from pyautocad import Autocad, APoint
    import math
    
    print("AutoCAD Dimension Tool - Quick Test")
    print("=" * 50)
    
    # Connect to AutoCAD
    print("Connecting to AutoCAD...")
    acad = Autocad(create_if_not_exists=True)
    print(f"Connected to: {acad.doc.Name}")
    
    # Get model space
    model = acad.model
    
    # Function to add dimension to a line
    def dimension_line(line):
        """Add dimension to a line object."""
        try:
            # Get line endpoints
            start = APoint(line.StartPoint)
            end = APoint(line.EndPoint)
            
            # Calculate midpoint
            mid_x = (start.x + end.x) / 2
            mid_y = (start.y + end.y) / 2
            
            # Calculate perpendicular offset
            angle = math.atan2(end.y - start.y, end.x - start.x)
            offset_angle = angle + math.pi / 2
            offset_distance = 500  # 500 units offset
            
            # Calculate dimension position
            dim_x = mid_x + offset_distance * math.cos(offset_angle)
            dim_y = mid_y + offset_distance * math.sin(offset_angle)
            dim_location = APoint(dim_x, dim_y)
            
            # Add dimension
            dim = model.AddDimAligned(start, end, dim_location)
            dim.TextHeight = 100
            
            return True
        except Exception as e:
            print(f"Error dimensioning line: {e}")
            return False
    
    # Test 1: Create and dimension a test line
    print("\nTest 1: Creating and dimensioning a test line...")
    test_line = model.AddLine(APoint(0, 0), APoint(5000, 0))
    test_line.Layer = "0"  # Default layer
    
    if dimension_line(test_line):
        print("Test line dimensioned successfully!")
    
    # Test 2: Dimension existing lines
    print("\nTest 2: Looking for existing lines to dimension...")
    line_count = 0
    dimensioned_count = 0
    
    for obj in acad.iter_objects("Line"):
        line_count += 1
        if obj.Layer.upper() in ["WALL", "WALLS", "0"]:
            if dimension_line(obj):
                dimensioned_count += 1
    
    print(f"Found {line_count} lines, dimensioned {dimensioned_count}")
    
    # Test 3: Create a simple rectangle and dimension it
    print("\nTest 3: Creating a rectangle and dimensioning all sides...")
    points = [
        (1000, 1000),
        (6000, 1000),
        (6000, 4000),
        (1000, 4000)
    ]
    
    # Create rectangle
    for i in range(4):
        start = APoint(points[i][0], points[i][1])
        end = APoint(points[(i + 1) % 4][0], points[(i + 1) % 4][1])
        line = model.AddLine(start, end)
        dimension_line(line)
    
    print("Rectangle created and dimensioned!")
    
    # Zoom to see everything
    acad.app.ZoomExtents()
    
    print("\nTest completed successfully!")
    print("Check your AutoCAD drawing to see the dimensions")
    
except ImportError as e:
    print("Error: Required packages not installed")
    print("Please run: pip install pyautocad pywin32")
    print(f"Details: {e}")
except Exception as e:
    print(f"Error: {e}")
    print("Make sure AutoCAD is running")
