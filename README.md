# AutoCAD Construction Engineering Toolkit

## 🎯 Project Overview

This is a **production-ready toolkit** for automating AutoCAD tasks in construction engineering. It saves 80-95% of time on repetitive tasks while ensuring 100% compliance with building codes.

**Target User**: Construction engineers who spend hours daily on manual AutoCAD tasks.

**Core Value**: What takes 2-3 hours manually happens in 30 seconds automatically.

## 🚨 IMPORTANT IMPLEMENTATION NOTES FOR CLAUDE CODE

### This is NOT a prototype - it's production software that must:
1. **Work with real AutoCAD drawings** (messy, complex, various standards)
2. **Be 100% accurate** (construction tolerances are strict)
3. **Have a beautiful modern GUI** (will be used 8 hours/day)
4. **Handle all edge cases** (real drawings have inconsistencies)

### Key Technical Points:
- We read **ACTUAL coordinates** from AutoCAD via COM interface (NOT image processing)
- We process **directly in memory** (no Excel export needed)
- We support **any drawing convention** (configurable layer names)
- We use **CustomTkinter** for modern GUI (not basic tkinter)

## 📋 Complete Feature Requirements

### 1. Automatic Dimensioning System

**Current Problem**: Manually dimensioning a floor plan with 50 walls takes 2-3 hours.

**Required Implementation**:
```python
# The system must:
1. Detect all geometric entities (lines, polylines, circles, arcs)
2. Identify walls (parallel line pairs = wall with thickness)
3. Place dimensions intelligently:
   - No overlaps
   - Consistent spacing (500mm default offset)
   - Proper text orientation
   - Dimension chains for connected walls
4. Support multiple entity types:
   - Lines → Linear dimensions
   - Circles → Diameter dimensions
   - Arcs → Radius dimensions
   - Polylines → Segment dimensions
5. Handle edge cases:
   - Very short walls (< min_length)
   - Angled walls
   - Curved walls
   - Overlapping geometry
```

**Example Usage**:
```python
# User workflow:
1. Open drawing with walls on layer "WALL"
2. Click "Connect" in toolkit
3. Click "Dimension All"
4. Done - all walls dimensioned in seconds
```

### 2. AI-Powered Compliance Checker

**Current Problem**: Manually checking if shelter designs meet building codes (PDF documents).

**Required Implementation**:
```python
# The system must:
1. Extract rules from PDF building codes:
   - Use OpenAI API to parse natural language
   - Fallback to regex patterns
   - Support multiple languages

2. Rule types to support:
   - Minimum values (wall thickness >= 300mm)
   - Maximum values (room area <= 50m²)
   - Exact values (door width == 900mm)
   - Range values (ceiling height 2.4-3.0m)
   - Boolean rules (must have emergency exit)

3. Check drawing elements:
   - Wall thickness (parallel line distance)
   - Door/window dimensions (block sizes)
   - Room areas (closed polyline areas)
   - Clearances and spacing
   - Structural requirements

4. Violation handling:
   - Mark violations visually (red circles)
   - Add text annotations
   - Group nearby violations
   - Color-code by severity

5. Generate reports:
   - JSON for data exchange
   - HTML for viewing
   - PDF for officials
   - Excel for analysis
```

**Example Rule Extraction**:
```
PDF Text: "All load-bearing walls must be at least 300mm thick"
Extracted Rule: ComplianceRule(
    name="load_bearing_wall_thickness",
    type=MINIMUM,
    value=300,
    unit="mm",
    category=WALL
)
```

### 3. Drawing Generation Tools

**Current Problem**: Creating standard drawings from scratch takes hours.

**Required Implementation**:
```python
# The system must generate:

1. Title Blocks:
   - All standard sizes (A0-A4)
   - Company information fields
   - Project details
   - Revision tracking
   - Drawing index

2. Floor Plans from specifications:
   Input: rooms = [
       {"name": "Living Room", "width": 5000, "height": 4000},
       {"name": "Kitchen", "width": 3000, "height": 3500}
   ]
   Output: Complete floor plan with:
   - Walls (double lines with thickness)
   - Doors (with swing arcs)
   - Windows (with sills)
   - Room labels and areas
   - Dimensions

3. Standard Construction Details:
   - Foundation sections
   - Wall sections
   - Door/window details
   - Structural connections
```

### 4. Modern GUI Requirements

**Must use CustomTkinter with**:
```python
# Main Window Structure:
- Sidebar navigation (fixed 250px width)
- Tab-based content area
- Status bar with progress
- Dark/Light theme toggle
- Connection status indicator

# Color Scheme:
colors = {
    'success': '#00b894',  # Green
    'warning': '#fdcb6e',  # Orange  
    'error': '#d63031',    # Red
    'info': '#0984e3',     # Blue
    'accent': '#6c5ce7'    # Purple
}

# Key Features:
- Smooth animations
- Progress indicators
- Tooltips on hover
- Keyboard shortcuts
- Drag-drop file support
- Real-time status updates
```

## 🏗️ Project Structure

```
autocad_toolkit/
├── README.md                 # This file
├── requirements.txt          # Dependencies
├── setup.py                  # Installation script
├── .env.example             # Environment variables template
├── src/
│   ├── main.py              # Entry point
│   ├── core/                # Business logic (no external dependencies)
│   │   ├── entities/        # Domain models
│   │   │   ├── rule.py      # ComplianceRule class
│   │   │   ├── violation.py # ComplianceViolation class
│   │   │   └── dimension.py # Dimension class
│   │   ├── interfaces/      # Abstract interfaces
│   │   └── exceptions/      # Custom exceptions
│   ├── infrastructure/      # External integrations
│   │   ├── autocad/        # AutoCAD COM interface
│   │   ├── ai/             # OpenAI integration
│   │   └── persistence/    # Data storage
│   ├── application/        # Use cases and services
│   │   ├── services/       # Business services
│   │   └── dto/           # Data transfer objects
│   ├── presentation/       # UI layer
│   │   ├── gui/           # CustomTkinter GUI
│   │   └── cli/           # Click CLI
│   └── utils/             # Helpers
├── tests/                 # Test suite
├── docs/                  # Documentation
└── examples/             # Example drawings
```

## 🔧 Implementation Priority Order

### Phase 1: Core Foundation (MUST complete first)
1. **AutoCAD Connection** (`infrastructure/autocad/`)
   - Establish COM connection
   - Entity reading/writing
   - Layer management
   - Error handling

2. **Basic Dimension Service** (`application/services/dimension_service.py`)
   - Line dimensioning
   - Offset calculation
   - Placement logic

3. **Simple GUI** (`presentation/gui/`)
   - Connection status
   - Dimension tab
   - Results display

### Phase 2: Full Dimensioning
1. **Complete dimension types**
   - Circles (diameter)
   - Arcs (radius)
   - Polylines (segments)
   - Angles

2. **Intelligent placement**
   - Collision detection
   - Dimension chains
   - Grouping logic

3. **Configuration options**
   - Styles
   - Precision
   - Units

### Phase 3: Compliance System
1. **Rule extraction**
   - PDF parser
   - AI integration
   - Rule validation

2. **Compliance checking**
   - Wall thickness
   - Door/window sizes
   - Room areas
   - Spacing requirements

3. **Violation handling**
   - Visual markers
   - Reports
   - Recommendations

### Phase 4: Drawing Generation
1. **Title blocks**
2. **Floor plans**
3. **Standard details**
4. **Templates**

### Phase 5: Polish & Advanced Features
1. **Pattern recognition**
2. **Batch processing**
3. **Cloud sync**
4. **Multi-language support**

## 💻 Code Examples for Key Features

### Dimension Placement Algorithm
```python
def calculate_dimension_position(self, start_point, end_point, offset=500):
    """Calculate optimal dimension placement avoiding overlaps."""
    
    # 1. Get midpoint
    mid_x = (start_point[0] + end_point[0]) / 2
    mid_y = (start_point[1] + end_point[1]) / 2
    
    # 2. Calculate perpendicular angle
    line_angle = math.atan2(
        end_point[1] - start_point[1],
        end_point[0] - start_point[0]
    )
    perp_angle = line_angle + math.pi / 2
    
    # 3. Check both sides for clearance
    left_pos = (
        mid_x + offset * math.cos(perp_angle),
        mid_y + offset * math.sin(perp_angle)
    )
    right_pos = (
        mid_x - offset * math.cos(perp_angle),
        mid_y - offset * math.sin(perp_angle)
    )
    
    # 4. Choose clearer side
    left_clear = self.check_clearance(left_pos)
    right_clear = self.check_clearance(right_pos)
    
    return left_pos if left_clear > right_clear else right_pos
```

### Wall Thickness Detection
```python
def find_wall_pairs(self, lines):
    """Detect parallel line pairs representing walls."""
    walls = []
    
    for i, line1 in enumerate(lines):
        for line2 in lines[i+1:]:
            # Check if parallel (within 1 degree)
            if self.are_parallel(line1, line2, tolerance=0.017):
                distance = self.get_parallel_distance(line1, line2)
                
                # Typical wall thickness: 100-400mm
                if 100 <= distance <= 400:
                    walls.append({
                        'inner': line1,
                        'outer': line2,
                        'thickness': distance,
                        'center_line': self.get_centerline(line1, line2)
                    })
    
    return walls
```

### Compliance Rule Extraction
```python
def extract_rules_from_pdf(self, pdf_path):
    """Extract compliance rules using AI."""
    
    # 1. Extract text from PDF
    text = self.extract_pdf_text(pdf_path)
    
    # 2. Send to AI with structured prompt
    prompt = f"""
    Extract all measurable construction requirements from this text.
    Return as JSON array with format:
    {{
        "name": "descriptive_name",
        "description": "what it checks",
        "type": "minimum|maximum|exact|range",
        "value": numeric_value,
        "unit": "mm|m|sqm",
        "category": "wall|door|window|room|safety"
    }}
    
    Text: {text}
    """
    
    # 3. Parse AI response
    rules = self.ai_client.extract_structured_data(prompt)
    
    # 4. Validate and store
    return [self.validate_rule(r) for r in rules if self.validate_rule(r)]
```

## 🚀 Testing Requirements

### Unit Tests Required:
- Geometry calculations
- Rule parsing
- Dimension placement
- Compliance checking

### Integration Tests Required:
- AutoCAD connection
- Drawing operations
- AI integration
- Report generation

### Test Drawings Needed:
1. Simple rectangle room
2. Multi-room floor plan
3. Complex building with curves
4. Drawing with existing dimensions
5. Non-standard layer names

## 📊 Performance Requirements

| Operation | Maximum Time | Current | Target |
|-----------|--------------|---------|--------|
| Connect to AutoCAD | 2 seconds | - | ✓ |
| Dimension 50 walls | 5 seconds | - | ✓ |
| Check 100 rules | 10 seconds | - | ✓ |
| Generate floor plan | 3 seconds | - | ✓ |
| Load 50-page PDF | 15 seconds | - | ✓ |

## 🐛 Known Issues to Address

1. **Unicode handling** - Windows encoding issues with emojis
2. **Large drawings** - Performance with 1000+ entities
3. **Memory usage** - Caching strategy needed
4. **Concurrent access** - Multiple AutoCAD instances

## 🔑 Environment Variables

Create `.env` file:
```bash
# Required for AI features
OPENAI_API_KEY=your_api_key_here

# Optional configuration
LOG_LEVEL=INFO
DATA_DIR=./data
AUTOCAD_VERSION=2024
DEFAULT_UNITS=metric
```

## 📦 Dependencies

```txt
# Core
pyautocad>=0.2.0     # AutoCAD COM interface
pywin32>=305         # Windows COM support

# GUI  
customtkinter>=5.2.0 # Modern UI framework
Pillow>=10.0.0      # Image processing

# CLI
click>=8.0.0        # Command line interface

# AI & Text
openai>=0.27.0      # AI rule extraction
PyPDF2>=3.0.0       # PDF processing

# Data
pandas>=1.5.0       # Data manipulation
openpyxl>=3.1.0     # Excel reports

# Utils
python-dotenv>=1.0.0 # Environment management
```

## 🎯 Success Metrics

The implementation is complete when:
1. ✅ All features work end-to-end
2. ✅ GUI is beautiful and responsive
3. ✅ Processing is fast (seconds not minutes)
4. ✅ Handles edge cases gracefully
5. ✅ Reports are professional quality
6. ✅ Zero manual intervention needed
7. ✅ Works with any drawing style

## 📞 Questions for Implementation

If you need clarification:
1. How should overlapping dimensions be handled?
2. What's the preferred report format for authorities?
3. Should dimensions update when geometry changes?
4. How to handle multi-language building codes?
5. What about imperial vs metric units?

---

**Remember**: This toolkit will be used daily by professional engineers. Every detail matters. Make it fast, accurate, and beautiful!