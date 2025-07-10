# AutoCAD Construction Toolkit - Claude Code Context

## Project Overview
Production-ready toolkit for automating AutoCAD tasks in construction engineering.
Target: Save 80-95% time on repetitive tasks, ensure 100% building code compliance.

## Current Implementation Status
- [x] Project structure created
- [x] Basic dimension_service.py started (106 lines)
- [ ] Complete dimension service implementation
- [ ] Compliance checker service
- [ ] Drawing generator service
- [ ] GUI with CustomTkinter
- [ ] CLI interface
- [ ] Integration tests

## Key Requirements
1. **Automatic Dimensioning**: One-click to dimension entire drawings
2. **AI Compliance Checking**: Extract rules from PDFs, check drawings
3. **Drawing Generation**: Title blocks, floor plans from specs
4. **Modern GUI**: Beautiful CustomTkinter interface

## Technical Notes
- We read ACTUAL coordinates from AutoCAD (not image processing)
- Direct COM interface via pyautocad
- Support any layer naming (configurable)
- Must handle real messy drawings
- Performance critical (seconds not minutes)

## Known Issues
- Previous Claude Code session had timeout errors
- May need to complete partial implementations

## Next Steps
1. Complete dimension_service.py implementation
2. Test basic dimensioning with AutoCAD
3. Implement compliance_service.py
4. Build GUI main window

## File Locations
- Services: src/application/services/
- AutoCAD interface: src/infrastructure/autocad/
- GUI: src/presentation/gui/
- Core entities: src/core/entities/

## Important
This is PRODUCTION software for daily professional use. Every detail matters.