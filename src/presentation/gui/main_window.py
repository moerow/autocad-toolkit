"""Modern GUI window for the AutoCAD Construction Toolkit."""
import customtkinter as ctk
from tkinter import messagebox
import logging
import threading
from typing import Optional

from src.infrastructure.autocad.connection import AutoCADConnection
from src.application.services.dimension_service import DimensionService
from src.application.services.compliance_service import ComplianceService, RuleCategory
from tkinter import Toplevel, Label

logger = logging.getLogger(__name__)

# Set appearance mode and color theme
ctk.set_appearance_mode("light")  # Clean light mode
ctk.set_default_color_theme("blue")

# Clean light theme colors
COLORS = {
    'primary': '#1976d2',
    'primary_light': '#42a5f5', 
    'primary_dark': '#0d47a1',
    'secondary': '#757575',
    'surface': '#f5f5f5',
    'background': '#ffffff',
    'text': '#212121',
    'text_secondary': '#757575',
    'success': '#4caf50',
    'warning': '#ff9800',
    'error': '#f44336',
    'divider': '#e0e0e0'
}

# Clean, modern typography with system fonts
FONTS = {
    'heading': ('Segoe UI', 24, 'normal'),
    'title': ('Segoe UI', 18, 'bold'),
    'body': ('Segoe UI', 14, 'normal'),
    'caption': ('Segoe UI', 12, 'normal'),
    'button': ('Segoe UI', 14, 'bold'),
    'mono': ('Consolas', 12, 'normal')
}


class MainWindow:
    """Clean, modern main application window."""
    
    def __init__(self):
        self.root = ctk.CTk()
        self.connection: Optional[AutoCADConnection] = None
        self.dimension_service: Optional[DimensionService] = None
        self.compliance_service: Optional[ComplianceService] = None
        self.loading_overlay = None
        self.loading_label = None
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the clean, modern user interface."""
        self.root.title("AutoCAD Construction Toolkit")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)
        
        # Main container with grid layout
        main_container = ctk.CTkFrame(self.root, fg_color=COLORS['background'])
        main_container.pack(fill="both", expand=True)
        main_container.grid_rowconfigure(1, weight=1)
        main_container.grid_columnconfigure(1, weight=1)
        
        # Top bar with connection status
        self.create_top_bar(main_container)
        
        # Sidebar navigation
        sidebar = ctk.CTkFrame(main_container, fg_color=COLORS['surface'], width=200)
        sidebar.grid(row=1, column=0, sticky="nsew", padx=(0, 1))
        sidebar.grid_propagate(False)
        self.create_sidebar(sidebar)
        
        # Main content area
        self.content_frame = ctk.CTkFrame(main_container, fg_color=COLORS['background'])
        self.content_frame.grid(row=1, column=1, sticky="nsew")
        self.content_frame.grid_rowconfigure(0, weight=1)
        self.content_frame.grid_columnconfigure(0, weight=1)
        
        # Activity log at bottom
        self.create_activity_log(main_container)
        
        # Show dimensions tab by default
        self.show_dimensions_tab()
        
    def create_top_bar(self, parent):
        """Create header with AutoCAD Construction Toolkit title and connection info."""
        top_bar = ctk.CTkFrame(parent, fg_color=COLORS['surface'], height=140)
        top_bar.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 1))
        top_bar.grid_propagate(False)
        
        # Main content container
        content_frame = ctk.CTkFrame(top_bar, fg_color="transparent")
        content_frame.pack(fill="both", expand=True, padx=32, pady=20)
        
        # Title section
        title_frame = ctk.CTkFrame(content_frame, fg_color="transparent")
        title_frame.pack(fill="x", pady=(0, 15))
        
        # Main title
        title_label = ctk.CTkLabel(
            title_frame,
            text="AutoCAD Construction Toolkit",
            font=ctk.CTkFont("Segoe UI", 24, "bold"),
            text_color=COLORS['text']
        )
        title_label.pack(anchor="w")
        
        # Subtitle
        subtitle_label = ctk.CTkLabel(
            title_frame,
            text="Professional automated dimensioning and building code compliance",
            font=ctk.CTkFont("Segoe UI", 14, "normal"),
            text_color=COLORS['text_secondary']
        )
        subtitle_label.pack(anchor="w", pady=(2, 0))
        
        # Connection section
        connection_frame = ctk.CTkFrame(content_frame, fg_color="transparent")
        connection_frame.pack(fill="x")
        
        # Connection status (left side)
        status_container = ctk.CTkFrame(connection_frame, fg_color="transparent")
        status_container.pack(side="left", fill="x", expand=True)
        
        # Status indicator and text
        status_row = ctk.CTkFrame(status_container, fg_color="transparent")
        status_row.pack(anchor="w", pady=(0, 8))
        
        self.status_indicator = ctk.CTkLabel(
            status_row,
            text="●",
            font=ctk.CTkFont(size=16),
            text_color="#e0e0e0"
        )
        self.status_indicator.pack(side="left", padx=(0, 8))
        
        self.status_label = ctk.CTkLabel(
            status_row,
            text="Not Connected",
            font=ctk.CTkFont(*FONTS['body']),
            text_color=COLORS['text']
        )
        self.status_label.pack(side="left")
        
        # Connect button (right next to status)
        self.connect_btn = ctk.CTkButton(
            status_row,
            text="Connect to AutoCAD",
            command=self.connect_autocad,
            font=ctk.CTkFont(*FONTS['button']),
            height=36,
            width=160,
            corner_radius=6,
            fg_color=COLORS['primary'],
            hover_color=COLORS['primary_light']
        )
        self.connect_btn.pack(side="left", padx=(15, 0))
        
        # Document dropdown
        document_row = ctk.CTkFrame(status_container, fg_color="transparent")
        document_row.pack(anchor="w")
        
        ctk.CTkLabel(
            document_row,
            text="Active Drawing:",
            font=ctk.CTkFont(*FONTS['body']),
            text_color=COLORS['text']
        ).pack(side="left", padx=(0, 10))
        
        self.document_dropdown = ctk.CTkComboBox(
            document_row,
            values=["No documents available"],
            command=self.on_document_selected,
            font=ctk.CTkFont(*FONTS['body']),
            height=36,
            state="disabled",
            width=250
        )
        self.document_dropdown.pack(side="left")
        self.document_dropdown.set("No documents available")
        
        
    def create_sidebar(self, parent):
        """Create sidebar navigation."""
        # Title
        title = ctk.CTkLabel(
            parent,
            text="Tools",
            font=ctk.CTkFont(*FONTS['title']),
            text_color=COLORS['text']
        )
        title.pack(pady=(20, 10))
        
        # Navigation buttons
        self.nav_buttons = {}
        
        # Dimensions tab
        self.nav_buttons['dimensions'] = ctk.CTkButton(
            parent,
            text="📐 Dimensions",
            command=self.show_dimensions_tab,
            font=ctk.CTkFont(*FONTS['body']),
            height=40,
            corner_radius=6,
            fg_color="transparent",
            hover_color=COLORS['divider'],
            text_color=COLORS['text'],
            anchor="w"
        )
        self.nav_buttons['dimensions'].pack(fill="x", padx=10, pady=(10, 5))
        
        # Compliance tab
        self.nav_buttons['compliance'] = ctk.CTkButton(
            parent,
            text="✓ Compliance",
            command=self.show_compliance_tab,
            font=ctk.CTkFont(*FONTS['body']),
            height=40,
            corner_radius=6,
            fg_color="transparent",
            hover_color=COLORS['divider'],
            text_color=COLORS['text'],
            anchor="w"
        )
        self.nav_buttons['compliance'].pack(fill="x", padx=10, pady=5)
        
        # Settings tab
        self.nav_buttons['settings'] = ctk.CTkButton(
            parent,
            text="⚙ Settings",
            command=self.show_settings_tab,
            font=ctk.CTkFont(*FONTS['body']),
            height=40,
            corner_radius=6,
            fg_color="transparent", 
            hover_color=COLORS['divider'],
            text_color=COLORS['text'],
            anchor="w"
        )
        self.nav_buttons['settings'].pack(fill="x", padx=10, pady=5)
        
    def create_activity_log(self, parent):
        """Create activity log at bottom."""
        log_frame = ctk.CTkFrame(parent, fg_color=COLORS['surface'], height=200)
        log_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(1, 0))
        log_frame.grid_propagate(False)
        
        # Header with clear button
        log_header = ctk.CTkFrame(log_frame, fg_color="transparent")
        log_header.pack(fill="x", padx=20, pady=(10, 5))
        
        log_title = ctk.CTkLabel(
            log_header,
            text="Activity Log",
            font=ctk.CTkFont(FONTS['body'][0], FONTS['body'][1], "bold"),
            text_color=COLORS['text']
        )
        log_title.pack(side="left")
        
        clear_btn = ctk.CTkButton(
            log_header,
            text="Clear",
            command=self.clear_log,
            font=ctk.CTkFont(*FONTS['caption']),
            height=24,
            width=60,
            corner_radius=4
        )
        clear_btn.pack(side="right")
        
        # Log text area
        self.results_text = ctk.CTkTextbox(
            log_frame,
            font=ctk.CTkFont(*FONTS['mono']),
            corner_radius=6,
            wrap="word",
            fg_color="white"
        )
        self.results_text.pack(fill="both", expand=True, padx=20, pady=(0, 10))
        
        # Initial message
        self.log_message("Ready to connect")
        
    def show_dimensions_tab(self):
        """Show dimensions tab content."""
        self.clear_content_frame()
        self.set_active_tab('dimensions')
        
        # Create dimensions content
        dimensions_frame = ctk.CTkFrame(self.content_frame, fg_color="transparent")
        dimensions_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Title
        title = ctk.CTkLabel(
            dimensions_frame,
            text="Automatic Dimensioning",
            font=ctk.CTkFont(*FONTS['heading']),
            text_color=COLORS['text']
        )
        title.pack(anchor="w", pady=(0, 20))
        
        # Configuration section
        config_frame = ctk.CTkFrame(dimensions_frame, fg_color=COLORS['surface'], corner_radius=8)
        config_frame.pack(fill="x", pady=(0, 20))
        
        config_title = ctk.CTkLabel(
            config_frame,
            text="Configuration",
            font=ctk.CTkFont(FONTS['body'][0], FONTS['body'][1], "bold"),
            text_color=COLORS['text']
        )
        config_title.pack(anchor="w", padx=20, pady=(15, 10))
        
        # Configuration fields
        fields_frame = ctk.CTkFrame(config_frame, fg_color="transparent")
        fields_frame.pack(fill="x", padx=20, pady=(0, 20))
        
        # Layer filter
        layer_label = ctk.CTkLabel(
            fields_frame,
            text="Layer Filter (optional):",
            font=ctk.CTkFont(*FONTS['body']),
            text_color=COLORS['text']
        )
        layer_label.grid(row=0, column=0, sticky="w", pady=(0, 5))
        
        self.layer_var = ctk.StringVar(value="")
        self.layer_entry = ctk.CTkEntry(
            fields_frame,
            textvariable=self.layer_var,
            placeholder_text="e.g., WALL (leave empty for all)",
            font=ctk.CTkFont(*FONTS['body']),
            height=36,
            width=300
        )
        self.layer_entry.grid(row=1, column=0, sticky="w", pady=(0, 15))
        
        # Offset distance
        offset_label = ctk.CTkLabel(
            fields_frame,
            text="Distance from lines (mm):",
            font=ctk.CTkFont(*FONTS['body']),
            text_color=COLORS['text']
        )
        offset_label.grid(row=2, column=0, sticky="w", pady=(0, 5))
        
        self.offset_var = ctk.StringVar(value="0.2")
        self.offset_entry = ctk.CTkEntry(
            fields_frame,
            textvariable=self.offset_var,
            placeholder_text="0.2",
            font=ctk.CTkFont(*FONTS['body']),
            height=36,
            width=150
        )
        self.offset_entry.grid(row=3, column=0, sticky="w")
        
        # Action buttons with loading indicators
        action_frame = ctk.CTkFrame(dimensions_frame, fg_color=COLORS['surface'], corner_radius=8)
        action_frame.pack(fill="x")
        
        buttons_frame = ctk.CTkFrame(action_frame, fg_color="transparent")
        buttons_frame.pack(padx=20, pady=20)
        
        # Button container with spinner
        add_container = ctk.CTkFrame(buttons_frame, fg_color="transparent")
        add_container.pack(side="left", padx=(0, 10))
        
        self.dimension_btn = ctk.CTkButton(
            add_container,
            text="Add Dimensions",
            command=self.add_dimensions,
            state="disabled",
            font=ctk.CTkFont(*FONTS['button']),
            height=44,
            corner_radius=6,
            width=160,
            fg_color=COLORS['primary']
        )
        self.dimension_btn.pack()
        
        # Spinner for add button
        self.add_spinner = ctk.CTkProgressBar(
            add_container,
            mode="indeterminate",
            width=160,
            height=3,
            corner_radius=2
        )
        self.add_spinner.pack(pady=(2, 0))
        self.add_spinner.pack_forget()  # Hide initially
        
        # Analyze button with spinner
        analyze_container = ctk.CTkFrame(buttons_frame, fg_color="transparent")
        analyze_container.pack(side="left", padx=(0, 10))
        
        self.analyze_btn = ctk.CTkButton(
            analyze_container,
            text="Analyze Existing",
            command=self.analyze_dimensions,
            state="disabled",
            font=ctk.CTkFont(*FONTS['button']),
            height=44,
            corner_radius=6,
            width=140
        )
        self.analyze_btn.pack()
        
        self.analyze_spinner = ctk.CTkProgressBar(
            analyze_container,
            mode="indeterminate",
            width=140,
            height=3,
            corner_radius=2
        )
        self.analyze_spinner.pack(pady=(2, 0))
        self.analyze_spinner.pack_forget()
        
        # Clear button
        clear_container = ctk.CTkFrame(buttons_frame, fg_color="transparent")
        clear_container.pack(side="left")
        
        self.clear_btn = ctk.CTkButton(
            clear_container,
            text="Clear All",
            command=self.clear_dimensions,
            state="disabled",
            font=ctk.CTkFont(*FONTS['button']),
            height=44,
            corner_radius=6,
            width=100,
            fg_color=COLORS['error']
        )
        self.clear_btn.pack()
        
        self.clear_spinner = ctk.CTkProgressBar(
            clear_container,
            mode="indeterminate", 
            width=100,
            height=3,
            corner_radius=2
        )
        self.clear_spinner.pack(pady=(2, 0))
        self.clear_spinner.pack_forget()
        
    def show_compliance_tab(self):
        """Show compliance tab content."""
        self.clear_content_frame()
        self.set_active_tab('compliance')
        
        # Implementation for compliance tab
        compliance_frame = ctk.CTkFrame(self.content_frame, fg_color="transparent")
        compliance_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        title = ctk.CTkLabel(
            compliance_frame,
            text="Building Code Compliance",
            font=ctk.CTkFont(*FONTS['heading']),
            text_color=COLORS['text']
        )
        title.pack(anchor="w", pady=(0, 20))
        
        # Rules section
        rules_frame = ctk.CTkFrame(compliance_frame, fg_color=COLORS['surface'], corner_radius=8)
        rules_frame.pack(fill="x", pady=(0, 20))
        
        rules_title = ctk.CTkLabel(
            rules_frame,
            text="Compliance Rules",
            font=ctk.CTkFont(FONTS['body'][0], FONTS['body'][1], "bold"),
            text_color=COLORS['text']
        )
        rules_title.pack(anchor="w", padx=20, pady=(15, 10))
        
        # Rules list
        rules_list_frame = ctk.CTkFrame(rules_frame, fg_color="transparent")
        rules_list_frame.pack(fill="x", padx=20, pady=(0, 20))
        
        # Sample rules display
        sample_rules = [
            "Minimum door width: 800mm",
            "Minimum ceiling height: 2400mm", 
            "Maximum stair riser: 190mm",
            "Minimum corridor width: 1200mm",
            "Window area minimum: 10% of floor area"
        ]
        
        for rule in sample_rules:
            rule_label = ctk.CTkLabel(
                rules_list_frame,
                text=f"• {rule}",
                font=ctk.CTkFont(*FONTS['body']),
                text_color=COLORS['text']
            )
            rule_label.pack(anchor="w", pady=2)
        
        # Actions section
        actions_frame = ctk.CTkFrame(compliance_frame, fg_color=COLORS['surface'], corner_radius=8)
        actions_frame.pack(fill="x", pady=(0, 20))
        
        actions_title = ctk.CTkLabel(
            actions_frame,
            text="Compliance Actions",
            font=ctk.CTkFont(FONTS['body'][0], FONTS['body'][1], "bold"),
            text_color=COLORS['text']
        )
        actions_title.pack(anchor="w", padx=20, pady=(15, 10))
        
        # Buttons for compliance actions
        buttons_frame = ctk.CTkFrame(actions_frame, fg_color="transparent")
        buttons_frame.pack(fill="x", padx=20, pady=(0, 20))
        
        # Check compliance button
        check_container = ctk.CTkFrame(buttons_frame, fg_color="transparent")
        check_container.pack(side="left", padx=(0, 20))
        
        self.check_compliance_btn = ctk.CTkButton(
            check_container,
            text="Check Compliance",
            command=self.check_compliance,
            state="normal" if self.connection and self.connection.is_connected() else "disabled",
            font=ctk.CTkFont(*FONTS['button']),
            height=44,
            corner_radius=6,
            width=160
        )
        self.check_compliance_btn.pack()
        
        self.check_spinner = ctk.CTkProgressBar(
            check_container,
            mode="indeterminate",
            width=160,
            height=3,
            corner_radius=2
        )
        self.check_spinner.pack(pady=(2, 0))
        self.check_spinner.pack_forget()
        
        # Load rules button
        load_container = ctk.CTkFrame(buttons_frame, fg_color="transparent")
        load_container.pack(side="left")
        
        self.load_rules_btn = ctk.CTkButton(
            load_container,
            text="Load Rules from PDF",
            command=self.load_rules_from_pdf,
            state="normal" if self.connection and self.connection.is_connected() else "disabled",
            font=ctk.CTkFont(*FONTS['button']),
            height=44,
            corner_radius=6,
            width=160,
            fg_color=COLORS['secondary']
        )
        self.load_rules_btn.pack()
        
        self.load_spinner = ctk.CTkProgressBar(
            load_container,
            mode="indeterminate",
            width=160,
            height=3,
            corner_radius=2
        )
        self.load_spinner.pack(pady=(2, 0))
        self.load_spinner.pack_forget()
        
    def show_settings_tab(self):
        """Show settings tab content."""
        self.clear_content_frame()
        self.set_active_tab('settings')
        
        # Implementation for settings tab
        settings_frame = ctk.CTkFrame(self.content_frame, fg_color="transparent")
        settings_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        title = ctk.CTkLabel(
            settings_frame,
            text="Settings",
            font=ctk.CTkFont(*FONTS['heading']),
            text_color=COLORS['text']
        )
        title.pack(anchor="w", pady=(0, 20))
        
        # Add settings content here...
        
    def clear_content_frame(self):
        """Clear the content frame."""
        for widget in self.content_frame.winfo_children():
            widget.destroy()
            
    def set_active_tab(self, tab_name):
        """Set the active tab appearance."""
        for name, button in self.nav_buttons.items():
            if name == tab_name:
                button.configure(fg_color=COLORS['primary'], text_color="white")
            else:
                button.configure(fg_color="transparent", text_color=COLORS['text'])
    
    def update_document_list(self):
        """Update the document dropdown with open drawings."""
        if not self.connection:
            return
            
        try:
            documents = self.connection.get_open_documents()
            if documents:
                doc_names = [doc['name'] for doc in documents]
                self.document_dropdown.configure(values=doc_names, state="readonly")
                
                # Set current active document as selected
                active_doc = next((doc['name'] for doc in documents if doc['is_active']), None)
                if active_doc:
                    self.document_dropdown.set(active_doc)
            else:
                self.document_dropdown.configure(values=["No documents open"], state="disabled")
                self.document_dropdown.set("No documents open")
        except Exception as e:
            self.log_message(f"Error updating document list: {e}")
        
        
    def connect_autocad(self):
        """Connect to AutoCAD."""
        self.log_message("Attempting to connect to AutoCAD...")
        try:
            self.connection = AutoCADConnection()
            if self.connection.connect():
                # Update UI
                self.status_label.configure(text=f"Connected to {self.connection.doc.Name}")
                self.status_indicator.configure(text_color="#4caf50")  # Green when connected
                
                # Get and populate open documents
                self.update_document_list()
                
                self.dimension_service = DimensionService(self.connection)
                self.compliance_service = ComplianceService(self.connection)
                self.enable_buttons()
                self.connect_btn.configure(text="Refresh")
                self.log_message(f"Connected successfully! Drawing: {self.connection.doc.Name}")
                
                # Log available documents
                docs = self.connection.get_open_documents()
                self.log_message(f"Found {len(docs)} open drawing(s)")
                for doc in docs:
                    status = " (ACTIVE)" if doc['is_active'] else ""
                    self.log_message(f"  • {doc['name']}{status}")
                    
            else:
                self.log_message("Failed to connect to AutoCAD")
                messagebox.showerror("Connection Error", "Failed to connect to AutoCAD.\\n\\nMake sure AutoCAD is running with a drawing open.")
        except Exception as e:
            self.log_message(f"Connection error: {e}")
            messagebox.showerror("Connection Error", f"Error connecting to AutoCAD:\\n\\n{e}")
            
    def update_document_list(self):
        """Update the document dropdown with open drawings."""
        if not self.connection:
            return
            
        try:
            documents = self.connection.get_open_documents()
            if documents:
                doc_names = [doc['name'] for doc in documents]
                self.document_dropdown.configure(values=doc_names, state="readonly")
                
                # Set current active document as selected
                active_doc = next((doc['name'] for doc in documents if doc['is_active']), None)
                if active_doc:
                    self.document_dropdown.set(active_doc)
            else:
                self.document_dropdown.configure(values=["No documents open"], state="disabled")
                self.document_dropdown.set("No documents open")
        except Exception as e:
            self.log_message(f"Error updating document list: {e}")
            
    def on_document_selected(self, selected_document):
        """Handle document selection from dropdown."""
        if not self.connection or selected_document == "No documents available":
            return
            
        try:
            self.log_message(f"Switching to drawing: {selected_document}")
            
            if self.connection.switch_to_document(selected_document):
                # Update services to use new document
                self.dimension_service = DimensionService(self.connection)
                self.compliance_service = ComplianceService(self.connection)
                
                # Update status
                self.status_label.configure(text=f"Connected to {selected_document}")
                self.log_message(f"Successfully switched to: {selected_document}")
                
                # Ensure buttons remain enabled
                self.enable_buttons()
                
            else:
                self.log_message(f"Failed to switch to: {selected_document}")
                self.log_message("Note: Document switching can fail if AutoCAD is busy")
                
        except Exception as e:
            self.log_message(f"Error switching documents: {e}")
            self.log_message("Note: This can happen if AutoCAD is processing or the document is protected")
            
    def test_connection(self):
        """Test and debug the AutoCAD connection."""
        self.log_message("=== CONNECTION DEBUG ===")
        
        # Test connection object
        if not self.connection:
            self.log_message("❌ No connection object")
            return
        else:
            self.log_message("✅ Connection object exists")
            
        # Test connection status
        try:
            is_connected = self.connection.is_connected()
            self.log_message(f"✅ Connection status: {is_connected}")
        except Exception as e:
            self.log_message(f"❌ Connection check failed: {e}")
            
        # Test model access
        try:
            model = self.connection.model
            if model:
                self.log_message("✅ Model object accessible")
                # Count entities
                entity_count = 0
                for entity in model:
                    entity_count += 1
                    if entity_count > 10:  # Don't count all
                        break
                self.log_message(f"✅ Found {entity_count}+ entities in drawing")
            else:
                self.log_message("❌ Model object is None")
        except Exception as e:
            self.log_message(f"❌ Model access failed: {e}")
            
        # Test dimension service
        if not self.dimension_service:
            self.log_message("❌ No dimension service")
        else:
            self.log_message("✅ Dimension service exists")
            
        # Analyze entities in drawing
        try:
            self.log_message("📊 ENTITY ANALYSIS:")
            entity_types = {}
            line_count = 0
            
            for entity in model:
                entity_type = entity.ObjectName
                entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
                
                # Count lines specifically
                if entity_type == 'AcDbLine':
                    line_count += 1
                    
            self.log_message(f"📋 Total entities: {sum(entity_types.values())}")
            self.log_message(f"📏 LINE entities (what we dimension): {line_count}")
            
            # Show top 5 entity types
            sorted_types = sorted(entity_types.items(), key=lambda x: x[1], reverse=True)
            for entity_type, count in sorted_types[:5]:
                friendly_name = {
                    'AcDbLine': 'Lines',
                    'AcDbPolyline': 'Polylines', 
                    'AcDb2dPolyline': 'Polylines',
                    'AcDbCircle': 'Circles',
                    'AcDbArc': 'Arcs',
                    'AcDbText': 'Text',
                    'AcDbMText': 'Text',
                    'AcDbBlockReference': 'Blocks',
                    'AcDbHatch': 'Hatches'
                }.get(entity_type, entity_type)
                self.log_message(f"  • {friendly_name}: {count}")
                
        except Exception as e:
            self.log_message(f"❌ Entity analysis failed: {e}")
            
        self.log_message("=== DEBUG COMPLETE ===")
        
    def test_first_line(self):
        """Test dimensioning just the first line found."""
        if not self.dimension_service:
            messagebox.showerror("Error", "Not connected to AutoCAD")
            return
            
        self.log_message("=== TESTING FIRST LINE ===")
        
        try:
            line_count = 0
            for entity in self.connection.model:
                if entity.ObjectName == 'AcDbLine':
                    line_count += 1
                    self.log_message(f"🔍 Found line {line_count} on layer '{entity.Layer}'")
                    
                    # Get line details
                    start = entity.StartPoint
                    end = entity.EndPoint
                    length = ((end[0] - start[0])**2 + (end[1] - start[1])**2)**0.5
                    
                    self.log_message(f"📏 Line length: {length:.2f} units")
                    self.log_message(f"📍 Start: ({start[0]:.1f}, {start[1]:.1f})")
                    self.log_message(f"📍 End: ({end[0]:.1f}, {end[1]:.1f})")
                    
                    # Try to dimension this one line
                    try:
                        if self.dimension_service._add_dimension_to_line(entity):
                            self.log_message("✅ Successfully added dimension to first line!")
                        else:
                            self.log_message("❌ Failed to add dimension to first line")
                            
                        # Only test the first line
                        break
                        
                    except Exception as dim_error:
                        self.log_message(f"❌ Error dimensioning first line: {dim_error}")
                        break
                        
            if line_count == 0:
                self.log_message("❌ No lines found in drawing")
                
        except Exception as e:
            self.log_message(f"❌ Error in line test: {e}")
            
        self.log_message("=== TEST COMPLETE ===")
            
    def add_dimensions(self):
        """Add dimensions to lines."""
        # Check connection and services
        if not self.connection:
            messagebox.showerror("Error", "Not connected to AutoCAD. Click 'Connect to AutoCAD' first.")
            return
            
        if not self.connection.is_connected():
            messagebox.showerror("Error", "Connection to AutoCAD lost. Please reconnect.")
            return
            
        if not self.dimension_service:
            messagebox.showerror("Error", "Dimension service not initialized. Please reconnect to AutoCAD.")
            return
            
        try:
            self.dimension_service.config['offset_distance'] = float(self.offset_var.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid offset distance. Please enter a numeric value.")
            return
            
        # Run synchronously (no threading)
        self.log_message("Starting automatic dimensioning...")
        self.show_spinner('add')
        self.dimension_btn.configure(state="disabled")
        
        try:
            layer_filter = self.layer_var.get().strip() if self.layer_var.get().strip() else None
            
            self.log_message("Processing drawing entities...")
            
            if layer_filter:
                self.log_message(f"Filtering by layer: {layer_filter}")
            else:
                self.log_message("Processing all layers")
                
            results = self.dimension_service.dimension_all_lines(layer_filter)
            
            self.log_message("Dimensioning completed!")
            self.log_message(f"Lines dimensioned: {results['lines']}")
            self.log_message(f"Total dimensions: {results['total']}")
            
            # Try to zoom to show all content (optional)
            try:
                if self.connection and self.connection.acad:
                    self.connection.acad.app.ZoomExtents()
                    self.log_message("Zoomed to show all content")
            except Exception as zoom_error:
                self.log_message(f"Note: Could not auto-zoom view (ZoomExtents)")
                
        except Exception as e:
            error_msg = f"Error during dimensioning: {e}"
            self.log_message(error_msg)
            messagebox.showerror("Dimensioning Error", str(e))
        finally:
            # Re-enable button and hide spinner
            self.hide_spinner('add')
            self.dimension_btn.configure(state="normal")
        
            
    def analyze_dimensions(self):
        """Analyze existing dimensions in the drawing."""
        if not self.dimension_service:
            messagebox.showerror("Error", "Not connected to AutoCAD")
            return
            
        self.log_message("=== ANALYZING EXISTING DIMENSIONS ===")
        
        # Show spinner for analyze button
        self.show_spinner('analyze')
        self.analyze_btn.configure(state="disabled")
        
        try:
            analysis = self.dimension_service.analyze_existing_dimensions()
            
            if analysis['total_count'] == 0:
                self.log_message("No dimensions found in the drawing")
                messagebox.showinfo("Analysis Complete", "No dimensions found in the drawing.")
                return
                
            # Log analysis results
            self.log_message(f"Total dimensions found: {analysis['total_count']}")
            
            # Dimensions by type
            if analysis['by_type']:
                self.log_message("\nDimensions by type:")
                for dim_type, count in analysis['by_type'].items():
                    self.log_message(f"  • {dim_type}: {count}")
                    
            # Dimensions by layer
            if analysis['by_layer']:
                self.log_message("\nDimensions by layer:")
                for layer, count in analysis['by_layer'].items():
                    self.log_message(f"  • Layer '{layer}': {count}")
                    
            # Dimension styles
            if analysis['dimension_styles']:
                self.log_message(f"\nDimension styles used: {', '.join(analysis['dimension_styles'])}")
                
            # Text height statistics
            if 'text_height_stats' in analysis:
                stats = analysis['text_height_stats']
                self.log_message(f"\nText height range: {stats['min']:.2f} - {stats['max']:.2f} (avg: {stats['avg']:.2f})")
                
            self.log_message("\n=== ANALYSIS COMPLETE ===")
            
            # Ask user if they want to clear dimensions
            result = messagebox.askyesnocancel(
                "Dimension Analysis",
                f"Found {analysis['total_count']} dimensions in the drawing.\n\n"
                f"Would you like to:\n"
                f"• YES - Clear ALL dimensions\n"
                f"• NO - Keep existing dimensions\n"
                f"• CANCEL - Close this dialog"
            )
                
            if result is True:  # User clicked Yes
                self.clear_dimensions()
            elif result is False:  # User clicked No
                self.log_message("Keeping existing dimensions")
                
        except Exception as e:
            self.log_message(f"Error analyzing dimensions: {e}")
            messagebox.showerror("Error", f"Error analyzing dimensions:\n\n{e}")
        finally:
            self.hide_spinner('analyze')
            self.analyze_btn.configure(state="normal")
            
    def clear_dimensions(self):
        """Clear all dimensions."""
        if not self.dimension_service:
            messagebox.showerror("Error", "Not connected to AutoCAD")
            return
            
        if messagebox.askyesno("Confirm", "Clear all dimensions? This cannot be undone."):
            self.show_spinner('clear')
            self.clear_btn.configure(state="disabled")
            
            try:
                self.log_message("Clearing all dimensions...")
                count = self.dimension_service.clear_all_dimensions()
                self.log_message(f"Cleared {count} dimensions")
                messagebox.showinfo("Success", f"Successfully cleared {count} dimensions")
            except Exception as e:
                self.log_message(f"Error clearing dimensions: {e}")
                messagebox.showerror("Error", f"Error clearing dimensions:\n\n{e}")
            finally:
                self.hide_spinner('clear')
                self.clear_btn.configure(state="normal")
                
    def check_compliance(self):
        """Check building code compliance."""
        if not self.compliance_service:
            messagebox.showerror("Error", "Not connected to AutoCAD")
            return
            
        # Show spinner
        self.show_spinner('check')
        self.check_compliance_btn.configure(state="disabled")
        
        try:
            self.log_message("Starting compliance checking...")
            self.log_message("Checking all available categories...")
            
            # Check all categories
            categories = [RuleCategory.FIRE_SAFETY, RuleCategory.ACCESSIBILITY, RuleCategory.STRUCTURAL]
            violations = self.compliance_service.check_compliance(categories)
            
            self.log_message(f"Compliance check completed!")
            self.log_message(f"Found {len(violations)} violations")
            
            if violations:
                for violation in violations[:3]:  # Show first 3 violations
                    severity_icon = "🔴" if violation.severity.value == "high" else "🟡" if violation.severity.value == "medium" else "🟢"
                    self.log_message(f"{severity_icon} {violation.rule_title}: {violation.description}")
                    
                if len(violations) > 3:
                    self.log_message(f"... and {len(violations) - 3} more violations")
                    
                report = self.compliance_service.generate_compliance_report(violations)
                self.log_message(f"Summary: {report['summary']['high_severity']} high, {report['summary']['medium_severity']} medium, {report['summary']['low_severity']} low severity")
            else:
                self.log_message("No compliance violations found!")
                
        except Exception as e:
            error_msg = f"Error during compliance checking: {e}"
            self.log_message(error_msg)
            messagebox.showerror("Compliance Check Error", str(e))
        finally:
            self.hide_spinner('check')
            self.check_compliance_btn.configure(state="normal")
            
    def load_rules_from_pdf(self):
        """Load building code rules from PDF file."""
        if not self.compliance_service:
            messagebox.showerror("Error", "Not connected to AutoCAD")
            return
            
        # Show spinner
        self.show_spinner('load')
        self.load_rules_btn.configure(state="disabled")
        
        try:
            from tkinter import filedialog
            
            # Open file dialog
            file_path = filedialog.askopenfilename(
                title="Select Building Code PDF",
                filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")]
            )
            
            if file_path:
                self.log_message(f"Loading rules from: {file_path}")
                self.log_message("Parsing PDF and extracting building code rules...")
                
                # Here you would implement PDF parsing logic
                # For now, show a placeholder message
                self.log_message("PDF parsing functionality not yet implemented")
                messagebox.showinfo("Info", "PDF rule loading will be implemented in a future version.")
            else:
                self.log_message("No file selected")
                
        except Exception as e:
            error_msg = f"Error loading rules from PDF: {e}"
            self.log_message(error_msg)
            messagebox.showerror("PDF Load Error", str(e))
        finally:
            self.hide_spinner('load')
            self.load_rules_btn.configure(state="normal")
        
    def log_message(self, message: str):
        """Add a message to the activity log with professional formatting."""
        import datetime
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        
        # Clean professional log formatting with proper newlines
        if message.startswith("==="):
            # Section headers
            formatted_message = f"\n{'─' * 50}\n{message}\n{'─' * 50}\n"
        elif "✅" in message or "❌" in message:
            # Status messages with proper indentation  
            formatted_message = f"[{timestamp}] {message}\n"
        elif message.startswith("  •"):
            # Bullet points with indentation
            formatted_message = f"[{timestamp}] {message}\n"
        elif "error" in message.lower() or "failed" in message.lower():
            # Error messages
            formatted_message = f"[{timestamp}] ❌ ERROR: {message}\n"
        elif "success" in message.lower() or "completed" in message.lower() or "connected" in message.lower():
            # Success messages
            formatted_message = f"[{timestamp}] ✅ {message}\n"
        elif "warning" in message.lower():
            # Warning messages
            formatted_message = f"[{timestamp}] ⚠️  {message}\n"
        else:
            # Regular messages
            formatted_message = f"[{timestamp}] ℹ️  {message}\n"
            
        self.results_text.insert("end", formatted_message)
        self.results_text.see("end")
        
    def clear_log(self):
        """Clear the activity log."""
        self.results_text.delete("1.0", "end")
        self.log_message("=== Log Cleared ===")
        
    def run(self):
        """Run the application."""
        self.root.mainloop()
        
    def show_spinner(self, spinner_name):
        """Show specific spinner."""
        spinners = {
            'add': getattr(self, 'add_spinner', None),
            'analyze': getattr(self, 'analyze_spinner', None),
            'clear': getattr(self, 'clear_spinner', None),
            'check': getattr(self, 'check_spinner', None),
            'load': getattr(self, 'load_spinner', None)
        }
        if spinner_name in spinners and spinners[spinner_name] is not None:
            spinners[spinner_name].pack(pady=(2, 0))
            spinners[spinner_name].start()
            self.root.update()
            
    def hide_spinner(self, spinner_name):
        """Hide specific spinner."""
        spinners = {
            'add': getattr(self, 'add_spinner', None),
            'analyze': getattr(self, 'analyze_spinner', None),
            'clear': getattr(self, 'clear_spinner', None),
            'check': getattr(self, 'check_spinner', None),
            'load': getattr(self, 'load_spinner', None)
        }
        if spinner_name in spinners and spinners[spinner_name] is not None:
            spinners[spinner_name].stop()
            spinners[spinner_name].pack_forget()
            
    def enable_buttons(self):
        """Enable all buttons after connection."""
        try:
            self.dimension_btn.configure(state="normal")
            self.analyze_btn.configure(state="normal")
            self.clear_btn.configure(state="normal")
            self.check_compliance_btn.configure(state="normal")
            self.load_rules_btn.configure(state="normal")
        except:
            pass  # Buttons may not exist yet
    
    def _add_hover_hint(self, widget, text):
        """Add a hover hint to a widget."""
        tooltip = None
        
        def show_hint(event):
            nonlocal tooltip
            x, y = widget.winfo_rootx() + 10, widget.winfo_rooty() + widget.winfo_height() + 5
            tooltip = Toplevel(self.root)
            tooltip.wm_overrideredirect(True)
            tooltip.wm_geometry(f"+{x}+{y}")
            
            hint_label = Label(
                tooltip,
                text=text,
                background="#ffffe0",
                foreground="#000000",
                relief="solid",
                borderwidth=1,
                font=("Segoe UI", 11, "normal"),
                padx=8,
                pady=4
            )
            hint_label.pack()
            
        def hide_hint(event):
            nonlocal tooltip
            if tooltip:
                tooltip.destroy()
                tooltip = None
                
        widget.bind("<Enter>", show_hint)
        widget.bind("<Leave>", hide_hint)
        
    def __del__(self):
        """Cleanup when closing."""
        if self.connection:
            self.connection.disconnect()


if __name__ == "__main__":
    app = MainWindow()
    app.run()