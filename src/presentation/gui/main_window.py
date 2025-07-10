"""Modern GUI window for the AutoCAD Construction Toolkit."""
import customtkinter as ctk
from tkinter import messagebox
import logging
import threading
from typing import Optional

from src.infrastructure.autocad.connection import AutoCADConnection
from src.application.services.dimension_service import DimensionService
from src.application.services.compliance_service import ComplianceService, RuleCategory

logger = logging.getLogger(__name__)

# Set appearance mode and color theme
ctk.set_appearance_mode("light")  # Clean light mode for professional appearance
ctk.set_default_color_theme("blue")

# Google Material Design inspired colors and typography
COLORS = {
    'primary': '#1976d2',
    'primary_light': '#63a4ff', 
    'primary_dark': '#004ba0',
    'secondary': '#757575',
    'surface': '#f8f9fa',
    'background': '#ffffff',
    'text': '#212121',
    'text_secondary': '#616161',
    'success': '#4caf50',
    'warning': '#ff9800',
    'error': '#f44336',
    'divider': '#e1e3e4'
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
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the clean, modern user interface."""
        self.root.title("AutoCAD Construction Toolkit")
        self.root.geometry("1000x800")
        self.root.minsize(900, 700)
        
        # Main container
        main_container = ctk.CTkFrame(self.root, fg_color=COLORS['background'])
        main_container.pack(fill="both", expand=True)
        
        # Header
        self.create_header(main_container)
        
        # Content area
        content_area = ctk.CTkScrollableFrame(
            main_container, 
            fg_color=COLORS['background'],
            corner_radius=0
        )
        content_area.pack(fill="both", expand=True, padx=24, pady=(0, 24))
        
        # Connection section
        self.create_connection_section(content_area)
        
        # Dimensioning section
        self.create_dimensioning_section(content_area)
        
        # Compliance section
        self.create_compliance_section(content_area)
        
        # Activity log
        self.create_activity_section(content_area)
        
    def create_header(self, parent):
        """Create clean header section."""
        header = ctk.CTkFrame(parent, fg_color=COLORS['surface'], corner_radius=0)
        header.pack(fill="x")
        
        # Title
        title = ctk.CTkLabel(
            header,
            text="AutoCAD Construction Toolkit",
            font=ctk.CTkFont(*FONTS['heading']),
            text_color=COLORS['text']
        )
        title.pack(anchor="w", padx=32, pady=(32, 8))
        
        # Subtitle
        subtitle = ctk.CTkLabel(
            header,
            text="Professional automated dimensioning and building code compliance",
            font=ctk.CTkFont(*FONTS['body']),
            text_color=COLORS['text_secondary']
        )
        subtitle.pack(anchor="w", padx=32, pady=(0, 24))
        
    def create_connection_section(self, parent):
        """Create connection section."""
        section = self.create_section(parent, "AutoCAD Connection")
        
        # Status container
        status_frame = ctk.CTkFrame(section, fg_color="transparent")
        status_frame.pack(fill="x", padx=24, pady=(0, 16))
        
        self.status_indicator = ctk.CTkLabel(
            status_frame,
            text="●",
            font=ctk.CTkFont(size=16),
            text_color=COLORS['error']
        )
        self.status_indicator.pack(side="left", padx=(0, 8))
        
        self.status_label = ctk.CTkLabel(
            status_frame,
            text="Not Connected",
            font=ctk.CTkFont(*FONTS['body']),
            text_color=COLORS['text_secondary']
        )
        self.status_label.pack(side="left")
        
        # Document selector (initially hidden)
        self.document_frame = ctk.CTkFrame(section, fg_color="transparent")
        self.document_frame.pack(fill="x", padx=24, pady=(0, 16))
        
        document_label = ctk.CTkLabel(
            self.document_frame,
            text="Active Drawing:",
            font=ctk.CTkFont(*FONTS['body']),
            text_color=COLORS['text']
        )
        document_label.pack(anchor="w", pady=(0, 8))
        
        self.document_dropdown = ctk.CTkComboBox(
            self.document_frame,
            values=["No documents available"],
            command=self.on_document_selected,
            font=ctk.CTkFont(*FONTS['body']),
            height=36,
            state="disabled"
        )
        self.document_dropdown.pack(fill="x", pady=(0, 8))
        
        # Initially hide the document selector
        self.document_frame.pack_forget()
        
        # Connect button
        self.connect_btn = ctk.CTkButton(
            section,
            text="Connect to AutoCAD",
            command=self.connect_autocad,
            font=ctk.CTkFont(*FONTS['button']),
            height=40,
            corner_radius=8,
            fg_color=COLORS['primary'],
            hover_color=COLORS['primary_dark']
        )
        self.connect_btn.pack(anchor="w", padx=24, pady=(0, 24))
        
    def create_dimensioning_section(self, parent):
        """Create dimensioning section."""
        section = self.create_section(parent, "Automatic Dimensioning")
        
        # Settings grid
        settings_grid = ctk.CTkFrame(section, fg_color="transparent")
        settings_grid.pack(fill="x", padx=24, pady=(0, 16))
        settings_grid.grid_columnconfigure((0, 1), weight=1)
        
        # Layer filter
        self.create_input_field(
            settings_grid, "Layer Filter", 
            "e.g., WALL, BEAM (leave empty for all layers)",
            row=0, column=0
        )
        self.layer_var = ctk.StringVar(value="")
        self.layer_entry = ctk.CTkEntry(
            settings_grid,
            textvariable=self.layer_var,
            placeholder_text="e.g., WALL, BEAM",
            font=ctk.CTkFont(*FONTS['body']),
            height=40
        )
        self.layer_entry.grid(row=1, column=0, sticky="ew", padx=(0, 8), pady=(4, 0))
        
        # Offset distance
        self.create_input_field(
            settings_grid, "Offset Distance (mm)", 
            "Distance from geometry",
            row=0, column=1
        )
        self.offset_var = ctk.StringVar(value="500")
        self.offset_entry = ctk.CTkEntry(
            settings_grid,
            textvariable=self.offset_var,
            placeholder_text="500",
            font=ctk.CTkFont(*FONTS['body']),
            height=40
        )
        self.offset_entry.grid(row=1, column=1, sticky="ew", padx=(8, 0), pady=(4, 0))
        
        # Action buttons
        button_frame = ctk.CTkFrame(section, fg_color="transparent")
        button_frame.pack(fill="x", padx=24, pady=(16, 0))
        
        self.dimension_btn = ctk.CTkButton(
            button_frame,
            text="Add Dimensions",
            command=self.add_dimensions,
            state="disabled",
            font=ctk.CTkFont(*FONTS['button']),
            height=44,
            corner_radius=8,
            fg_color=COLORS['success'],
            hover_color="#388e3c"
        )
        self.dimension_btn.pack(side="left", padx=(0, 12))
        
        self.clear_btn = ctk.CTkButton(
            button_frame,
            text="Clear All",
            command=self.clear_dimensions,
            state="disabled",
            font=ctk.CTkFont(*FONTS['button']),
            height=44,
            corner_radius=8,
            fg_color=COLORS['error'],
            hover_color="#d32f2f"
        )
        self.clear_btn.pack(side="left")
        
        # Progress bar
        self.progress = ctk.CTkProgressBar(
            section,
            mode="indeterminate",
            height=4,
            corner_radius=2,
            fg_color=COLORS['divider'],
            progress_color=COLORS['primary']
        )
        self.progress.pack(fill="x", padx=24, pady=(16, 24))
        self.progress.set(0)
        
    def create_compliance_section(self, parent):
        """Create compliance checking section."""
        section = self.create_section(parent, "Building Code Compliance")
        
        # Category selection
        categories_frame = ctk.CTkFrame(section, fg_color="transparent")
        categories_frame.pack(fill="x", padx=24, pady=(0, 16))
        
        ctk.CTkLabel(
            categories_frame,
            text="Check Categories:",
            font=ctk.CTkFont(*FONTS['body']),
            text_color=COLORS['text']
        ).pack(anchor="w", pady=(0, 8))
        
        checkbox_frame = ctk.CTkFrame(categories_frame, fg_color="transparent")
        checkbox_frame.pack(fill="x")
        
        self.fire_safety_var = ctk.BooleanVar(value=True)
        self.fire_safety_cb = ctk.CTkCheckBox(
            checkbox_frame,
            text="Fire Safety",
            variable=self.fire_safety_var,
            font=ctk.CTkFont(*FONTS['body']),
            checkbox_width=20,
            checkbox_height=20
        )
        self.fire_safety_cb.pack(side="left", padx=(0, 24))
        
        self.accessibility_var = ctk.BooleanVar(value=True)
        self.accessibility_cb = ctk.CTkCheckBox(
            checkbox_frame,
            text="Accessibility (ADA)",
            variable=self.accessibility_var,
            font=ctk.CTkFont(*FONTS['body']),
            checkbox_width=20,
            checkbox_height=20
        )
        self.accessibility_cb.pack(side="left", padx=(0, 24))
        
        self.structural_var = ctk.BooleanVar(value=False)
        self.structural_cb = ctk.CTkCheckBox(
            checkbox_frame,
            text="Structural",
            variable=self.structural_var,
            font=ctk.CTkFont(*FONTS['body']),
            checkbox_width=20,
            checkbox_height=20
        )
        self.structural_cb.pack(side="left")
        
        # Action buttons
        button_frame = ctk.CTkFrame(section, fg_color="transparent")
        button_frame.pack(fill="x", padx=24, pady=(16, 0))
        
        self.compliance_btn = ctk.CTkButton(
            button_frame,
            text="Check Compliance",
            command=self.check_compliance,
            state="disabled",
            font=ctk.CTkFont(*FONTS['button']),
            height=44,
            corner_radius=8,
            fg_color=COLORS['warning'],
            hover_color="#f57c00"
        )
        self.compliance_btn.pack(side="left")
        
        # Compliance progress
        self.compliance_progress = ctk.CTkProgressBar(
            section,
            mode="indeterminate",
            height=4,
            corner_radius=2,
            fg_color=COLORS['divider'],
            progress_color=COLORS['warning']
        )
        self.compliance_progress.pack(fill="x", padx=24, pady=(16, 24))
        self.compliance_progress.set(0)
        
    def create_activity_section(self, parent):
        """Create activity log section."""
        section = self.create_section(parent, "Activity Log")
        
        self.results_text = ctk.CTkTextbox(
            section,
            font=ctk.CTkFont(*FONTS['mono']),
            corner_radius=8,
            wrap="word",
            height=200,
            fg_color=COLORS['surface']
        )
        self.results_text.pack(fill="both", expand=True, padx=24, pady=(0, 24))
        
        # Welcome message
        self.log_message("Welcome to AutoCAD Construction Toolkit")
        self.log_message("Connect to AutoCAD to start automated dimensioning and compliance checking")
        
    def create_section(self, parent, title):
        """Create a clean section card."""
        section = ctk.CTkFrame(parent, fg_color=COLORS['surface'], corner_radius=12)
        section.pack(fill="x", pady=(0, 16))
        
        # Section title
        title_label = ctk.CTkLabel(
            section,
            text=title,
            font=ctk.CTkFont(*FONTS['title']),
            text_color=COLORS['text']
        )
        title_label.pack(anchor="w", padx=24, pady=(20, 12))
        
        return section
        
    def create_input_field(self, parent, label, placeholder, row, column):
        """Create an input field with label."""
        label_widget = ctk.CTkLabel(
            parent,
            text=label,
            font=ctk.CTkFont(*FONTS['body']),
            text_color=COLORS['text']
        )
        label_widget.grid(row=row, column=column, sticky="w", padx=(0 if column == 0 else 8, 8 if column == 0 else 0))
        
    def connect_autocad(self):
        """Connect to AutoCAD."""
        self.log_message("Attempting to connect to AutoCAD...")
        try:
            self.connection = AutoCADConnection()
            if self.connection.connect():
                # Update UI
                self.status_label.configure(text=f"Connected to {self.connection.doc.Name}")
                self.status_indicator.configure(text_color=COLORS['success'])
                
                # Get and populate open documents
                self.update_document_list()
                
                # Show document selector
                self.document_frame.pack(fill="x", padx=24, pady=(0, 16))
                
                self.dimension_service = DimensionService(self.connection)
                self.compliance_service = ComplianceService(self.connection)
                self.dimension_btn.configure(state="normal")
                self.clear_btn.configure(state="normal")
                self.compliance_btn.configure(state="normal")
                self.connect_btn.configure(text="Refresh Connection")
                self.log_message(f"Connected successfully! Drawing: {self.connection.doc.Name}")
                self.log_message(f"Compliance service initialized with {len(self.compliance_service.rules)} rules")
                
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
        if not self.connection or selected_document == "No documents open":
            return
            
        try:
            if self.connection.switch_to_document(selected_document):
                # Update services to use new document
                self.dimension_service = DimensionService(self.connection)
                self.compliance_service = ComplianceService(self.connection)
                
                # Update status
                self.status_label.configure(text=f"Connected to {selected_document}")
                self.log_message(f"Switched to drawing: {selected_document}")
                self.log_message(f"Services reinitialized for new drawing")
            else:
                self.log_message(f"Failed to switch to: {selected_document}")
                messagebox.showerror("Error", f"Failed to switch to drawing: {selected_document}")
        except Exception as e:
            self.log_message(f"Error switching documents: {e}")
            messagebox.showerror("Error", f"Error switching documents:\\n\\n{e}")
            
    def add_dimensions(self):
        """Add dimensions to lines."""
        if not self.dimension_service:
            messagebox.showerror("Error", "Not connected to AutoCAD")
            return
            
        try:
            self.dimension_service.config['offset_distance'] = float(self.offset_var.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid offset distance. Please enter a numeric value.")
            return
            
        threading.Thread(target=self._dimension_worker, daemon=True).start()
        
    def _dimension_worker(self):
        """Worker thread for dimensioning."""
        try:
            self.root.after(0, self._start_progress)
            
            layer_filter = self.layer_var.get().strip() if self.layer_var.get().strip() else None
            
            self.root.after(0, self.log_message, "Starting dimensioning process...")
            if layer_filter:
                self.root.after(0, self.log_message, f"Filtering by layer: {layer_filter}")
            else:
                self.root.after(0, self.log_message, "Processing all layers")
                
            results = self.dimension_service.dimension_all_lines(layer_filter)
            
            self.root.after(0, self.log_message, "Dimensioning completed!")
            self.root.after(0, self.log_message, f"Lines dimensioned: {results['lines']}")
            self.root.after(0, self.log_message, f"Total dimensions: {results['total']}")
            
            if self.connection and self.connection.acad:
                self.connection.acad.app.ZoomExtents()
                self.root.after(0, self.log_message, "Zoomed to extents")
                
        except Exception as e:
            error_msg = f"Error during dimensioning: {e}"
            self.root.after(0, self.log_message, error_msg)
            self.root.after(0, messagebox.showerror, "Dimensioning Error", str(e))
        finally:
            self.root.after(0, self._stop_progress)
            
    def clear_dimensions(self):
        """Clear all dimensions."""
        if not self.dimension_service:
            messagebox.showerror("Error", "Not connected to AutoCAD")
            return
            
        if messagebox.askyesno("Confirm", "Clear all dimensions? This cannot be undone."):
            try:
                self.log_message("Clearing all dimensions...")
                count = self.dimension_service.clear_all_dimensions()
                self.log_message(f"Cleared {count} dimensions")
                messagebox.showinfo("Success", f"Successfully cleared {count} dimensions")
            except Exception as e:
                self.log_message(f"Error clearing dimensions: {e}")
                messagebox.showerror("Error", f"Error clearing dimensions:\\n\\n{e}")
                
    def check_compliance(self):
        """Check building code compliance."""
        if not self.compliance_service:
            messagebox.showerror("Error", "Not connected to AutoCAD")
            return
            
        categories = []
        if self.fire_safety_var.get():
            categories.append(RuleCategory.FIRE_SAFETY)
        if self.accessibility_var.get():
            categories.append(RuleCategory.ACCESSIBILITY)
        if self.structural_var.get():
            categories.append(RuleCategory.STRUCTURAL)
            
        if not categories:
            messagebox.showwarning("Warning", "Please select at least one compliance category.")
            return
            
        threading.Thread(target=self._compliance_worker, args=(categories,), daemon=True).start()
        
    def _compliance_worker(self, categories):
        """Worker thread for compliance checking."""
        try:
            self.root.after(0, self._start_compliance_progress)
            
            self.root.after(0, self.log_message, "Starting compliance checking...")
            self.root.after(0, self.log_message, f"Checking categories: {', '.join([c.value for c in categories])}")
            
            violations = self.compliance_service.check_compliance(categories)
            
            self.root.after(0, self.log_message, f"Compliance check completed!")
            self.root.after(0, self.log_message, f"Found {len(violations)} violations")
            
            if violations:
                for violation in violations[:3]:  # Show first 3 violations
                    severity_icon = "🔴" if violation.severity.value == "high" else "🟡" if violation.severity.value == "medium" else "🟢"
                    self.root.after(0, self.log_message, f"{severity_icon} {violation.rule_title}: {violation.description}")
                    
                if len(violations) > 3:
                    self.root.after(0, self.log_message, f"... and {len(violations) - 3} more violations")
                    
                report = self.compliance_service.generate_compliance_report(violations)
                self.root.after(0, self.log_message, f"Summary: {report['summary']['high_severity']} high, {report['summary']['medium_severity']} medium, {report['summary']['low_severity']} low severity")
            else:
                self.root.after(0, self.log_message, "No compliance violations found!")
                
        except Exception as e:
            error_msg = f"Error during compliance checking: {e}"
            self.root.after(0, self.log_message, error_msg)
            self.root.after(0, messagebox.showerror, "Compliance Check Error", str(e))
        finally:
            self.root.after(0, self._stop_compliance_progress)
            
    def _start_progress(self):
        """Start progress bar."""
        self.progress.start()
        self.dimension_btn.configure(state="disabled")
        self.clear_btn.configure(state="disabled")
        
    def _stop_progress(self):
        """Stop progress bar."""
        self.progress.stop()
        self.progress.set(0)
        self.dimension_btn.configure(state="normal")
        self.clear_btn.configure(state="normal")
        
    def _start_compliance_progress(self):
        """Start compliance progress bar."""
        self.compliance_progress.start()
        self.compliance_btn.configure(state="disabled")
        
    def _stop_compliance_progress(self):
        """Stop compliance progress bar."""
        self.compliance_progress.stop()
        self.compliance_progress.set(0)
        self.compliance_btn.configure(state="normal")
        
    def log_message(self, message: str):
        """Add a message to the activity log."""
        import datetime
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}\\n"
        self.results_text.insert("end", formatted_message)
        self.results_text.see("end")
        
    def run(self):
        """Run the application."""
        self.root.mainloop()
        
    def __del__(self):
        """Cleanup when closing."""
        if self.connection:
            self.connection.disconnect()


if __name__ == "__main__":
    app = MainWindow()
    app.run()