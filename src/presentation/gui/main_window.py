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
ctk.set_appearance_mode("dark")  # Modes: "System" (standard), "Dark", "Light"
ctk.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

# Google-style modern fonts
PRIMARY_FONT = "Segoe UI"  # Clean, modern system font (fallback for Google fonts)
SECONDARY_FONT = "Segoe UI"
MONO_FONT = "Consolas"


class MainWindow:
    """Modern main application window."""
    
    def __init__(self):
        self.root = ctk.CTk()
        self.connection: Optional[AutoCADConnection] = None
        self.dimension_service: Optional[DimensionService] = None
        self.compliance_service: Optional[ComplianceService] = None
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the modern user interface."""
        self.root.title("AutoCAD Construction Toolkit")
        self.root.geometry("800x700")
        self.root.minsize(700, 600)
        
        # Configure grid layout
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(2, weight=1)
        
        # Header
        header_frame = ctk.CTkFrame(self.root, corner_radius=15)
        header_frame.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="ew")
        header_frame.grid_columnconfigure(0, weight=1)
        
        title_label = ctk.CTkLabel(
            header_frame,
            text="🏗️ AutoCAD Construction Toolkit",
            font=ctk.CTkFont(family=PRIMARY_FONT, size=26, weight="normal")
        )
        title_label.grid(row=0, column=0, padx=20, pady=15)
        
        subtitle_label = ctk.CTkLabel(
            header_frame,
            text="Professional automated dimensioning for construction drawings",
            font=ctk.CTkFont(family=PRIMARY_FONT, size=13, weight="normal"),
            text_color="gray60"
        )
        subtitle_label.grid(row=1, column=0, padx=20, pady=(0, 15))
        
        # Connection Panel
        conn_frame = ctk.CTkFrame(self.root, corner_radius=15)
        conn_frame.grid(row=1, column=0, padx=20, pady=10, sticky="ew")
        conn_frame.grid_columnconfigure(1, weight=1)
        
        conn_title = ctk.CTkLabel(
            conn_frame,
            text="🔗 AutoCAD Connection",
            font=ctk.CTkFont(family=PRIMARY_FONT, size=16, weight="normal")
        )
        conn_title.grid(row=0, column=0, columnspan=3, padx=20, pady=(15, 10), sticky="w")
        
        # Status indicator
        self.status_frame = ctk.CTkFrame(conn_frame, corner_radius=10)
        self.status_frame.grid(row=1, column=0, padx=(20, 10), pady=10, sticky="ew")
        
        self.status_indicator = ctk.CTkLabel(
            self.status_frame,
            text="●",
            font=ctk.CTkFont(size=20),
            text_color="red"
        )
        self.status_indicator.grid(row=0, column=0, padx=10, pady=10)
        
        self.status_label = ctk.CTkLabel(
            self.status_frame,
            text="Not Connected",
            font=ctk.CTkFont(family=PRIMARY_FONT, size=13, weight="normal")
        )
        self.status_label.grid(row=0, column=1, padx=(0, 10), pady=10, sticky="w")
        
        # Connect button
        self.connect_btn = ctk.CTkButton(
            conn_frame,
            text="Connect to AutoCAD",
            command=self.connect_autocad,
            font=ctk.CTkFont(family=PRIMARY_FONT, size=13, weight="normal"),
            height=40,
            corner_radius=8,
            fg_color="#4a90e2",
            hover_color="#357abd"
        )
        self.connect_btn.grid(row=1, column=2, padx=20, pady=10)
        
        # Main Controls Panel
        main_frame = ctk.CTkFrame(self.root, corner_radius=15)
        main_frame.grid(row=2, column=0, padx=20, pady=10, sticky="nsew")
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_rowconfigure(3, weight=1)
        
        # Controls title
        controls_title = ctk.CTkLabel(
            main_frame,
            text="⚙️ Automatic Dimensioning",
            font=ctk.CTkFont(family=PRIMARY_FONT, size=16, weight="normal")
        )
        controls_title.grid(row=0, column=0, padx=20, pady=(15, 10), sticky="w")
        
        # Settings Panel
        settings_frame = ctk.CTkFrame(main_frame, corner_radius=10)
        settings_frame.grid(row=1, column=0, padx=20, pady=10, sticky="ew")
        settings_frame.grid_columnconfigure((0, 1, 2), weight=1)
        
        # Layer filter
        layer_label = ctk.CTkLabel(
            settings_frame,
            text="Layer Filter:",
            font=ctk.CTkFont(family=PRIMARY_FONT, size=13, weight="normal")
        )
        layer_label.grid(row=0, column=0, padx=15, pady=15, sticky="w")
        
        self.layer_var = ctk.StringVar(value="")
        self.layer_entry = ctk.CTkEntry(
            settings_frame,
            textvariable=self.layer_var,
            placeholder_text="e.g., WALL, BEAM (leave empty for all layers)",
            font=ctk.CTkFont(family=PRIMARY_FONT, size=12, weight="normal"),
            height=35
        )
        self.layer_entry.grid(row=0, column=1, columnspan=2, padx=15, pady=15, sticky="ew")
        
        # Offset distance
        offset_label = ctk.CTkLabel(
            settings_frame,
            text="Offset Distance:",
            font=ctk.CTkFont(family=PRIMARY_FONT, size=13, weight="normal")
        )
        offset_label.grid(row=1, column=0, padx=15, pady=(0, 15), sticky="w")
        
        self.offset_var = ctk.StringVar(value="500")
        self.offset_entry = ctk.CTkEntry(
            settings_frame,
            textvariable=self.offset_var,
            placeholder_text="500",
            font=ctk.CTkFont(family=PRIMARY_FONT, size=12, weight="normal"),
            width=120,
            height=35
        )
        self.offset_entry.grid(row=1, column=1, padx=15, pady=(0, 15), sticky="w")
        
        # Min length
        min_length_label = ctk.CTkLabel(
            settings_frame,
            text="Min Length:",
            font=ctk.CTkFont(family=PRIMARY_FONT, size=13, weight="normal")
        )
        min_length_label.grid(row=1, column=1, padx=(160, 15), pady=(0, 15), sticky="w")
        
        self.min_length_var = ctk.StringVar(value="100")
        self.min_length_entry = ctk.CTkEntry(
            settings_frame,
            textvariable=self.min_length_var,
            placeholder_text="100",
            font=ctk.CTkFont(family=PRIMARY_FONT, size=12, weight="normal"),
            width=120,
            height=35
        )
        self.min_length_entry.grid(row=1, column=2, padx=15, pady=(0, 15), sticky="w")
        
        # Action buttons
        btn_frame = ctk.CTkFrame(main_frame, corner_radius=10)
        btn_frame.grid(row=2, column=0, padx=20, pady=10, sticky="ew")
        btn_frame.grid_columnconfigure((0, 1, 2), weight=1)
        
        self.dimension_btn = ctk.CTkButton(
            btn_frame,
            text="🔧 Add Dimensions",
            command=self.add_dimensions,
            state="disabled",
            font=ctk.CTkFont(family=PRIMARY_FONT, size=14, weight="normal"),
            height=45,
            corner_radius=8,
            fg_color="#34c759",
            hover_color="#30b750"
        )
        self.dimension_btn.grid(row=0, column=0, padx=15, pady=15, sticky="ew")
        
        self.clear_btn = ctk.CTkButton(
            btn_frame,
            text="🗑️ Clear Dimensions",
            command=self.clear_dimensions,
            state="disabled",
            font=ctk.CTkFont(family=PRIMARY_FONT, size=14, weight="normal"),
            height=45,
            corner_radius=8,
            fg_color="#ff6b6b",
            hover_color="#ff5252"
        )
        self.clear_btn.grid(row=0, column=1, padx=15, pady=15, sticky="ew")
        
        # Progress bar
        self.progress = ctk.CTkProgressBar(
            btn_frame,
            mode="indeterminate",
            height=20,
            corner_radius=10
        )
        self.progress.grid(row=1, column=0, columnspan=2, padx=15, pady=(0, 15), sticky="ew")
        self.progress.set(0)
        
        # Results Panel
        results_frame = ctk.CTkFrame(main_frame, corner_radius=10)
        results_frame.grid(row=3, column=0, padx=20, pady=(0, 20), sticky="nsew")
        results_frame.grid_columnconfigure(0, weight=1)
        results_frame.grid_rowconfigure(1, weight=1)
        
        results_title = ctk.CTkLabel(
            results_frame,
            text="📊 Activity Log",
            font=ctk.CTkFont(family=PRIMARY_FONT, size=14, weight="normal")
        )
        results_title.grid(row=0, column=0, padx=15, pady=(15, 10), sticky="w")
        
        self.results_text = ctk.CTkTextbox(
            results_frame,
            font=ctk.CTkFont(family=MONO_FONT, size=11, weight="normal"),
            corner_radius=8,
            wrap="word"
        )
        self.results_text.grid(row=1, column=0, padx=15, pady=(0, 15), sticky="nsew")
        
        # Compliance Checking Panel
        compliance_frame = ctk.CTkFrame(self.root, corner_radius=15)
        compliance_frame.grid(row=3, column=0, padx=20, pady=10, sticky="ew")
        compliance_frame.grid_columnconfigure(0, weight=1)
        
        # Compliance title
        compliance_title = ctk.CTkLabel(
            compliance_frame,
            text="🏛️ Building Code Compliance",
            font=ctk.CTkFont(family=PRIMARY_FONT, size=16, weight="normal")
        )
        compliance_title.grid(row=0, column=0, padx=20, pady=(15, 10), sticky="w")
        
        # Compliance settings
        compliance_settings_frame = ctk.CTkFrame(compliance_frame, corner_radius=10)
        compliance_settings_frame.grid(row=1, column=0, padx=20, pady=10, sticky="ew")
        compliance_settings_frame.grid_columnconfigure((0, 1), weight=1)
        
        # Rule categories
        categories_label = ctk.CTkLabel(
            compliance_settings_frame,
            text="Check Categories:",
            font=ctk.CTkFont(family=PRIMARY_FONT, size=13, weight="normal")
        )
        categories_label.grid(row=0, column=0, padx=15, pady=15, sticky="w")
        
        # Category checkboxes
        self.fire_safety_var = ctk.BooleanVar(value=True)
        self.fire_safety_cb = ctk.CTkCheckBox(
            compliance_settings_frame,
            text="Fire Safety",
            variable=self.fire_safety_var,
            font=ctk.CTkFont(family=PRIMARY_FONT, size=12, weight="normal")
        )
        self.fire_safety_cb.grid(row=0, column=1, padx=15, pady=15, sticky="w")
        
        self.accessibility_var = ctk.BooleanVar(value=True)
        self.accessibility_cb = ctk.CTkCheckBox(
            compliance_settings_frame,
            text="Accessibility (ADA)",
            variable=self.accessibility_var,
            font=ctk.CTkFont(family=PRIMARY_FONT, size=12, weight="normal")
        )
        self.accessibility_cb.grid(row=1, column=1, padx=15, pady=(0, 15), sticky="w")
        
        self.structural_var = ctk.BooleanVar(value=False)
        self.structural_cb = ctk.CTkCheckBox(
            compliance_settings_frame,
            text="Structural",
            variable=self.structural_var,
            font=ctk.CTkFont(family=PRIMARY_FONT, size=12, weight="normal")
        )
        self.structural_cb.grid(row=2, column=1, padx=15, pady=(0, 15), sticky="w")
        
        # PDF upload option
        pdf_label = ctk.CTkLabel(
            compliance_settings_frame,
            text="Load Rules from PDF:",
            font=ctk.CTkFont(family=PRIMARY_FONT, size=13, weight="normal")
        )
        pdf_label.grid(row=1, column=0, padx=15, pady=(0, 15), sticky="w")
        
        self.pdf_path_var = ctk.StringVar(value="")
        self.pdf_entry = ctk.CTkEntry(
            compliance_settings_frame,
            textvariable=self.pdf_path_var,
            placeholder_text="Path to building code PDF (optional)",
            font=ctk.CTkFont(family=PRIMARY_FONT, size=12, weight="normal"),
            height=35
        )
        self.pdf_entry.grid(row=2, column=0, padx=15, pady=(0, 15), sticky="ew")
        
        # Compliance action buttons
        compliance_btn_frame = ctk.CTkFrame(compliance_frame, corner_radius=10)
        compliance_btn_frame.grid(row=2, column=0, padx=20, pady=10, sticky="ew")
        compliance_btn_frame.grid_columnconfigure((0, 1), weight=1)
        
        self.compliance_btn = ctk.CTkButton(
            compliance_btn_frame,
            text="🔍 Check Compliance",
            command=self.check_compliance,
            state="disabled",
            font=ctk.CTkFont(family=PRIMARY_FONT, size=14, weight="normal"),
            height=45,
            corner_radius=8,
            fg_color="#9b59b6",
            hover_color="#8e44ad"
        )
        self.compliance_btn.grid(row=0, column=0, padx=15, pady=15, sticky="ew")
        
        self.load_pdf_btn = ctk.CTkButton(
            compliance_btn_frame,
            text="📄 Load PDF Rules",
            command=self.load_pdf_rules,
            state="disabled",
            font=ctk.CTkFont(family=PRIMARY_FONT, size=14, weight="normal"),
            height=45,
            corner_radius=8,
            fg_color="#3498db",
            hover_color="#2980b9"
        )
        self.load_pdf_btn.grid(row=0, column=1, padx=15, pady=15, sticky="ew")
        
        # Compliance progress bar
        self.compliance_progress = ctk.CTkProgressBar(
            compliance_btn_frame,
            mode="indeterminate",
            height=20,
            corner_radius=10
        )
        self.compliance_progress.grid(row=1, column=0, columnspan=2, padx=15, pady=(0, 15), sticky="ew")
        self.compliance_progress.set(0)
        
        # Welcome message
        self.log_message("🚀 Welcome to AutoCAD Construction Toolkit")
        self.log_message("💡 Connect to AutoCAD to start automatic dimensioning and compliance checking")
        self.log_message("📋 Features: Smart dimensioning, AI compliance checking, building code validation")
        
    def connect_autocad(self):
        """Connect to AutoCAD."""
        self.log_message("🔄 Attempting to connect to AutoCAD...")
        try:
            self.connection = AutoCADConnection()
            if self.connection.connect():
                # Update UI to show connected status
                self.status_label.configure(text=f"Connected to {self.connection.doc.Name}")
                self.status_indicator.configure(text_color="green")
                self.dimension_service = DimensionService(self.connection)
                self.compliance_service = ComplianceService(self.connection)
                self.dimension_btn.configure(state="normal")
                self.clear_btn.configure(state="normal")
                self.compliance_btn.configure(state="normal")
                self.load_pdf_btn.configure(state="normal")
                self.connect_btn.configure(text="Reconnect")
                self.log_message("✅ Connected to AutoCAD successfully!")
                self.log_message(f"📄 Drawing: {self.connection.doc.Name}")
                self.log_message(f"🏛️ Compliance service initialized with {len(self.compliance_service.rules)} rules")
            else:
                self.log_message("❌ Failed to connect to AutoCAD")
                messagebox.showerror("Connection Error", "Failed to connect to AutoCAD.\n\nMake sure AutoCAD is running with a drawing open.")
        except Exception as e:
            self.log_message(f"❌ Connection error: {e}")
            messagebox.showerror("Connection Error", f"Error connecting to AutoCAD:\n\n{e}")
            
    def add_dimensions(self):
        """Add dimensions to lines."""
        if not self.dimension_service:
            messagebox.showerror("Error", "Not connected to AutoCAD")
            return
            
        # Update dimension service config
        try:
            self.dimension_service.config['offset_distance'] = float(self.offset_var.get())
            self.dimension_service.config['min_length'] = float(self.min_length_var.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid offset distance or minimum length.\n\nPlease enter numeric values.")
            return
            
        # Start dimensioning in a separate thread
        threading.Thread(target=self._dimension_worker, daemon=True).start()
        
    def _dimension_worker(self):
        """Worker thread for dimensioning."""
        try:
            self.root.after(0, self._start_progress)
            
            layer_filter = self.layer_var.get().strip() if self.layer_var.get().strip() else None
            
            self.root.after(0, self.log_message, "🚀 Starting dimensioning process...")
            if layer_filter:
                self.root.after(0, self.log_message, f"🎯 Filtering by layer: {layer_filter}")
            else:
                self.root.after(0, self.log_message, "🌐 Processing all layers")
                
            results = self.dimension_service.dimension_all_lines(layer_filter)
            
            self.root.after(0, self.log_message, "🎉 Dimensioning completed!")
            self.root.after(0, self.log_message, f"📏 Lines dimensioned: {results['lines']}")
            self.root.after(0, self.log_message, f"📊 Total dimensions: {results['total']}")
            
            # Zoom to extents
            if self.connection and self.connection.acad:
                self.connection.acad.app.ZoomExtents()
                self.root.after(0, self.log_message, "🔍 Zoomed to extents")
                
        except Exception as e:
            error_msg = f"❌ Error during dimensioning: {e}"
            self.root.after(0, self.log_message, error_msg)
            self.root.after(0, messagebox.showerror, "Dimensioning Error", str(e))
        finally:
            self.root.after(0, self._stop_progress)
            
    def clear_dimensions(self):
        """Clear all dimensions."""
        if not self.dimension_service:
            messagebox.showerror("Error", "Not connected to AutoCAD")
            return
            
        if messagebox.askyesno("Confirm", "Are you sure you want to clear all dimensions?\n\nThis action cannot be undone."):
            try:
                self.log_message("🗑️ Clearing all dimensions...")
                count = self.dimension_service.clear_all_dimensions()
                self.log_message(f"✅ Cleared {count} dimensions")
                messagebox.showinfo("Success", f"Successfully cleared {count} dimensions")
            except Exception as e:
                self.log_message(f"❌ Error clearing dimensions: {e}")
                messagebox.showerror("Error", f"Error clearing dimensions:\n\n{e}")
                
    def check_compliance(self):
        """Check building code compliance."""
        if not self.compliance_service:
            messagebox.showerror("Error", "Not connected to AutoCAD")
            return
            
        # Get selected categories
        categories = []
        if self.fire_safety_var.get():
            categories.append(RuleCategory.FIRE_SAFETY)
        if self.accessibility_var.get():
            categories.append(RuleCategory.ACCESSIBILITY)
        if self.structural_var.get():
            categories.append(RuleCategory.STRUCTURAL)
            
        if not categories:
            messagebox.showwarning("Warning", "Please select at least one compliance category to check.")
            return
            
        # Start compliance checking in a separate thread
        threading.Thread(target=self._compliance_worker, args=(categories,), daemon=True).start()
        
    def _compliance_worker(self, categories):
        """Worker thread for compliance checking."""
        try:
            self.root.after(0, self._start_compliance_progress)
            
            self.root.after(0, self.log_message, "🔍 Starting compliance checking...")
            self.root.after(0, self.log_message, f"📋 Checking categories: {', '.join([c.value for c in categories])}")
            
            violations = self.compliance_service.check_compliance(categories)
            
            self.root.after(0, self.log_message, f"🏛️ Compliance check completed!")
            self.root.after(0, self.log_message, f"⚠️ Found {len(violations)} violations")
            
            # Generate detailed report
            if violations:
                for violation in violations[:5]:  # Show first 5 violations
                    severity_emoji = "🔴" if violation.severity.value == "high" else "🟡" if violation.severity.value == "medium" else "🟢"
                    self.root.after(0, self.log_message, f"{severity_emoji} {violation.rule_title}: {violation.description}")
                    
                if len(violations) > 5:
                    self.root.after(0, self.log_message, f"... and {len(violations) - 5} more violations")
                    
                # Generate summary report
                report = self.compliance_service.generate_compliance_report(violations)
                self.root.after(0, self.log_message, f"📊 Summary: {report['summary']['high_severity']} high, {report['summary']['medium_severity']} medium, {report['summary']['low_severity']} low severity")
                
            else:
                self.root.after(0, self.log_message, "✅ No compliance violations found!")
                
        except Exception as e:
            error_msg = f"❌ Error during compliance checking: {e}"
            self.root.after(0, self.log_message, error_msg)
            self.root.after(0, messagebox.showerror, "Compliance Check Error", str(e))
        finally:
            self.root.after(0, self._stop_compliance_progress)
            
    def load_pdf_rules(self):
        """Load compliance rules from PDF."""
        if not self.compliance_service:
            messagebox.showerror("Error", "Not connected to AutoCAD")
            return
            
        pdf_path = self.pdf_path_var.get().strip()
        if not pdf_path:
            messagebox.showwarning("Warning", "Please enter the path to a building code PDF file.")
            return
            
        # Start PDF loading in a separate thread
        threading.Thread(target=self._pdf_worker, args=(pdf_path,), daemon=True).start()
        
    def _pdf_worker(self, pdf_path):
        """Worker thread for PDF rule extraction."""
        try:
            self.root.after(0, self._start_compliance_progress)
            
            self.root.after(0, self.log_message, f"📄 Loading rules from PDF: {pdf_path}")
            
            rules_count = self.compliance_service.load_rules_from_pdf(pdf_path)
            
            if rules_count > 0:
                self.root.after(0, self.log_message, f"✅ Successfully extracted {rules_count} rules from PDF")
                self.root.after(0, self.log_message, f"📋 Total rules available: {len(self.compliance_service.rules)}")
            else:
                self.root.after(0, self.log_message, "⚠️ No rules could be extracted from the PDF")
                
        except Exception as e:
            error_msg = f"❌ Error loading PDF rules: {e}"
            self.root.after(0, self.log_message, error_msg)
            self.root.after(0, messagebox.showerror, "PDF Loading Error", str(e))
        finally:
            self.root.after(0, self._stop_compliance_progress)
            
    def _start_compliance_progress(self):
        """Start compliance progress bar."""
        self.compliance_progress.start()
        self.compliance_btn.configure(state="disabled")
        self.load_pdf_btn.configure(state="disabled")
        
    def _stop_compliance_progress(self):
        """Stop compliance progress bar."""
        self.compliance_progress.stop()
        self.compliance_progress.set(0)
        self.compliance_btn.configure(state="normal")
        self.load_pdf_btn.configure(state="normal")
                
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
        
    def log_message(self, message: str):
        """Add a message to the results text."""
        import datetime
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}\n"
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