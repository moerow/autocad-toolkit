"""Modern GUI window for the AutoCAD Construction Toolkit."""
import customtkinter as ctk
from tkinter import messagebox
import logging
import threading
from typing import Optional

from src.infrastructure.autocad.connection import AutoCADConnection
from src.application.services.dimension_service import DimensionService

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
        
        # Welcome message
        self.log_message("🚀 Welcome to AutoCAD Construction Toolkit")
        self.log_message("💡 Connect to AutoCAD to start automatic dimensioning")
        self.log_message("📋 Features: Smart line detection, configurable layers, professional dimensions")
        
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
                self.dimension_btn.configure(state="normal")
                self.clear_btn.configure(state="normal")
                self.connect_btn.configure(text="Reconnect")
                self.log_message("✅ Connected to AutoCAD successfully!")
                self.log_message(f"📄 Drawing: {self.connection.doc.Name}")
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