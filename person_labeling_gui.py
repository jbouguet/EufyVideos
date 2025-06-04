#!/usr/bin/env python3
"""
Person Labeling GUI Tool

A comprehensive GUI application for manually labeling person detections and managing
person recognition databases. This tool provides:

1. Visual browsing of person crops with thumbnail display
2. Manual labeling interface for assigning names to detections
3. Cluster visualization showing similar persons grouped together
4. Batch editing capabilities for efficient labeling
5. Integration with PersonDatabase and embedding systems

Key Features:
- Load person crops from directories or embedding files
- Display crops in organized thumbnail grids
- Manual labeling with autocomplete person name suggestions
- Cluster view showing similar detections grouped together
- Batch operations for labeling multiple crops at once
- Real-time search and filtering
- Export labeled data for training

Example Usage:
    python person_labeling_gui.py --crops_dir /path/to/crops --database persons.json
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from PIL import Image, ImageTk
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import argparse
from datetime import datetime

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from logging_config import create_logger
from person_database import PersonDatabase, PersonIdentity
from person_embedding import PersonEmbeddingGenerator, PersonEmbedding

logger = create_logger(__name__)


class PersonCropData:
    """Data structure for person crop with labeling information."""
    
    def __init__(self, image_path: str, embedding: Optional[np.ndarray] = None):
        self.image_path = image_path
        self.filename = os.path.basename(image_path)
        self.embedding = embedding
        
        # Parse filename to extract metadata
        self._parse_filename()
        
        # Labeling information
        self.person_name: Optional[str] = None
        self.person_id: Optional[str] = None
        self.confidence: float = 1.0
        self.labeled_by: str = "manual"
        self.notes: str = ""
        self.cluster_id: Optional[int] = None
        
        # UI state
        self.selected: bool = False
        self.thumbnail: Optional[ImageTk.PhotoImage] = None
    
    def _parse_filename(self):
        """Parse filename to extract video, track, and frame information."""
        try:
            # Expected format: videoname_trackXXX_frameXXXXXX.jpg
            parts = self.filename.replace('.jpg', '').split('_')
            
            # Find track and frame parts first
            self.track_id = None
            self.frame_number = None
            track_frame_parts = []
            
            for part in parts:
                if part.startswith('track'):
                    self.track_id = part.replace('track', '')
                    track_frame_parts.append(part)
                elif part.startswith('frame'):
                    self.frame_number = int(part.replace('frame', ''))
                    track_frame_parts.append(part)
            
            # Video name is everything except track and frame parts
            video_parts = [part for part in parts if part not in track_frame_parts]
            self.video_name = '_'.join(video_parts) if video_parts else "unknown"
                    
        except Exception as e:
            logger.warning(f"Failed to parse filename {self.filename}: {e}")
            self.video_name = "unknown"
            self.track_id = None
            self.frame_number = None
    
    def get_display_text(self) -> str:
        """Get text for display in UI."""
        name = self.person_name or "Unlabeled"
        track_info = f"Track {self.track_id}" if self.track_id else "No Track"
        return f"{name}\n{track_info}"


class ThumbnailGrid(tk.Frame):
    """Scrollable grid widget for displaying person crop thumbnails."""
    
    def __init__(self, parent, thumbnail_size=(100, 100), columns=6):
        super().__init__(parent)
        
        self.thumbnail_size = thumbnail_size
        self.columns = columns
        self.crops: List[PersonCropData] = []
        self.selected_crops: List[PersonCropData] = []
        self.on_selection_changed = None
        self.on_crop_double_click = None
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup the thumbnail grid UI."""
        # Create scrollable frame
        self.canvas = tk.Canvas(self, bg='white')
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        # Pack components
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        
        # Bind mouse wheel
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.bind_all("<Control-a>", self._select_all)
    
    def _on_mousewheel(self, event):
        """Handle mouse wheel scrolling."""
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
    
    def _select_all(self, event):
        """Select all visible crops."""
        for crop in self.crops:
            crop.selected = True
        self.refresh_display()
        if self.on_selection_changed:
            self.on_selection_changed(self.crops[:])
    
    def set_crops(self, crops: List[PersonCropData]):
        """Set the crops to display."""
        self.crops = crops
        self.selected_crops = []
        self.refresh_display()
    
    def refresh_display(self):
        """Refresh the thumbnail display."""
        # Clear existing widgets
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        
        # Create thumbnail grid
        for i, crop in enumerate(self.crops):
            row = i // self.columns
            col = i % self.columns
            
            self._create_thumbnail_widget(crop, row, col)
        
        # Update scroll region
        self.scrollable_frame.update_idletasks()
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
    
    def _create_thumbnail_widget(self, crop: PersonCropData, row: int, col: int):
        """Create a thumbnail widget for a crop."""
        # Create frame for this thumbnail
        frame = ttk.Frame(self.scrollable_frame, relief="solid", borderwidth=1)
        frame.grid(row=row, column=col, padx=2, pady=2, sticky="nsew")
        
        # Load and resize image
        try:
            img = Image.open(crop.image_path)
            img = img.resize(self.thumbnail_size, Image.Resampling.LANCZOS)
            
            # Add selection border if selected
            if crop.selected:
                # Create a border by expanding image with colored background
                border_size = 3
                bordered_size = (self.thumbnail_size[0] + 2*border_size, 
                               self.thumbnail_size[1] + 2*border_size)
                bordered_img = Image.new('RGB', bordered_size, color='blue')
                bordered_img.paste(img, (border_size, border_size))
                img = bordered_img
            
            crop.thumbnail = ImageTk.PhotoImage(img)
            
            # Create image label
            img_label = tk.Label(frame, image=crop.thumbnail, cursor="hand2")
            img_label.pack()
            
            # Create text label
            text_label = tk.Label(frame, text=crop.get_display_text(), 
                                font=("Arial", 8), wraplength=self.thumbnail_size[0])
            text_label.pack()
            
            # Bind click events
            for widget in [img_label, text_label, frame]:
                widget.bind("<Button-1>", lambda e, c=crop: self._on_crop_click(c))
                widget.bind("<Double-Button-1>", lambda e, c=crop: self._on_crop_double_click(c))
                widget.bind("<Control-Button-1>", lambda e, c=crop: self._on_crop_ctrl_click(c))
            
        except Exception as e:
            logger.error(f"Failed to load thumbnail for {crop.image_path}: {e}")
            # Create error placeholder
            error_label = tk.Label(frame, text="Error\nLoading\nImage", 
                                 width=10, height=6, bg="lightgray")
            error_label.pack()
    
    def _on_crop_click(self, crop: PersonCropData):
        """Handle single click on crop."""
        # Clear other selections
        for c in self.crops:
            c.selected = False
        
        crop.selected = True
        self.selected_crops = [crop]
        self.refresh_display()
        
        if self.on_selection_changed:
            self.on_selection_changed([crop])
    
    def _on_crop_ctrl_click(self, crop: PersonCropData):
        """Handle Ctrl+click on crop for multi-selection."""
        crop.selected = not crop.selected
        
        if crop.selected:
            if crop not in self.selected_crops:
                self.selected_crops.append(crop)
        else:
            if crop in self.selected_crops:
                self.selected_crops.remove(crop)
        
        self.refresh_display()
        
        if self.on_selection_changed:
            self.on_selection_changed(self.selected_crops[:])
    
    def _on_crop_double_click(self, crop: PersonCropData):
        """Handle double click on crop."""
        if self.on_crop_double_click:
            self.on_crop_double_click(crop)


class PersonLabelingGUI:
    """Main GUI application for person labeling."""
    
    def __init__(self, crops_dir: Optional[str] = None, 
                 database_file: Optional[str] = None,
                 embeddings_file: Optional[str] = None):
        
        self.crops_dir = crops_dir
        self.database_file = database_file or "persons.json"
        self.embeddings_file = embeddings_file
        
        # Data
        self.crops: List[PersonCropData] = []
        self.filtered_crops: List[PersonCropData] = []
        self.clusters: List[List[PersonCropData]] = []
        self.person_db: Optional[PersonDatabase] = None
        self.embedder: Optional[PersonEmbeddingGenerator] = None
        
        # UI State
        self.current_view = "all"  # "all", "clusters", "unlabeled"
        self.search_filter = ""
        
        self._setup_ui()
        self._load_data()
    
    def _setup_ui(self):
        """Setup the main GUI."""
        self.root = tk.Tk()
        self.root.title("Person Labeling Tool")
        self.root.geometry("1200x800")
        
        # Setup menu
        self._setup_menu()
        
        # Setup toolbar
        self._setup_toolbar()
        
        # Setup main content area
        self._setup_main_area()
        
        # Setup status bar
        self._setup_status_bar()
    
    def _setup_menu(self):
        """Setup the application menu."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load Crops Directory...", command=self._load_crops_directory)
        file_menu.add_command(label="Load Embeddings File...", command=self._load_embeddings_file)
        file_menu.add_separator()
        file_menu.add_command(label="Save Database", command=self._save_database)
        file_menu.add_command(label="Export Labels...", command=self._export_labels)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Show All", command=lambda: self._change_view("all"))
        view_menu.add_command(label="Show Clusters", command=lambda: self._change_view("clusters"))
        view_menu.add_command(label="Show Unlabeled", command=lambda: self._change_view("unlabeled"))
        view_menu.add_separator()
        view_menu.add_command(label="Generate Clusters", command=self._generate_clusters)
        
        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Add Person...", command=self._add_person)
        tools_menu.add_command(label="Manage Persons...", command=self._manage_persons)
        tools_menu.add_separator()
        tools_menu.add_command(label="Auto-Label Similar", command=self._auto_label_similar)
    
    def _setup_toolbar(self):
        """Setup the toolbar."""
        toolbar = ttk.Frame(self.root)
        toolbar.pack(side="top", fill="x", padx=5, pady=5)
        
        # Search
        ttk.Label(toolbar, text="Search:").pack(side="left", padx=(0, 5))
        self.search_var = tk.StringVar()
        self.search_var.trace("w", self._on_search_changed)
        search_entry = ttk.Entry(toolbar, textvariable=self.search_var, width=20)
        search_entry.pack(side="left", padx=(0, 10))
        
        # View selector
        ttk.Label(toolbar, text="View:").pack(side="left", padx=(0, 5))
        self.view_var = tk.StringVar(value="all")
        view_combo = ttk.Combobox(toolbar, textvariable=self.view_var, 
                                 values=["all", "clusters", "unlabeled"], 
                                 state="readonly", width=10)
        view_combo.pack(side="left", padx=(0, 10))
        view_combo.bind("<<ComboboxSelected>>", self._on_view_changed)
        
        # Action buttons
        ttk.Button(toolbar, text="Label Selected", 
                  command=self._label_selected).pack(side="left", padx=5)
        ttk.Button(toolbar, text="Clear Labels", 
                  command=self._clear_selected_labels).pack(side="left", padx=5)
        ttk.Button(toolbar, text="Generate Clusters", 
                  command=self._generate_clusters).pack(side="left", padx=5)
    
    def _setup_main_area(self):
        """Setup the main content area."""
        # Create paned window for resizable layout
        paned = ttk.PanedWindow(self.root, orient="horizontal")
        paned.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Left panel - thumbnail grid
        left_frame = ttk.Frame(paned)
        paned.add(left_frame, weight=3)
        
        ttk.Label(left_frame, text="Person Crops", font=("Arial", 12, "bold")).pack(pady=(0, 5))
        
        self.thumbnail_grid = ThumbnailGrid(left_frame, thumbnail_size=(120, 120), columns=5)
        self.thumbnail_grid.pack(fill="both", expand=True)
        self.thumbnail_grid.on_selection_changed = self._on_selection_changed
        self.thumbnail_grid.on_crop_double_click = self._on_crop_double_click
        
        # Right panel - details and actions
        right_frame = ttk.Frame(paned)
        paned.add(right_frame, weight=1)
        
        self._setup_details_panel(right_frame)
    
    def _setup_details_panel(self, parent):
        """Setup the details panel."""
        ttk.Label(parent, text="Selection Details", font=("Arial", 12, "bold")).pack(pady=(0, 10))
        
        # Selection info
        self.selection_info = ttk.Label(parent, text="No crops selected")
        self.selection_info.pack(pady=(0, 10))
        
        # Labeling section
        label_frame = ttk.LabelFrame(parent, text="Labeling", padding=10)
        label_frame.pack(fill="x", pady=(0, 10))
        
        # Person name entry with autocomplete
        ttk.Label(label_frame, text="Person Name:").pack(anchor="w")
        self.person_name_var = tk.StringVar()
        self.person_name_entry = ttk.Entry(label_frame, textvariable=self.person_name_var)
        self.person_name_entry.pack(fill="x", pady=(0, 5))
        
        # Quick person buttons
        self.person_buttons_frame = ttk.Frame(label_frame)
        self.person_buttons_frame.pack(fill="x", pady=(0, 5))
        
        # Action buttons
        ttk.Button(label_frame, text="Apply Label", 
                  command=self._apply_label_to_selected).pack(fill="x", pady=2)
        ttk.Button(label_frame, text="Remove Label", 
                  command=self._remove_label_from_selected).pack(fill="x", pady=2)
        
        # Cluster info
        cluster_frame = ttk.LabelFrame(parent, text="Cluster Info", padding=10)
        cluster_frame.pack(fill="x", pady=(0, 10))
        
        self.cluster_info = ttk.Label(cluster_frame, text="No cluster analysis")
        self.cluster_info.pack()
        
        ttk.Button(cluster_frame, text="Label All in Cluster", 
                  command=self._label_cluster).pack(fill="x", pady=2)
        
        # Statistics
        stats_frame = ttk.LabelFrame(parent, text="Statistics", padding=10)
        stats_frame.pack(fill="x", pady=(0, 10))
        
        self.stats_text = tk.Text(stats_frame, height=6, wrap="word")
        self.stats_text.pack(fill="both", expand=True)
    
    def _setup_status_bar(self):
        """Setup the status bar."""
        self.status_bar = ttk.Label(self.root, text="Ready", relief="sunken")
        self.status_bar.pack(side="bottom", fill="x")
    
    def _load_data(self):
        """Load initial data."""
        try:
            # Load person database
            self.person_db = PersonDatabase(self.database_file)
            
            # Load crops if directory specified
            if self.crops_dir:
                self._load_crops_from_directory(self.crops_dir)
            
            # Load embeddings if file specified
            if self.embeddings_file:
                self._load_embeddings_from_file(self.embeddings_file)
            
            self._update_person_buttons()
            self._update_display()
            self._update_statistics()
            
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            messagebox.showerror("Error", f"Failed to load data: {e}")
    
    def _load_crops_from_directory(self, directory: str):
        """Load person crops from directory."""
        if not os.path.exists(directory):
            return
        
        crop_files = list(Path(directory).glob("*.jpg"))
        self.crops = []
        
        for crop_file in crop_files:
            crop_data = PersonCropData(str(crop_file))
            self.crops.append(crop_data)
        
        logger.info(f"Loaded {len(self.crops)} crop images from {directory}")
        self._update_status(f"Loaded {len(self.crops)} crops from directory")
    
    def _load_embeddings_from_file(self, filename: str):
        """Load embeddings and associate with crops."""
        if not os.path.exists(filename):
            return
        
        try:
            embeddings = PersonEmbeddingGenerator.load_embeddings(filename)
            
            # Match embeddings to crops by track_id and frame_number
            embedding_map = {}
            for emb in embeddings:
                key = f"{emb.track_id}_{emb.frame_number}"
                embedding_map[key] = emb.embedding
            
            # Associate embeddings with crops
            matched_count = 0
            for crop in self.crops:
                if crop.track_id and crop.frame_number:
                    key = f"{crop.track_id}_{crop.frame_number}"
                    if key in embedding_map:
                        crop.embedding = embedding_map[key]
                        matched_count += 1
            
            logger.info(f"Matched {matched_count}/{len(self.crops)} crops with embeddings")
            self._update_status(f"Loaded embeddings: {matched_count} matches")
            
        except Exception as e:
            logger.error(f"Failed to load embeddings: {e}")
    
    def _update_person_buttons(self):
        """Update quick person selection buttons."""
        # Clear existing buttons
        for widget in self.person_buttons_frame.winfo_children():
            widget.destroy()
        
        # Add buttons for existing persons
        if self.person_db:
            persons = self.person_db.list_persons()
            for i, person in enumerate(persons[:6]):  # Limit to 6 buttons
                btn = ttk.Button(self.person_buttons_frame, text=person.name,
                               command=lambda name=person.name: self._set_person_name(name))
                btn.grid(row=i//2, column=i%2, padx=2, pady=2, sticky="ew")
            
            # Configure grid weights
            self.person_buttons_frame.grid_columnconfigure(0, weight=1)
            self.person_buttons_frame.grid_columnconfigure(1, weight=1)
    
    def _set_person_name(self, name: str):
        """Set the person name in the entry field."""
        self.person_name_var.set(name)
    
    def _update_display(self):
        """Update the main display based on current view and filters."""
        # Apply search filter
        if self.search_filter:
            self.filtered_crops = [crop for crop in self.crops 
                                 if self.search_filter.lower() in (crop.person_name or "").lower()
                                 or self.search_filter.lower() in crop.filename.lower()]
        else:
            self.filtered_crops = self.crops[:]
        
        # Apply view filter
        if self.current_view == "unlabeled":
            self.filtered_crops = [crop for crop in self.filtered_crops if not crop.person_name]
        elif self.current_view == "clusters":
            # Show crops grouped by clusters
            if self.clusters:
                self.filtered_crops = []
                for cluster in self.clusters:
                    self.filtered_crops.extend(cluster)
        
        # Update thumbnail grid
        self.thumbnail_grid.set_crops(self.filtered_crops)
        
        # Update status
        total = len(self.crops)
        filtered = len(self.filtered_crops)
        labeled = len([c for c in self.crops if c.person_name])
        self._update_status(f"Showing {filtered}/{total} crops ({labeled} labeled)")
    
    def _update_statistics(self):
        """Update the statistics display."""
        if not self.crops:
            self.stats_text.delete(1.0, tk.END)
            self.stats_text.insert(tk.END, "No crops loaded")
            return
        
        total_crops = len(self.crops)
        labeled_crops = len([c for c in self.crops if c.person_name])
        unlabeled_crops = total_crops - labeled_crops
        
        # Count persons
        person_counts = {}
        for crop in self.crops:
            if crop.person_name:
                person_counts[crop.person_name] = person_counts.get(crop.person_name, 0) + 1
        
        stats_text = f"Total Crops: {total_crops}\n"
        stats_text += f"Labeled: {labeled_crops}\n"
        stats_text += f"Unlabeled: {unlabeled_crops}\n"
        stats_text += f"Unique Persons: {len(person_counts)}\n\n"
        
        if person_counts:
            stats_text += "Person Counts:\n"
            for person, count in sorted(person_counts.items()):
                stats_text += f"  {person}: {count}\n"
        
        if self.clusters:
            stats_text += f"\nClusters: {len(self.clusters)}\n"
            avg_cluster_size = np.mean([len(cluster) for cluster in self.clusters])
            stats_text += f"Avg Cluster Size: {avg_cluster_size:.1f}"
        
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(tk.END, stats_text)
    
    def _update_status(self, message: str):
        """Update the status bar."""
        self.status_bar.config(text=message)
        self.root.update_idletasks()
    
    # Event handlers
    def _on_search_changed(self, *args):
        """Handle search text change."""
        self.search_filter = self.search_var.get()
        self._update_display()
    
    def _on_view_changed(self, event):
        """Handle view selection change."""
        self._change_view(self.view_var.get())
    
    def _change_view(self, view: str):
        """Change the current view."""
        self.current_view = view
        self.view_var.set(view)
        self._update_display()
    
    def _on_selection_changed(self, selected_crops: List[PersonCropData]):
        """Handle selection change in thumbnail grid."""
        count = len(selected_crops)
        if count == 0:
            self.selection_info.config(text="No crops selected")
        elif count == 1:
            crop = selected_crops[0]
            info_text = f"Selected: {crop.filename}\n"
            if crop.person_name:
                info_text += f"Label: {crop.person_name}\n"
            if crop.track_id:
                info_text += f"Track: {crop.track_id}\n"
            if crop.cluster_id is not None:
                info_text += f"Cluster: {crop.cluster_id}"
            self.selection_info.config(text=info_text)
            
            # Set person name in entry if labeled
            if crop.person_name:
                self.person_name_var.set(crop.person_name)
        else:
            self.selection_info.config(text=f"{count} crops selected")
        
        # Update cluster info if applicable
        if selected_crops and hasattr(selected_crops[0], 'cluster_id') and selected_crops[0].cluster_id is not None:
            cluster_id = selected_crops[0].cluster_id
            if cluster_id < len(self.clusters):
                cluster = self.clusters[cluster_id]
                cluster_text = f"Cluster {cluster_id}: {len(cluster)} crops"
                
                # Show cluster labels
                cluster_labels = set(c.person_name for c in cluster if c.person_name)
                if cluster_labels:
                    cluster_text += f"\nLabels: {', '.join(cluster_labels)}"
                
                self.cluster_info.config(text=cluster_text)
            else:
                self.cluster_info.config(text="Invalid cluster")
        else:
            self.cluster_info.config(text="No cluster analysis")
    
    def _on_crop_double_click(self, crop: PersonCropData):
        """Handle double click on crop."""
        # Open labeling dialog
        self._open_labeling_dialog(crop)
    
    # Action methods
    def _load_crops_directory(self):
        """Load crops from selected directory."""
        directory = filedialog.askdirectory(title="Select Crops Directory")
        if directory:
            self.crops_dir = directory
            self._load_crops_from_directory(directory)
            self._update_display()
            self._update_statistics()
    
    def _load_embeddings_file(self):
        """Load embeddings from selected file."""
        filename = filedialog.askopenfilename(
            title="Select Embeddings File",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            self.embeddings_file = filename
            self._load_embeddings_from_file(filename)
    
    def _save_database(self):
        """Save the person database."""
        if self.person_db:
            try:
                self.person_db.save_database()
                messagebox.showinfo("Success", "Database saved successfully")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save database: {e}")
    
    def _export_labels(self):
        """Export labeled data."""
        filename = filedialog.asksaveasfilename(
            title="Export Labels",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            self._export_labels_to_file(filename)
    
    def _export_labels_to_file(self, filename: str):
        """Export labels to file."""
        try:
            labeled_crops = [c for c in self.crops if c.person_name]
            
            export_data = {
                'metadata': {
                    'exported_at': datetime.now().isoformat(),
                    'total_crops': len(self.crops),
                    'labeled_crops': len(labeled_crops),
                    'source_directory': self.crops_dir
                },
                'labels': []
            }
            
            for crop in labeled_crops:
                label_data = {
                    'filename': crop.filename,
                    'image_path': crop.image_path,
                    'person_name': crop.person_name,
                    'person_id': crop.person_id,
                    'track_id': crop.track_id,
                    'frame_number': crop.frame_number,
                    'confidence': crop.confidence,
                    'labeled_by': crop.labeled_by,
                    'notes': crop.notes
                }
                export_data['labels'].append(label_data)
            
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            messagebox.showinfo("Success", f"Exported {len(labeled_crops)} labels to {filename}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export labels: {e}")
    
    def _apply_label_to_selected(self):
        """Apply current person name to selected crops."""
        person_name = self.person_name_var.get().strip()
        if not person_name:
            messagebox.showwarning("Warning", "Please enter a person name")
            return
        
        selected_crops = [c for c in self.crops if c.selected]
        if not selected_crops:
            messagebox.showwarning("Warning", "No crops selected")
            return
        
        # Get or create person in database
        person_id = None
        if self.person_db:
            person_id = self.person_db.get_person_id_by_name(person_name)
            if not person_id:
                person_id = self.person_db.add_person(person_name, "Added via labeling tool")
        
        # Apply label to selected crops
        for crop in selected_crops:
            crop.person_name = person_name
            crop.person_id = person_id
            crop.labeled_by = "manual"
        
        self._update_display()
        self._update_statistics()
        self._update_person_buttons()
        
        messagebox.showinfo("Success", f"Labeled {len(selected_crops)} crops as '{person_name}'")
    
    def _remove_label_from_selected(self):
        """Remove labels from selected crops."""
        selected_crops = [c for c in self.crops if c.selected]
        if not selected_crops:
            messagebox.showwarning("Warning", "No crops selected")
            return
        
        for crop in selected_crops:
            crop.person_name = None
            crop.person_id = None
            crop.notes = ""
        
        self._update_display()
        self._update_statistics()
        
        messagebox.showinfo("Success", f"Removed labels from {len(selected_crops)} crops")
    
    def _clear_selected_labels(self):
        """Clear labels from selected crops."""
        self._remove_label_from_selected()
    
    def _generate_clusters(self):
        """Generate clusters based on embeddings."""
        if not any(crop.embedding is not None for crop in self.crops):
            messagebox.showwarning("Warning", "No embeddings available for clustering")
            return
        
        try:
            # Initialize embedder if needed
            if not self.embedder:
                self.embedder = PersonEmbeddingGenerator(device="mps")
            
            # Create PersonEmbedding objects for crops with embeddings
            embeddings = []
            crop_mapping = {}
            
            for i, crop in enumerate(self.crops):
                if crop.embedding is not None:
                    # Create dummy PersonEmbedding for clustering
                    person_emb = PersonEmbedding(
                        track_id=crop.track_id or 0,
                        frame_number=crop.frame_number or 0,
                        video_filename=crop.video_name,
                        embedding=crop.embedding
                    )
                    embeddings.append(person_emb)
                    crop_mapping[len(embeddings)-1] = i
            
            if not embeddings:
                messagebox.showwarning("Warning", "No valid embeddings found")
                return
            
            # Generate clusters
            cluster_indices = self.embedder.cluster_embeddings(embeddings, similarity_threshold=0.75)
            
            # Convert to crop clusters
            self.clusters = []
            for cluster_idx, indices in enumerate(cluster_indices):
                cluster_crops = []
                for emb_idx in indices:
                    if emb_idx in crop_mapping:
                        crop_idx = crop_mapping[emb_idx]
                        crop = self.crops[crop_idx]
                        crop.cluster_id = cluster_idx
                        cluster_crops.append(crop)
                
                if cluster_crops:
                    self.clusters.append(cluster_crops)
            
            messagebox.showinfo("Success", f"Generated {len(self.clusters)} clusters")
            self._update_display()
            self._update_statistics()
            
        except Exception as e:
            logger.error(f"Clustering failed: {e}")
            messagebox.showerror("Error", f"Clustering failed: {e}")
    
    def _label_cluster(self):
        """Label all crops in the same cluster as selected."""
        selected_crops = [c for c in self.crops if c.selected]
        if not selected_crops:
            messagebox.showwarning("Warning", "No crops selected")
            return
        
        person_name = self.person_name_var.get().strip()
        if not person_name:
            messagebox.showwarning("Warning", "Please enter a person name")
            return
        
        # Get cluster ID from first selected crop
        cluster_id = selected_crops[0].cluster_id
        if cluster_id is None:
            messagebox.showwarning("Warning", "Selected crop is not in a cluster")
            return
        
        # Find all crops in the same cluster
        cluster_crops = [c for c in self.crops if c.cluster_id == cluster_id]
        
        if not cluster_crops:
            messagebox.showwarning("Warning", "No crops found in cluster")
            return
        
        # Get or create person in database
        person_id = None
        if self.person_db:
            person_id = self.person_db.get_person_id_by_name(person_name)
            if not person_id:
                person_id = self.person_db.add_person(person_name, "Added via labeling tool")
        
        # Apply label to all crops in cluster
        for crop in cluster_crops:
            crop.person_name = person_name
            crop.person_id = person_id
            crop.labeled_by = "manual_cluster"
        
        self._update_display()
        self._update_statistics()
        self._update_person_buttons()
        
        messagebox.showinfo("Success", f"Labeled {len(cluster_crops)} crops in cluster as '{person_name}'")
    
    def _auto_label_similar(self):
        """Automatically label crops similar to labeled ones."""
        messagebox.showinfo("Info", "Auto-labeling feature not yet implemented")
    
    def _add_person(self):
        """Add a new person to the database."""
        name = simpledialog.askstring("Add Person", "Enter person name:")
        if name and name.strip():
            if self.person_db:
                try:
                    self.person_db.add_person(name.strip(), "Added via labeling tool")
                    self._update_person_buttons()
                    messagebox.showinfo("Success", f"Added person: {name}")
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to add person: {e}")
    
    def _manage_persons(self):
        """Open person management dialog."""
        messagebox.showinfo("Info", "Person management dialog not yet implemented")
    
    def _open_labeling_dialog(self, crop: PersonCropData):
        """Open detailed labeling dialog for a crop."""
        dialog = tk.Toplevel(self.root)
        dialog.title(f"Label Crop: {crop.filename}")
        dialog.geometry("400x300")
        
        # Image display
        try:
            img = Image.open(crop.image_path)
            img = img.resize((200, 200), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            img_label = tk.Label(dialog, image=photo)
            img_label.image = photo  # Keep a reference
            img_label.pack(pady=10)
        except Exception as e:
            tk.Label(dialog, text="Error loading image").pack(pady=10)
        
        # Labeling form
        form_frame = ttk.Frame(dialog)
        form_frame.pack(fill="x", padx=20, pady=10)
        
        ttk.Label(form_frame, text="Person Name:").grid(row=0, column=0, sticky="w", pady=2)
        name_var = tk.StringVar(value=crop.person_name or "")
        name_entry = ttk.Entry(form_frame, textvariable=name_var, width=30)
        name_entry.grid(row=0, column=1, sticky="ew", pady=2)
        
        ttk.Label(form_frame, text="Notes:").grid(row=1, column=0, sticky="nw", pady=2)
        notes_text = tk.Text(form_frame, width=30, height=4)
        notes_text.grid(row=1, column=1, sticky="ew", pady=2)
        notes_text.insert(1.0, crop.notes)
        
        form_frame.grid_columnconfigure(1, weight=1)
        
        # Buttons
        button_frame = ttk.Frame(dialog)
        button_frame.pack(fill="x", padx=20, pady=10)
        
        def save_label():
            person_name = name_var.get().strip()
            if person_name:
                # Get or create person
                person_id = None
                if self.person_db:
                    person_id = self.person_db.get_person_id_by_name(person_name)
                    if not person_id:
                        person_id = self.person_db.add_person(person_name, "Added via labeling tool")
                
                crop.person_name = person_name
                crop.person_id = person_id
                crop.notes = notes_text.get(1.0, tk.END).strip()
                crop.labeled_by = "manual"
                
                self._update_display()
                self._update_statistics()
                self._update_person_buttons()
            
            dialog.destroy()
        
        def remove_label():
            crop.person_name = None
            crop.person_id = None
            crop.notes = ""
            
            self._update_display()
            self._update_statistics()
            dialog.destroy()
        
        ttk.Button(button_frame, text="Save", command=save_label).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Remove Label", command=remove_label).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side="left", padx=5)
    
    def run(self):
        """Run the GUI application."""
        self.root.mainloop()


def main():
    """Main function for the person labeling GUI."""
    parser = argparse.ArgumentParser(description="Person Labeling GUI Tool")
    parser.add_argument("--crops_dir", type=str, 
                       help="Directory containing person crop images")
    parser.add_argument("--database", type=str, default="persons.json",
                       help="Person database file")
    parser.add_argument("--embeddings", type=str,
                       help="Embeddings file to load")
    
    args = parser.parse_args()
    
    # Create and run GUI
    app = PersonLabelingGUI(
        crops_dir=args.crops_dir,
        database_file=args.database,
        embeddings_file=args.embeddings
    )
    
    app.run()


if __name__ == "__main__":
    main()