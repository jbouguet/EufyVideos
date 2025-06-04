#!/usr/bin/env python3
"""
Streamlit Person Labeling GUI

A comprehensive web-based GUI application for manually labeling person detections
and managing person recognition databases using Streamlit.

Key Features:
- Visual browsing of person crops with image gallery
- Manual labeling interface for assigning names to detections
- Cluster visualization showing similar persons grouped together
- Batch editing capabilities for efficient labeling
- Real-time search and filtering
- Integration with PersonDatabase and embedding systems

Usage:
    streamlit run person_labeling_streamlit.py -- --crops_dir /path/to/crops --database persons.json
"""

import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import argparse
from datetime import datetime
import base64
from io import BytesIO

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
            
            # Handle negative track IDs (convert to string properly)
            if self.track_id and self.track_id.startswith('-'):
                # Keep negative track IDs as-is for consistency
                pass
                    
        except Exception as e:
            logger.warning(f"Failed to parse filename {self.filename}: {e}")
            self.video_name = "unknown"
            self.track_id = None
            self.frame_number = None
    
    def get_display_info(self) -> Dict[str, Any]:
        """Get information for display in UI."""
        return {
            'filename': self.filename,
            'person_name': self.person_name or "Unlabeled",
            'track_id': self.track_id or "Unknown",
            'frame_number': self.frame_number or 0,
            'video_name': self.video_name,
            'cluster_id': self.cluster_id,
            'notes': self.notes
        }


class PersonLabelingApp:
    """Streamlit application for person labeling."""
    
    def __init__(self):
        self.crops: List[PersonCropData] = []
        self.filtered_crops: List[PersonCropData] = []
        self.clusters: List[List[PersonCropData]] = []
        self.person_db: Optional[PersonDatabase] = None
        self.embedder: Optional[PersonEmbeddingGenerator] = None
        
        # Initialize session state
        self._init_session_state()
    
    def _init_session_state(self):
        """Initialize Streamlit session state."""
        if 'crops_loaded' not in st.session_state:
            st.session_state.crops_loaded = False
        if 'current_view' not in st.session_state:
            st.session_state.current_view = "all"
        if 'selected_crops' not in st.session_state:
            st.session_state.selected_crops = []
        if 'clusters_generated' not in st.session_state:
            st.session_state.clusters_generated = False
        if 'label_changes' not in st.session_state:
            st.session_state.label_changes = {}
    
    def run(self):
        """Run the Streamlit application."""
        st.set_page_config(
            page_title="Person Labeling Tool",
            page_icon="👤",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.title("👤 Person Labeling Tool")
        st.markdown("---")
        
        # Sidebar for configuration and controls
        self._setup_sidebar()
        
        # Main content area
        if st.session_state.crops_loaded and self.crops:
            self._display_main_interface()
        else:
            self._display_welcome_screen()
    
    def _setup_sidebar(self):
        """Setup the sidebar with controls."""
        st.sidebar.title("Controls")
        
        # Data loading section
        st.sidebar.subheader("📁 Data Loading")
        
        # Check for demo data
        demo_crops = os.environ.get('DEMO_CROPS_DIR', '')
        demo_database = os.environ.get('DEMO_DATABASE_FILE', '')
        demo_embeddings = os.environ.get('DEMO_EMBEDDINGS_FILE', '')
        
        # If demo data not in env, check for local demo directory
        if not demo_crops:
            demo_dir = Path(__file__).parent / "person_recognition_demo_output"
            if demo_dir.exists():
                demo_crops = str(demo_dir / "person_crops")
                demo_database = str(demo_dir / "persons.json")
                demo_embeddings = str(demo_dir / "person_embeddings.json")
        
        # Directory input
        crops_dir = st.sidebar.text_input(
            "Crops Directory:",
            value=getattr(st.session_state, 'crops_dir', demo_crops),
            help="Path to directory containing person crop images"
        )
        
        database_file = st.sidebar.text_input(
            "Database File:",
            value=getattr(st.session_state, 'database_file', demo_database or 'persons.json'),
            help="Path to person database JSON file"
        )
        
        embeddings_file = st.sidebar.text_input(
            "Embeddings File (optional):",
            value=getattr(st.session_state, 'embeddings_file', demo_embeddings),
            help="Path to embeddings JSON file for clustering"
        )
        
        # Quick load demo data button
        if demo_crops and os.path.exists(demo_crops):
            if st.sidebar.button("🎯 Load Demo Data", help="Load data from person recognition demo"):
                self._load_data(demo_crops, demo_database, demo_embeddings)
        
        # Load data button
        if st.sidebar.button("🔄 Load Data", type="primary"):
            self._load_data(crops_dir, database_file, embeddings_file)
        
        if st.session_state.crops_loaded:
            st.sidebar.success(f"✅ Loaded {len(self.crops)} crops")
            
            # View controls
            st.sidebar.subheader("👁️ View Controls")
            
            view_options = ["all", "unlabeled", "labeled", "clusters"]
            current_view = st.sidebar.selectbox(
                "View:",
                options=view_options,
                index=view_options.index(st.session_state.current_view)
            )
            
            if current_view != st.session_state.current_view:
                st.session_state.current_view = current_view
                st.rerun()
            
            # Search filter
            search_filter = st.sidebar.text_input(
                "🔍 Search:",
                help="Filter by person name or filename"
            )
            
            # Clustering controls
            st.sidebar.subheader("🔗 Clustering")
            
            if st.sidebar.button("Generate Clusters", help="Group similar faces together"):
                self._generate_clusters()
            
            if st.session_state.clusters_generated:
                st.sidebar.success(f"✅ {len(self.clusters)} clusters generated")
            
            # Statistics
            self._display_sidebar_stats()
            
            # Actions
            st.sidebar.subheader("💾 Actions")
            
            if st.sidebar.button("Save Database"):
                self._save_database()
            
            if st.sidebar.button("Export Labels"):
                self._export_labels()
            
            # Apply search filter
            if search_filter:
                self._apply_search_filter(search_filter)
            else:
                self._update_filtered_crops()
    
    def _display_sidebar_stats(self):
        """Display statistics in sidebar."""
        st.sidebar.subheader("📊 Statistics")
        
        total_crops = len(self.crops)
        labeled_crops = len([c for c in self.crops if c.person_name])
        unlabeled_crops = total_crops - labeled_crops
        
        # Count persons
        person_counts = {}
        for crop in self.crops:
            if crop.person_name:
                person_counts[crop.person_name] = person_counts.get(crop.person_name, 0) + 1
        
        st.sidebar.metric("Total Crops", total_crops)
        st.sidebar.metric("Labeled", labeled_crops)
        st.sidebar.metric("Unlabeled", unlabeled_crops)
        st.sidebar.metric("Unique Persons", len(person_counts))
        
        if person_counts:
            st.sidebar.write("**Person Counts:**")
            for person, count in sorted(person_counts.items()):
                st.sidebar.write(f"• {person}: {count}")
    
    def _display_welcome_screen(self):
        """Display welcome screen when no data is loaded."""
        # Check if demo data exists
        demo_dir = Path(__file__).parent / "person_recognition_demo_output"
        has_demo_data = demo_dir.exists()
        
        st.markdown("""
        ## Welcome to the Person Labeling Tool! 👋
        
        This tool helps you manually label person detections for building person recognition datasets.
        
        ### Features:
        - 📸 **Visual browsing** of person crops in grid and cluster views
        - 🏷️ **Manual labeling** with individual and batch name assignment
        - 🔗 **Smart clustering** of similar faces using AI embeddings
        - 📊 **Batch operations** for efficient labeling workflows
        - 💾 **Database integration** with automatic saving and export
        - 📈 **Real-time statistics** showing labeling progress
        """)
        
        if has_demo_data:
            st.success("""
            ### 🎯 Demo Data Available!
            
            Found demo data from your person recognition pipeline. Use the **"Load Demo Data"** button in the sidebar to get started immediately with:
            - 20 person crop images from surveillance video
            - Pre-generated embeddings for clustering
            - Existing person database with family members
            """)
        
        st.markdown("""
        ### Getting Started:
        
        **Option 1: Use Demo Data (Recommended)**
        1. Click **"🎯 Load Demo Data"** in the sidebar to load example data
        2. Explore the interface with real person crops
        3. Try clustering and labeling features
        
        **Option 2: Load Your Own Data**
        1. Enter the path to your person crops directory in the sidebar
        2. Specify your person database file (will be created if it doesn't exist) 
        3. Optionally add an embeddings file for clustering capabilities
        4. Click **"🔄 Load Data"** to begin
        
        ### Workflow Tips:
        - **Start with clustering** to group similar faces together
        - **Use batch labeling** to efficiently label entire clusters
        - **Save frequently** to preserve your work
        - **Export labels** when ready for training ML models
        
        ### Example Paths:
        ```
        Crops Directory: person_recognition_demo_output/person_crops
        Database File: person_recognition_demo_output/persons.json  
        Embeddings File: person_recognition_demo_output/person_embeddings.json
        ```
        """)
        
        if not has_demo_data:
            st.info("""
            💡 **No demo data found.** Run the person recognition demo first to generate sample data:
            
            ```bash
            python person_recognition_demo.py
            ```
            """)
    
    def _load_data(self, crops_dir: str, database_file: str, embeddings_file: str = ""):
        """Load data from specified paths."""
        try:
            # Store paths in session state
            st.session_state.crops_dir = crops_dir
            st.session_state.database_file = database_file
            st.session_state.embeddings_file = embeddings_file
            
            # Load person database
            if database_file:
                self.person_db = PersonDatabase(database_file)
                st.sidebar.success(f"Loaded database: {len(self.person_db.persons)} persons")
            
            # Load crops
            if crops_dir and os.path.exists(crops_dir):
                self._load_crops_from_directory(crops_dir)
                st.sidebar.success(f"Loaded {len(self.crops)} crops")
            else:
                st.sidebar.error("Crops directory not found")
                return
            
            # Load embeddings if specified
            if embeddings_file and os.path.exists(embeddings_file):
                self._load_embeddings_from_file(embeddings_file)
                st.sidebar.success("Embeddings loaded")
            
            # Load existing labels from database
            self._load_existing_labels()
            
            st.session_state.crops_loaded = True
            self._update_filtered_crops()
            st.rerun()
            
        except Exception as e:
            st.sidebar.error(f"Failed to load data: {e}")
            logger.error(f"Data loading error: {e}")
    
    def _load_crops_from_directory(self, directory: str):
        """Load person crops from directory."""
        crop_files = list(Path(directory).glob("*.jpg"))
        self.crops = []
        
        for crop_file in crop_files:
            crop_data = PersonCropData(str(crop_file))
            self.crops.append(crop_data)
        
        logger.info(f"Loaded {len(self.crops)} crop images from {directory}")
    
    def _load_embeddings_from_file(self, filename: str):
        """Load embeddings and associate with crops."""
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
            
        except Exception as e:
            logger.error(f"Failed to load embeddings: {e}")
    
    def _load_existing_labels(self):
        """Load existing labels from database."""
        if not self.person_db:
            return
        
        # Get all track labels
        for crop in self.crops:
            if crop.track_id and crop.video_name:
                track_label = self.person_db.get_track_label(
                    f"{crop.video_name}.mp4", crop.track_id
                )
                if track_label:
                    crop.person_name = track_label.person_name
                    crop.person_id = track_label.person_id
                    crop.confidence = track_label.confidence
                    crop.labeled_by = track_label.labeled_by
                    crop.notes = track_label.notes
    
    def _apply_search_filter(self, search_filter: str):
        """Apply search filter to crops."""
        search_lower = search_filter.lower()
        self.filtered_crops = [
            crop for crop in self.crops
            if search_lower in (crop.person_name or "").lower()
            or search_lower in crop.filename.lower()
            or search_lower in crop.video_name.lower()
        ]
        self._apply_view_filter()
    
    def _update_filtered_crops(self):
        """Update filtered crops based on current view."""
        self.filtered_crops = self.crops[:]
        self._apply_view_filter()
    
    def _apply_view_filter(self):
        """Apply view filter to filtered crops."""
        if st.session_state.current_view == "unlabeled":
            self.filtered_crops = [crop for crop in self.filtered_crops if not crop.person_name]
        elif st.session_state.current_view == "labeled":
            self.filtered_crops = [crop for crop in self.filtered_crops if crop.person_name]
        elif st.session_state.current_view == "clusters":
            if self.clusters:
                cluster_crops = []
                for cluster in self.clusters:
                    cluster_crops.extend(cluster)
                self.filtered_crops = [crop for crop in self.filtered_crops if crop in cluster_crops]
    
    def _display_main_interface(self):
        """Display the main labeling interface."""
        # Update filtered crops based on current view
        self._update_filtered_crops()
        
        # Display view information
        st.subheader(f"📋 {st.session_state.current_view.title()} View")
        st.write(f"Showing {len(self.filtered_crops)} of {len(self.crops)} crops")
        
        if not self.filtered_crops:
            st.info("No crops to display with current filters.")
            return
        
        # Batch labeling controls
        self._display_batch_controls()
        
        # Display crops based on current view
        if st.session_state.current_view == "clusters" and self.clusters:
            self._display_cluster_view()
        else:
            self._display_grid_view()
    
    def _display_batch_controls(self):
        """Display batch labeling controls."""
        st.markdown("---")
        col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
        
        with col1:
            # Person name input for batch labeling
            person_name = st.text_input(
                "Person Name for Batch Labeling:",
                key="batch_person_name",
                help="Enter name to apply to selected crops"
            )
        
        with col2:
            # Quick person buttons
            if self.person_db:
                persons = self.person_db.list_persons()
                if persons:
                    selected_person = st.selectbox(
                        "Quick Select Person:",
                        options=[""] + [p.name for p in persons[:10]],
                        key="quick_person_select"
                    )
                    if selected_person:
                        st.session_state.batch_person_name = selected_person
        
        with col3:
            if st.button("🏷️ Apply to Selected", help="Apply name to selected crops"):
                self._apply_batch_label()
        
        with col4:
            if st.button("🗑️ Clear Selected", help="Remove labels from selected crops"):
                self._clear_selected_labels()
        
        st.markdown("---")
    
    def _display_grid_view(self):
        """Display crops in a grid layout."""
        # Calculate grid layout
        cols_per_row = 4
        crops_to_show = self.filtered_crops[:100]  # Limit for performance
        
        if len(self.filtered_crops) > 100:
            st.warning(f"Showing first 100 of {len(self.filtered_crops)} crops for performance.")
        
        # Display crops in grid
        for i in range(0, len(crops_to_show), cols_per_row):
            cols = st.columns(cols_per_row)
            
            for j, col in enumerate(cols):
                if i + j < len(crops_to_show):
                    crop = crops_to_show[i + j]
                    self._display_crop_card(crop, col, i + j)
    
    def _display_cluster_view(self):
        """Display crops grouped by clusters."""
        st.subheader("🔗 Clusters")
        
        for cluster_idx, cluster in enumerate(self.clusters):
            # Filter cluster crops by current filters
            filtered_cluster = [crop for crop in cluster if crop in self.filtered_crops]
            
            if not filtered_cluster:
                continue
            
            # Cluster header
            st.markdown(f"### Cluster {cluster_idx + 1} ({len(filtered_cluster)} crops)")
            
            # Show cluster labels
            cluster_labels = set(crop.person_name for crop in filtered_cluster if crop.person_name)
            if cluster_labels:
                st.write(f"**Labels in cluster:** {', '.join(cluster_labels)}")
            else:
                st.write("**No labels in cluster**")
            
            # Cluster labeling controls
            col1, col2 = st.columns([3, 1])
            
            with col1:
                cluster_name = st.text_input(
                    f"Label all in cluster {cluster_idx + 1}:",
                    key=f"cluster_name_{cluster_idx}"
                )
            
            with col2:
                if st.button(f"Apply to Cluster", key=f"apply_cluster_{cluster_idx}"):
                    if cluster_name:
                        self._label_cluster(filtered_cluster, cluster_name)
            
            # Display cluster crops
            cols_per_row = 6
            for i in range(0, len(filtered_cluster), cols_per_row):
                cols = st.columns(cols_per_row)
                
                for j, col in enumerate(cols):
                    if i + j < len(filtered_cluster):
                        crop = filtered_cluster[i + j]
                        self._display_crop_card(crop, col, f"cluster_{cluster_idx}_{i + j}", show_checkbox=False)
            
            st.markdown("---")
    
    def _display_crop_card(self, crop: PersonCropData, col, key_suffix, show_checkbox=True):
        """Display a single crop card."""
        with col:
            try:
                # Load and display image
                img = Image.open(crop.image_path)
                img = img.resize((150, 150), Image.Resampling.LANCZOS)
                
                col.image(img, caption=crop.filename, use_container_width=True)
                
                # Selection checkbox
                if show_checkbox:
                    is_selected = col.checkbox(
                        "Select",
                        key=f"select_{key_suffix}",
                        value=crop in st.session_state.selected_crops
                    )
                    
                    # Update selection
                    if is_selected and crop not in st.session_state.selected_crops:
                        st.session_state.selected_crops.append(crop)
                    elif not is_selected and crop in st.session_state.selected_crops:
                        st.session_state.selected_crops.remove(crop)
                
                # Current label
                if crop.person_name:
                    col.success(f"👤 {crop.person_name}")
                else:
                    col.warning("❓ Unlabeled")
                
                # Individual labeling
                new_name = col.text_input(
                    "Name:",
                    value=crop.person_name or "",
                    key=f"name_{key_suffix}",
                    placeholder="Enter name..."
                )
                
                if col.button("Apply", key=f"apply_{key_suffix}", help="Apply name to this crop"):
                    if new_name != crop.person_name:
                        self._apply_individual_label(crop, new_name)
                
                # Show metadata
                if col.button("ℹ️", key=f"info_{key_suffix}", help="Show details"):
                    self._show_crop_details(crop)
                
            except Exception as e:
                col.error(f"Error loading image: {crop.filename}")
                logger.error(f"Failed to load image {crop.image_path}: {e}")
    
    def _apply_individual_label(self, crop: PersonCropData, person_name: str):
        """Apply label to individual crop."""
        if person_name.strip():
            # Get or create person in database
            person_id = None
            if self.person_db:
                person_id = self.person_db.get_person_id_by_name(person_name)
                if not person_id:
                    person_id = self.person_db.add_person(person_name, "Added via labeling tool")
                
                # Add track label to database
                if crop.track_id and crop.video_name:
                    self.person_db.label_track(
                        video_filename=f"{crop.video_name}.mp4",
                        track_id=crop.track_id,
                        person_id=person_id,
                        labeled_by="manual",
                        notes="Labeled via Streamlit GUI"
                    )
            
            crop.person_name = person_name
            crop.person_id = person_id
            crop.labeled_by = "manual"
            
            st.success(f"✅ Labeled {crop.filename} as '{person_name}'")
        else:
            # Remove label
            crop.person_name = None
            crop.person_id = None
            
            # Remove from database
            if self.person_db and crop.track_id and crop.video_name:
                self.person_db.remove_track_label(f"{crop.video_name}.mp4", crop.track_id)
            
            st.info(f"Removed label from {crop.filename}")
        
        st.rerun()
    
    def _apply_batch_label(self):
        """Apply batch label to selected crops."""
        person_name = st.session_state.get('batch_person_name', '').strip()
        selected_crops = st.session_state.selected_crops
        
        if not person_name:
            st.error("Please enter a person name")
            return
        
        if not selected_crops:
            st.error("No crops selected")
            return
        
        # Get or create person in database
        person_id = None
        if self.person_db:
            person_id = self.person_db.get_person_id_by_name(person_name)
            if not person_id:
                person_id = self.person_db.add_person(person_name, "Added via labeling tool")
        
        # Apply labels
        for crop in selected_crops:
            crop.person_name = person_name
            crop.person_id = person_id
            crop.labeled_by = "manual_batch"
            
            # Add to database
            if self.person_db and crop.track_id and crop.video_name:
                self.person_db.label_track(
                    video_filename=f"{crop.video_name}.mp4",
                    track_id=crop.track_id,
                    person_id=person_id,
                    labeled_by="manual_batch",
                    notes="Batch labeled via Streamlit GUI"
                )
        
        st.success(f"✅ Labeled {len(selected_crops)} crops as '{person_name}'")
        st.session_state.selected_crops = []
        st.rerun()
    
    def _clear_selected_labels(self):
        """Clear labels from selected crops."""
        selected_crops = st.session_state.selected_crops
        
        if not selected_crops:
            st.error("No crops selected")
            return
        
        for crop in selected_crops:
            crop.person_name = None
            crop.person_id = None
            crop.notes = ""
            
            # Remove from database
            if self.person_db and crop.track_id and crop.video_name:
                self.person_db.remove_track_label(f"{crop.video_name}.mp4", crop.track_id)
        
        st.success(f"✅ Cleared labels from {len(selected_crops)} crops")
        st.session_state.selected_crops = []
        st.rerun()
    
    def _label_cluster(self, cluster_crops: List[PersonCropData], person_name: str):
        """Label all crops in a cluster."""
        if not person_name.strip():
            st.error("Please enter a person name")
            return
        
        # Get or create person in database
        person_id = None
        if self.person_db:
            person_id = self.person_db.get_person_id_by_name(person_name)
            if not person_id:
                person_id = self.person_db.add_person(person_name, "Added via labeling tool")
        
        # Apply labels to all crops in cluster
        for crop in cluster_crops:
            crop.person_name = person_name
            crop.person_id = person_id
            crop.labeled_by = "manual_cluster"
            
            # Add to database
            if self.person_db and crop.track_id and crop.video_name:
                self.person_db.label_track(
                    video_filename=f"{crop.video_name}.mp4",
                    track_id=crop.track_id,
                    person_id=person_id,
                    labeled_by="manual_cluster",
                    notes="Cluster labeled via Streamlit GUI"
                )
        
        st.success(f"✅ Labeled {len(cluster_crops)} crops in cluster as '{person_name}'")
        st.rerun()
    
    def _generate_clusters(self):
        """Generate clusters based on embeddings."""
        if not any(crop.embedding is not None for crop in self.crops):
            st.error("No embeddings available for clustering")
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
                    person_emb = PersonEmbedding(
                        track_id=crop.track_id or 0,
                        frame_number=crop.frame_number or 0,
                        video_filename=crop.video_name,
                        embedding=crop.embedding
                    )
                    embeddings.append(person_emb)
                    crop_mapping[len(embeddings)-1] = i
            
            if not embeddings:
                st.error("No valid embeddings found")
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
            
            st.session_state.clusters_generated = True
            st.success(f"✅ Generated {len(self.clusters)} clusters")
            st.rerun()
            
        except Exception as e:
            logger.error(f"Clustering failed: {e}")
            st.error(f"Clustering failed: {e}")
    
    def _show_crop_details(self, crop: PersonCropData):
        """Show detailed information about a crop."""
        info = crop.get_display_info()
        st.info(f"""
        **Crop Details:**
        - Filename: {info['filename']}
        - Person: {info['person_name']}
        - Track ID: {info['track_id']}
        - Frame: {info['frame_number']}
        - Video: {info['video_name']}
        - Cluster: {info['cluster_id'] if info['cluster_id'] is not None else 'None'}
        - Notes: {info['notes'] or 'None'}
        """)
    
    def _save_database(self):
        """Save the person database."""
        if self.person_db:
            try:
                self.person_db.save_database()
                st.sidebar.success("✅ Database saved successfully")
            except Exception as e:
                st.sidebar.error(f"❌ Failed to save database: {e}")
        else:
            st.sidebar.error("No database loaded")
    
    def _export_labels(self):
        """Export labeled data."""
        try:
            labeled_crops = [c for c in self.crops if c.person_name]
            
            export_data = {
                'metadata': {
                    'exported_at': datetime.now().isoformat(),
                    'total_crops': len(self.crops),
                    'labeled_crops': len(labeled_crops),
                    'source_directory': getattr(st.session_state, 'crops_dir', '')
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
            
            # Convert to JSON for download
            json_str = json.dumps(export_data, indent=2)
            
            st.sidebar.download_button(
                label="📥 Download Labels JSON",
                data=json_str,
                file_name=f"person_labels_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
            
            st.sidebar.success(f"✅ Ready to download {len(labeled_crops)} labels")
            
        except Exception as e:
            st.sidebar.error(f"❌ Failed to export labels: {e}")


def main():
    """Main function to run the Streamlit app."""
    app = PersonLabelingApp()
    app.run()


if __name__ == "__main__":
    main()