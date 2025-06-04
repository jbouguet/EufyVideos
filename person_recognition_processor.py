#!/usr/bin/env python3
"""
Person Recognition Tag Processor

This module extends the existing tag processing system to include person recognition
capabilities. It integrates seamlessly with the current workflow while adding:

1. Person detection and crop extraction during tag processing
2. Embedding generation for person tracks
3. Automatic person identification using existing database
4. Enhanced tag output with person identity information

Key Features:
- Extends existing TaggerConfig with person recognition options
- Integrates with PersonDetector and PersonEmbeddingGenerator
- Maintains backward compatibility with existing tag processing
- Adds person identity tags to VideoTags output
- Supports both manual and automatic person labeling

Example Usage:
    # Create enhanced configuration
    config = PersonRecognitionConfig(
        model="Yolo11x_Optimized",
        task="Track",
        enable_person_recognition=True,
        person_database_file="persons.json"
    )
    
    # Process videos with person recognition
    processor = PersonRecognitionProcessor(config)
    video_tags = processor.run(video_metadata)
"""

import os
import json
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

from logging_config import create_logger
from tag_processor import TaggerConfig, TagProcessor, VideoTags
from person_detector import PersonDetector, PersonTrack, PersonCrop
from person_embedding import PersonEmbeddingGenerator, PersonEmbedding
from person_database import PersonDatabase
from video_metadata import VideoMetadata

logger = create_logger(__name__)


@dataclass
class PersonRecognitionConfig(TaggerConfig):
    """
    Extended configuration for tag processing with person recognition.
    
    Inherits all standard tag processing options and adds person recognition
    specific parameters.
    """
    
    # Person recognition settings
    enable_person_recognition: bool = False
    person_database_file: Optional[str] = None
    person_embeddings_file: Optional[str] = None
    person_crops_dir: Optional[str] = None
    
    # Person detection parameters
    person_crop_size: tuple = (224, 224)
    person_min_confidence: float = 0.6
    person_min_bbox_area: int = 2000
    max_crops_per_track: int = 10
    
    # Embedding generation parameters
    embedding_device: Optional[str] = None  # "mps", "cuda", "cpu", or None for auto
    embedding_dim: int = 512
    clip_weight: float = 0.7
    reid_weight: float = 0.3
    
    # Person identification parameters
    similarity_threshold: float = 0.75
    auto_label_confidence: float = 0.8
    enable_auto_labeling: bool = True
    
    def get_identifier(self) -> str:
        """Generate a unique identifier including person recognition settings."""
        base_id = super().get_identifier()
        if self.enable_person_recognition:
            return f"{base_id}_PersonRec"
        return base_id
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "PersonRecognitionConfig":
        """Create configuration from dictionary representation."""
        # Get base TaggerConfig parameters
        base_config = TaggerConfig.from_dict(config_dict)
        
        # Extract person recognition specific parameters
        person_params = {
            'enable_person_recognition': config_dict.get('enable_person_recognition', False),
            'person_database_file': config_dict.get('person_database_file'),
            'person_embeddings_file': config_dict.get('person_embeddings_file'),
            'person_crops_dir': config_dict.get('person_crops_dir'),
            'person_crop_size': tuple(config_dict.get('person_crop_size', (224, 224))),
            'person_min_confidence': config_dict.get('person_min_confidence', 0.6),
            'person_min_bbox_area': config_dict.get('person_min_bbox_area', 2000),
            'max_crops_per_track': config_dict.get('max_crops_per_track', 10),
            'embedding_device': config_dict.get('embedding_device'),
            'embedding_dim': config_dict.get('embedding_dim', 512),
            'clip_weight': config_dict.get('clip_weight', 0.7),
            'reid_weight': config_dict.get('reid_weight', 0.3),
            'similarity_threshold': config_dict.get('similarity_threshold', 0.75),
            'auto_label_confidence': config_dict.get('auto_label_confidence', 0.8),
            'enable_auto_labeling': config_dict.get('enable_auto_labeling', True),
        }
        
        # Create enhanced config with both base and person recognition parameters
        return cls(
            model=base_config.model,
            task=base_config.task,
            num_frames_per_second=base_config.num_frames_per_second,
            conf_threshold=base_config.conf_threshold,
            batch_size=base_config.batch_size,
            **person_params
        )


class PersonRecognitionProcessor:
    """
    Enhanced tag processor with integrated person recognition capabilities.
    
    This class extends the standard tag processing workflow to include:
    1. Person detection and tracking
    2. Person crop extraction and embedding generation
    3. Person identification using similarity matching
    4. Enhanced tag output with person identity information
    """
    
    def __init__(self, config: PersonRecognitionConfig):
        """
        Initialize the person recognition processor.
        
        Args:
            config: PersonRecognitionConfig with all processing parameters
        """
        self.config = config
        
        # Initialize base tag processor
        base_config = TaggerConfig(
            model=config.model,
            task=config.task,
            num_frames_per_second=config.num_frames_per_second,
            conf_threshold=config.conf_threshold,
            batch_size=config.batch_size
        )
        self.tag_processor = TagProcessor(base_config)
        
        # Initialize person recognition components if enabled
        self.person_detector = None
        self.embedder = None
        self.person_db = None
        
        if self.config.enable_person_recognition:
            self._initialize_person_recognition()
        
        logger.info(f"PersonRecognitionProcessor initialized:")
        logger.info(f"  Base model: {config.model}")
        logger.info(f"  Person recognition: {'Enabled' if config.enable_person_recognition else 'Disabled'}")
        if config.enable_person_recognition:
            logger.info(f"  Database: {config.person_database_file}")
            logger.info(f"  Auto-labeling: {'Enabled' if config.enable_auto_labeling else 'Disabled'}")
    
    def _initialize_person_recognition(self):
        """Initialize person recognition components."""
        try:
            # Initialize person detector
            self.person_detector = PersonDetector(
                base_detector_config=TaggerConfig(
                    model=self.config.model,
                    task=self.config.task,
                    num_frames_per_second=self.config.num_frames_per_second,
                    conf_threshold=self.config.conf_threshold,
                    batch_size=self.config.batch_size
                ),
                crop_size=self.config.person_crop_size,
                min_confidence=self.config.person_min_confidence,
                min_bbox_area=self.config.person_min_bbox_area
            )
            
            # Initialize embedding generator
            self.embedder = PersonEmbeddingGenerator(
                device=self.config.embedding_device,
                embedding_dim=self.config.embedding_dim,
                clip_weight=self.config.clip_weight,
                reid_weight=self.config.reid_weight
            )
            
            # Initialize person database if specified
            if self.config.person_database_file:
                self.person_db = PersonDatabase(self.config.person_database_file)
                logger.info(f"Loaded person database: {len(self.person_db.persons)} persons, "
                           f"{len(self.person_db.track_labels)} labeled tracks")
            
            logger.info("✅ Person recognition components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize person recognition: {e}")
            self.config.enable_person_recognition = False
            raise
    
    def run(self, video_metadata: VideoMetadata) -> VideoTags:
        """
        Process video with enhanced person recognition capabilities.
        
        Args:
            video_metadata: VideoMetadata object for the video to process
            
        Returns:
            VideoTags with standard object detection tags plus person identity information
        """
        logger.info(f"Processing {video_metadata.filename} with person recognition")
        
        # Step 1: Run standard tag processing
        video_tags = self.tag_processor.run(video_metadata)
        
        # Step 2: Add person recognition if enabled
        if self.config.enable_person_recognition and self.person_detector:
            video_tags = self._enhance_with_person_recognition(video_metadata, video_tags)
        
        return video_tags
    
    def _enhance_with_person_recognition(self, video_metadata: VideoMetadata, video_tags: VideoTags) -> VideoTags:
        """
        Enhance video tags with person recognition information.
        
        Args:
            video_metadata: Original video metadata
            video_tags: Base video tags from standard processing
            
        Returns:
            Enhanced video tags with person identity information
        """
        logger.info("Enhancing tags with person recognition...")
        
        try:
            # Extract person tracks from existing tags
            person_tracks = self._extract_person_tracks_from_tags(video_tags, video_metadata.filename)
            
            if not person_tracks:
                logger.info("No person tracks found in video tags")
                return video_tags
            
            logger.info(f"Found {len(person_tracks)} person tracks for recognition")
            
            # Extract person crops
            if self.config.person_crops_dir:
                # Save crops to specified directory
                crops_dir = os.path.join(self.config.person_crops_dir, 
                                       os.path.splitext(video_metadata.filename)[0])
                person_tracks = self.person_detector.extract_person_crops(
                    video_path=video_metadata.full_path,
                    person_tracks=person_tracks,
                    output_dir=crops_dir,
                    max_crops_per_track=self.config.max_crops_per_track
                )
            else:
                # Extract crops without saving
                person_tracks = self.person_detector.extract_person_crops(
                    video_path=video_metadata.full_path,
                    person_tracks=person_tracks,
                    output_dir=None,
                    max_crops_per_track=self.config.max_crops_per_track
                )
            
            # Generate embeddings
            embeddings = self.embedder.generate_embeddings_for_tracks(person_tracks)
            
            if self.config.person_embeddings_file:
                # Save embeddings
                embeddings_file = self._get_embeddings_filename(video_metadata.filename)
                self.embedder.save_embeddings(embeddings, embeddings_file)
                logger.info(f"Saved {len(embeddings)} embeddings to {embeddings_file}")
            
            # Perform person identification if database is available
            if self.person_db and self.config.enable_auto_labeling:
                identified_tracks = self._identify_persons(person_tracks, embeddings)
                # Add person identity tags to video_tags
                video_tags = self._add_person_identity_tags(video_tags, identified_tracks)
            
            return video_tags
            
        except Exception as e:
            logger.error(f"Person recognition enhancement failed: {e}")
            return video_tags
    
    def _extract_person_tracks_from_tags(self, video_tags: VideoTags, video_filename: str) -> List[PersonTrack]:
        """
        Extract person tracks from existing video tags.
        
        Args:
            video_tags: VideoTags containing object detection results
            video_filename: Name of the video file
            
        Returns:
            List of PersonTrack objects extracted from tags
        """
        tracks_dict = {}
        
        # Process all tags for this video
        for filename, frames in video_tags.tags.items():
            if not filename.endswith(video_filename):
                continue
                
            for frame_num, frame_tags in frames.items():
                for tag_id, tag_data in frame_tags.items():
                    # Filter for person detections only
                    if (tag_data.get('value') == 'person' and 
                        tag_data.get('type') == 'TRACKED_OBJECT'):
                        
                        confidence = float(tag_data.get('confidence', 0))
                        track_id = tag_data.get('track_id')
                        bbox = tag_data.get('bounding_box', {})
                        
                        # Apply filters
                        if (confidence >= self.config.person_min_confidence and 
                            track_id is not None and
                            self._is_valid_bbox(bbox)):
                            
                            # Create person crop data structure
                            crop = PersonCrop(
                                track_id=track_id,
                                frame_number=int(frame_num),
                                confidence=confidence,
                                bbox=bbox,
                                video_filename=video_filename
                            )
                            
                            # Add to tracks
                            if track_id not in tracks_dict:
                                tracks_dict[track_id] = PersonTrack(
                                    track_id=track_id,
                                    video_filename=video_filename
                                )
                            tracks_dict[track_id].add_crop(crop)
        
        return list(tracks_dict.values())
    
    def _is_valid_bbox(self, bbox: Dict[str, int]) -> bool:
        """Validate bounding box meets minimum requirements."""
        if not all(key in bbox for key in ['x1', 'y1', 'x2', 'y2']):
            return False
        
        width = bbox['x2'] - bbox['x1']
        height = bbox['y2'] - bbox['y1']
        area = width * height
        
        return area >= self.config.person_min_bbox_area and width > 0 and height > 0
    
    def _identify_persons(self, person_tracks: List[PersonTrack], embeddings: List[PersonEmbedding]) -> List[PersonTrack]:
        """
        Identify persons using similarity matching against database.
        
        Args:
            person_tracks: List of person tracks to identify
            embeddings: Generated embeddings for the tracks
            
        Returns:
            Person tracks with identity labels added
        """
        logger.info("Performing person identification...")
        
        # Load existing embeddings from database if available
        known_embeddings = self._load_known_embeddings()
        
        if not known_embeddings:
            logger.info("No known embeddings available for identification")
            return person_tracks
        
        # Group embeddings by track_id
        track_embeddings = {}
        for emb in embeddings:
            if emb.track_id not in track_embeddings:
                track_embeddings[emb.track_id] = []
            track_embeddings[emb.track_id].append(emb)
        
        # Identify each track
        identified_count = 0
        for track in person_tracks:
            if track.track_id not in track_embeddings:
                continue
            
            # Get best embedding for this track
            track_embs = track_embeddings[track.track_id]
            best_emb = max(track_embs, key=lambda e: e.embedding_quality)
            
            # Find best match in known embeddings
            best_match, best_similarity = self._find_best_match(best_emb, known_embeddings)
            
            if best_match and best_similarity >= self.config.auto_label_confidence:
                track.person_label = best_match['person_name']
                identified_count += 1
                logger.info(f"Identified track {track.track_id} as {best_match['person_name']} "
                           f"(similarity: {best_similarity:.3f})")
                
                # Add to database
                if self.person_db:
                    self.person_db.label_track(
                        video_filename=track.video_filename,
                        track_id=track.track_id,
                        person_name=track.person_label,
                        confidence=best_similarity,
                        labeled_by="automatic",
                        notes=f"Auto-identified with {best_similarity:.3f} similarity"
                    )
        
        logger.info(f"Successfully identified {identified_count}/{len(person_tracks)} person tracks")
        return person_tracks
    
    def _load_known_embeddings(self) -> List[Dict[str, Any]]:
        """Load known person embeddings from various sources."""
        known_embeddings = []
        
        # Try to load from specified embeddings file
        if self.config.person_embeddings_file and os.path.exists(self.config.person_embeddings_file):
            try:
                embeddings = PersonEmbeddingGenerator.load_embeddings(self.config.person_embeddings_file)
                for emb in embeddings:
                    # Get person name from database
                    track_label = self.person_db.get_track_label(emb.video_filename, emb.track_id)
                    if track_label and track_label.person_name:
                        known_embeddings.append({
                            'embedding': emb.embedding,
                            'person_name': track_label.person_name,
                            'person_id': track_label.person_id,
                            'source': 'embeddings_file'
                        })
            except Exception as e:
                logger.warning(f"Failed to load embeddings from {self.config.person_embeddings_file}: {e}")
        
        # TODO: Add more sources (e.g., pre-computed embeddings directory)
        
        return known_embeddings
    
    def _find_best_match(self, query_embedding: PersonEmbedding, known_embeddings: List[Dict[str, Any]]) -> tuple:
        """
        Find the best matching person for a query embedding.
        
        Args:
            query_embedding: PersonEmbedding to match
            known_embeddings: List of known person embeddings
            
        Returns:
            Tuple of (best_match_dict, similarity_score) or (None, 0.0)
        """
        best_match = None
        best_similarity = 0.0
        
        for known in known_embeddings:
            similarity = self.embedder.compute_similarity(
                query_embedding.embedding, 
                known['embedding']
            )
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = known
        
        return best_match, best_similarity
    
    def _add_person_identity_tags(self, video_tags: VideoTags, identified_tracks: List[PersonTrack]) -> VideoTags:
        """
        Add person identity information to video tags.
        
        Args:
            video_tags: Original video tags
            identified_tracks: Person tracks with identity labels
            
        Returns:
            Enhanced video tags with person identity information
        """
        # Create mapping of track_id to person_label
        track_labels = {track.track_id: track.person_label 
                       for track in identified_tracks if track.person_label}
        
        if not track_labels:
            return video_tags
        
        # Add person identity tags to existing tags
        for filename, frames in video_tags.tags.items():
            for frame_num, frame_tags in frames.items():
                for tag_id, tag_data in frame_tags.items():
                    if (tag_data.get('value') == 'person' and 
                        tag_data.get('type') == 'TRACKED_OBJECT'):
                        
                        track_id = tag_data.get('track_id')
                        if track_id in track_labels:
                            # Add person identity information
                            tag_data['person_identity'] = track_labels[track_id]
                            tag_data['identity_confidence'] = tag_data.get('confidence', 0.0)
                            tag_data['identification_method'] = 'automatic'
        
        return video_tags
    
    def _get_embeddings_filename(self, video_filename: str) -> str:
        """Generate filename for saving embeddings."""
        base_name = os.path.splitext(video_filename)[0]
        config_id = self.config.get_identifier()
        return f"{base_name}_{config_id}_embeddings.json"
    
    def run_batch(self, video_metadata_list: List[VideoMetadata]) -> Dict[str, VideoTags]:
        """
        Process a batch of videos with person recognition.
        
        Args:
            video_metadata_list: List of VideoMetadata objects to process
            
        Returns:
            Dictionary mapping video filenames to VideoTags
        """
        results = {}
        
        logger.info(f"Processing batch of {len(video_metadata_list)} videos with person recognition")
        
        for video_metadata in video_metadata_list:
            try:
                video_tags = self.run(video_metadata)
                results[video_metadata.filename] = video_tags
            except Exception as e:
                logger.error(f"Failed to process {video_metadata.filename}: {e}")
                # Continue with other videos
        
        logger.info(f"Completed batch processing: {len(results)}/{len(video_metadata_list)} successful")
        return results