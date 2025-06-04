#!/usr/bin/env python3
"""
Person Detection and Crop Extraction Module

This module provides specialized person detection capabilities built on top of
the existing YOLO detection system. It focuses specifically on extracting and
managing person detections for identification purposes.

Key Features:
- Person-specific filtering from YOLO detections
- Person crop extraction with consistent sizing
- Track-based person grouping for temporal consistency
- Integration with existing tag processing pipeline

Example Usage:
    # Create person detector
    person_detector = PersonDetector(
        base_detector_config=TaggerConfig(model="Yolo11x_Optimized"),
        crop_size=(224, 224),
        min_confidence=0.5
    )
    
    # Process video for person detection
    person_tracks = person_detector.detect_persons_in_video(video_path)
    
    # Extract crops for a specific track
    crops = person_detector.extract_person_crops(video_path, track_id=123)
"""

import os
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import json
from datetime import datetime

from logging_config import create_logger
from tag_processor import TaggerConfig, TagProcessor, VideoTags
from video_metadata import VideoMetadata

logger = create_logger(__name__)


@dataclass
class PersonCrop:
    """Data structure for a person crop extracted from video."""
    
    track_id: int
    frame_number: int
    confidence: float
    bbox: Dict[str, int]  # {x1, y1, x2, y2}
    crop_image: Optional[np.ndarray] = None
    crop_path: Optional[str] = None
    video_filename: str = ""
    timestamp: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
    
    @property
    def crop_size(self) -> Tuple[int, int]:
        """Get the size of the crop (width, height)."""
        if self.crop_image is not None:
            return self.crop_image.shape[1], self.crop_image.shape[0]
        return (0, 0)
    
    @property
    def bbox_area(self) -> int:
        """Calculate the area of the bounding box."""
        return (self.bbox['x2'] - self.bbox['x1']) * (self.bbox['y2'] - self.bbox['y1'])


@dataclass
class PersonTrack:
    """Data structure for a person track across multiple frames."""
    
    track_id: int
    video_filename: str
    crops: List[PersonCrop] = field(default_factory=list)
    person_label: Optional[str] = None  # For manual labeling
    confidence_avg: float = 0.0
    
    def __post_init__(self):
        self.update_statistics()
    
    def add_crop(self, crop: PersonCrop):
        """Add a crop to this track."""
        if crop.track_id != self.track_id:
            raise ValueError(f"Crop track_id {crop.track_id} doesn't match track {self.track_id}")
        self.crops.append(crop)
        self.update_statistics()
    
    def update_statistics(self):
        """Update track statistics based on current crops."""
        if self.crops:
            self.confidence_avg = sum(crop.confidence for crop in self.crops) / len(self.crops)
    
    @property
    def duration_frames(self) -> int:
        """Get the duration of this track in frames."""
        if not self.crops:
            return 0
        frame_numbers = [crop.frame_number for crop in self.crops]
        return max(frame_numbers) - min(frame_numbers) + 1
    
    @property
    def best_crop(self) -> Optional[PersonCrop]:
        """Get the crop with highest confidence."""
        if not self.crops:
            return None
        return max(self.crops, key=lambda c: c.confidence)


class PersonDetector:
    """
    Specialized person detector built on top of YOLO detection system.
    
    This class provides person-specific functionality for:
    - Filtering person detections from general object detection
    - Extracting person crops from video frames
    - Managing person tracks across time
    - Preparing data for person recognition
    """
    
    def __init__(self, 
                 base_detector_config: Optional[TaggerConfig] = None,
                 crop_size: Tuple[int, int] = (224, 224),
                 min_confidence: float = 0.3,
                 min_bbox_area: int = 1000):
        """
        Initialize person detector.
        
        Args:
            base_detector_config: Configuration for underlying YOLO detector
            crop_size: Target size for extracted person crops (width, height)
            min_confidence: Minimum confidence threshold for person detections
            min_bbox_area: Minimum bounding box area (pixels) to filter small detections
        """
        self.base_detector_config = base_detector_config or TaggerConfig(
            model="Yolo11x_Optimized",
            task="Track",
            conf_threshold=0.2,
            batch_size=8
        )
        self.crop_size = crop_size
        self.min_confidence = min_confidence
        self.min_bbox_area = min_bbox_area
        
        # Initialize the base detector
        self.tag_processor = TagProcessor(self.base_detector_config)
        
        logger.info(f"PersonDetector initialized:")
        logger.info(f"  Base model: {self.base_detector_config.model}")
        logger.info(f"  Crop size: {crop_size}")
        logger.info(f"  Min confidence: {min_confidence}")
        logger.info(f"  Min bbox area: {min_bbox_area}")
    
    def detect_persons_in_video(self, video_path: str) -> List[PersonTrack]:
        """
        Detect all persons in a video and organize by tracks.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            List of PersonTrack objects containing person detections
        """
        logger.info(f"Detecting persons in video: {os.path.basename(video_path)}")
        
        # Create VideoMetadata for the file
        video_metadata = VideoMetadata.from_video_file(video_path)
        
        if video_metadata is None:
            raise ValueError(f"Could not load video metadata for: {video_path}")
        
        # Run detection using the tag processor
        video_tags = self.tag_processor.run(video_metadata)
        
        # Extract person detections from tags
        person_tracks = self._extract_person_tracks(video_tags, video_path)
        
        logger.info(f"Found {len(person_tracks)} person tracks:")
        for track in person_tracks:
            logger.info(f"  Track {track.track_id}: {len(track.crops)} detections, "
                       f"avg confidence: {track.confidence_avg:.2f}")
        
        return person_tracks
    
    def _extract_person_tracks(self, video_tags: VideoTags, video_path: str) -> List[PersonTrack]:
        """
        Extract person tracks from video tags.
        
        Args:
            video_tags: VideoTags object containing all detections
            video_path: Path to the video file
            
        Returns:
            List of PersonTrack objects
        """
        video_filename = os.path.basename(video_path)
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
                        if (confidence >= self.min_confidence and 
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
        """
        Validate bounding box meets minimum requirements.
        
        Args:
            bbox: Bounding box dictionary with x1, y1, x2, y2
            
        Returns:
            True if bbox meets minimum requirements
        """
        if not all(key in bbox for key in ['x1', 'y1', 'x2', 'y2']):
            return False
        
        width = bbox['x2'] - bbox['x1']
        height = bbox['y2'] - bbox['y1']
        area = width * height
        
        return area >= self.min_bbox_area and width > 0 and height > 0
    
    def extract_person_crops(self, 
                            video_path: str, 
                            person_tracks: List[PersonTrack],
                            output_dir: Optional[str] = None,
                            max_crops_per_track: int = 10) -> List[PersonTrack]:
        """
        Extract actual image crops for person tracks.
        
        Args:
            video_path: Path to the video file
            person_tracks: List of person tracks to extract crops for
            output_dir: Directory to save crop images (optional)
            max_crops_per_track: Maximum number of crops to extract per track
            
        Returns:
            Updated person tracks with crop images
        """
        logger.info(f"Extracting person crops from {os.path.basename(video_path)}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Group crops by frame number for efficient processing
        frame_crops = {}
        for track in person_tracks:
            # Limit crops per track
            crops_to_process = sorted(track.crops, key=lambda c: c.confidence, reverse=True)[:max_crops_per_track]
            
            for crop in crops_to_process:
                frame_num = crop.frame_number
                if frame_num not in frame_crops:
                    frame_crops[frame_num] = []
                frame_crops[frame_num].append(crop)
        
        # Extract crops frame by frame
        for frame_num in sorted(frame_crops.keys()):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            
            if not ret:
                logger.warning(f"Could not read frame {frame_num}")
                continue
            
            # Extract all crops for this frame
            for crop in frame_crops[frame_num]:
                crop_image = self._extract_crop_from_frame(frame, crop.bbox)
                crop.crop_image = crop_image
                
                # Save crop image if output directory specified
                if output_dir and crop_image is not None:
                    crop.crop_path = self._save_crop_image(crop, output_dir)
        
        cap.release()
        
        # Update tracks with extracted crops
        extracted_count = sum(len([c for c in track.crops if c.crop_image is not None]) 
                            for track in person_tracks)
        logger.info(f"Extracted {extracted_count} person crops")
        
        return person_tracks
    
    def _extract_crop_from_frame(self, frame: np.ndarray, bbox: Dict[str, int]) -> Optional[np.ndarray]:
        """
        Extract and resize crop from frame using bounding box.
        
        Args:
            frame: Video frame as numpy array
            bbox: Bounding box coordinates
            
        Returns:
            Resized crop image or None if extraction failed
        """
        try:
            x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
            
            # Ensure coordinates are within frame bounds
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 <= x1 or y2 <= y1:
                return None
            
            # Extract crop
            crop = frame[y1:y2, x1:x2]
            
            # Resize to target size
            resized_crop = cv2.resize(crop, self.crop_size, interpolation=cv2.INTER_AREA)
            
            return resized_crop
            
        except Exception as e:
            logger.warning(f"Failed to extract crop: {e}")
            return None
    
    def _save_crop_image(self, crop: PersonCrop, output_dir: str) -> str:
        """
        Save crop image to disk.
        
        Args:
            crop: PersonCrop with image data
            output_dir: Directory to save image
            
        Returns:
            Path to saved image file
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate filename: video_trackID_frameNUM.jpg
        video_name = os.path.splitext(crop.video_filename)[0]
        filename = f"{video_name}_track{crop.track_id:03d}_frame{crop.frame_number:06d}.jpg"
        filepath = os.path.join(output_dir, filename)
        
        # Save image
        cv2.imwrite(filepath, crop.crop_image)
        
        return filepath
    
    def save_person_tracks(self, person_tracks: List[PersonTrack], output_file: str):
        """
        Save person tracks to JSON file for later processing.
        
        Args:
            person_tracks: List of person tracks to save
            output_file: Path to output JSON file
        """
        data = {
            'timestamp': datetime.now().isoformat(),
            'detector_config': {
                'model': self.base_detector_config.model,
                'crop_size': self.crop_size,
                'min_confidence': self.min_confidence,
                'min_bbox_area': self.min_bbox_area
            },
            'tracks': []
        }
        
        for track in person_tracks:
            track_data = {
                'track_id': track.track_id,
                'video_filename': track.video_filename,
                'person_label': track.person_label,
                'confidence_avg': track.confidence_avg,
                'duration_frames': track.duration_frames,
                'crops': []
            }
            
            for crop in track.crops:
                crop_data = {
                    'track_id': crop.track_id,
                    'frame_number': crop.frame_number,
                    'confidence': crop.confidence,
                    'bbox': crop.bbox,
                    'crop_path': crop.crop_path,
                    'crop_size': crop.crop_size,
                    'timestamp': crop.timestamp
                }
                track_data['crops'].append(crop_data)
            
            data['tracks'].append(track_data)
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved {len(person_tracks)} person tracks to {output_file}")
    
    @classmethod
    def load_person_tracks(cls, input_file: str) -> List[PersonTrack]:
        """
        Load person tracks from JSON file.
        
        Args:
            input_file: Path to JSON file containing person tracks
            
        Returns:
            List of PersonTrack objects
        """
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        tracks = []
        for track_data in data['tracks']:
            track = PersonTrack(
                track_id=track_data['track_id'],
                video_filename=track_data['video_filename'],
                person_label=track_data.get('person_label')
            )
            
            for crop_data in track_data['crops']:
                crop = PersonCrop(
                    track_id=crop_data['track_id'],
                    frame_number=crop_data['frame_number'],
                    confidence=crop_data['confidence'],
                    bbox=crop_data['bbox'],
                    crop_path=crop_data.get('crop_path'),
                    video_filename=track.video_filename,
                    timestamp=crop_data.get('timestamp')
                )
                track.add_crop(crop)
            
            tracks.append(track)
        
        logger.info(f"Loaded {len(tracks)} person tracks from {input_file}")
        return tracks