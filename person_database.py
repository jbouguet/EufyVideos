#!/usr/bin/env python3
"""
Person Database Module

This module provides database functionality for managing person identities,
labels, and embeddings in the surveillance video analysis system.

Key Features:
- Person identity management and labeling
- Track-to-person mapping
- Embedding storage and retrieval
- Label verification and correction workflows
- Statistics and reporting

Example Usage:
    # Create database
    db = PersonDatabase("persons.json")
    
    # Add person identity
    person_id = db.add_person("Jean-Yves Bouguet", description="Father")
    
    # Label a track
    db.label_track(video="video1.mp4", track_id=123, person_id=person_id)
    
    # Get all tracks for a person
    tracks = db.get_person_tracks("Jean-Yves Bouguet")
"""

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple
from pathlib import Path

from logging_config import create_logger

logger = create_logger(__name__)


@dataclass
class PersonIdentity:
    """Data structure for a person's identity information."""
    
    person_id: str
    name: str
    description: str = ""
    aliases: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def update_timestamp(self):
        """Update the last modified timestamp."""
        self.updated_at = datetime.now().isoformat()


@dataclass
class TrackLabel:
    """Data structure for labeling a track with a person identity."""
    
    video_filename: str
    track_id: int
    person_id: Optional[str] = None
    person_name: Optional[str] = None  # Denormalized for quick access
    confidence: float = 1.0  # Manual labels start with 100% confidence
    labeled_by: str = "manual"  # "manual" or "automatic"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    notes: str = ""
    
    @property
    def track_key(self) -> str:
        """Unique key for this track."""
        return f"{self.video_filename}:{self.track_id}"
    
    def update_timestamp(self):
        """Update the last modified timestamp."""
        self.updated_at = datetime.now().isoformat()


@dataclass
class PersonStats:
    """Statistics for a person in the database."""
    
    person_id: str
    name: str
    total_tracks: int = 0
    total_detections: int = 0
    videos_appeared: Set[str] = field(default_factory=set)
    first_seen: Optional[str] = None
    last_seen: Optional[str] = None
    confidence_avg: float = 0.0
    manual_labels: int = 0
    automatic_labels: int = 0


class PersonDatabase:
    """
    Database for managing person identities and track labels.
    
    This class provides a file-based database for storing and managing:
    - Person identities and metadata
    - Track-to-person mappings
    - Labeling confidence and provenance
    - Statistics and analytics
    """
    
    def __init__(self, database_file: str):
        """
        Initialize person database.
        
        Args:
            database_file: Path to JSON file for persistent storage
        """
        self.database_file = database_file
        self.persons: Dict[str, PersonIdentity] = {}
        self.track_labels: Dict[str, TrackLabel] = {}  # Key: "video:track_id"
        self.load_database()
        
        logger.info(f"PersonDatabase initialized with {len(self.persons)} persons "
                   f"and {len(self.track_labels)} labeled tracks")
    
    def load_database(self):
        """Load database from JSON file."""
        if os.path.exists(self.database_file):
            try:
                with open(self.database_file, 'r') as f:
                    data = json.load(f)
                
                # Load persons
                self.persons = {}
                for person_data in data.get('persons', []):
                    person = PersonIdentity(**person_data)
                    self.persons[person.person_id] = person
                
                # Load track labels
                self.track_labels = {}
                for label_data in data.get('track_labels', []):
                    label = TrackLabel(**label_data)
                    self.track_labels[label.track_key] = label
                
                logger.info(f"Loaded database: {len(self.persons)} persons, "
                           f"{len(self.track_labels)} track labels")
                
            except Exception as e:
                logger.error(f"Failed to load database: {e}")
                self.persons = {}
                self.track_labels = {}
        else:
            logger.info("Database file not found, starting with empty database")
            self.persons = {}
            self.track_labels = {}
    
    def save_database(self):
        """Save database to JSON file."""
        try:
            data = {
                'metadata': {
                    'version': '1.0',
                    'created_at': datetime.now().isoformat(),
                    'total_persons': len(self.persons),
                    'total_track_labels': len(self.track_labels)
                },
                'persons': [asdict(person) for person in self.persons.values()],
                'track_labels': [asdict(label) for label in self.track_labels.values()]
            }
            
            # Create backup of existing file
            if os.path.exists(self.database_file):
                backup_file = f"{self.database_file}.backup"
                os.rename(self.database_file, backup_file)
            
            with open(self.database_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            logger.info(f"Saved database: {len(self.persons)} persons, "
                       f"{len(self.track_labels)} track labels")
            
        except Exception as e:
            logger.error(f"Failed to save database: {e}")
            raise
    
    def add_person(self, 
                   name: str, 
                   description: str = "", 
                   aliases: Optional[List[str]] = None) -> str:
        """
        Add a new person to the database.
        
        Args:
            name: Person's name
            description: Optional description
            aliases: Optional list of alternative names
            
        Returns:
            Generated person_id
        """
        # Check if person already exists
        for person in self.persons.values():
            if person.name.lower() == name.lower():
                logger.warning(f"Person '{name}' already exists with ID {person.person_id}")
                return person.person_id
        
        # Generate unique person ID
        person_id = self._generate_person_id(name)
        
        # Create person identity
        person = PersonIdentity(
            person_id=person_id,
            name=name,
            description=description,
            aliases=aliases or []
        )
        
        self.persons[person_id] = person
        self.save_database()
        
        logger.info(f"Added person: {name} (ID: {person_id})")
        return person_id
    
    def _generate_person_id(self, name: str) -> str:
        """Generate a unique person ID from name."""
        # Create base ID from name
        base_id = name.lower().replace(" ", "_").replace("-", "_")
        base_id = "".join(c for c in base_id if c.isalnum() or c == "_")
        
        # Ensure uniqueness
        person_id = base_id
        counter = 1
        while person_id in self.persons:
            person_id = f"{base_id}_{counter}"
            counter += 1
        
        return person_id
    
    def label_track(self, 
                    video_filename: str, 
                    track_id: int, 
                    person_id: Optional[str] = None,
                    person_name: Optional[str] = None,
                    confidence: float = 1.0,
                    labeled_by: str = "manual",
                    notes: str = ""):
        """
        Label a track with a person identity.
        
        Args:
            video_filename: Name of the video file
            track_id: Track ID within the video
            person_id: Person ID (if known)
            person_name: Person name (alternative to person_id)
            confidence: Confidence in the label (0-1)
            labeled_by: Who/what created the label
            notes: Additional notes
        """
        # Resolve person_id from name if needed
        if person_id is None and person_name is not None:
            person_id = self.get_person_id_by_name(person_name)
            if person_id is None:
                logger.warning(f"Person '{person_name}' not found in database")
                return
        
        # Get person name if we have ID
        if person_id is not None and person_name is None:
            person = self.persons.get(person_id)
            person_name = person.name if person else None
        
        # Create track label
        track_key = f"{video_filename}:{track_id}"
        
        if track_key in self.track_labels:
            # Update existing label
            label = self.track_labels[track_key]
            label.person_id = person_id
            label.person_name = person_name
            label.confidence = confidence
            label.labeled_by = labeled_by
            label.notes = notes
            label.update_timestamp()
            logger.info(f"Updated label for track {track_key}: {person_name}")
        else:
            # Create new label
            label = TrackLabel(
                video_filename=video_filename,
                track_id=track_id,
                person_id=person_id,
                person_name=person_name,
                confidence=confidence,
                labeled_by=labeled_by,
                notes=notes
            )
            self.track_labels[track_key] = label
            logger.info(f"Added label for track {track_key}: {person_name}")
        
        self.save_database()
    
    def get_person_id_by_name(self, name: str) -> Optional[str]:
        """Get person ID by name (case-insensitive)."""
        name_lower = name.lower()
        for person in self.persons.values():
            if (person.name.lower() == name_lower or 
                any(alias.lower() == name_lower for alias in person.aliases)):
                return person.person_id
        return None
    
    def get_person_tracks(self, person_identifier: str) -> List[TrackLabel]:
        """
        Get all tracks for a person.
        
        Args:
            person_identifier: Person ID or name
            
        Returns:
            List of TrackLabel objects for this person
        """
        # Try as person ID first, then as name
        person_id = person_identifier
        if person_id not in self.persons:
            person_id = self.get_person_id_by_name(person_identifier)
        
        if person_id is None:
            return []
        
        return [label for label in self.track_labels.values() 
                if label.person_id == person_id]
    
    def get_track_label(self, video_filename: str, track_id: int) -> Optional[TrackLabel]:
        """Get label for a specific track."""
        track_key = f"{video_filename}:{track_id}"
        return self.track_labels.get(track_key)
    
    def remove_person(self, person_identifier: str) -> bool:
        """
        Remove a person and all their track labels.
        
        Args:
            person_identifier: Person ID or name
            
        Returns:
            True if person was removed
        """
        person_id = person_identifier
        if person_id not in self.persons:
            person_id = self.get_person_id_by_name(person_identifier)
        
        if person_id is None:
            return False
        
        # Remove person
        person_name = self.persons[person_id].name
        del self.persons[person_id]
        
        # Remove all track labels for this person
        tracks_to_remove = [key for key, label in self.track_labels.items() 
                           if label.person_id == person_id]
        for key in tracks_to_remove:
            del self.track_labels[key]
        
        self.save_database()
        logger.info(f"Removed person {person_name} and {len(tracks_to_remove)} track labels")
        return True
    
    def remove_track_label(self, video_filename: str, track_id: int) -> bool:
        """Remove label for a specific track."""
        track_key = f"{video_filename}:{track_id}"
        if track_key in self.track_labels:
            del self.track_labels[track_key]
            self.save_database()
            logger.info(f"Removed label for track {track_key}")
            return True
        return False
    
    def get_unlabeled_tracks(self, video_filename: Optional[str] = None) -> List[str]:
        """
        Get list of track keys that are not labeled.
        
        Note: This requires external track information to be useful.
        Currently returns empty list as we don't have track discovery here.
        """
        # This would need integration with track discovery
        # For now, return empty list
        return []
    
    def get_person_statistics(self, person_identifier: str) -> Optional[PersonStats]:
        """
        Get comprehensive statistics for a person.
        
        Args:
            person_identifier: Person ID or name
            
        Returns:
            PersonStats object or None if person not found
        """
        person_id = person_identifier
        if person_id not in self.persons:
            person_id = self.get_person_id_by_name(person_identifier)
        
        if person_id is None:
            return None
        
        person = self.persons[person_id]
        tracks = self.get_person_tracks(person_id)
        
        if not tracks:
            return PersonStats(person_id=person_id, name=person.name)
        
        # Calculate statistics
        videos = set(track.video_filename for track in tracks)
        confidences = [track.confidence for track in tracks]
        manual_count = sum(1 for track in tracks if track.labeled_by == "manual")
        automatic_count = len(tracks) - manual_count
        
        # Sort by creation time to get first/last seen
        sorted_tracks = sorted(tracks, key=lambda t: t.created_at)
        
        stats = PersonStats(
            person_id=person_id,
            name=person.name,
            total_tracks=len(tracks),
            total_detections=len(tracks),  # TODO: Get actual detection count
            videos_appeared=videos,
            first_seen=sorted_tracks[0].created_at,
            last_seen=sorted_tracks[-1].created_at,
            confidence_avg=sum(confidences) / len(confidences),
            manual_labels=manual_count,
            automatic_labels=automatic_count
        )
        
        return stats
    
    def get_database_statistics(self) -> Dict[str, Any]:
        """Get overall database statistics."""
        total_persons = len(self.persons)
        total_tracks = len(self.track_labels)
        
        if total_tracks == 0:
            return {
                'total_persons': total_persons,
                'total_tracks': 0,
                'manual_labels': 0,
                'automatic_labels': 0,
                'videos_with_labels': 0,
                'average_confidence': 0.0
            }
        
        manual_labels = sum(1 for label in self.track_labels.values() 
                          if label.labeled_by == "manual")
        automatic_labels = total_tracks - manual_labels
        
        videos = set(label.video_filename for label in self.track_labels.values())
        confidences = [label.confidence for label in self.track_labels.values()]
        
        return {
            'total_persons': total_persons,
            'total_tracks': total_tracks,
            'manual_labels': manual_labels,
            'automatic_labels': automatic_labels,
            'videos_with_labels': len(videos),
            'average_confidence': sum(confidences) / len(confidences),
            'persons_list': [p.name for p in self.persons.values()]
        }
    
    def export_labels_for_training(self, output_file: str):
        """
        Export labeled tracks in format suitable for ML training.
        
        Args:
            output_file: Path to output JSON file
        """
        training_data = {
            'metadata': {
                'exported_at': datetime.now().isoformat(),
                'total_samples': len(self.track_labels),
                'persons': list(self.persons.keys())
            },
            'samples': []
        }
        
        for label in self.track_labels.values():
            if label.person_id is not None:  # Only export labeled tracks
                sample = {
                    'video_filename': label.video_filename,
                    'track_id': label.track_id,
                    'person_id': label.person_id,
                    'person_name': label.person_name,
                    'confidence': label.confidence,
                    'labeled_by': label.labeled_by
                }
                training_data['samples'].append(sample)
        
        with open(output_file, 'w') as f:
            json.dump(training_data, f, indent=2)
        
        logger.info(f"Exported {len(training_data['samples'])} labeled samples to {output_file}")
    
    def list_persons(self) -> List[PersonIdentity]:
        """Get list of all persons in database."""
        return list(self.persons.values())
    
    def search_persons(self, query: str) -> List[PersonIdentity]:
        """
        Search persons by name or alias.
        
        Args:
            query: Search query (case-insensitive)
            
        Returns:
            List of matching PersonIdentity objects
        """
        query_lower = query.lower()
        matches = []
        
        for person in self.persons.values():
            # Check name
            if query_lower in person.name.lower():
                matches.append(person)
                continue
            
            # Check aliases
            if any(query_lower in alias.lower() for alias in person.aliases):
                matches.append(person)
                continue
            
            # Check description
            if query_lower in person.description.lower():
                matches.append(person)
        
        return matches