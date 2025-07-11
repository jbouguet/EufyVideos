#!/usr/bin/env python3
"""
This module provides Database management through VideoDatabase and VideoDatabaseList classes

The library supports both direct video file scanning and metadata file operations,
offering flexibility in how video collections are managed and accessed.

Example Usage:
    # Load videos from a directory
    videos = VideoMetadata.load_videos_from_directories('/path/to/videos')

    # Create a selector for filtering
    selector = VideoSelector(
        devices=['Backyard'],
        date_range=DateRange(start='2023-01-01', end='2023-12-31'),
        time_range=TimeRange(start='08:00:00', end='17:00:00')
    )

    # Filter videos
    filtered_videos = VideoFilter.by_selectors(videos, [selector])

    # Export to metadata file
    VideoMetadata.export_videos_to_metadata_file(filtered_videos, 'metadata.csv')
"""
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Union, cast

import yaml

from logging_config import create_logger
from video_metadata import VideoMetadata

logger = create_logger(__name__)


def clean_none_values(d):
    """Remove None values from dict and convert tuples to lists."""
    if isinstance(d, dict):
        return {k: clean_none_values(v) for k, v in d.items() if v is not None}
    elif isinstance(d, list):
        return [clean_none_values(item) for item in d]
    elif isinstance(d, tuple):
        return list(d)
    return d


@dataclass
class VideoDatabase:
    """
    Manages a collection of videos from directories or metadata files.

    Provides flexibility in how videos are loaded:
    - Can load directly from video directories
    - Can load from pre-generated metadata files
    - Can force directory scanning even when metadata file exists

    Example:
        db = VideoDatabase(
            video_directories=['/path/to/videos'],
            video_metadata_file='metadata.csv',
            force_video_directories_scanning=False
        )
        videos = db.load_videos()
    """

    video_directories: Optional[Union[str, List[str]]]
    video_metadata_file: Optional[str]
    force_video_directories_scanning: bool = False

    @classmethod
    def from_dict(cls, video_database_dict: Any) -> "VideoDatabase":
        """Create VideoDatabase from dictionary representation."""
        if not isinstance(video_database_dict, dict):
            raise ValueError("Input must be a dictionary")
        video_metadata_file = None
        video_directories = None
        if "video_metadata_file" in video_database_dict:
            video_metadata_file = video_database_dict["video_metadata_file"]
        if "video_directories" in video_database_dict:
            video_directories = video_database_dict["video_directories"]
            video_directories = (
                [video_directories]
                if isinstance(video_directories, str)
                else video_directories
            )
        return cls(
            video_directories=video_directories,
            video_metadata_file=video_metadata_file,
            force_video_directories_scanning=video_database_dict.get(
                "force_video_directories_scanning", False
            ),
        )

    def load_videos(
        self, corrupted_files: Optional[List[str]] = None
    ) -> Union[List[VideoMetadata], None]:
        """
        Load videos based on configuration.

        The loading strategy is determined by:
        1. Existence of metadata file
        2. Existence of video directories
        3. force_video_directories_scanning flag

        Returns None if no valid source is available.

        Args:
            corrupted_files: List to store paths of corrupted/unreadable files

        Returns:
            List of VideoMetadata objects or None if no valid source

        Raises:
            VideoLoadError: If there's an error loading videos
            MetadataFileError: If there's an error with metadata files
        """
        if corrupted_files is None:
            corrupted_files = []
        valid_video_meta_file: bool = (
            self.video_metadata_file is not None
            and os.path.exists(self.video_metadata_file)
        )
        valid_video_directories: bool = False
        if self.video_directories is not None:
            directories = (
                [self.video_directories]
                if isinstance(self.video_directories, str)
                else self.video_directories
            )
            valid_video_directories = all(os.path.exists(dir) for dir in directories)

        nothing_can_be_done: bool = (
            not valid_video_meta_file and not valid_video_directories
        )
        scan_directories: bool = valid_video_directories and (
            self.force_video_directories_scanning or not valid_video_meta_file
        )

        if nothing_can_be_done:
            return None

        videos: List[VideoMetadata] = []
        if scan_directories and self.video_directories is not None:
            directories = (
                [self.video_directories]
                if isinstance(self.video_directories, str)
                else self.video_directories
            )
            videos = VideoMetadata.load_videos_from_directories(
                directories, corrupted_files
            )
            logger.info(
                f"{len(videos):,} videos loaded from {len(directories)} directories"
            )
            if self.video_metadata_file is not None:
                logger.info(f"Saving videos to {self.video_metadata_file}")
                VideoMetadata.export_videos_to_metadata_file(
                    videos, self.video_metadata_file
                )
        elif self.video_metadata_file is not None:
            videos = VideoMetadata.load_videos_from_metadata_files(
                self.video_metadata_file, corrupted_files
            )
            logger.info(
                f"{len(videos):,} videos loaded from {self.video_metadata_file}"
            )

        return VideoMetadata.clean_and_sort(videos) if videos else None

    def to_file(self, video_database_filename: str) -> None:
        """
        Save database configuration to YAML file.

        Raises:
            IOError: If there's an error writing to the file
        """
        with open(video_database_filename, "w") as f:
            yaml.dump(clean_none_values(asdict(self)), f, default_flow_style=False)

    @classmethod
    def from_file(cls, video_database_filename: str) -> "VideoDatabase":
        """
        Load database configuration from YAML file.

        Raises:
            IOError: If there's an error reading from the file
            ValueError: If the file contains invalid YAML or missing required fields
        """
        with open(video_database_filename, "r") as f:
            config = yaml.safe_load(f)
            if not isinstance(config, dict):
                raise ValueError(f"Invalid YAML format in {video_database_filename}")
            return cls.from_dict(config)


@dataclass
class VideoDatabaseList:
    """
    Manages multiple VideoDatabase instances.

    Allows for flexible organization of video collections:
    - Some collections can be configured for frequent scanning
    - Others can use cached metadata for efficiency
    - Provides unified interface for loading from multiple sources

    Example:
        db_list = VideoDatabaseList([
            VideoDatabase(video_directories=['/path/to/active/videos'], force_video_directories_scanning=True),
            VideoDatabase(video_metadata_file='archived_videos.csv', force_video_directories_scanning=False)
        ])
        all_videos = db_list.load_videos()
    """

    video_database_list: List[VideoDatabase]

    @classmethod
    def from_dict(cls, video_database_list_dict: Any) -> "VideoDatabaseList":
        """Create VideoDatabaseList from dictionary representation."""
        if not isinstance(video_database_list_dict, (list, tuple)):
            raise ValueError("Input must be a list or tuple")
        video_database_list_dict = cast(
            Sequence[Dict[str, Any]], video_database_list_dict
        )
        video_database_list: List[VideoDatabase] = []
        for video_database_dict in video_database_list_dict:
            video_database_list.append(VideoDatabase.from_dict(video_database_dict))
        return cls(video_database_list=video_database_list)

    def load_videos(
        self, corrupted_files: Optional[List[str]] = None
    ) -> Union[List[VideoMetadata], None]:
        """
        Load videos from all databases in the list.

        Returns:
            Combined list of videos from all databases, sorted by datetime
        """
        if corrupted_files is None:
            corrupted_files = []
        videos_all: List[VideoMetadata] = []
        for video_dir in self.video_database_list:
            videos = video_dir.load_videos(corrupted_files)
            if videos is not None:
                videos_all.extend(videos)
        return VideoMetadata.clean_and_sort(videos_all) if videos_all else None

    def to_file(self, video_database_list_filename: str):
        """Save database list configuration to YAML file."""
        with open(video_database_list_filename, "w") as f:
            yaml.dump(
                clean_none_values(asdict(self)),
                f,
                default_flow_style=False,
                sort_keys=False,
            )

    @classmethod
    def from_file(cls, video_database_list_filename: str) -> "VideoDatabaseList":
        """
        Load database list configuration from YAML file.

        Raises:
            IOError: If there's an error reading from the file
            ValueError: If the file contains invalid YAML or missing required fields
        """
        with open(video_database_list_filename, "r") as f:
            config = yaml.safe_load(f)
            if not isinstance(config, list):
                raise ValueError(
                    f"Invalid YAML format in {video_database_list_filename}"
                )
            return cls.from_dict(config)
