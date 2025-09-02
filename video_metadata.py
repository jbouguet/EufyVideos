#!/usr/bin/env python3

"""
A comprehensive library for managing video metadata and collections.

This module provides a robust framework for handling video metadata, including
core video metadata management through the VideoMetadata class

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

import contextlib
import csv
import io
import os
import sys
from dataclasses import dataclass, field
from datetime import date as date_type
from datetime import datetime as datetime_type
from datetime import time as time_type
from datetime import timedelta as timedelta_type
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import av
import av.container
import av.error
import dateutil.parser
from dateutil.tz import tzlocal, tzutc
from tqdm import tqdm

from config import Config
from logging_config import create_logger

logger = create_logger(__name__)


@contextlib.contextmanager
def capture_stderr():
    """Context manager to capture stderr output."""
    stderr_capture = io.StringIO()
    original_stderr = sys.stderr
    sys.stderr = stderr_capture
    try:
        yield stderr_capture
    finally:
        sys.stderr = original_stderr


class VideoError(Exception):
    """Base exception for all video-related operations."""

    def __init__(self, message: str):
        super().__init__(message)


@dataclass
class VideoMetadata:
    """
    Core class representing metadata for a single video file.

    This class encapsulates all relevant metadata for a video file, including:
    - Basic file information (filename, path, size)
    - Video properties (dimensions, frame count, duration, codec)
    - Temporal information (creation date/time)
    - Device information (device name, serial number)
    - Optional metadata (tags)

    The class provides methods for:
    - Loading metadata from video files
    - Exporting/importing metadata to/from files
    - Managing collections of videos
    - Comparing and sorting video entries

    Example:
        video = VideoMetadata.from_video_file('path/to/video.mp4')
        print(f"Video duration: {video.duration}")
        print(f"Recorded on: {video.datetime_str}")
    """

    filename: str
    full_path: str
    device: str
    datetime_obj: datetime_type
    serial: str
    file_size: float
    width: int
    height: int
    frame_count: int
    duration: timedelta_type
    fps: float
    video_codec: str

    # tags[frame_number][hash_key] is a tag in frame frame_number in the video with the hash key hash_key.
    # The choice of a nested dictionary data structuire, combined with a suitable construction of the
    # hash_key (via the function VideoTags.compute_tag_hash()) guarrantee no exact tag duplication.
    # The frame_number keys are kept sorted in the dictionary to facilitate its use by the the visualization
    # method TagVisualizer.generate_video().
    # The tags are populated via the method merge_new_tags() that is called by VideoTags.to_videos().
    tags: Dict[int, Dict[int, Dict[str, Any]]] = field(default_factory=dict)

    @property
    def datetime(self) -> datetime_type:
        """Get the video's datetime object."""
        return self.datetime_obj

    @property
    def date(self) -> date_type:
        """Get the video's date component."""
        return self.datetime_obj.date()

    @property
    def time_obj(self) -> time_type:
        """Get the video's time component."""
        return self.datetime_obj.time()

    @property
    def time(self) -> time_type:
        """Get the video's time component (for backward compatibility)."""
        return self.datetime_obj.time()

    @property
    def date_str(self) -> str:
        """Get formatted date string (YYYY-MM-DD)."""
        return f"{self.datetime_obj.strftime('%Y-%m-%d')}"

    @property
    def time_str(self) -> str:
        """Get formatted time string (HH:MM:SS)."""
        return f"{self.datetime_obj.strftime('%H:%M:%S')}"

    @property
    def datetime_str(self) -> str:
        """Get formatted datetime string (YYYY-MM-DD HH:MM:SS)."""
        return f"{self.datetime_obj.strftime('%Y-%m-%d %H:%M:%S')}"

    def __eq__(self, other: object) -> bool:
        """Compare videos based on filename and datetime."""
        if not isinstance(other, VideoMetadata):
            return NotImplemented
        return self.filename == other.filename and self.datetime == other.datetime

    def __hash__(self) -> int:
        """Hash based on filename and datetime."""
        return hash((self.filename, self.datetime))

    def __lt__(self, other: Union["VideoMetadata", Any]) -> bool:
        """Compare videos for sorting based on datetime."""
        if not isinstance(other, VideoMetadata):
            return NotImplemented
        return self.datetime < other.datetime

    def merge_new_tags(self, more_tags: Dict[int, Dict[int, Dict[str, Any]]]) -> int:
        """
        Merge an incoming set of new tags into the existing set self.tags.
        Any new tag with identical hash_key will be skipped, and frame numbers are kept sorted
        post merge operation.
        """
        num_tags_added = 0
        new_tags = {
            int(frame): {
                int(hash_key): tag
                for hash_key, tag in frame_tags.items()
                if int(hash_key) not in self.tags.get(int(frame), {})
            }
            for frame, frame_tags in more_tags.items()
        }
        for frame, frame_tags in new_tags.items():
            self.tags.setdefault(int(frame), {}).update(frame_tags)
            num_tags_added += len(frame_tags)

        # Sort tags by frame number
        self.tags = dict(sorted(self.tags.items(), key=lambda x: int(x[0])))

        return num_tags_added

    @staticmethod
    def _get_creation_time(file_path: str) -> datetime_type:
        """Extract creation time from file system metadata."""
        try:
            stat = os.stat(file_path)
            ctime = (
                stat.st_birthtime if hasattr(stat, "st_birthtime") else stat.st_mtime
            )
            return datetime_type.fromtimestamp(ctime)
        except OSError as e:
            logger.warning(f"Failed to get file creation time: {e}")
            return datetime_type.strptime("19000101000000", "%Y%m%d%H%M%S")

    @staticmethod
    def _get_video_metadata_time(
        container: av.container.Container,
    ) -> Optional[datetime_type]:
        """Extract datetime from video container metadata."""
        try:
            if hasattr(container, "metadata") and "creation_time" in container.metadata:
                creation_time = container.metadata.get("creation_time")
                if creation_time:
                    utc_time = dateutil.parser.parse(creation_time)
                    if utc_time.tzinfo is None:
                        utc_time = utc_time.replace(tzinfo=tzutc())
                    local_time = utc_time.astimezone(tzlocal())
                    logger.debug(
                        f"Converted UTC time {utc_time} to local time {local_time}"
                    )
                    return local_time.replace(tzinfo=None)
        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to extract metadata datetime: {e}")
        return None

    @classmethod
    def from_video_file(cls, file_path: str) -> Optional["VideoMetadata"]:
        """
        Create a VideoMetadata instance from a video file.

        Args:
            file_path: Path to the video file

        Returns:
            VideoMetadata instance or None if file cannot be processed

        Raises:
            VideoError: If there's an error processing the video file
        """
        try:
            filename = os.path.basename(file_path)
            serial_and_datetime = filename.split("_")
            num_parts = len(serial_and_datetime)

            serial: str = serial_and_datetime[0] if num_parts >= 1 else ""
            datetime_obj: datetime_type = datetime_type.strptime(
                "19000101000000", "%Y%m%d%H%M%S"
            )

            if num_parts >= 2:
                try:
                    datetime_part = serial_and_datetime[1]
                    datetime_obj = datetime_type.strptime(
                        datetime_part[:14], "%Y%m%d%H%M%S"
                    )
                    logger.debug(
                        f"Successfully parsed datetime from filename: {datetime_obj}"
                    )
                except (ValueError, TypeError) as e:
                    logger.debug(f"Failed to parse datetime from filename: {e}")

            # Get device name using date-aware lookup
            device: str = Config.get_device_for_date(serial, datetime_obj)

            result = None
            with capture_stderr() as stderr_capture:
                try:
                    with av.open(file_path) as container:
                        stream = container.streams.video[0]
                        duration = (
                            container.duration / 1000000 if container.duration else 0
                        )
                        codec_name = stream.codec_context.name

                        if datetime_obj is None:
                            datetime_obj = cls._get_video_metadata_time(container)
                            if datetime_obj:
                                logger.debug(
                                    f"Using datetime from video metadata: {datetime_obj}"
                                )

                        if datetime_obj is None:
                            datetime_obj = cls._get_creation_time(file_path)
                            logger.debug(f"Using file creation time: {datetime_obj}")

                        result = cls(
                            filename=filename,
                            full_path=file_path,
                            device=device,
                            datetime_obj=datetime_obj,
                            serial=serial,
                            file_size=os.path.getsize(file_path) / (1024 * 1024),
                            width=stream.width,
                            height=stream.height,
                            frame_count=stream.frames,
                            duration=timedelta_type(seconds=float(duration)),
                            fps=float(stream.average_rate or 0.0),
                            video_codec=codec_name,
                        )

                except (av.error.FFmpegError, OSError, Exception) as e:
                    raise VideoError(f"Error processing {file_path}") from e

            # Check for any stderr output after restoring stderr
            error_output = stderr_capture.getvalue()
            if error_output:
                logger.warning(
                    f"LibAV warnings/errors for file {file_path}:\n{error_output}"
                )

            return result

        except Exception as e:
            raise VideoError(f"Error processing {file_path}") from e

    @staticmethod
    def clean_and_sort(videos: List["VideoMetadata"]) -> List["VideoMetadata"]:
        """Remove duplicates and sort videos by datetime."""
        unique_videos: Set["VideoMetadata"] = set(videos)
        return sorted(unique_videos)

    @staticmethod
    def load_videos_from_directories(
        directories: Union[str, List[str]], corrupted_files: Optional[List[str]] = None
    ) -> List["VideoMetadata"]:
        """
        Load videos from one or more directories.

        Args:
            directories: Single directory path or list of paths
            corrupted_files: List to store paths of corrupted/unreadable files

        Returns:
            List of VideoMetadata objects, sorted by datetime

        Raises:
            VideoError: If there's an error loading videos
        """
        videos: List[VideoMetadata] = []
        if corrupted_files is None:
            corrupted_files = []
        if isinstance(directories, str):
            directories = [directories]

        supported_extensions = (".mp4", ".mov")

        for directory in directories:
            if not os.path.isdir(directory):
                logger.warning(f"Directory not found: {directory}")
                continue

            video_files = [
                file
                for file in os.listdir(directory)
                if file.lower().endswith(supported_extensions)
            ]
            if not video_files:
                logger.warning(
                    f"No supported video files found in directory: {directory}"
                )
                continue

            for file in tqdm(
                video_files,
                desc=f"Scanning videos in {directory}",
                unit="file",
                colour="green",
                position=0,
                leave=True,
            ):
                try:
                    video = VideoMetadata.from_video_file(os.path.join(directory, file))
                    if video:
                        videos.append(video)
                    else:
                        corrupted_files.append(os.path.join(directory, file))
                except VideoError as e:
                    logger.error(str(e))
                    corrupted_files.append(os.path.join(directory, file))

        return VideoMetadata.clean_and_sort(videos)

    @staticmethod
    def load_videos_from_metadata_files(
        video_metadata_files: Union[str, List[str]],
        corrupted_files: Optional[List[str]] = None,
    ) -> List["VideoMetadata"]:
        """
        Load videos from one or more metadata files.

        Args:
            video_metadata_files: Single metadata file path or list of paths
            corrupted_files: List to store paths of corrupted/unreadable files

        Returns:
            List of VideoMetadata objects, sorted by datetime

        Raises:
            VideoError: If there's an error reading metadata files
        """
        videos: List[VideoMetadata] = []
        if corrupted_files is None:
            corrupted_files = []
        if isinstance(video_metadata_files, str):
            video_metadata_files = [video_metadata_files]

        for video_metadata_file in video_metadata_files:
            if not os.path.isfile(video_metadata_file):
                logger.warning(f"Metadata file not found: {video_metadata_file}")
                corrupted_files.append(video_metadata_file)
                continue

            try:
                with open(video_metadata_file, "r", encoding="utf-8-sig") as file:
                    reader = csv.DictReader(file)
                    for row in reader:
                        try:
                            videos.append(
                                VideoMetadata(
                                    filename=row["Filename"],
                                    full_path=row["Full Path"],
                                    video_codec=row["Video Codec"],
                                    file_size=float(row["File Size"]),
                                    width=int(row["Width"]),
                                    height=int(row["Height"]),
                                    frame_count=int(row["Frame Count"]),
                                    duration=timedelta_type(
                                        seconds=float(row["Duration"])
                                    ),
                                    fps=float(row["FPS"]),
                                    device=row["Device"],
                                    datetime_obj=datetime_type.strptime(
                                        row["Date Time"], "%Y-%m-%d %H:%M:%S"
                                    ),
                                    serial=row["Serial"],
                                )
                            )
                        except (ValueError, KeyError) as e:
                            logger.error(
                                f"Error parsing row in {video_metadata_file}: {str(e)}"
                            )
                            continue

            except (IOError, csv.Error) as e:
                raise VideoError(
                    f"Error reading metadata file {video_metadata_file}"
                ) from e

        return VideoMetadata.clean_and_sort(videos)

    @staticmethod
    def export_videos_to_metadata_file(
        videos: List["VideoMetadata"], video_metadata_file: str
    ) -> None:
        """
        Export video metadata to a CSV file.

        Args:
            videos: List of VideoMetadata objects to export
            video_metadata_file: Path to the output CSV file

        Raises:
            VideoError: If there's an error writing to the file
        """
        try:
            with open(
                video_metadata_file, "w", newline="", encoding="utf-8-sig"
            ) as file:
                writer = csv.writer(file)
                writer.writerow(
                    [
                        "Date Time",
                        "Duration",
                        "Device",
                        "Serial",
                        "Filename",
                        "Full Path",
                        "File Size",
                        "Width",
                        "Height",
                        "Frame Count",
                        "FPS",
                        "Video Codec",
                    ]
                )
                for video in videos:
                    writer.writerow(
                        [
                            video.datetime_str,
                            video.duration.total_seconds(),
                            video.device,
                            video.serial,
                            video.filename,
                            video.full_path,
                            video.file_size,
                            video.width,
                            video.height,
                            video.frame_count,
                            video.fps,
                            video.video_codec,
                        ]
                    )
        except IOError as e:
            raise VideoError(
                f"Error saving video metafile {video_metadata_file}"
            ) from e

    @staticmethod
    def export_videos_to_playlist_file(
        videos: List["VideoMetadata"], playlist_filename: str
    ) -> None:
        """
        Export video list to M3U playlist format.

        Args:
            videos: List of VideoMetadata objects to export
            playlist_filename: Path to the output M3U file

        Raises:
            VideoError: If there's an error writing to the file
        """
        try:
            with open(playlist_filename, "w", encoding="utf-8") as file:
                file.write("#EXTM3U\n")
                for video in videos:
                    file.write(
                        f"#EXTINF:-1,{video.date_str} {video.time_str} {video.device} - {video.filename} - {video.width}x{video.height}\n{video.full_path}\n"
                    )
        except IOError as e:
            raise VideoError(f"Error saving playlist file {playlist_filename}") from e

    @staticmethod
    def differences(
        list1: List["VideoMetadata"], list2: List["VideoMetadata"]
    ) -> Tuple[List["VideoMetadata"], List["VideoMetadata"]]:
        """
        Find videos present in one list but not the other.

        Returns:
            Tuple of (videos in list1 not in list2, videos in list2 not in list1)
        """
        set1: Set["VideoMetadata"] = set(list1)
        set2: Set["VideoMetadata"] = set(list2)
        return sorted(set1 - set2), sorted(set2 - set1)

    @staticmethod
    def repeats(
        source: List["VideoMetadata"], other: List["VideoMetadata"]
    ) -> List["VideoMetadata"]:
        """Find videos from 'other' that already exist in 'source'."""
        source_set: Set["VideoMetadata"] = set(source)
        return sorted(video for video in other if video in source_set)
