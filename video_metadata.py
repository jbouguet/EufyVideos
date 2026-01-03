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
from config import Config
from dateutil.tz import tzlocal, tzutc
from logging_config import create_logger
from tqdm import tqdm

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

    Examples:
        # Simple constructor - auto-populates from video file
        video = VideoMetadata(full_path='path/to/video.mp4')

        # Explicit constructor - all fields specified
        video = VideoMetadata(
            filename='video.mp4',
            full_path='path/to/video.mp4',
            device='Camera1',
            ...
        )

        # Classmethod (still available for clarity)
        video = VideoMetadata.from_video_file('path/to/video.mp4')
    """

    # Only full_path is required; all others have defaults
    full_path: str
    filename: str = ""
    device: str = ""
    datetime_obj: datetime_type = field(default_factory=lambda: datetime_type.min)
    serial: str = ""
    file_size: float = 0.0
    width: int = 0
    height: int = 0
    frame_count: int = 0
    duration: timedelta_type = field(default_factory=lambda: timedelta_type(0))
    fps: float = 0.0
    video_codec: str = ""

    # tags[frame_number][hash_key] is a tag in frame frame_number in the video with the hash key hash_key.
    # The choice of a nested dictionary data structuire, combined with a suitable construction of the
    # hash_key (via the function VideoTags.compute_tag_hash()) guarrantee no exact tag duplication.
    # The frame_number keys are kept sorted in the dictionary to facilitate its use by the the visualization
    # method TagVisualizer.generate_video().
    # The tags are populated via the method merge_new_tags() that is called by VideoTags.to_videos().
    tags: Dict[int, Dict[int, Dict[str, Any]]] = field(default_factory=dict)

    # Internal flag to track if we should auto-populate
    _auto_populate: bool = field(default=False, init=False, repr=False)

    def __post_init__(self):
        """Auto-populate fields from video file if only full_path was provided."""
        # Skip auto-population if this is already a populated instance
        # (prevents infinite recursion when from_video_file creates an instance)
        if self._auto_populate:
            return

        # Detect if only full_path was provided (all other fields at defaults)
        if (
            self.filename == ""
            and self.device == ""
            and self.serial == ""
            and self.file_size == 0.0
            and self.width == 0
            and self.height == 0
            and self.frame_count == 0
            and self.fps == 0.0
            and self.video_codec == ""
        ):
            # Auto-populate from video file
            populated = VideoMetadata.from_video_file(full_path=self.full_path)
            if populated:
                # Copy all fields from populated instance
                self.filename = populated.filename
                self.device = populated.device
                self.datetime_obj = populated.datetime_obj
                self.serial = populated.serial
                self.file_size = populated.file_size
                self.width = populated.width
                self.height = populated.height
                self.frame_count = populated.frame_count
                self.duration = populated.duration
                self.fps = populated.fps
                self.video_codec = populated.video_codec
                self.tags = populated.tags
                # Mark as populated to avoid re-population
                self._auto_populate = True

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

    def __str__(self) -> str:
        """Return a nicely formatted string representation of the video metadata."""
        return (
            f"VideoMetadata(\n"
            f"  filename      = {self.filename}\n"
            f"  full_path     = {self.full_path}\n"
            f"  device        = {self.device}\n"
            f"  datetime      = {self.datetime_str}\n"
            f"  serial        = {self.serial}\n"
            f"  file_size     = {self.file_size:.2f} MB\n"
            f"  width         = {self.width}\n"
            f"  height        = {self.height}\n"
            f"  frame_count   = {self.frame_count}\n"
            f"  duration      = {self.duration}\n"
            f"  fps           = {self.fps:.2f}\n"
            f"  video_codec   = {self.video_codec}\n"
            f")"
        )

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
    def _is_finite(value: float) -> bool:
        """Check if a value is finite (not NaN or Inf)."""
        import math

        return not (math.isnan(value) or math.isinf(value))

    @staticmethod
    def _parse_full_path(full_path: str) -> Tuple[str, str, Optional[datetime_type]]:
        """Extract filename, serial and time from full_path."""
        filename = os.path.basename(full_path)
        serial_and_datetime = filename.split("_")
        num_parts = len(serial_and_datetime)
        serial: str = serial_and_datetime[0] if num_parts >= 1 else ""
        datetime_obj: Optional[datetime_type] = None
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
        return (filename, serial, datetime_obj)

    @staticmethod
    def _get_file_creation_time(full_path: str) -> Optional[datetime_type]:
        """Extract creation time from file system metadata."""
        try:
            stat = os.stat(full_path)
            ctime = (
                stat.st_birthtime if hasattr(stat, "st_birthtime") else stat.st_mtime
            )
            return datetime_type.fromtimestamp(ctime)
        except OSError as e:
            logger.warning(f"Failed to get file creation time: {e}")
            return None

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
    def from_video_file(cls, full_path: str) -> Optional["VideoMetadata"]:
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
            datetime_obj_default: datetime_type = datetime_type.strptime(
                "19000101000000", "%Y%m%d%H%M%S"
            )
            filename, serial, datetime_obj = cls._parse_full_path(full_path=full_path)

            # Get device name using date-aware lookup
            device: str = serial
            if datetime_obj is None:
                device = Config.get_device_for_date(serial, datetime_obj_default)
            else:
                device = Config.get_device_for_date(serial, datetime_obj)

            result = None
            with capture_stderr() as stderr_capture:
                try:
                    with av.open(full_path) as container:
                        stream = container.streams.video[0]

                        # Get container duration (in seconds)
                        container_duration = (
                            container.duration / 1000000 if container.duration else 0
                        )

                        # Get fps
                        fps_value = float(stream.average_rate or 0)

                        # Get number of frames
                        frame_count = stream.frames or 0

                        # Calculate expected duration from frame count and FPS
                        # Use frames-1 because duration is time between first and last frame
                        calculated_duration = None
                        if frame_count > 0 and fps_value > 0:
                            calculated_duration = float(frame_count - 1) / fps_value

                        logger.debug(f"Container duration: {container_duration}s")
                        logger.debug(f"Frame count: {frame_count}")
                        logger.debug(f"Average FPS: {fps_value}")
                        logger.debug(
                            f"Calculated duration from frames: {calculated_duration}s"
                            if calculated_duration
                            else "Cannot calculate duration from frames"
                        )

                        # Determine which duration to use based on validity checks
                        duration = container_duration
                        duration_source = "container"

                        # Check for various corruption patterns
                        is_corrupted = False
                        corruption_reason = None

                        if container_duration <= 0:
                            is_corrupted = True
                            corruption_reason = "zero or negative"
                        elif not cls._is_finite(container_duration):
                            is_corrupted = True
                            corruption_reason = "not finite (NaN or Inf)"
                        elif calculated_duration is not None:
                            # Check that the container duration is within acceptable bounds.
                            max_reasonable_container_duration = 86400
                            if container_duration > max_reasonable_container_duration:
                                is_corrupted = True
                                corruption_reason = f"container duration {container_duration:.2f}s > {max_reasonable_container_duration:.2f}s"

                        # Use calculated duration if container duration is corrupted
                        if is_corrupted:
                            if calculated_duration is not None:
                                logger.warning(
                                    f"Container duration appears corrupted ({corruption_reason}). "
                                    f"Using calculated duration from frame count: {calculated_duration:.2f}s. "
                                    f"Video: {full_path}"
                                )
                                duration = calculated_duration
                                duration_source = "calculated from frames"
                            else:
                                logger.error(
                                    f"Container duration appears corrupted ({corruption_reason}) "
                                    f"and cannot calculate from frames. Using corrupted value: {container_duration:.2f}s. "
                                    f"Video: {full_path}"
                                )

                        logger.debug(
                            f"Final duration: {duration}s (source: {duration_source})"
                        )

                        codec_name = stream.codec_context.name

                        if datetime_obj is None:
                            datetime_obj = cls._get_video_metadata_time(container)
                            if datetime_obj:
                                logger.debug(
                                    f"Using datetime from video metadata: {datetime_obj}"
                                )

                        if datetime_obj is None:
                            datetime_obj = cls._get_file_creation_time(full_path)
                            if datetime_obj:
                                logger.debug(
                                    f"Using file creation time: {datetime_obj}"
                                )

                        if datetime_obj is None:
                            datetime_obj = datetime_obj_default

                        result = cls(
                            filename=filename,
                            full_path=full_path,
                            device=device,
                            datetime_obj=datetime_obj,
                            serial=serial,
                            file_size=os.path.getsize(full_path) / (1024 * 1024),
                            width=stream.width,
                            height=stream.height,
                            frame_count=frame_count,
                            duration=timedelta_type(seconds=float(duration)),
                            fps=fps_value,
                            video_codec=codec_name,
                        )

                except (av.error.FFmpegError, OSError, Exception) as e:
                    raise VideoError(f"Error processing {full_path}") from e

            # Check for any stderr output after restoring stderr
            error_output = stderr_capture.getvalue()
            if error_output:
                logger.warning(
                    f"LibAV warnings/errors for file {full_path}:\n{error_output}"
                )

            return result

        except Exception as e:
            raise VideoError(f"Error processing {full_path}") from e

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


if __name__ == "__main__":
    # Testing code for the module.
    import logging

    from logging_config import set_logger_level_and_format

    # Set extended logging for this module only.
    set_logger_level_and_format(logger, level=logging.DEBUG, extended_format=True)

    def run_example_1():
        logger.info("*** EXAMPLE 1:")
        video_hevc: str = (
            "/Volumes/SSK Drive/record/Batch044/T8160T1224250195_20251022121824.mp4"
        )
        video_hevc_meta = VideoMetadata(full_path=video_hevc)
        logger.info("Video original:")
        logger.info(video_hevc_meta)

    run_example_1()
