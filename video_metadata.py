#!/usr/bin/env python3

"""
A comprehensive library for managing video metadata and collections.

This module provides a robust framework for handling video metadata, including:
- Core video metadata management through the VideoMetadata class
- Filtering capabilities via VideoFilter and VideoSelector classes
- Date and time range specifications using DateRange and TimeRange
- Database management through VideoDatabase and VideoDatabaseList classes

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

import copy
import csv
import io
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import av
import dateutil.parser
import yaml
from dateutil.tz import tzlocal, tzutc
from tqdm import tqdm

from config import Config
from logging_config import create_logger

logger = create_logger(__name__)


class VideoMetadataError(Exception):
    """Base exception for video metadata operations."""

    pass


class VideoLoadError(VideoMetadataError):
    """Raised when a video file cannot be loaded."""

    pass


class MetadataFileError(VideoMetadataError):
    """Raised when there's an error with metadata file operations."""

    pass


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
    datetime_obj: datetime
    serial: str
    file_size: float
    width: int
    height: int
    frame_count: int
    duration: timedelta
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
    def datetime(self) -> datetime:
        """Get the video's datetime object."""
        return self.datetime_obj

    @property
    def date(self):
        """Get the video's date component."""
        return self.datetime_obj.date()

    def time(self):
        """Get the video's time component."""
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

    def __lt__(self, other: "VideoMetadata") -> bool:
        """Compare videos for sorting based on datetime."""
        if not isinstance(other, VideoMetadata):
            return NotImplemented
        return self.datetime < other.datetime

    # Merged an incoming set of new tags more_tags into the existing set self.tags.
    # Any new tag with identical hash_key will be skipped, and frame numbers are kept sorted
    # post merge operation.
    def merge_new_tags(self, more_tags: Dict[int, Dict[int, Dict[str, Any]]]) -> int:
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
    def _get_creation_time(file_path: str) -> datetime:
        """Extract creation time from file system metadata."""
        try:
            stat = os.stat(file_path)
            ctime = (
                stat.st_birthtime if hasattr(stat, "st_birthtime") else stat.st_mtime
            )
            return datetime.fromtimestamp(ctime)
        except OSError as e:
            logger.warning(f"Failed to get file creation time: {e}")
            return datetime.strptime("19000101000000", "%Y%m%d%H%M%S")

    @staticmethod
    def _get_video_metadata_time(
        container: av.container.Container,
    ) -> Optional[datetime]:
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
            VideoLoadError: If there's an error processing the video file
        """
        try:
            filename = os.path.basename(file_path)
            serial_and_datetime = filename.split("_")
            num_parts = len(serial_and_datetime)

            serial = serial_and_datetime[0] if num_parts >= 1 else ""
            device = Config.get_device_dict().get(serial, serial)
            datetime_obj = None

            if num_parts >= 2:
                try:
                    datetime_part = serial_and_datetime[1]
                    datetime_obj = datetime.strptime(datetime_part[:14], "%Y%m%d%H%M%S")
                    logger.debug(
                        f"Successfully parsed datetime from filename: {datetime_obj}"
                    )
                except (ValueError, TypeError) as e:
                    logger.debug(f"Failed to parse datetime from filename: {e}")

            old_stderr = sys.stderr
            sys.stderr = io.StringIO()

            try:
                with av.open(file_path) as container:
                    stream = container.streams.video[0]
                    duration = container.duration / 1000000 if container.duration else 0
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

                    error_output = sys.stderr.getvalue()
                    if error_output:
                        logger.warning(
                            f"LibAV warnings/errors for file {file_path}:\n{error_output}"
                        )

                    return cls(
                        filename=filename,
                        full_path=file_path,
                        device=device,
                        datetime_obj=datetime_obj,
                        serial=serial,
                        file_size=os.path.getsize(file_path) / (1024 * 1024),
                        width=stream.width,
                        height=stream.height,
                        frame_count=stream.frames,
                        duration=timedelta(seconds=float(duration)),
                        fps=float(stream.average_rate),
                        video_codec=codec_name,
                    )

            except av.AVError as e:
                raise VideoLoadError(f"LibAV error processing {file_path}: {str(e)}")
            except OSError as e:
                raise VideoLoadError(f"OS error accessing {file_path}: {str(e)}")
            except Exception as e:
                raise VideoLoadError(
                    f"Unexpected error processing {file_path}: {str(e)}"
                )

        finally:
            sys.stderr = old_stderr

    @staticmethod
    def clean_and_sort(videos: List["VideoMetadata"]) -> List["VideoMetadata"]:
        """Remove duplicates and sort videos by datetime."""
        return sorted(set(videos))

    @staticmethod
    def load_videos_from_directories(
        directories: Union[str, List[str]], corrupted_files: List[str] = []
    ) -> List["VideoMetadata"]:
        """
        Load videos from one or more directories.

        Args:
            directories: Single directory path or list of paths
            corrupted_files: List to store paths of corrupted/unreadable files

        Returns:
            List of VideoMetadata objects, sorted by datetime

        Raises:
            VideoLoadError: If there's an error loading videos
        """
        videos: List[VideoMetadata] = []
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
                except VideoLoadError as e:
                    logger.error(str(e))
                    corrupted_files.append(os.path.join(directory, file))

        return VideoMetadata.clean_and_sort(videos)

    @staticmethod
    def load_videos_from_metadata_files(
        video_metadata_files: Union[str, List[str]], corrupted_files: List[str] = []
    ) -> List["VideoMetadata"]:
        """
        Load videos from one or more metadata files.

        Args:
            video_metadata_files: Single metadata file path or list of paths
            corrupted_files: List to store paths of corrupted/unreadable files

        Returns:
            List of VideoMetadata objects, sorted by datetime

        Raises:
            MetadataFileError: If there's an error reading metadata files
        """
        videos: List[VideoMetadata] = []
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
                                    duration=timedelta(seconds=float(row["Duration"])),
                                    fps=float(row["FPS"]),
                                    device=row["Device"],
                                    datetime_obj=datetime.strptime(
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
                raise MetadataFileError(
                    f"Error reading metadata file {video_metadata_file}: {str(e)}"
                )

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
            MetadataFileError: If there's an error writing to the file
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
            raise MetadataFileError(
                f"Error saving video metafile {video_metadata_file}: {str(e)}"
            )

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
            MetadataFileError: If there's an error writing to the file
        """
        try:
            with open(playlist_filename, "w", encoding="utf-8") as file:
                file.write("#EXTM3U\n")
                for video in videos:
                    file.write(
                        f"#EXTINF:-1,{video.date_str} {video.time_str} {video.device} - {video.filename}\n{video.full_path}\n"
                    )
        except IOError as e:
            raise MetadataFileError(
                f"Error saving playlist file {playlist_filename}: {str(e)}"
            )

    @staticmethod
    def differences(
        list1: List["VideoMetadata"], list2: List["VideoMetadata"]
    ) -> Tuple[List["VideoMetadata"], List["VideoMetadata"]]:
        """
        Find videos present in one list but not the other.

        Returns:
            Tuple of (videos in list1 not in list2, videos in list2 not in list1)
        """
        set1 = set(list1)
        set2 = set(list2)
        return sorted(set1 - set2), sorted(set2 - set1)

    @staticmethod
    def intersection(
        list1: List["VideoMetadata"], list2: List["VideoMetadata"]
    ) -> List["VideoMetadata"]:
        """Find videos present in both lists."""
        return sorted(set(list1).intersection(set(list2)))

    @staticmethod
    def repeats(
        source: List["VideoMetadata"], other: List["VideoMetadata"]
    ) -> List["VideoMetadata"]:
        """Find videos from 'other' that already exist in 'source'."""
        source_set = set(source)
        return sorted(video for video in other if video in source_set)


@dataclass
class DateRange:
    """
    Represents a date range for video filtering.

    Used by VideoSelector to define date-based filtering criteria.
    Dates should be in 'YYYY-MM-DD' format.
    """

    start: str
    end: str

    @classmethod
    def from_dict(cls, date_range_dict: Dict[str, Any]) -> "DateRange":
        """Create DateRange from dictionary representation."""
        return cls(start=date_range_dict["start"], end=date_range_dict["end"])


@dataclass
class TimeRange:
    """
    Represents a time range for video filtering.

    Used by VideoSelector to define time-based filtering criteria.
    Times should be in 'HH:MM:SS' format.
    """

    start: str
    end: str

    @classmethod
    def from_dict(cls, time_range_dict: Dict[str, Any]) -> "TimeRange":
        """Create TimeRange from dictionary representation."""
        return cls(start=time_range_dict["start"], end=time_range_dict["end"])


@dataclass
class VideoSelector:
    """
    Defines criteria for filtering videos.

    Combines multiple filtering criteria (devices, dates, times, filenames, weekdays)
    that are applied together (AND condition). When multiple VideoSelector
    objects are used, they act as OR conditions.

    Example:
        selector = VideoSelector(
            devices=['Front Yard'],
            date_range=DateRange(start='2023-01-01', end='2023-12-31'),
            time_range=TimeRange(start='08:00:00', end='17:00:00'),
            weekdays=['monday', 'wednesday', 'friday']
        )
    """

    devices: Optional[List[str]] = None
    date_range: Optional[DateRange] = None
    time_range: Optional[TimeRange] = None
    filenames: Optional[List[str]] = None
    weekdays: Optional[List[str]] = None

    def _validate_weekdays(self, weekdays):
        """Validate and normalize weekday inputs."""
        if weekdays is None:
            return None

        valid_weekdays = [
            "monday",
            "tuesday",
            "wednesday",
            "thursday",
            "friday",
            "saturday",
            "sunday",
        ]
        if not isinstance(weekdays, (list, tuple)):
            raise ValueError("Weekdays must be provided as a list or tuple")

        normalized_weekdays = [day.lower() for day in weekdays]
        invalid_days = [day for day in normalized_weekdays if day not in valid_weekdays]
        if invalid_days:
            raise ValueError(
                f"Invalid weekdays provided: {invalid_days}. Valid values are: {valid_weekdays}"
            )

        return normalized_weekdays

    @classmethod
    def from_dict(cls, selector_dict: Dict[str, Any]) -> "VideoSelector":
        """
        Create VideoSelector from dictionary representation.

        Handles device list validation and range parsing.

        Raises:
            ValueError: If unknown devices are specified
        """
        date_range = None
        if "date_range" in selector_dict:
            date_range_dict = selector_dict["date_range"]
            if (
                isinstance(date_range_dict, dict)
                and "start" in date_range_dict
                and "end" in date_range_dict
            ):
                date_range = DateRange.from_dict(date_range_dict)
            else:
                logger.warning(
                    "Invalid date_range format. Expected 'start' and 'end' keys."
                )

        time_range = None
        if "time_range" in selector_dict:
            time_range_dict = selector_dict["time_range"]
            if (
                isinstance(time_range_dict, dict)
                and "start" in time_range_dict
                and "end" in time_range_dict
            ):
                time_range = TimeRange.from_dict(time_range_dict)
            else:
                logger.warning(
                    "Invalid time_range format. Expected 'start' and 'end' keys."
                )

        devices = selector_dict.get("devices")
        if devices is None:
            devices = Config.get_all_devices()
        elif isinstance(devices, str):
            devices = [devices]

        unknown_devices = [
            device for device in devices if device not in Config.get_all_devices()
        ]
        if unknown_devices:
            raise ValueError(f"Unknown devices: {', '.join(unknown_devices)}")

        filenames = selector_dict.get("filenames")
        if filenames and not isinstance(filenames, list):
            filenames = [filenames]

        weekdays = selector_dict.get("weekdays")
        if weekdays and not isinstance(weekdays, list):
            weekdays = [weekdays]

        return cls(
            devices=devices,
            date_range=date_range,
            time_range=time_range,
            filenames=filenames,
            weekdays=weekdays,
        )

    @staticmethod
    def log(selectors: Union["VideoSelector", List["VideoSelector"]]) -> None:
        """Log the criteria defined in one or more selectors."""
        selectors = [selectors] if isinstance(selectors, VideoSelector) else selectors
        for i, selector in enumerate(selectors, 1):
            logger.info(f"Selector {i}:")
            if selector.devices:
                logger.info(f"  Devices: {', '.join(selector.devices)}")
            if selector.date_range:
                logger.info(
                    f"  Date range: {selector.date_range.start} to {selector.date_range.end}"
                )
            if selector.time_range:
                logger.info(
                    f"  Time range: {selector.time_range.start} to {selector.time_range.end}"
                )
            if selector.filenames:
                logger.info(f"  Filenames: {', '.join(selector.filenames)}")
            if selector.weekdays:
                logger.info(f"  Weekdays: {', '.join(selector.weekdays)}")


class VideoFilter:
    """
    Provides static methods for filtering video collections.

    Methods can be used individually for simple filtering or combined
    through the by_selector method for complex filtering criteria.

    Example:
        # Filter by single criterion
        daytime_videos = VideoFilter.by_time(videos, '08:00:00', '17:00:00')

        # Filter by multiple criteria using selector
        selector = VideoSelector(
            devices=['Garage'],
            date_range=DateRange(start='2023-01-01', end='2023-12-31')
        )
        filtered_videos = VideoFilter.by_selectors(videos, [selector])
    """

    @staticmethod
    def by_devices(
        videos: List[VideoMetadata], devices: List[str]
    ) -> List[VideoMetadata]:
        """Filter videos by device names."""
        return VideoMetadata.clean_and_sort([v for v in videos if v.device in devices])

    @staticmethod
    def by_date(
        videos: List[VideoMetadata], start_date: str, end_date: str
    ) -> List[VideoMetadata]:
        """Filter videos within a date range."""
        start = datetime.strptime(start_date, "%Y-%m-%d").date()
        end = datetime.strptime(end_date, "%Y-%m-%d").date()
        return VideoMetadata.clean_and_sort(
            [v for v in videos if start <= v.date <= end]
        )

    @staticmethod
    def by_time(
        videos: List[VideoMetadata], start_time: str, end_time: str
    ) -> List[VideoMetadata]:
        """Filter videos within a time range."""
        start = datetime.strptime(start_time, "%H:%M:%S").time()
        end = datetime.strptime(end_time, "%H:%M:%S").time()
        return VideoMetadata.clean_and_sort(
            [v for v in videos if VideoFilter.is_time_in_range(v.time(), start, end)]
        )

    @staticmethod
    def by_filenames(
        videos: List[VideoMetadata], filenames: List[str]
    ) -> List[VideoMetadata]:
        """Filter videos by specific filenames."""
        return VideoMetadata.clean_and_sort(
            [v for v in videos if v.filename in filenames]
        )

    @staticmethod
    def by_weekdays(
        videos: List[VideoMetadata], weekdays: List[str]
    ) -> List[VideoMetadata]:
        """Filter videos by weekdays."""
        normalized_weekdays = [day.lower() for day in weekdays]
        return VideoMetadata.clean_and_sort(
            [
                v
                for v in videos
                if v.datetime.strftime("%A").lower() in normalized_weekdays
            ]
        )

    @staticmethod
    def is_time_in_range(time, start, end) -> bool:
        """Check if time falls within range, handling midnight crossing."""
        if start <= end:
            return start <= time <= end
        else:  # Crosses midnight
            return start <= time or time <= end

    @staticmethod
    def by_selectors(
        videos: List[VideoMetadata],
        selectors: Union[VideoSelector, List[VideoSelector]],
    ) -> List[VideoMetadata]:
        """
        Filter videos using multiple selectors.

        Each selector's criteria are combined with AND logic.
        Multiple selectors are combined with OR logic.

        Example:
            # This will return videos from 'Front Door' within the date range OR videos from 'Back Alleyway'] within the time range
            selector1 = VideoSelector(devices=['Front Door'], date_range=DateRange('2023-01-01', '2023-12-31'))
            selector2 = VideoSelector(devices=['Back Alleyway'], time_range=TimeRange('08:00:00', '17:00:00'))
            filtered_videos = VideoFilter.by_selectors(videos, [selector1, selector2])
        """
        if not selectors:
            return copy.deepcopy(videos)
        selectors = [selectors] if isinstance(selectors, VideoSelector) else selectors

        output_videos: List[VideoMetadata] = []
        for selector in selectors:
            # Make a deep copy for this selector's filtering
            temp_videos = copy.deepcopy(videos)

            if selector.devices:
                temp_videos = VideoFilter.by_devices(temp_videos, selector.devices)
            if selector.date_range:
                temp_videos = VideoFilter.by_date(
                    temp_videos, selector.date_range.start, selector.date_range.end
                )
            if selector.time_range:
                temp_videos = VideoFilter.by_time(
                    temp_videos, selector.time_range.start, selector.time_range.end
                )
            if selector.filenames:
                temp_videos = VideoFilter.by_filenames(temp_videos, selector.filenames)
            if selector.weekdays:
                temp_videos = VideoFilter.by_weekdays(temp_videos, selector.weekdays)

            output_videos.extend(temp_videos)

        return VideoMetadata.clean_and_sort(output_videos)


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

    video_directories: Union[Union[str, List[str]], None]
    video_metadata_file: Optional[str]
    force_video_directories_scanning: bool = False

    @classmethod
    def from_dict(
        cls, video_database_dict: Dict[str, Any]
    ) -> Union["VideoDatabase", None]:
        """Create VideoDatabase from dictionary representation."""
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
        self, corrupted_files: List[str] = []
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
        valid_video_meta_file: bool = (
            self.video_metadata_file is not None
            and os.path.exists(self.video_metadata_file)
        )
        valid_video_directories: bool = self.video_directories is not None
        if valid_video_directories:
            if isinstance(self.video_directories, str):
                self.video_directories = [self.video_directories]
            for dir in self.video_directories:
                if not os.path.exists(dir):
                    valid_video_directories = False
                    continue

        nothing_can_be_done: bool = (
            not valid_video_meta_file and not valid_video_directories
        )
        scan_directories: bool = valid_video_directories and (
            self.force_video_directories_scanning or not valid_video_meta_file
        )

        if nothing_can_be_done:
            return None
        elif scan_directories:
            videos = VideoMetadata.load_videos_from_directories(
                self.video_directories, corrupted_files
            )
            logger.info(
                f"{len(videos)} videos loaded from {len(self.video_directories)} directories"
            )
            if self.video_metadata_file is not None:
                logger.info(f"Saving videos to {self.video_metadata_file}")
                VideoMetadata.export_videos_to_metadata_file(
                    videos, self.video_metadata_file
                )
        else:
            videos = VideoMetadata.load_videos_from_metadata_files(
                self.video_metadata_file, corrupted_files
            )
            logger.info(f"{len(videos)} videos loaded from {self.video_metadata_file}")

        return VideoMetadata.clean_and_sort(videos)

    def to_file(self, video_database_filename: str) -> None:
        """
        Save database configuration to YAML file.

        Raises:
            IOError: If there's an error writing to the file
        """
        with open(video_database_filename, "w") as f:
            yaml.dump(asdict(self), f)

    @classmethod
    def from_file(cls, video_database_filename: str) -> "VideoDatabase":
        """
        Load database configuration from YAML file.

        Raises:
            IOError:  If there's an error reading from the file
        """
        with open(video_database_filename, "r") as f:
            return cls.from_dict(yaml.safe_load(f))


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
    def from_dict(cls, video_database_list_dict: Dict[str, Any]) -> "VideoDatabaseList":
        """Create VideoDatabaseList from dictionary representation."""
        video_database_list: List[VideoDatabase] = []
        for video_database_dict in video_database_list_dict:
            video_database_list.append(VideoDatabase.from_dict(video_database_dict))
        return cls(video_database_list=video_database_list)

    def load_videos(
        self, corrupted_files: List[str] = []
    ) -> Union[List[VideoMetadata], None]:
        """
        Load videos from all databases in the list.

        Returns:
            Combined list of videos from all databases, sorted by datetime
        """
        videos_all = None
        for video_dir in self.video_database_list:
            videos = video_dir.load_videos(corrupted_files)
            if videos is not None:
                if videos_all is None:
                    videos_all = []
                videos_all.extend(videos)
        return VideoMetadata.clean_and_sort(videos_all)

    def to_file(self, video_database_list_filename: str):
        """Save database list configuration to YAML file."""
        with open(video_database_list_filename, "w") as f:
            yaml.dump(asdict(self), f)

    @classmethod
    def from_file(cls, video_database_list_filename: str) -> "VideoDatabaseList":
        """Load database list configuration from YAML file."""
        with open(video_database_list_filename, "r") as f:
            return cls.from_dict(yaml.safe_load(f))


if __name__ == "__main__":
    import glob

    # Script run a health check of the video database and its backup copies.

    def print_dirs(data_dirs: List[str]):
        logger.info("Video_directories:")
        for dir in data_dirs:
            logger.info(f"  - '{dir}'")

    def load_videos(
        data_dirs: List[str], video_file: str, force_directory_scanning: bool = False
    ) -> List[VideoMetadata]:
        print_dirs(data_dirs)
        corrupted_files: List[str] = []
        return VideoDatabase(
            data_dirs, video_file, force_directory_scanning
        ).load_videos(corrupted_files)

    def get_batch_directories(base_dir: str) -> List[str]:
        batch_pattern = os.path.join(base_dir, "Batch*")
        batch_dirs = sorted(glob.glob(batch_pattern))
        return batch_dirs

    def load_videos_batches_on_mac(
        video_files_dir: str, force_directory_scanning: bool = False
    ) -> List[VideoMetadata]:
        data_base_dir = os.path.expanduser(
            "~/Documents/EufySecurityVideos/EufyVideos/record"
        )
        data_dirs = get_batch_directories(data_base_dir)
        video_file = os.path.join(video_files_dir, "videos_in_batches.csv")
        return load_videos(data_dirs, video_file, force_directory_scanning)

    def load_videos_backup_on_mac(
        video_files_dir: str, force_directory_scanning: bool = False
    ) -> List[VideoMetadata]:
        data_dirs = [
            os.path.expanduser(
                "~/Documents/EufySecurityVideos/EufyVideos/record/backup"
            )
        ]
        video_file = os.path.join(video_files_dir, "videos_in_backup.csv")
        return load_videos(data_dirs, video_file, force_directory_scanning)

    def load_all_videos_on_mac(
        video_files_dir: str, force_directory_scanning: bool = False
    ) -> List[VideoMetadata]:
        data_base_dir = os.path.expanduser(
            "~/Documents/EufySecurityVideos/EufyVideos/record"
        )
        data_dirs = get_batch_directories(data_base_dir)
        backup_dir = os.path.expanduser(
            "~/Documents/EufySecurityVideos/EufyVideos/record/backup"
        )
        if os.path.exists(backup_dir):
            data_dirs.append(backup_dir)
        video_file = os.path.join(video_files_dir, "videos_all_mac.csv")
        return load_videos(data_dirs, video_file, force_directory_scanning)

    def load_all_videos_seagate(
        video_files_dir: str, force_directory_scanning: bool = False
    ) -> List[VideoMetadata]:
        data_base_dir = os.path.join(
            "/Volumes", "Seagate Hub", "EufySecurityVideos", "EufyVideos", "record"
        )
        data_dirs = get_batch_directories(data_base_dir)
        backup_dir = os.path.join(data_base_dir, "backup")
        if os.path.exists(backup_dir):
            data_dirs.append(backup_dir)
        video_file = os.path.join(video_files_dir, "videos_all_seagate.csv")
        return load_videos(data_dirs, video_file, force_directory_scanning)

    def load_all_videos_usb_key(
        video_files_dir: str, force_directory_scanning: bool = False
    ) -> List[VideoMetadata]:
        data_base_dir = os.path.join("/Volumes", "Eufy Videos", "record")
        data_dirs = get_batch_directories(data_base_dir)
        backup_dir = os.path.join(data_base_dir, "backup")
        if os.path.exists(backup_dir):
            data_dirs.append(backup_dir)
        video_file = os.path.join(video_files_dir, "videos_all_usb_key.csv")
        return load_videos(data_dirs, video_file, force_directory_scanning)

    # Control the scan of the video directories in every database location
    force_directory_scanning_on_mac: bool = True
    force_directory_scanning_on_usb_key: bool = True
    force_directory_scanning_on_seagate: bool = True

    # Action selection
    check_mac_database: bool = False
    check_usb_key_database: bool = True
    check_seagate_database: bool = True

    if (
        not check_mac_database
        and not check_usb_key_database
        and not check_seagate_database
    ):
        logger.info(
            "No database check is done. Set one of the flags check_mac_database, "
            "check_usb_key_database or check_seagate_database to True to complete an action."
        )

    video_files_dir = os.path.expanduser(
        "~/Documents/EufySecurityVideos/EufyVideos/record"
    )
    out_dir = os.path.expanduser("~/Documents/EufySecurityVideos/stories")

    # PART 1: Check the state of the reference database on the mac:
    if check_mac_database:
        logger.info(
            "Checking the health of the mac video database "
            "(partition between batches and backup folders)"
        )
        videos_batches_on_mac = load_videos_batches_on_mac(
            video_files_dir, force_directory_scanning_on_mac
        )
        videos_backup_on_mac = load_videos_backup_on_mac(
            video_files_dir, force_directory_scanning_on_mac
        )
        # Make sure that there is no overlap in videos between batches and backup
        videos_batches_and_backup_on_mac = VideoMetadata.repeats(
            videos_batches_on_mac, videos_backup_on_mac
        )
        videos_backup_and_batches_on_mac = VideoMetadata.repeats(
            videos_backup_on_mac, videos_batches_on_mac
        )

        video_file_batches_and_backup = os.path.join(
            out_dir, "batches_and_backup_videos_on_mac.csv"
        )
        video_file_backup_and_batches = os.path.join(
            out_dir, "backup_and_batches_videos_on_mac.csv"
        )
        VideoMetadata.export_videos_to_metadata_file(
            videos_batches_and_backup_on_mac, video_file_batches_and_backup
        )
        VideoMetadata.export_videos_to_metadata_file(
            videos_backup_and_batches_on_mac, video_file_backup_and_batches
        )
        logger.info(
            f"{len(videos_batches_and_backup_on_mac)} videos in backup that are "
            f"also present in batches are saved in {video_file_batches_and_backup}"
        )
        logger.info(
            f"{len(videos_backup_and_batches_on_mac)} videos in batches that are "
            f"also present in backup are saved in {video_file_backup_and_batches}"
        )

    # PART 2: Check that the reference database is identical the ones on the usb_key and seagate
    if check_usb_key_database or check_seagate_database:
        logger.info("Loading reference video database on mac")
        all_videos_mac = load_all_videos_on_mac(
            video_files_dir, force_directory_scanning_on_mac
        )

    # Compare mac and usb_key:
    if check_usb_key_database:
        logger.info("Checking video backup on usb-key")
        all_videos_usb_key = load_all_videos_usb_key(
            video_files_dir, force_directory_scanning_on_usb_key
        )
        if all_videos_usb_key is None or len(all_videos_usb_key) == 0:
            logger.info("No videos found on usb-key. The drive could be disconnected.")
        else:  # Compare mac and usb_key:
            videos_missing_on_usb_key, videos_missing_on_mac_from_usb_key = (
                VideoMetadata.differences(all_videos_mac, all_videos_usb_key)
            )
            videos_file_missing_on_usb_key = os.path.join(
                out_dir, "videos_missing_on_usb_key.csv"
            )
            videos_file_missing_on_mac_from_usb_key = os.path.join(
                out_dir, "videos_missing_on_mac_from_usb_key.csv"
            )

            VideoMetadata.export_videos_to_metadata_file(
                videos_missing_on_usb_key, videos_file_missing_on_usb_key
            )
            VideoMetadata.export_videos_to_metadata_file(
                videos_missing_on_mac_from_usb_key,
                videos_file_missing_on_mac_from_usb_key,
            )

            logger.info(
                f"{len(videos_missing_on_usb_key)} videos on mac that are not on usb_key "
                f"are saved in {videos_file_missing_on_usb_key}"
            )
            logger.info(
                f"{len(videos_missing_on_mac_from_usb_key)} videos on usb_key that are not "
                f"on mac are saved in {videos_file_missing_on_mac_from_usb_key}"
            )

    # Compare mac and seagate:
    if check_seagate_database:
        logger.info("Checking video backup on seagate")
        all_videos_seagate = load_all_videos_seagate(
            video_files_dir, force_directory_scanning_on_seagate
        )
        if all_videos_seagate is None or len(all_videos_seagate) == 0:
            logger.info("No videos found on seagate. The drive could be disconnected.")
        else:
            videos_missing_on_seagate, videos_missing_on_mac_from_seagate = (
                VideoMetadata.differences(all_videos_mac, all_videos_seagate)
            )
            videos_file_missing_on_seagate = os.path.join(
                out_dir, "videos_missing_on_seagate.csv"
            )
            videos_file_missing_on_mac_from_seagate = os.path.join(
                out_dir, "videos_missing_on_mac_from_seagate.csv"
            )

            VideoMetadata.export_videos_to_metadata_file(
                videos_missing_on_seagate, videos_file_missing_on_seagate
            )
            VideoMetadata.export_videos_to_metadata_file(
                videos_missing_on_mac_from_seagate,
                videos_file_missing_on_mac_from_seagate,
            )

            logger.info(
                f"{len(videos_missing_on_seagate)} videos on mac that are not on seagate "
                f"are saved in {videos_file_missing_on_seagate}"
            )
            logger.info(
                f"{len(videos_missing_on_mac_from_seagate)} videos on seagate that are not "
                f"on mac are saved in {videos_file_missing_on_mac_from_seagate}"
            )

    sys.exit()
