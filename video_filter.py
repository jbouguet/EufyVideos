#!/usr/bin/env python3
"""
This module provides filtering capabilities via VideoFilter and VideoSelector classes

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
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from config import Config
from logging_config import create_logger
from video_metadata import VideoMetadata

logger = create_logger(__name__)


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
            logger.info(f"  - Selector {i}:")
            if selector.devices:
                logger.info(f"     Devices: {', '.join(selector.devices)}")
            if selector.date_range:
                logger.info(
                    f"     Date range: {selector.date_range.start} to {selector.date_range.end}"
                )
            if selector.time_range:
                logger.info(
                    f"     Time range: {selector.time_range.start} to {selector.time_range.end}"
                )
            if selector.filenames:
                logger.info(f"     Filenames: {', '.join(selector.filenames)}")
            if selector.weekdays:
                logger.info(f"     Weekdays: {', '.join(selector.weekdays)}")


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
