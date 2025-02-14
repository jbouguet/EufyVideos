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
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

# import video_analyzer
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
    Defines criteria for filtering videos and implements the matching check.

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
        filtered_videos = VideoMetadata.clean_and_sort(
            list(v for v in videos if selector.matches(v))
            )
    """

    devices: Optional[List[str]] = None
    date_range: Optional[DateRange] = None
    time_range: Optional[TimeRange] = None
    filenames: Optional[List[str]] = None
    weekdays: Optional[List[str]] = None

    # Placeholder for a new criteria for selection:
    # new_criteria: Optional[T] = None

    # Cached values for runtime optimization of the method matches
    _devices_set: Optional[set[str]] = None
    _start_date = None
    _end_date = None
    _start_time = None
    _end_time = None
    _filenames_set: Optional[set[str]] = None
    _weekdays_set: Optional[set[str]] = None

    #  Placeholder for cached values for a new criteria for selection:
    # _new_criteria: Optional[T] = None

    def __post_init__(self):
        """Initialize cached values for faster lookups"""
        # Convert lists to sets for O(1) lookups
        self._devices_set = set(self.devices) if self.devices is not None else None
        self._filenames_set = (
            set(self.filenames) if self.filenames is not None else None
        )
        self._weekdays_set = (
            set(day.lower() for day in self.weekdays)
            if self.weekdays is not None
            else None
        )

        # Pre-compute date and time objects
        if self.date_range is not None:
            self._start_date = datetime.strptime(
                self.date_range.start, "%Y-%m-%d"
            ).date()
            self._end_date = datetime.strptime(self.date_range.end, "%Y-%m-%d").date()

        if self.time_range is not None:
            self._start_time = datetime.strptime(
                self.time_range.start, "%H:%M:%S"
            ).time()
            self._end_time = datetime.strptime(self.time_range.end, "%H:%M:%S").time()

        # Placeholder for computations of cached values for a new criteria for selection:
        # if self.new_criteria is not None:
        #    self._new_criteria = ...

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

        devices = None
        if "devices" in selector_dict:
            devices = selector_dict["devices"]
        if devices and isinstance(devices, str):
            devices = [devices]

        if devices is not None:
            unknown_devices = [
                device for device in devices if device not in Config.get_all_devices()
            ]
            if unknown_devices:
                raise ValueError(f"Unknown devices: {', '.join(unknown_devices)}")

        filenames = None
        if "filenames" in selector_dict:
            filenames = selector_dict["filenames"]
        if filenames and isinstance(filenames, str):
            filenames = [filenames]

        weekdays = None
        if "weekdays" in selector_dict:
            weekdays = selector_dict["weekdays"]
        if weekdays and isinstance(weekdays, str):
            weekdays = [weekdays]

        # Placeholder for a new criteria for selection:
        # new_criteria = None
        # if "new_criteria" in selector_dict:
        #     new_criteria_dict = selector_dict["new_criteria"]
        #     new_criteria = F(new_criteria_dict)

        return cls(
            devices=devices,
            date_range=date_range,
            time_range=time_range,
            filenames=filenames,
            weekdays=weekdays,
            # new_criteria=new_criteria,  # Placeholder for a new criteria for selection:\
        )

    @staticmethod
    def is_time_in_range(time, start, end) -> bool:
        """Check if time falls within range, handling midnight crossing."""
        if start <= end:
            return start <= time <= end
        else:  # Crosses midnight
            return start <= time or time <= end

    def matches(self, video: VideoMetadata) -> bool:
        """
        Returns True only if video satisfies all filtering conditions listed in selector.
        """
        # Check device (fastest operation - simple set lookup)
        if self._devices_set is not None and video.device not in self._devices_set:
            return False

        # Check filename (fast set lookup)
        if (
            self._filenames_set is not None
            and video.filename not in self._filenames_set
        ):
            return False

        # Check weekday (fast set lookup)
        if self._weekdays_set is not None:
            weekday = video.datetime.strftime("%A").lower()
            if weekday not in self._weekdays_set:
                return False

        # Check date range (more expensive - date comparison)
        if (
            self._start_date is not None
            and self._end_date is not None
            and (video.date < self._start_date or video.date > self._end_date)
        ):
            return False

        # Check time range (most expensive - time comparison with midnight handling)
        if self._start_time is not None and not VideoSelector.is_time_in_range(
            video.time, self._start_time, self._end_time
        ):
            return False

        # Placeholder for checking a new criteria for selection:
        # if self.new_criteria is not None and (video does not pass self.new_criteria):
        #     return False

        return True

    def log_str(self) -> List[str]:
        output_str: List[str] = []
        if self.devices is not None:
            output_str.append(f"Devices: {', '.join(self.devices)}")
        if self.date_range is not None:
            output_str.append(
                f"Date range: {self.date_range.start} to {self.date_range.end}"
            )
        if self.time_range is not None:
            output_str.append(
                f"Time range: {self.time_range.start} to {self.time_range.end}"
            )
        if self.filenames is not None:
            output_str.append(f"Filenames: {', '.join(self.filenames)}")
        if self.weekdays is not None:
            output_str.append(f"Weekdays: {', '.join(self.weekdays)}")

        # Placeholder for logging a new criteria for selection:
        # if self.new_criteria is not None:
        #     output_str.append(f"New_criteria: {self.new_criteria}")

        return output_str

    @staticmethod
    def log(selectors: Union["VideoSelector", Optional[List["VideoSelector"]]]) -> None:
        """Log the criteria defined in one or more selectors."""
        if selectors is None:
            return
        selectors = [selectors] if isinstance(selectors, VideoSelector) else selectors
        for i, selector in enumerate(selectors, 1):
            if len(selectors) > 1:
                logger.info(f"  - Selector {i}:")
            else:
                logger.info("  - Selector:")
            selector_log_str = selector.log_str()
            if not selector_log_str:
                logger.info("     None")
            else:
                for s in selector_log_str:
                    logger.info(f"     {s}")


class VideoFilter:
    """
    Provides static methods for filtering video collections.

    Methods can be used individually for simple filtering or combined
    through the by_selector method for complex filtering criteria.

    Example:
        # Filter by multiple criteria in two separate selectors
        selector1 = VideoSelector(
            devices=['Garage'],
            date_range=DateRange(start='2023-01-01', end='2023-12-31')
        )
        selector2 = VideoSelector(
            devices=['Front Yard'],
            date_range=DateRange(start='2023-01-01', end='2023-12-31'),
            time_range=TimeRange(start='08:00:00', end='17:00:00'),
            weekdays=['monday', 'wednesday', 'friday']
        )
        # Filter the videos that match the criteria of either one of the two selectors:
        filtered_videos = VideoFilter.by_selectors(videos, [selector1, selector2])
    """

    @staticmethod
    def matches_any_selectors(
        video: VideoMetadata,
        selectors: Union[Optional[VideoSelector], List[VideoSelector]] = None,
    ) -> bool:
        """
        Returns True only if video satisfies the selection criteria of at least one of the selectors.
        Returns True if selector is None.
        """
        # Early exit for common cases
        if selectors is None:
            return True

        # Normalize to list and handle empty case
        selectors = [selectors] if isinstance(selectors, VideoSelector) else selectors
        if not selectors:  # Empty list case
            return True

        # Check each selector until a match is found
        return any(selector.matches(video) for selector in selectors)

    @staticmethod
    def by_selectors(
        videos: List[VideoMetadata],
        selectors: Union[Optional[VideoSelector], List[VideoSelector]] = None,
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
        # Early exit for common cases
        if not videos:
            return []
        if selectors is None:
            return VideoMetadata.clean_and_sort(videos)

        # Use generator expression for memory efficiency
        filtered = (
            v for v in videos if VideoFilter.matches_any_selectors(v, selectors)
        )
        return VideoMetadata.clean_and_sort(list(filtered))


if __name__ == "__main__":

    # Testing code for the module.
    import logging
    import os

    from config import Config
    from logging_config import set_logger_level_and_format
    from video_analyzer import AnalysisConfig, VideoAnalyzer

    set_logger_level_and_format(logger, level=logging.DEBUG, extended_format=False)

    # Load the config "analysis_config.yaml" containing up to date information about the database
    analyzer = VideoAnalyzer(
        AnalysisConfig.from_file(
            os.path.join(os.path.dirname(__file__), "analysis_config.yaml")
        )
    )
    analyzer.load_all_databases()
    analyzer.log_statistics()

    if not analyzer.videos_database:
        logger.error("No videos found in database")
        sys.exit(1)

    start_date = analyzer.videos_database[0].date_str
    end_date = analyzer.videos_database[-1].date_str
    start_time = "12:00:00"
    end_time = "14:00:00"
    devices = ["Backyard"]

    selector = VideoSelector(
        date_range=DateRange(start=start_date, end=end_date),
        time_range=TimeRange(start=start_time, end=end_time),
        devices=devices,
    )

    VideoSelector.log(selector)

    # Filter the database
    videos = VideoFilter.by_selectors(analyzer.videos_database, selector)

    logger.debug(f"Number of selected videos: {len(videos)}")

    VideoMetadata.export_videos_to_metadata_file(
        videos, os.path.join(analyzer.config.output_directory, "filtered_videos.csv")
    )
