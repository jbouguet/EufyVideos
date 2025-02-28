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
import re  # Used for Pattern type and regex operations
import sys
from dataclasses import dataclass, field  # field is used to configure dataclass fields
from datetime import date, datetime, time
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
    Either start or end can be None to specify only a lower or upper bound.
    """

    start: Optional[str] = None
    end: Optional[str] = None

    @classmethod
    def from_dict(cls, date_range_dict: Dict[str, Any]) -> "DateRange":
        """Create DateRange from dictionary representation."""
        return cls(start=date_range_dict.get("start"), end=date_range_dict.get("end"))


@dataclass
class TimeRange:
    """
    Represents a time range for video filtering.

    Used by VideoSelector to define time-based filtering criteria.
    Times should be in 'HH:MM:SS' format.
    If start is larger than end, the time interval crosses midnight.
    Either start or end can be None to specify only a lower or upper bound.
    """

    start: Optional[str] = None  # Optional, in HH:MM:SS format
    end: Optional[str] = None  # Optional, in HH:MM:SS format

    @classmethod
    def from_dict(cls, time_range_dict: Dict[str, Any]) -> "TimeRange":
        """
        Create TimeRange from dictionary representation.

        Args:
            time_range_dict: Dictionary containing optional 'start' and 'end' time strings

        Returns:
            TimeRange instance

        Raises:
            ValueError: If time strings are not in HH:MM:SS format
        """
        start = time_range_dict.get("start")
        end = time_range_dict.get("end")

        # Validate time format if provided
        if start is not None:
            try:
                datetime.strptime(start, "%H:%M:%S")
            except ValueError as e:
                raise ValueError(f"Invalid start time format. Expected HH:MM:SS: {e}")

        if end is not None:
            try:
                datetime.strptime(end, "%H:%M:%S")
            except ValueError as e:
                raise ValueError(f"Invalid end time format. Expected HH:MM:SS: {e}")

        return cls(start=start, end=end)


@dataclass
class DurationRange:
    """
    Represents a duration range for video filtering.

    Used by VideoSelector to define duration-based filtering criteria.
    Durations should be in seconds format.
    Either min or max can be None to specify only a lower or upper bound.
    """

    min: Optional[float] = None
    max: Optional[float] = None

    @classmethod
    def from_dict(cls, duration_range_dict: Dict[str, Any]) -> "DurationRange":
        """Create DurationRange from dictionary representation."""
        return cls(
            min=duration_range_dict.get("min"), max=duration_range_dict.get("max")
        )


@dataclass
class VideoSelector:
    """
    Defines criteria for filtering videos and implements the matching check.

    Combines multiple filtering criteria (devices, dates, times, duration, filenames, weekdays)
    that are applied together (AND condition).
    When multiple VideoSelector objects are used, they act as OR conditions.

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

    devices: Optional[List[str]] = field(default=None)
    date_range: Optional[DateRange] = field(default=None)
    time_range: Optional[TimeRange] = field(default=None)
    filenames: Optional[List[str]] = field(default=None)
    weekdays: Optional[List[str]] = field(default=None)
    duration_range: Optional[DurationRange] = field(default=None)
    date_regex: Optional[str] = field(default=None)

    # Placeholder for a new criteria for selection:
    # new_criteria: Optional[T] = None

    # Cached values for runtime optimization of the method matches (initialized in __post_init__)
    _devices_set: Optional[set[str]] = field(
        default=None, metadata={"exclude_from_dict": True}
    )
    _start_date: Optional[date] = field(
        default=None, metadata={"exclude_from_dict": True}
    )
    _end_date: Optional[date] = field(
        default=None, metadata={"exclude_from_dict": True}
    )
    _start_time: Optional[time] = field(
        default=None, metadata={"exclude_from_dict": True}
    )
    _end_time: Optional[time] = field(
        default=None, metadata={"exclude_from_dict": True}
    )
    _filenames_set: Optional[set[str]] = field(
        default=None, metadata={"exclude_from_dict": True}
    )
    _weekdays_set: Optional[set[str]] = field(
        default=None, metadata={"exclude_from_dict": True}
    )
    _date_pattern: Optional[re.Pattern] = field(
        default=None, metadata={"exclude_from_dict": True}
    )
    _duration_range_min: Optional[float] = field(
        default=None, metadata={"exclude_from_dict": True}
    )
    _duration_range_max: Optional[float] = field(
        default=None, metadata={"exclude_from_dict": True}
    )

    #  Placeholder for cached values for a new criteria for selection:
    # _new_criteria: Optional[T] = None

    def __post_init__(self):
        """Initialize cached values for faster lookups and validate inputs"""
        # Convert lists to sets for O(1) lookups, treating None as empty list
        self._devices_set = (
            set(self.devices or []) if self.devices is not None else None
        )
        self._filenames_set = (
            set(self.filenames or []) if self.filenames is not None else None
        )
        self._weekdays_set = (
            set(day.lower() for day in (self.weekdays or []))
            if self.weekdays is not None
            else None
        )

        # Validate and cache duration_range
        if self.duration_range is not None:
            if self.duration_range.min is None and self.duration_range.max is None:
                raise ValueError(
                    "At least one of min or max must be specified in duration_range"
                )
            self._duration_range_min = self.duration_range.min
            self._duration_range_max = self.duration_range.max

        # Initialize and validate date range
        self._start_date = None
        self._end_date = None
        if self.date_range is not None:
            if self.date_range.start is None and self.date_range.end is None:
                raise ValueError(
                    "At least one of start or end must be specified in date_range"
                )
            if self.date_range.start is not None:
                self._start_date = datetime.strptime(
                    self.date_range.start, "%Y-%m-%d"
                ).date()
            if self.date_range.end is not None:
                self._end_date = datetime.strptime(
                    self.date_range.end, "%Y-%m-%d"
                ).date()

        # Compile date regex pattern if provided
        self._date_pattern = None
        if self.date_regex is not None:
            try:
                self._date_pattern = re.compile(f"^{self.date_regex}$")
            except re.error as e:
                raise ValueError(
                    f"Invalid date regex pattern '{self.date_regex}': {e}. "
                ) from e

        # Initialize and validate time range
        self._start_time = None
        self._end_time = None
        if self.time_range is not None:
            if self.time_range.start is None and self.time_range.end is None:
                raise ValueError(
                    "At least one of start or end must be specified in time_range"
                )
            try:
                if self.time_range.start is not None:
                    self._start_time = datetime.strptime(
                        self.time_range.start, "%H:%M:%S"
                    ).time()
                if self.time_range.end is not None:
                    self._end_time = datetime.strptime(
                        self.time_range.end, "%H:%M:%S"
                    ).time()
            except ValueError as e:
                raise ValueError(
                    f"Invalid time format in time_range. Expected HH:MM:SS format: {e}"
                ) from e

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
        # Handle date range
        date_range = None
        date_range_dict = selector_dict.get("date_range")
        if isinstance(date_range_dict, dict):
            # Create DateRange even if only one of start/end is specified
            date_range = DateRange.from_dict(date_range_dict)

        # Handle time range
        time_range = None
        time_range_dict = selector_dict.get("time_range")
        if isinstance(time_range_dict, dict):
            # Create TimeRange even if only one of start/end is specified
            time_range = TimeRange.from_dict(time_range_dict)

        # Handle devices
        devices = selector_dict.get("devices")
        if devices is not None:
            if isinstance(devices, str):
                devices = [devices]
            if devices:  # Only validate non-empty list
                unknown_devices = [
                    device
                    for device in devices
                    if device not in Config.get_all_devices()
                ]
                if unknown_devices:
                    raise ValueError(f"Unknown devices: {', '.join(unknown_devices)}")

        # Handle filenames
        filenames = selector_dict.get("filenames")
        if filenames is not None and isinstance(filenames, str):
            filenames = [filenames]

        # Handle weekdays
        weekdays = selector_dict.get("weekdays")
        if weekdays is not None and isinstance(weekdays, str):
            weekdays = [weekdays]

        # Handle duration range
        duration_range = None
        duration_range_dict = selector_dict.get("duration_range")
        if isinstance(duration_range_dict, dict):
            # Create DurationRange even if only one of min/max is specified
            duration_range = DurationRange.from_dict(duration_range_dict)

        # Handle date regex pattern
        date_regex = selector_dict.get("date_regex")

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
            duration_range=duration_range,
            date_regex=date_regex,
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
        if self.devices is not None and not self.devices:
            return True  # Empty list matches everything
        if self._devices_set is not None and video.device not in self._devices_set:
            return False

        # Check filename (fast set lookup)
        if self.filenames is not None and not self.filenames:
            return True  # Empty list matches everything
        if (
            self._filenames_set is not None
            and video.filename not in self._filenames_set
        ):
            return False

        # Check weekday (fast set lookup)
        if self.weekdays is not None and not self.weekdays:
            return True  # Empty list matches everything
        if self._weekdays_set is not None:
            weekday = video.datetime.strftime("%A").lower()
            if weekday not in self._weekdays_set:
                return False

        # Check date range (more expensive - date comparison)
        if self.date_range is not None:
            if self._start_date is not None and video.date < self._start_date:
                return False
            if self._end_date is not None and video.date > self._end_date:
                return False

        # Check date regex pattern
        if self._date_pattern is not None:
            if not self._date_pattern.match(video.date_str):
                return False

        # Check duration range (more expensive)
        duration_seconds = video.duration.total_seconds()
        if (
            self._duration_range_min is not None
            and duration_seconds < self._duration_range_min
        ):
            return False
        if (
            self._duration_range_max is not None
            and duration_seconds > self._duration_range_max
        ):
            return False

        # Check time range (most expensive - time comparison with midnight handling)
        if self.time_range is not None:
            if self._start_time is not None and self._end_time is not None:
                # Both bounds exist - use is_time_in_range to handle midnight crossing
                if not VideoSelector.is_time_in_range(
                    video.time, self._start_time, self._end_time
                ):
                    return False
            else:
                # Single bound - simple comparison
                if self._start_time is not None and video.time < self._start_time:
                    return False
                if self._end_time is not None and video.time > self._end_time:
                    return False

        # Placeholder for checking a new criteria for selection:
        # if self.new_criteria is not None and (video does not pass self.new_criteria):
        #     return False

        return True

    def log_str(self) -> List[str]:
        output_str: List[str] = []
        if self.devices is not None:
            if not self.devices:
                output_str.append("Devices: []")
            else:
                output_str.append(f"Devices: {', '.join(self.devices)}")

        if self.date_range is not None:
            start_str = (
                self.date_range.start if self.date_range.start is not None else "-∞"
            )
            end_str = self.date_range.end if self.date_range.end is not None else "∞"
            output_str.append(f"Date range: {start_str} to {end_str}")

        if self.date_regex is not None:
            output_str.append(f"Date regex: {self.date_regex}")

        if self.time_range is not None:
            start_str = (
                self.time_range.start if self.time_range.start is not None else "-∞"
            )
            end_str = self.time_range.end if self.time_range.end is not None else "∞"
            output_str.append(f"Time range: {start_str} to {end_str}")

        if self.duration_range is not None:
            min_str = (
                str(self.duration_range.min)
                if self.duration_range.min is not None
                else "-∞"
            )
            max_str = (
                str(self.duration_range.max)
                if self.duration_range.max is not None
                else "∞"
            )
            output_str.append(f"Duration range: [{min_str} - {max_str}] seconds")

        if self.filenames is not None:
            if not self.filenames:
                output_str.append("Filenames: []")
            else:
                output_str.append(f"Filenames: {', '.join(self.filenames)}")

        if self.weekdays is not None:
            if not self.weekdays:
                output_str.append("Weekdays: []")
            else:
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
