#!/usr/bin/env python3

"""
VideoDataAggregator module for creating aggregates of video metadata in prepareation for
interactive Plotly graphs for visdualization.

This module interfaces with video_metadata.py to create aggregated views of metadata of video collections.
It processes VideoMetadata objects to generate various types of data matrices showing temporal distributions
and device-specific patterns.

Key interfaces with video_metadata.py:
1. Takes List[VideoMetadata] as input
2. Uses VideoMetadata properties:
   - date: For daily aggregations
   - time(): For hourly aggregations
   - device: For device-specific grouping
   - duration: For duration-based metrics
   - file_size: For storage-based metrics

Example usage:
    from video_metadata import VideoMetadata
    
    # Load videos using VideoMetadata's methods
    videos = VideoMetadata.load_videos_from_directories('/path/to/videos')
    
    # Create dashboard and generate graphs
    data_aggregator = VideoDataAggregator()
    daily_data = self.data_aggregator.get_daily_aggregates(videos)
    hourly_data = self.data_aggregator.get_hourly_aggregates(videos)
"""

from datetime import datetime
from typing import Dict, List, Literal

import pandas as pd

from logging_config import create_logger
from video_metadata import VideoMetadata  # Main interface point

logger = create_logger(__name__)


class VideoDataAggregator:
    """
    Handles data aggregation logic for video metadata.

    This class processes VideoMetadata objects to create aggregated datasets
    for visualization. It interfaces with video_metadata.py by using the following
    VideoMetadata properties:
    - date: For daily grouping
    - time(): For hourly grouping
    - device: For device-based grouping
    - duration: For duration metrics
    - file_size: For storage metrics

    Example:
        videos = VideoMetadata.load_videos_from_directories('videos/')
        aggregator = VideoDataAggregator()
        daily_data = aggregator.get_daily_aggregates(videos)
        # daily_data now contains aggregated metrics by date and device
    """

    def __init__(self, metrics: List[str] = None, config: Dict[str, bool | int] = None):
        """Initialize aggregarator optionally specifying metrics of interest"""
        if metrics is None:
            # By default, aggregate across all metrics
            self.metrics = ["activity", "duration", "filesize"]
        else:
            # Only aggregate across a subset of specified metrics
            self.metrics = metrics
        bins_per_hour = config.get("bins_per_hour", 4)
        if config is None:
            self.config = {}
        else:
            self.config = config
        self.config["bins_per_hour"] = bins_per_hour

    @staticmethod
    def _aggregate_by_metric(
        videos: List[VideoMetadata],
        time_key: Literal["date", "hour"],
        value_key: Literal["activity", "duration", "filesize"],
        bins_per_hour: int = 4,
    ) -> pd.DataFrame:
        """
        Generic aggregation function that processes VideoMetadata objects.

        Example usage with different metrics:
            # For daily video activity
            df = _aggregate_by_metric(videos, 'date', 'activity')

            # For hourly duration totals
            df = _aggregate_by_metric(videos, 'hour', 'duration')

            # For daily storage usage
            df = _aggregate_by_metric(videos, 'date', 'filesize')
        """

        def get_time_value(video: VideoMetadata) -> int | datetime | float:
            # Interface with VideoMetadata's date and time methods
            if time_key == "date":
                return video.date
            else:
                time = video.time()
                # Convert to fractional hour based on bins_per_hour
                # First determine which bin the minutes fall into
                minutes_per_bin = 60 / bins_per_hour
                bin_index = time.minute // minutes_per_bin
                # Note that the last offset 1/(2 * bins_per_hour) set the mid point of the bin
                # as representative.
                return time.hour + bin_index / bins_per_hour + 1 / (2 * bins_per_hour)

        def get_metric_value(video: VideoMetadata) -> float:
            # Interface with VideoMetadata's properties for different metrics
            if value_key == "activity":
                return 1
            elif value_key == "duration":
                return video.duration.total_seconds() / 60  # Convert to minutes
            else:  # filesize
                return video.file_size

        df = pd.DataFrame(
            [
                {
                    "TimeKey": get_time_value(video),
                    "Device": video.device,
                    "Value": get_metric_value(video),
                }
                for video in videos
            ]
        )

        return (
            df.groupby(["TimeKey", "Device"])["Value"]
            .sum()
            .unstack(fill_value=0)
            .reset_index()
        )

    def get_daily_aggregates(
        self, videos: List[VideoMetadata]
    ) -> Dict[str, pd.DataFrame]:
        """
        Returns daily aggregates for all metrics.

        Example:
            daily_data = get_daily_aggregates(videos)
            activity_by_date = daily_data['activity']  # Videos per day
            duration_by_date = daily_data['duration']  # Hours per day
            storage_by_date = daily_data['filesize']   # MB per day
        """
        output = {}
        if "activity" in self.metrics:
            output["activity"] = self._aggregate_by_metric(
                videos, "date", "activity"
            ).rename(columns={"TimeKey": "Date"})
        if "duration" in self.metrics:
            output["duration"] = self._aggregate_by_metric(
                videos, "date", "duration"
            ).rename(columns={"TimeKey": "Date"})
        if "filesize" in self.metrics:
            output["filesize"] = self._aggregate_by_metric(
                videos, "date", "filesize"
            ).rename(columns={"TimeKey": "Date"})
        return output

    def get_hourly_aggregates(
        self,
        videos: List[VideoMetadata],
    ) -> Dict[str, pd.DataFrame]:
        """
        Returns hourly aggregates for all metrics with configurable temporal binning.

        Args:
            videos: List of VideoMetadata objects to aggregate
            bins_per_hour: Number of bins per hour (default=1). Common values:
                1: Hour-level bins (00:00, 01:00, etc.)
                2: 30-minute bins (00:00, 00:30, 01:00, etc.)
                4: 15-minute bins (00:00, 00:15, 00:30, 00:45, etc.)

        Example:
            # Get standard hourly aggregates
            hourly_data = get_hourly_aggregates(videos)

            # Get 30-minute aggregates
            half_hour_data = get_hourly_aggregates(videos, bins_per_hour=2)

            # Get 15-minute aggregates
            quarter_hour_data = get_hourly_aggregates(videos, bins_per_hour=4)
        """
        bins_per_hour = self.config.get("bins_per_hour", 4)
        output = {}
        if "activity" in self.metrics:
            output["activity"] = self._aggregate_by_metric(
                videos, "hour", "activity", bins_per_hour
            ).rename(columns={"TimeKey": "Hour"})
        if "duration" in self.metrics:
            output["duration"] = self._aggregate_by_metric(
                videos, "hour", "duration", bins_per_hour
            ).rename(columns={"TimeKey": "Hour"})
        if "filesize" in self.metrics:
            output["filesize"] = self._aggregate_by_metric(
                videos, "hour", "filesize", bins_per_hour
            ).rename(columns={"TimeKey": "Hour"})
        return output
