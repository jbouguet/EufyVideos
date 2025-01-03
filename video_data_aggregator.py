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

    def __init__(self, metrics: List[str] = None):
        """Initialize aggregarator optionally specifying metrics of interest"""
        if metrics is None:
            # By default, aggregate across all metrics
            self.metrics = ["activity", "duration", "filesize"]
        else:
            # Only aggregate across a subset of specified metrics
            self.metrics = metrics

    @staticmethod
    def _aggregate_by_metric(
        videos: List[VideoMetadata],
        time_key: Literal["date", "hour"],
        value_key: Literal["count", "duration", "filesize"],
    ) -> pd.DataFrame:
        """
        Generic aggregation function that processes VideoMetadata objects.

        Example usage with different metrics:
            # For daily video counts
            df = _aggregate_by_metric(videos, 'date', 'count')

            # For hourly duration totals
            df = _aggregate_by_metric(videos, 'hour', 'duration')

            # For daily storage usage
            df = _aggregate_by_metric(videos, 'date', 'filesize')
        """

        def get_time_value(video: VideoMetadata) -> int | datetime:
            # Interface with VideoMetadata's date and time methods
            return video.date if time_key == "date" else video.time().hour

        def get_metric_value(video: VideoMetadata) -> float:
            # Interface with VideoMetadata's properties for different metrics
            if value_key == "count":
                return 1
            elif value_key == "duration":
                return video.duration.total_seconds() / 3600  # Convert to hours
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
                videos, "date", "count"
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
        self, videos: List[VideoMetadata]
    ) -> Dict[str, pd.DataFrame]:
        """
        Returns hourly aggregates for all metrics.

        Example:
            hourly_data = get_hourly_aggregates(videos)
            activity_by_hour = hourly_data['activity']  # Videos per hour
            duration_by_hour = hourly_data['duration']  # Hours per hour slot
            storage_by_hour = hourly_data['filesize']   # MB per hour
        """
        output = {}
        if "activity" in self.metrics:
            output["activity"] = self._aggregate_by_metric(
                videos, "hour", "count"
            ).rename(columns={"TimeKey": "Hour"})
        if "duration" in self.metrics:
            output["duration"] = self._aggregate_by_metric(
                videos, "hour", "duration"
            ).rename(columns={"TimeKey": "Hour"})
        if "filesize" in self.metrics:
            output["filesize"] = self._aggregate_by_metric(
                videos, "hour", "filesize"
            ).rename(columns={"TimeKey": "Hour"})
        return output
