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
from typing import Dict, List, Literal, Tuple

import pandas as pd

from logging_config import create_logger
from video_metadata import VideoMetadata

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
    - filesize: For storage metrics

        Example:
            videos = VideoMetadata.load_videos_from_directories('videos/')
            metrics = ["activity", "duration", "filesize"]
            config = {"bins_per_hours": 4} # Temporal binning set to 15 minutes (60/4)
            aggregator = VideoDataAggregator(metrics, config)
            daily_data, hourly_data = aggregator.run(videos)
            activity_by_date = daily_data['activity']  # Videos per day
            duration_by_date = daily_data['duration']  # Minutes per day
            storage_by_date = daily_data['filesize']   # MB per day
            activity_by_15_minutes = hourly_data['activity']  # Videos per 15 minutes
            duration_by_15_minutes = hourly_data['duration']  # Minutes per 15 minutes
            storage_by_15_minutes = hourly_data['filesize']   # MB per 15 minutes
    """

    def __init__(self, metrics: List[str] = None, config: Dict[str, bool | int] = None):
        """Initialize aggregarator optionally specifying metrics of interest"""
        if metrics is None:
            # By default, aggregate across all metrics
            self.metrics = ["activity", "duration", "filesize"]
        else:
            # Only aggregate across a subset of specified metrics
            self.metrics = metrics
        # By default, set time interval to 15 minutes = 1 hour / 4 bin_per_hour
        if config is None:
            self.config = {"bins_per_hour": 4}
        else:
            self.config = config
        bins_per_hour = self.config.get("bins_per_hour", 4)
        self.config["bins_per_hour"] = bins_per_hour

    def run(
        self, videos: List[VideoMetadata]
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
        """
        Returns daily and hourly aggregates for all metrics with configurable temporal binning.

        Args:
            videos: List of VideoMetadata objects to aggregate
            bins_per_hour: Number of bins per hour (default=1). Common values:
                1: Hour-level bins (00:00, 01:00, etc.)
                2: 30-minute bins (00:00, 00:30, 01:00, etc.)
                4: 15-minute bins (00:00, 00:15, 00:30, 00:45, etc.)
        """
        daily_data = {}
        hourly_data = {}
        # By default, set time interval to 15 minutes = 1 hour / 4 bin_per_hour
        bins_per_hour = self.config.get("bins_per_hour", 4)
        for metric in self.metrics:
            daily_data[metric] = self._aggregate_by_metric(
                videos, "date", metric
            ).rename(columns={"TimeKey": "Date"})
            hourly_data[metric] = self._aggregate_by_metric(
                videos, "hour", metric, bins_per_hour
            ).rename(columns={"TimeKey": "Hour"})
        return daily_data, hourly_data

    @staticmethod
    def _aggregate_by_metric(
        videos: List[VideoMetadata],
        time_key: Literal["date", "hour"],
        metric: Literal["activity", "duration", "filesize"],
        bins_per_hour: int = 4,
    ) -> pd.DataFrame:
        """
        Generic aggregation function that processes VideoMetadata objects.
        """

        def get_time_value(video: VideoMetadata) -> int | datetime | float:
            # Interface with VideoMetadata's date and time methods
            if time_key == "date":
                return video.date
            else:
                time = video.time()
                # Convert to fractional hour based on bins_per_hour and set
                # the value to mid point of the quantized time bin.
                # First determine which bin the minutes fall into
                minutes_per_bin = 60 / bins_per_hour
                bin_index = (time.minute + (time.second / 60.0)) // minutes_per_bin
                # Returns a decimal representation of the temporal quantized bin
                # with the last offset 1 / (2 * bins_per_hour) setting the value
                # to the mid point of the quantized bin.
                return time.hour + bin_index / bins_per_hour + 1 / (2 * bins_per_hour)

        def get_metric_value(video: VideoMetadata) -> float:
            # Interface with VideoMetadata's properties for different metrics
            if metric == "activity":
                return 1
            elif metric == "duration":
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


if __name__ == "__main__":

    # Testing code for the module.
    import logging
    import os

    from config import Config
    from logging_config import set_logger_level_and_format
    from video_database import VideoDatabase, VideoDatabaseList
    from video_filter import DateRange, TimeRange, VideoFilter, VideoSelector

    set_logger_level_and_format(logger, level=logging.DEBUG, extended_format=True)

    # Load video database
    root_database = (
        "/Users/jbouguet/Documents/EufySecurityVideos/record/"
    )
    metadata_files = [
        os.path.join(root_database, "videos_in_batches.csv"),
        os.path.join(root_database, "videos_in_backup.csv"),
        # Add more metadata files as needed
    ]
    out_dir: str = "/Users/jbouguet/Documents/EufySecurityVideos/stories"

    video_database = VideoDatabaseList(
        [
            VideoDatabase(video_directories=None, video_metadata_file=file)
            for file in metadata_files
        ]
    ).load_videos()

    bins_per_hour = 1

    start_date = "2024-12-30"
    end_date = "2025-01-05"
    start_time = "00:00:00"
    end_time = "23:59:59"
    devices = Config.get_all_devices()
    weekdays = [
        "monday",
        "wednesday",
        "friday",
        "sunday",
    ]

    logger.debug(f"Date range: {start_date} to {end_date}")
    logger.debug(f"Time range: {start_time} to {end_time}")
    logger.debug(f"Devices: {devices}")
    logger.debug(f"Weekdays: {weekdays}")

    selector = VideoSelector(
        date_range=DateRange(start=start_date, end=end_date),
        time_range=TimeRange(start=start_time, end=end_time),
        devices=devices,
        weekdays=weekdays,
    )
    # Filter the database
    videos = VideoFilter.by_selectors(video_database, selector)

    logger.debug(f"Number of videos: {len(videos)}")

    metrics = ["activity"]

    # Get aggregated data
    data_aggregator = VideoDataAggregator(
        metrics=metrics, config={"bins_per_hour": bins_per_hour}
    )

    daily_data, hourly_data = data_aggregator.run(videos)

    print(hourly_data["activity"])
    print(daily_data["activity"])
