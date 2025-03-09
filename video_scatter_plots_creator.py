#!/usr/bin/env python3

"""
VideoScatterPlotsCreator module for creating scatter plots from individual video metadata.

This module creates scatter plots showing the distribution of video durations and filesizes
as a function of day and time, color-coded by device type. Unlike the aggregated plots in
video_graph_creator.py, these plots show individual data points for each video.

Example usage:
    from video_metadata import VideoMetadata
    from video_scatter_plots_creator import VideoScatterPlotsCreator

    # Load videos
    videos = VideoMetadata.load_videos_from_directories('/path/to/videos')

    # Create scatter plots
    duration_fig = VideoScatterPlotsCreator.create_duration_scatter(videos)
    filesize_fig = VideoScatterPlotsCreator.create_filesize_scatter(videos)
"""

import warnings
from typing import List, cast

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from dashboard_config import DashboardConfig
from logging_config import create_logger
from video_metadata import VideoMetadata

logger = create_logger(__name__)


class VideoScatterPlotsCreator:
    """
    Creates scatter plots showing individual video metadata points.

    This class generates scatter plots that visualize:
    1. Video duration vs. time of day, color-coded by device
    2. Video filesize vs. time of day, color-coded by device

    Unlike the aggregated plots in VideoGraphCreator, these plots show individual
    data points for each video, providing a more detailed view of the distribution.
    """

    _version_checked = False

    @classmethod
    def _check_versions(cls):
        """Check for compatible library versions"""
        if cls._version_checked:
            return

        from version_config import verify_environment

        incompatible = verify_environment(raise_on_error=False)

        if incompatible:
            versions_str = "\n".join(
                f"  {pkg}: found {cast(dict, info).get('current', 'unknown')}, requires {cast(dict, info).get('required', 'unknown')}"
                for pkg, info in incompatible.items()
            )
            warnings.warn(
                f"Incompatible package versions detected:\n{versions_str}\n"
                "Visual inconsistencies may occur. See requirements.txt for supported versions.",
                RuntimeWarning,
            )

        cls._version_checked = True

    @staticmethod
    def _prepare_data(videos: List[VideoMetadata]) -> pd.DataFrame:
        """
        Prepare data for scatter plots from a list of VideoMetadata objects.

        Args:
            videos: List of VideoMetadata objects

        Returns:
            DataFrame with columns for time, date, hour, device, duration, and filesize
        """
        data = []

        for video in videos:
            # Extract time components
            time_obj = video.time_obj
            hour_float = time_obj.hour + time_obj.minute / 60 + time_obj.second / 3600

            # Create a row for each video
            data.append(
                {
                    "Date": video.date,
                    "Time": time_obj,
                    "Hour": hour_float,
                    "Device": video.device,
                    "Duration": video.duration.total_seconds(),
                    "Filesize": video.file_size,  # In MB
                    "DateTime": video.datetime_obj,
                    "DayOfWeek": video.datetime_obj.strftime("%A"),
                }
            )

        return pd.DataFrame(data)

    @classmethod
    def create_duration_datetime_scatter(
        cls,
        videos: List[VideoMetadata],
        **kwargs,
    ) -> go.Figure:
        """
        Create a scatter plot of video durations vs. continuous datetime.

        Args:
            videos: List of VideoMetadata objects
            **kwargs: Additional keyword arguments to pass to the figure layout

        Returns:
            Plotly figure object
        """
        cls._check_versions()

        # Prepare data
        df = cls._prepare_data(videos)

        # Get device colors - use the exact same color scheme as video_graph_creator.py
        colors = DashboardConfig.get_device_colors()

        # Create a figure from scratch with device coloring
        fig = go.Figure()

        # Add a trace for each device
        for device in df["Device"].unique():
            device_df = df[df["Device"] == device]

            fig.add_trace(
                go.Scatter(
                    x=device_df["DateTime"],
                    y=device_df["Duration"],
                    mode="markers",
                    marker={
                        "color": colors[device],
                        "size": 4,
                        "opacity": 1.0,
                        "symbol": "square",
                    },
                    name=device,
                    customdata=np.stack(
                        (
                            device_df["DateTime"]
                            .dt.strftime("%Y-%m-%d %H:%M:%S")
                            .values,
                            device_df["DayOfWeek"].values,
                        ),
                        axis=-1,
                    ),
                    hovertemplate=(
                        "<b>%{customdata[1]}</b><br>"
                        "<b>Time:</b> %{customdata[0]}<br>"
                        f"<b>{device}</b><br>"
                        "<b>Duration:</b> %{y:.2f} seconds<extra></extra>"
                    ),
                )
            )

        # Get default figure config
        fig_config = DashboardConfig.get_figure_config()

        # Update layout with defaults plus any custom settings
        fig_height = DashboardConfig.get_figure_height()
        layout_config = {
            "title": "Video Duration by Date & Time",
            "height": fig_height,
            **fig_config["layout"],
            **kwargs,
        }
        fig.update_layout(**layout_config)

        # Update axes
        fig.update_xaxes(
            title="Date & Time",
            **fig_config["axes"]["grid"],
            type="date",
        )
        fig.update_yaxes(
            title="Duration (seconds)",
            **fig_config["axes"]["grid"],
            **fig_config["axes"]["yaxis"],
        )

        return fig

    @classmethod
    def create_filesize_datetime_scatter(
        cls,
        videos: List[VideoMetadata],
        **kwargs,
    ) -> go.Figure:
        """
        Create a scatter plot of video filesizes vs. continuous datetime.

        Args:
            videos: List of VideoMetadata objects
            **kwargs: Additional keyword arguments to pass to the figure layout

        Returns:
            Plotly figure object
        """
        cls._check_versions()

        # Prepare data
        df = cls._prepare_data(videos)

        # Get device colors - use the exact same color scheme as video_graph_creator.py
        colors = DashboardConfig.get_device_colors()

        # Create a figure from scratch with device coloring
        fig = go.Figure()

        # Add a trace for each device
        for device in df["Device"].unique():
            device_df = df[df["Device"] == device]

            fig.add_trace(
                go.Scatter(
                    x=device_df["DateTime"],
                    y=device_df["Filesize"],
                    mode="markers",
                    marker={
                        "color": colors[device],
                        "size": 4,
                        "opacity": 1.0,
                        "symbol": "square",
                    },
                    name=device,
                    customdata=np.stack(
                        (
                            device_df["DateTime"]
                            .dt.strftime("%Y-%m-%d %H:%M:%S")
                            .values,
                            device_df["DayOfWeek"].values,
                        ),
                        axis=-1,
                    ),
                    hovertemplate=(
                        "<b>%{customdata[1]}</b><br>"
                        "<b>Time:</b> %{customdata[0]}<br>"
                        f"<b>{device}</b><br>"
                        "<b>Filesize:</b> %{y:.2f} MB<extra></extra>"
                    ),
                )
            )

        # Get default figure config
        fig_config = DashboardConfig.get_figure_config()

        # Update layout with defaults plus any custom settings
        fig_height = DashboardConfig.get_figure_height()
        layout_config = {
            "title": "Video File Size by Date & Time",
            "height": fig_height,
            **fig_config["layout"],
            **kwargs,
        }
        fig.update_layout(**layout_config)

        # Update axes
        fig.update_xaxes(
            title="Date & Time",
            **fig_config["axes"]["grid"],
            type="date",
        )
        fig.update_yaxes(
            title="File Size (MB)",
            **fig_config["axes"]["grid"],
            **fig_config["axes"]["yaxis"],
        )

        return fig


if __name__ == "__main__":
    # Testing code for the module
    import logging
    import os
    import sys

    from config import Config
    from logging_config import set_logger_level_and_format
    from video_database import VideoDatabase, VideoDatabaseList
    from video_filter import DateRange, TimeRange, VideoFilter, VideoSelector

    set_logger_level_and_format(logger, level=logging.DEBUG, extended_format=True)

    # Load video database
    root_database = "/Users/jbouguet/Documents/EufySecurityVideos/record/"
    metadata_files = [
        os.path.join(root_database, "videos_in_batches.csv"),
        os.path.join(root_database, "videos_in_backup.csv"),
    ]
    out_dir: str = "/Users/jbouguet/Documents/EufySecurityVideos/stories"

    video_database = VideoDatabaseList(
        [
            VideoDatabase(video_directories=None, video_metadata_file=file)
            for file in metadata_files
        ]
    ).load_videos()

    if video_database is None:
        logger.error("Failed to load video database")
        sys.exit(1)

    start_date = "2024-01-01"
    end_date = "2025-12-31"
    start_time = "00:00:00"
    end_time = "23:59:59"
    devices = Config.get_all_devices()
    weekdays = [
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
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

    # Create datetime scatter plots
    duration_datetime_fig = VideoScatterPlotsCreator.create_duration_datetime_scatter(
        videos
    )
    filesize_datetime_fig = VideoScatterPlotsCreator.create_filesize_datetime_scatter(
        videos
    )

    # Save figures to HTML
    duration_datetime_fig.write_html(
        os.path.join(out_dir, "duration_datetime_scatter.html")
    )
    filesize_datetime_fig.write_html(
        os.path.join(out_dir, "filesize_datetime_scatter.html")
    )
