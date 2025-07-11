#!/usr/bin/env python3

"""
PlotCreator module for creating scatter plots from individual video metadata.

This module creates scatter plots showing the distribution of video metrics such as duration
and filesize as a function of day and time, color-coded by device type. Unlike the aggregated
plots in video_graph_creator.py, these plots show individual data points for each video.

Example usage:
    from video_metadata import VideoMetadata
    from video_scatter_plots_creator import PlotCreator

    # Load videos
    videos = VideoMetadata.load_videos_from_directories('/path/to/videos')

    # Create scatter plots
    duration_vs_datetime_fig = PlotCreator.metric_vs_datetime(videos, metric="duration")
    filesize_vs_datetime_fig = PlotCreator.metric_vs_datetime(videos, metric="filesize")
    filesize_vs_duration_fig = PlotCreator.filesize_vs_duration(videos)
    filesize_vs_pixel_fig = PlotCreator.filesize_vs_pixel(videos)

    # Save all graphs to a single HTML file
    PlotCreator.create_graphs_file(videos, 'output_graphs.html')
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


class PlotCreator:
    """
    Creates scatter plots showing individual video metadata points.

    This class generates scatter plots that visualize:
    1. Video metrics (duration/filesize) vs. datetime, color-coded by device
    2. Video filesize vs. duration, color-coded by device
    3. Video filesize vs. total pixel count, color-coded by device

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
            DataFrame with columns for device, duration, filesize, pixel count, datetime, and day of week
        """
        data = []

        for video in videos:
            # Calculate total pixel count (width * height * frame_count)
            pixel_count = video.width * video.height * video.frame_count

            # Create a row for each video with only the necessary fields
            data.append(
                {
                    "Device": video.device,
                    "Duration": video.duration.total_seconds(),
                    "Filesize": video.file_size,  # In MB
                    "PixelCount": pixel_count,
                    "Resolution": f"{video.width}x{video.height}",
                    "DateTime": video.datetime_obj,
                    "DayOfWeek": video.datetime_obj.strftime("%A"),
                }
            )

        return pd.DataFrame(data)

    @classmethod
    def metric_vs_datetime(
        cls,
        videos: List[VideoMetadata],
        metric: str = "duration",
        use_log_scale: bool = True,
        **kwargs,
    ) -> go.Figure:
        """
        Create a scatter plot of video metrics vs. continuous datetime.

        Args:
            videos: List of VideoMetadata objects
            metric: The metric to plot on the y-axis, either "duration" or "filesize"
            use_log_scale: If True, use logarithmic scale for the y-axis
            **kwargs: Additional keyword arguments to pass to the figure layout

        Returns:
            Plotly figure object
        """
        cls._check_versions()

        # Validate metric parameter
        if metric not in ["duration", "filesize"]:
            raise ValueError('Metric must be either "duration" or "filesize"')

        # Prepare data
        df = cls._prepare_data(videos)

        # Get device colors - use the exact same color scheme as video_graph_creator.py
        colors = DashboardConfig.get_device_colors()

        # Create a figure from scratch with device coloring
        fig = go.Figure()

        # Set up metric-specific configurations
        metric_column = "Duration" if metric == "duration" else "Filesize"
        metric_title = (
            "Duration (seconds)" if metric == "duration" else "File Size (MB)"
        )
        metric_hover_label = "Duration" if metric == "duration" else "Filesize"
        metric_units = "seconds" if metric == "duration" else "MB"
        plot_title = f"Video {metric.capitalize()} by Date & Time"

        # Add a trace for each device
        for device in df["Device"].unique():
            device_df = df[df["Device"] == device]

            fig.add_trace(
                go.Scatter(
                    x=device_df["DateTime"],
                    y=device_df[metric_column],
                    mode="markers",
                    marker={
                        "color": colors[device],
                        "size": 3,
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
                        f"<b>{metric_hover_label}:</b> %{{y:.2f}} {metric_units}<extra></extra>"
                    ),
                )
            )

        # Get default figure config
        fig_config = DashboardConfig.get_figure_config()

        # Update layout with defaults plus any custom settings
        fig_height = DashboardConfig.get_figure_height()
        layout_config = {
            "title": plot_title,
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

        # Set y-axis with optional log scale
        y_axis_config = {
            "title": metric_title,
            **fig_config["axes"]["grid"],
            **fig_config["axes"]["yaxis"],
        }

        if use_log_scale:
            y_axis_config["type"] = "log"

        fig.update_yaxes(**y_axis_config)

        return fig

    @classmethod
    def filesize_vs_pixel(
        cls,
        videos: List[VideoMetadata],
        use_log_scale: bool = True,
        **kwargs,
    ) -> go.Figure:
        """
        Create a scatter plot of video filesize vs. total pixel count.

        This plot shows the relationship between total pixel count (width * height * frame_count)
        and filesize for each device, which can help identify compression efficiency differences.

        Args:
            videos: List of VideoMetadata objects
            use_log_scale: If True, use logarithmic scale for both axes (recommended for large numbers)
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
                    x=device_df["PixelCount"],
                    y=device_df["Filesize"],
                    mode="markers",
                    marker={
                        "color": colors[device],
                        "size": 3,
                        "opacity": 1.0,
                        "symbol": "square",
                    },
                    name=device,
                    customdata=np.stack(
                        (
                            device_df["Resolution"].values,
                            device_df["Duration"].values,
                            device_df["DateTime"]
                            .dt.strftime("%Y-%m-%d %H:%M:%S")
                            .values,
                        ),
                        axis=-1,
                    ),
                    hovertemplate=(
                        "<b>%{customdata[2]}</b><br>"
                        f"<b>{device}</b><br>"
                        "<b>Resolution:</b> %{customdata[0]}<br>"
                        "<b>Duration:</b> %{customdata[1]:.2f} seconds<br>"
                        "<b>Pixel Count:</b> %{x:,.0f} pixels<br>"
                        "<b>Filesize:</b> %{y:.2f} MB<extra></extra>"
                    ),
                )
            )

        # Get default figure config
        fig_config = DashboardConfig.get_figure_config()

        # Update layout with defaults plus any custom settings
        fig_height = DashboardConfig.get_figure_height()
        layout_config = {
            "title": "Video File Size vs. Total Pixel Count",
            "height": fig_height,
            **fig_config["layout"],
            **kwargs,
        }
        fig.update_layout(**layout_config)

        # Set axis types (log or linear)
        axis_type = "log" if use_log_scale else "linear"

        # Update axes
        fig.update_xaxes(
            title="Total Pixel Count (width × height × frames)",
            type=axis_type,
            **fig_config["axes"]["grid"],
        )

        fig.update_yaxes(
            title="File Size (MB)",
            type=axis_type,
            **fig_config["axes"]["grid"],
            **fig_config["axes"]["yaxis"],
        )

        return fig

    @classmethod
    def filesize_vs_duration(
        cls,
        videos: List[VideoMetadata],
        use_log_scale: bool = True,
        **kwargs,
    ) -> go.Figure:
        """
        Create a scatter plot of video filesize vs. duration.

        This plot shows the relationship between video duration and filesize
        for each device, collapsing the datetime dimension.

        Args:
            videos: List of VideoMetadata objects
            use_log_scale: If True, use logarithmic scale for both axes
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
                    x=device_df["Duration"],
                    y=device_df["Filesize"],
                    mode="markers",
                    marker={
                        "color": colors[device],
                        "size": 3,
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
                        "<b>Duration:</b> %{x:.2f} seconds<br>"
                        "<b>Filesize:</b> %{y:.2f} MB<extra></extra>"
                    ),
                )
            )

        # Get default figure config
        fig_config = DashboardConfig.get_figure_config()

        # Update layout with defaults plus any custom settings
        fig_height = DashboardConfig.get_figure_height()
        layout_config = {
            "title": "Video File Size vs. Duration",
            "height": fig_height,
            **fig_config["layout"],
            **kwargs,
        }
        fig.update_layout(**layout_config)

        # Set axis types (log or linear)
        axis_type = "log" if use_log_scale else "linear"

        # Update axes
        fig.update_xaxes(
            title="Duration (seconds)",
            type=axis_type,
            **fig_config["axes"]["grid"],
        )

        fig.update_yaxes(
            title="File Size (MB)",
            type=axis_type,
            **fig_config["axes"]["grid"],
            **fig_config["axes"]["yaxis"],
        )

        return fig

    @staticmethod
    def save_graphs_to_html(figures: List[go.Figure], output_file: str):
        """Saves all graphs to a single HTML file using shared configuration."""
        fig_height = DashboardConfig.get_figure_height()
        styles = DashboardConfig.get_html_styles()

        # Convert style dictionaries to CSS strings
        title_style = "; ".join(f"{k}: {v}" for k, v in styles["title"].items())
        container_style = "; ".join(f"{k}: {v}" for k, v in styles["container"].items())
        graph_style = "; ".join(
            f"{k}: {v}" for k, v in styles["graph_container"].items()
        )

        html_template = f"""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Video Statistics</title>
                <link href="https://cdnjs.cloudflare.com/ajax/libs/bootstrap/5.3.2/css/bootstrap.min.css" rel="stylesheet">
                <script src='https://cdn.plot.ly/plotly-2.20.0.min.js'></script>
                <style>
                    :root {{
                        --bs-font-sans-serif: system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
                    }}

                    body {{
                        font-family: var(--bs-font-sans-serif);
                        margin: 0;
                        padding: 0;
                    }}

                    .dashboard-title {{ {title_style} }}
                    .container-fluid {{ {container_style} }}
                    .graph-container {{ {graph_style} }}

                    .plotly-graph-div {{
                        width: 100%;
                        height: {fig_height}px;
                    }}
                </style>
            </head>
            <body>
                <div class="container-fluid">
                    <h2 class="dashboard-title">Video Statistics Dashboard</h2>
                    <div class="row">
                        <div class="col-12">
        """

        with open(output_file, "w") as file:
            file.write(html_template)

            for fig in figures:
                file.write('<div class="graph-container">')
                file.write(fig.to_html(full_html=False, include_plotlyjs=False))
                file.write("</div>")

            file.write(
                """
                        </div>
                    </div>
                </div>
            </body>
            </html>
            """
            )

    @staticmethod
    def create_graphs_file(
        videos: List[VideoMetadata],
        output_file: str,
    ):
        """
        Create a comprehensive HTML dashboard with multiple scatter plots and save to a file.

        This method generates four different scatter plots:
        1. Video duration vs. datetime
        2. Video filesize vs. datetime
        3. Video filesize vs. duration
        4. Video filesize vs. total pixel count

        All plots are combined into a single HTML file with consistent styling and layout.

        Args:
            videos: List of VideoMetadata objects to visualize
            output_file: Path where the HTML file will be saved

        Returns:
            None
        """
        # Create datetime scatter plots
        duration_vs_datetime_fig = PlotCreator.metric_vs_datetime(
            videos, metric="duration"
        )
        filesize_vs_datetime_fig = PlotCreator.metric_vs_datetime(
            videos, metric="filesize"
        )

        # Create duration vs. filesize scatter plot
        filesize_vs_duration_fig = PlotCreator.filesize_vs_duration(videos)

        # Create filesize vs. pixel count scatter plot
        filesize_vs_pixel_fig = PlotCreator.filesize_vs_pixel(videos)

        # Save figures to HTML
        PlotCreator.save_graphs_to_html(
            [
                duration_vs_datetime_fig,
                filesize_vs_datetime_fig,
                filesize_vs_duration_fig,
                filesize_vs_pixel_fig,
            ],
            output_file,
        )


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
        os.path.join(root_database, "archive_000.csv"),
        os.path.join(root_database, "archive_001.csv"),
        os.path.join(root_database, "backup.csv"),
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

    PlotCreator.create_graphs_file(videos, os.path.join(out_dir, "stats_graphs.html"))
