#!/usr/bin/env python3

"""
VideoGraphCreator module for creating Plotly graphs from aggregated video metadata.

This module interfaces with video_data_aggregator.py to create visualizations of video statistical data.
"""

import warnings
from typing import Dict, Optional, Union, cast

import pandas as pd
import plotly.graph_objects as go

from config import Config
from dashboard_config import DashboardConfig
from logging_config import create_logger
from video_data_aggregator import TimeKeyType

logger = create_logger(__name__)


class VideoGraphCreator:
    """
    Handles creation of plotly graphs from aggregated video data.

    While this class doesn't directly interface with VideoMetadata objects,
    it processes the aggregated data that was derived from VideoMetadata properties.
    It uses Config.get_all_devices() and DashboardConfig.get_device_colors() to maintain
    consistent device representation across graphs.

    Example:
        videos = VideoMetadata.load_videos_from_directories('videos/')
        config = {"bins_per_hours": 4} # Temporal binning set to 15 minutes (60/4)
        metrics = ['activity']
        aggregator = VideoDataAggregator(metrics=metrics, config=config)
        daily_data, hourly_data = aggregator.run(videos)

        fig = VideoGraphCreator.create_figure(
            daily_data[metrics[0]],
            title="Daily Video Count",
            config=config,
        )

        Version Requirements:
        - Python: >=3.12,<3.14
        - Plotly: ==5.18.0
        - Pandas: >=2.2.0,<3.0.0

    See requirements.txt for complete dependency specifications.
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
    def create_figure(
        data: pd.DataFrame,
        title: str,
        config: Optional[Dict[str, Union[TimeKeyType, int]]] = None,
        **kwargs,
    ) -> go.Figure:
        """
        Creates a plotly figure with consistent styling using shared configuration.

        Version Notes:
            This method requires Plotly 5.18.0 for correct visualization. Using other
            versions may result in incorrect graph orientation or styling.
        """
        VideoGraphCreator._check_versions()
        if config is None:
            config = {"time_key": "date", "bins_per_hour": 4}

        def decimal_hour_to_time(decimal_hour):
            hours = int(decimal_hour)
            minutes = int((decimal_hour - hours) * 60)
            seconds = int(((decimal_hour - hours) * 60 - minutes) * 60)
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

        fig = go.Figure()
        x_column = "Hour" if config.get("time_key") == "hour" else "Date"
        devices = [dev for dev in Config.get_all_devices() if dev in data.columns]
        colors = DashboardConfig.get_device_colors()

        # For date-based charts, ensure all dates are present with zeros for missing dates
        if config.get("time_key") == "date":
            # Convert Date column to datetime
            data[x_column] = pd.to_datetime(data[x_column])

            # Get min and max dates
            min_date = data[x_column].min()
            max_date = data[x_column].max()

            # Create a complete date range DataFrame
            complete_dates = pd.date_range(start=min_date, end=max_date, freq="D")
            complete_df = pd.DataFrame({x_column: complete_dates})

            # Merge with original data, filling NaN with 0
            data = pd.merge(complete_df, data, on=x_column, how="left")
            data = data.fillna(0)

        # Add device traces
        for device in devices:
            if config.get("time_key") == "hour":
                hovertemplate = (
                    "<b>Time:</b> %{customdata}<br>"
                    + f"<b>{device}:</b> %{{y:.0f}}<extra></extra>"
                )
                customdata = [decimal_hour_to_time(h) for h in data[x_column]]
            else:
                hovertemplate = (
                    f"<b>Date:</b> %{{x|%Y-%m-%d}}<br>"
                    + f"<b>{device}:</b> %{{y:.0f}}<extra></extra>"
                )
                customdata = None

            fig.add_trace(
                go.Bar(
                    x=data[x_column],
                    y=data[device],
                    name=device,
                    marker_color=colors[device],
                    text=data[device].apply(lambda x: f"{int(x)}"),
                    textposition="inside",
                    hovertemplate=hovertemplate,
                    customdata=customdata,
                )
            )

        # Add total line with gaps for zero values
        total = data[devices].sum(axis=1)
        # Replace zeros with None to create discontinuities in the line
        total_with_gaps = total.replace(0, None)

        if config.get("time_key") == "hour":
            hovertemplate = (
                "<b>Time:</b> %{customdata}<br><b>Total:</b> %{y:.0f}<extra></extra>"
            )
            customdata = [decimal_hour_to_time(h) for h in data[x_column]]
        else:
            hovertemplate = (
                "<b>Date:</b> %{x|%Y-%m-%d}<br><b>Total:</b> %{y:.0f}<extra></extra>"
            )
            customdata = None

        fig.add_trace(
            go.Scatter(
                x=data[x_column],
                y=total_with_gaps,  # Use the version with None values
                mode="lines",
                name="Total",
                line=dict(color="red", width=1, shape="linear"),
                hovertemplate=hovertemplate,
                customdata=customdata,
                connectgaps=False,  # Don't connect across null values
            )
        )

        # Get default figure config
        fig_config = DashboardConfig.get_figure_config()

        # Update layout with defaults plus any custom settings
        fig_height = DashboardConfig.get_figure_height()
        layout_config = {
            "title": title,
            "height": fig_height,
            **fig_config["layout"],
            **kwargs,
        }
        fig.update_layout(**layout_config)

        # Apply axis styling
        fig.update_xaxes(**fig_config["axes"]["grid"])
        fig.update_yaxes(**fig_config["axes"]["grid"], **fig_config["axes"]["yaxis"])

        if config.get("time_key") == "hour":
            # By default, set time interval to 15 minutes = 1 hour / 4 bin_per_hour
            bins_per_hour = int(config.get("bins_per_hour", 4))

            # Get the actual range of hours from the data
            min_hour = float(min(data[x_column]))
            max_hour = float(max(data[x_column]))

            # Round down min_hour and up max_hour to the nearest bin
            min_hour_binned = int(min_hour * bins_per_hour) / bins_per_hour
            max_hour_binned = int(max_hour * bins_per_hour) / bins_per_hour

            plot_range = [
                min_hour_binned,
                max_hour_binned + (1.0 / bins_per_hour),
            ]

            # Create tick values based on actual data range
            total_bins = int((max_hour_binned - min_hour_binned) * bins_per_hour)
            tick_values = [
                min_hour_binned + (float(i) / bins_per_hour)
                for i in range(total_bins + 1)
            ]

            logger.debug(f"min_hour = {min_hour}")
            logger.debug(f"max_hour = {max_hour}")
            logger.debug(f"min_hour_binned = {min_hour_binned}")
            logger.debug(f"max_hour_binned = {max_hour_binned}")
            logger.debug(f"plot_range = {plot_range}")
            logger.debug(f"total_bins = {total_bins}")

            # Create time labels for each tick
            tick_labels = []
            for val in tick_values:
                hours = int(val)
                minutes = int((val - hours) * 60)
                time_str = f"{hours:02d}:{minutes:02d}"
                tick_labels.append(time_str)

            fig.update_xaxes(
                tickmode="array",
                tickvals=tick_values,
                ticktext=tick_labels,
                range=plot_range,
                type="linear",
                tickfont=dict(size=8),
                tickangle=-90,
            )
        else:
            # Get min and max dates from the data
            min_date = data[x_column].min()
            max_date = data[x_column].max()
            logger.debug(f"min_date = {min_date}")
            logger.debug(f"max_date = {max_date}")
            logger.debug(f"Number of days = {(max_date - min_date).days + 1}")

            # Generate all dates between min and max
            all_dates = pd.date_range(start=min_date, end=max_date, freq="D")

            # Configure x-axis with explicit ticks for all dates
            fig.update_xaxes(
                tickmode="array",
                tickvals=all_dates,
                ticktext=[d.strftime("%a %Y-%m-%d") for d in all_dates],
                **fig_config["axes"]["xaxis_date"],
            )

        return fig


if __name__ == "__main__":

    # Testing code for the module.
    import logging
    import os
    import sys

    from logging_config import set_logger_level_and_format
    from video_data_aggregator import VideoDataAggregator
    from video_database import VideoDatabase, VideoDatabaseList
    from video_filter import DateRange, TimeRange, VideoFilter, VideoSelector

    set_logger_level_and_format(logger, level=logging.DEBUG, extended_format=True)

    # Load video database
    root_database = "/Users/jbouguet/Documents/EufySecurityVideos/record/"
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

    if video_database is None:
        logger.error("Failed to load video database")
        sys.exit(1)

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
    if video_database is None:
        videos = []
    else:
        videos = VideoFilter.by_selectors(video_database, selector)

    logger.debug(f"Number of videos: {len(videos)}")

    metric = "activity"

    # Get aggregated data
    data_aggregator = VideoDataAggregator(
        metrics=[metric], config={"bins_per_hour": bins_per_hour}
    )

    daily_data, hourly_data = data_aggregator.run(videos)

    print(daily_data[metric])

    df: pd.DataFrame = daily_data[metric]

    fig_daily = VideoGraphCreator.create_figure(
        df,
        title="Daily Video " + metric.capitalize(),
    )

    # fig_daily.show()
