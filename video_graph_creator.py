#!/usr/bin/env python3

"""
VideoGraphCreator module for creating Plotly graphs from aggregated video metadata.

This module interfaces with video_data_aggregator.py to create visualizations of video statistical data.
"""

from typing import Dict, List

import pandas as pd
import plotly.graph_objects as go

from config import Config
from dashboard_config import DashboardConfig
from logging_config import create_logger

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
    """

    @staticmethod
    def create_figure(
        data: pd.DataFrame, title: str, config: Dict[str, bool | int] = None, **kwargs
    ) -> go.Figure:
        """Creates a plotly figure with consistent styling using shared configuration"""
        if config is None:
            config = {
                "is_cumulative": False,
                "is_hourly": False,
                "bins_per_hour": 4,
            }

        def decimal_hour_to_time(decimal_hour):
            hours = int(decimal_hour)
            minutes = int((decimal_hour - hours) * 60)
            seconds = int(((decimal_hour - hours) * 60 - minutes) * 60)
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

        fig = go.Figure()
        x_column = "Hour" if config.get("is_hourly") else "Date"
        devices = [dev for dev in Config.get_all_devices() if dev in data.columns]
        colors = DashboardConfig.get_device_colors()

        # Add device traces
        for device in devices:
            if config.get("is_hourly"):
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

        # Add total line
        total = data[devices].sum(axis=1)
        if config.get("is_hourly"):
            hovertemplate = (
                "<b>Time:</b> %{customdata}<br>"
                + "<b>Total:</b> %{y:.0f}<extra></extra>"
            )
            customdata = [decimal_hour_to_time(h) for h in data[x_column]]
        else:
            hovertemplate = (
                f"<b>Date:</b> %{{x|%Y-%m-%d}}<br>"
                + "<b>Total:</b> %{y:.0f}<extra></extra>"
            )
            customdata = None

        fig.add_trace(
            go.Scatter(
                x=data[x_column],
                y=total,
                mode="lines",
                name="Total",
                line=dict(color="red", width=1, shape="linear"),
                hovertemplate=hovertemplate,
                customdata=customdata,
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

        if config.get("is_hourly"):
            # By default, set time interval to 15 minutes = 1 hour / 4 bin_per_hour
            bins_per_hour = config.get("bins_per_hour", 4)

            # Get the actual range of hours from the data
            min_hour = min(data[x_column])
            max_hour = max(data[x_column])

            # Round down min_hour and up max_hour to the nearest bin
            min_hour_binned = int(min_hour * bins_per_hour) / bins_per_hour
            max_hour_binned = int(max_hour * bins_per_hour) / bins_per_hour

            plot_range = [
                min_hour_binned,
                max_hour_binned + (1 / (bins_per_hour)),
            ]

            # Create tick values based on actual data range
            total_bins = int((max_hour_binned - min_hour_binned) * bins_per_hour)
            tick_values = [
                min_hour_binned + (i / bins_per_hour) for i in range(total_bins + 1)
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
            fig.update_xaxes(**fig_config["axes"]["xaxis_date"])

        return fig

    @staticmethod
    def create_graphs(
        daily_data: Dict[str, pd.DataFrame],
        hourly_data: Dict[str, pd.DataFrame],
        metric_to_graph: str = "activity",
        bins_per_hour: int = 4,
    ) -> List[go.Figure]:
        """Creates daily and hourly activity graphs from aggregated data."""

        daily_fig = VideoGraphCreator.create_figure(
            daily_data[metric_to_graph],
            title="Daily Video " + metric_to_graph.capitalize(),
        )

        hourly_fig = VideoGraphCreator.create_figure(
            hourly_data[metric_to_graph],
            title="Hourly Video " + metric_to_graph.capitalize(),
            config={"is_hourly": True, "bins_per_hour": bins_per_hour},
        )

        cumulative_fig = VideoGraphCreator.create_figure(
            daily_data[metric_to_graph].set_index("Date").cumsum().reset_index(),
            title="Cumulative Daily Video " + metric_to_graph.capitalize(),
            config={"is_cumulative": True},
        )

        return [daily_fig, hourly_fig, cumulative_fig]
