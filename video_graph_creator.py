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
        creator = VideoGraphCreator()
        fig = creator.create_figure(
            daily_data['activity'],
            "Daily Video Count",
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
            bins_per_hour = config.get("bins_per_hour", 4)
            total_bins = 24 * bins_per_hour
            tick_values = [i / bins_per_hour for i in range(total_bins)]

            # Create time labels based on bin size
            tick_labels = []
            for i in range(total_bins):
                hours = int(i / bins_per_hour)
                minutes = int((i % bins_per_hour) * (60 / bins_per_hour))
                time_str = f"{hours:02d}:{minutes:02d}"
                tick_labels.append(time_str)

            fig.update_xaxes(
                tickmode="array",
                tickvals=tick_values,
                ticktext=tick_labels,
                range=[-0.5 / bins_per_hour, 23.5],
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
