#!/usr/bin/env python3

"""
VideoGraphCreator module for creating Plotly graphs from aggregated video metadata.

This module interfaces with video_data_aggregator.py to create visualizations of video statistical data.
"""

from typing import Dict

import pandas as pd
import plotly.graph_objects as go

from config import Config
from logging_config import create_logger

logger = create_logger(__name__)


class VideoGraphCreator:
    """
    Handles creation of plotly graphs from aggregated video data.

    While this class doesn't directly interface with VideoMetadata objects,
    it processes the aggregated data that was derived from VideoMetadata properties.
    It uses Config.get_device_order() and Config.get_device_colors() to maintain
    consistent device representation across graphs.

    Example:
        creator = VideoGraphCreator()
        fig = creator.create_figure(
            daily_data['activity'],
            "Daily Video Count",
            "Count"
        )
    """

    @staticmethod
    def create_figure(
        data: pd.DataFrame,
        title: str,
        y_axis_title: str,
        config: Dict[str, bool | int] = None,
    ) -> go.Figure:
        """Creates a plotly figure with consistent styling"""
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
        devices = [dev for dev in Config.get_device_order() if dev in data.columns]
        colors = Config.get_device_colors()

        # Add device traces
        for device in devices:
            if config.get("is_hourly"):
                hovertemplate = (
                    "<b>Time:</b> %{customdata}<br>"
                    + f"<b>{device}:</b> %{{y:.0f}}<extra></extra>"
                )
            else:
                hovertemplate = (
                    f"<b>Date:</b> %{{x|%Y-%m-%d}}<br>"
                    + f"<b>{device}:</b> %{{y:.0f}}<extra></extra>"
                )

            if config.get("is_hourly"):
                customdata = [decimal_hour_to_time(h) for h in data[x_column]]
            else:
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
            # hovertemplate = (
            #    "<b>Hour:</b> %{x:.2f}<br>" + "<b>Total:</b> %{y:.0f}<extra></extra>"
            # )
            hovertemplate = (
                "<b>Time:</b> %{customdata}<br>"
                + f"<b>{device}:</b> %{{y:.0f}}<extra></extra>"
            )
        else:
            hovertemplate = (
                f"<b>Date:</b> %{{x|%Y-%m-%d}}<br>"
                + "<b>Total:</b> %{y:.0f}<extra></extra>"
            )

        if config.get("is_hourly"):
            customdata = [decimal_hour_to_time(h) for h in data[x_column]]
        else:
            customdata = None

        fig.add_trace(
            go.Scatter(
                x=data[x_column],
                y=total,
                mode="lines",
                name="Total",
                line=dict(color="red", width=1, shape="spline"),
                hovertemplate=hovertemplate,
                customdata=customdata,
            )
        )

        # Update layout and axes
        fig_height = Config.get_figure_height()
        fig.update_layout(
            title=title,
            xaxis_title="Time" if config.get("is_hourly") else "Date",
            yaxis_title=y_axis_title,
            barmode="stack",
            bargap=0,
            plot_bgcolor="white",
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
            height=fig_height,
            margin=dict(l=50, r=50, t=80, b=50),
        )

        fig.update_xaxes(
            showgrid=True, gridwidth=1, gridcolor="lightgrey", zeroline=False
        )

        fig.update_yaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor="lightgrey",
            zeroline=False,
            rangemode="nonnegative",
        )

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
            fig.update_xaxes(
                dtick="D1",
                tickformat="%a %Y/%m/%d",
                tickangle=-90,
                tickfont=dict(size=6),
                rangeslider=dict(visible=False),
                type="date",
            )

        return fig
