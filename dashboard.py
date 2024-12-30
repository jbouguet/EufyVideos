#!/usr/bin/env python3

"""
Dashboard module for creating interactive Plotly graphs from video metadata.

This module interfaces with video_metadata.py to create statistical visualizations of video data.
It processes VideoMetadata objects to generate various types of graphs showing temporal distributions
and device-specific patterns.

Key interfaces with video_metadata.py:
1. Takes List[VideoMetadata] as input for visualization
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
    dashboard = Dashboard()
    dashboard.create_graphs_file(videos, 'video_analytics.html')
"""

from datetime import datetime
from typing import Dict, List, Literal

import pandas as pd
import plotly.graph_objects as go

from config import Config
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

    @classmethod
    def get_daily_aggregates(
        cls, videos: List[VideoMetadata]
    ) -> Dict[str, pd.DataFrame]:
        """
        Returns daily aggregates for all metrics.

        Example:
            daily_data = get_daily_aggregates(videos)
            activity_by_date = daily_data['activity']  # Videos per day
            duration_by_date = daily_data['duration']  # Hours per day
            storage_by_date = daily_data['filesize']   # MB per day
        """
        return {
            "activity": cls._aggregate_by_metric(videos, "date", "count").rename(
                columns={"TimeKey": "Date"}
            ),
            "duration": cls._aggregate_by_metric(videos, "date", "duration").rename(
                columns={"TimeKey": "Date"}
            ),
            "filesize": cls._aggregate_by_metric(videos, "date", "filesize").rename(
                columns={"TimeKey": "Date"}
            ),
        }

    @classmethod
    def get_hourly_aggregates(
        cls, videos: List[VideoMetadata]
    ) -> Dict[str, pd.DataFrame]:
        """
        Returns hourly aggregates for all metrics.

        Example:
            hourly_data = get_hourly_aggregates(videos)
            activity_by_hour = hourly_data['activity']  # Videos per hour
            duration_by_hour = hourly_data['duration']  # Hours per hour slot
            storage_by_hour = hourly_data['filesize']   # MB per hour
        """
        return {
            "activity": cls._aggregate_by_metric(videos, "hour", "count").rename(
                columns={"TimeKey": "Hour"}
            ),
            "duration": cls._aggregate_by_metric(videos, "hour", "duration").rename(
                columns={"TimeKey": "Hour"}
            ),
            "filesize": cls._aggregate_by_metric(videos, "hour", "filesize").rename(
                columns={"TimeKey": "Hour"}
            ),
        }


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
        config: Dict[str, bool] = None,
    ) -> go.Figure:
        """Creates a plotly figure with consistent styling"""
        if config is None:
            config = {
                "is_cumulative": False,
                "is_hourly": False,
            }

        fig = go.Figure()
        x_column = "Hour" if config.get("is_hourly") else "Date"
        devices = [dev for dev in Config.get_device_order() if dev in data.columns]
        colors = Config.get_device_colors()

        # Add device traces
        for device in devices:
            hovertemplate = (
                f"<b>Hour:</b> %{{x:02d}}:00<br>"
                if config.get("is_hourly")
                else f"<b>Date:</b> %{{x|%Y-%m-%d}}<br>"
            ) + f"<b>{device}:</b> %{{y:.0f}}<extra></extra>"

            fig.add_trace(
                go.Bar(
                    x=data[x_column],
                    y=data[device],
                    name=device,
                    marker_color=colors[device],
                    text=data[device].apply(lambda x: f"{int(x)}"),
                    textposition="inside",
                    hovertemplate=hovertemplate,
                )
            )

        # Add total line
        total = data[devices].sum(axis=1)
        hovertemplate = (
            f"<b>Hour:</b> %{{x:02d}}:00<br>"
            if config.get("is_hourly")
            else f"<b>Date:</b> %{{x|%Y-%m-%d}}<br>"
        ) + f"<b>Total:</b> %{{y:.0f}}<extra></extra>"

        fig.add_trace(
            go.Scatter(
                x=data[x_column],
                y=total,
                mode="lines",
                name="Total",
                line=dict(color="red", width=1, shape="spline"),
                hovertemplate=hovertemplate,
            )
        )

        # Update layout and axes
        fig_height = Config.get_figure_height()
        fig.update_layout(
            title=title,
            xaxis_title="",
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
            hours = list(range(24))
            fig.update_xaxes(
                tickmode="array",
                tickvals=hours,
                ticktext=[f"{h:02d}:00" for h in hours],
                range=[-0.5, 23.5],
                type="linear",
            )
        else:
            fig.update_xaxes(
                dtick="D1",
                tickmode="auto",
                nticks=100,
                tickformat="%a %Y-%m-%d",
                tickangle=-90,
                tickfont=dict(size=10),
                rangeslider=dict(visible=False),
                type="date",
            )
        return fig


class Dashboard:
    """
    Main class for generating and saving video analytics visualizations.

    This class orchestrates the entire visualization process:
    1. Takes a list of VideoMetadata objects as input
    2. Uses VideoDataAggregator to process the metadata into aggregated datasets
    3. Uses VideoGraphCreator to generate visualizations
    4. Saves the results to an interactive HTML file

    Example:
        # Load videos using VideoMetadata's methods
        videos = VideoMetadata.load_videos_from_directories('videos/')

        # Create dashboard
        dashboard = Dashboard()

        # Generate graphs file
        dashboard.create_graphs_file(
            videos,
            'video_analytics.html'
        )
    """

    def __init__(self):
        """Initialize with data aggregator and graph creator instances"""
        self.data_aggregator = VideoDataAggregator()
        self.graph_creator = VideoGraphCreator()

    def create_graphs(
        self, daily_data: Dict[str, pd.DataFrame], hourly_data: Dict[str, pd.DataFrame]
    ) -> List[go.Figure]:
        """
        Creates daily and hourly activity graphs from aggregated data.
        """
        return [
            self.graph_creator.create_figure(
                daily_data["activity"], "Event Video Count per Device per Day", "Count"
            ),
            self.graph_creator.create_figure(
                daily_data["activity"].set_index("Date").cumsum().reset_index(),
                "Cumulative Event Video Count per Device",
                "Cumulative Count",
                {"is_cumulative": True},
            ),
            self.graph_creator.create_figure(
                hourly_data["activity"],
                "Hourly Video Count per Device",
                "Count",
                {"is_hourly": True},
            ),
        ]

    @staticmethod
    def save_graphs_to_html(figures: List[go.Figure], output_file: str):
        """
        Saves all graphs to a single HTML file.

        Creates an interactive HTML page with all graphs stacked vertically.
        Uses plotly's modular approach to minimize file size by including
        the plotly.js library only once.

        Example:
            dashboard.save_graphs_to_html(
                daily_graphs + hourly_graphs,
                'video_analytics.html'
            )
        """
        fig_height = Config.get_figure_height()

        html_template = f"""
            <html>
            <head>
                <title>Video Analytics</title>
                <script src='https://cdn.plot.ly/plotly-2.20.0.min.js'></script>
                <meta name='viewport' content='width=device-width, initial-scale=1.0'>
                <style>
                    body {{ margin: 0; padding: 20px; }}
                    .plotly-graph-div {{
                        width: 100%;
                        height: {fig_height}px;
                        margin-bottom: 20px;
                    }}
                </style>
            </head>
            <body>
        """

        with open(output_file, "w") as file:
            file.write(html_template)
            for fig in figures:
                file.write(fig.to_html(full_html=False, include_plotlyjs=False))
            file.write("</body></html>")

    def create_graphs_file(self, videos: List[VideoMetadata], output_file: str):
        """
        Main entry point for creating all graphs and saving to file.

        This method orchestrates the entire visualization process:
        1. Aggregates video metadata into daily and hourly statistics
        2. Creates various types of visualizations
        3. Saves everything to an interactive HTML file

        Example:
            # Load videos using VideoMetadata's methods
            videos = VideoMetadata.load_videos_from_directories('videos/')

            # Create graphs
            dashboard = Dashboard()
            dashboard.create_graphs_file(videos, 'video_analytics.html')
        """
        # Get aggregated data
        daily_data = self.data_aggregator.get_daily_aggregates(videos)
        hourly_data = self.data_aggregator.get_hourly_aggregates(videos)

        # Create all graphs
        graphs = self.create_graphs(daily_data, hourly_data)

        # Save to file
        self.save_graphs_to_html(graphs, output_file)


if __name__ == "__main__":

    # Testing code for the module.
    import logging
    import os
    import sys

    from logging_config import set_logger_level_and_format
    from video_database import VideoDatabase, VideoDatabaseList
    from video_filter import DateRange, VideoFilter, VideoSelector

    set_logger_level_and_format(logger, level=logging.DEBUG, extended_format=True)

    root_database: str = (
        "/Users/jeanyves.bouguet/Documents/EufySecurityVideos/EufyVideos/record/"
    )
    out_dir: str = "/Users/jeanyves.bouguet/Documents/EufySecurityVideos/stories"

    video_metadata_file1: str = os.path.join(root_database, "videos_in_batches.csv")
    video_metadata_file2: str = os.path.join(root_database, "videos_in_backup.csv")

    video_database = VideoDatabaseList(
        [
            VideoDatabase(
                video_directories=None, video_metadata_file=video_metadata_file1
            ),
            VideoDatabase(
                video_directories=None, video_metadata_file=video_metadata_file2
            ),
        ]
    ).load_videos()

    # Creating a partial filtered view of the total database, filtering by date range, timreange and devices.
    # Note that any of those filtering conditions could be removed by setting the entries date_range, time_range or devices to None.
    # The range of acceptable dates can be computed from the min and maximum dates of videos in video_database.
    # Since the dates are ordered, the minimum date is video_database[0].date_str and video_database[-1].date_str
    # The range of times is between "00:00:00" and "23:59:59" and the start and end times do not need to be ordered
    # (to selected videos crossing midnight)
    # The set of devices to pick from for filtering can be picked from Config.get_all_devices imported from config.py
    videos = VideoFilter.by_selectors(
        video_database,
        [
            VideoSelector(
                date_range=DateRange(start="2024-05-05", end="2024-06-05"),
            ),
            VideoSelector(
                date_range=DateRange(start="2024-11-11", end="2024-12-31"),
            ),
        ],
    )

    # Get aggregated data
    data_aggregator = VideoDataAggregator()
    daily_data = data_aggregator.get_daily_aggregates(videos)
    hourly_data = data_aggregator.get_hourly_aggregates(videos)

    # Create dashboard
    dashboard = Dashboard()

    # Create all graphs
    graphs = dashboard.create_graphs(daily_data, hourly_data)

    # Save to file
    dashboard.save_graphs_to_html(
        graphs, os.path.join(out_dir, "video_analytics_partial.html")
    )

    sys.exit()
