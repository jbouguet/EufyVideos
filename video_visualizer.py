#!/usr/bin/env python3

"""
Video visualization module for creating interactive Plotly graphs from video metadata.

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
    
    # Create visualizer and generate graphs
    visualizer = VideoVisualizer()
    visualizer.create_graphs_file(videos, 'video_analytics.html')
"""

from datetime import datetime
from typing import Dict, List, Literal, Tuple

import pandas as pd
import plotly.graph_objects as go

from config import Config
from video_metadata import VideoMetadata  # Main interface point


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
                "is_percentage": False,
                "is_hourly": False,
            }

        fig = go.Figure()
        x_column = "Hour" if config.get("is_hourly") else "Date"
        devices = [dev for dev in Config.get_device_order() if dev in data.columns]
        colors = Config.get_device_colors()

        # Add device traces
        for device in devices:
            hovertemplate = (
                f"<b>{device}</b><br>"
                + (
                    f"<b>Hour:</b> %{{x}}<br>"
                    if config.get("is_hourly")
                    else f"<b>Date:</b> %{{x|%Y-%m-%d}}<br>"
                )
                + f"<b>Value:</b> %{{y:.2f}}<extra></extra>"
            )

            fig.add_trace(
                go.Bar(
                    x=data[x_column],
                    y=data[device],
                    name=device,
                    marker_color=colors[device],
                    text=data[device].apply(lambda x: f"{x:.2f}"),
                    textposition="inside",
                    hovertemplate=hovertemplate,
                )
            )

        # Add total line if appropriate
        if not config.get("is_cumulative") and not config.get("is_percentage"):
            total = data[devices].sum(axis=1)
            hovertemplate = (
                "<b>Total</b><br>"
                + (
                    f"<b>Hour:</b> %{{x}}<br>"
                    if config.get("is_hourly")
                    else f"<b>Date:</b> %{{x|%Y-%m-%d}}<br>"
                )
                + f"<b>Value:</b> %{{y:.2f}}<extra></extra>"
            )

            fig.add_trace(
                go.Scatter(
                    x=data[x_column],
                    y=total,
                    mode="lines+markers",
                    name="Total",
                    line=dict(color="red", width=2, shape="spline"),
                    marker=dict(size=6, color="red"),
                    hovertemplate=hovertemplate,
                )
            )

        # Update layout and axes
        fig_height = Config.get_figure_height()
        fig.update_layout(
            title=title,
            xaxis_title="" if config.get("is_hourly") else "",
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
            fig.update_xaxes(tickmode="linear", tick0=0, dtick=1, range=[-0.5, 23.5])
        else:
            fig.update_xaxes(
                dtick="D1",
                tickformat="%a %Y/%m/%d",
                tickangle=-90,
                tickfont=dict(size=8),
                rangeslider=dict(visible=False),
                type="date",
            )

        return fig


class VideoVisualizer:
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

        # Create visualizer
        visualizer = VideoVisualizer()

        # Generate graphs file
        visualizer.create_graphs_file(
            videos,
            'video_analytics.html'
        )
    """

    def __init__(self):
        """Initialize with data aggregator and graph creator instances"""
        self.data_aggregator = VideoDataAggregator()
        self.graph_creator = VideoGraphCreator()

    def create_daily_graphs(
        self, daily_data: Dict[str, pd.DataFrame]
    ) -> List[go.Figure]:
        """
        Creates all daily graphs from aggregated data.

        This method generates three types of visualizations for each metric:
        1. Daily values (stacked bar charts)
        2. Cumulative totals (stacked bar charts with running totals)
        3. Percentage distributions (showing relative device contributions)

        Example:
            daily_graphs = visualizer.create_daily_graphs(daily_data)
            # Returns list of figures in order: activity, duration, filesize,
            # followed by their cumulative and percentage distribution variants
        """
        figures = []

        # Basic daily graphs
        figures.append(
            self.graph_creator.create_figure(
                daily_data["activity"], "Event Video Count per Device per Day", "Count"
            )
        )

        figures.append(
            self.graph_creator.create_figure(
                daily_data["duration"],
                "Video Capture Time in Hours per Device per Day",
                "Duration (Hours)",
            )
        )

        figures.append(
            self.graph_creator.create_figure(
                daily_data["filesize"],
                "Video File Size in MB per Device per Day",
                "File Size (MB)",
            )
        )

        # Cumulative graphs and percentage distributions
        for metric, df in daily_data.items():
            cumulative = df.set_index("Date").cumsum().reset_index()
            title_map = {
                "activity": (
                    "Cumulative Event Video Count per Device",
                    "Cumulative Count",
                ),
                "duration": (
                    "Cumulative Video Capture Time in Hours per Device",
                    "Cumulative Duration (Hours)",
                ),
                "filesize": (
                    "Cumulative Video Disk Space in MB per Device",
                    "Cumulative File Size (MB)",
                ),
            }

            figures.append(
                self.graph_creator.create_figure(
                    cumulative,
                    title_map[metric][0],
                    title_map[metric][1],
                    {"is_cumulative": True},
                )
            )

            # Add percentage distribution
            devices = [dev for dev in Config.get_device_order() if dev in df.columns]
            total = cumulative[devices].sum(axis=1)
            percentage_dist = cumulative[devices].div(total, axis=0) * 100
            percentage_dist["Date"] = cumulative["Date"]

            figures.append(
                self.graph_creator.create_figure(
                    percentage_dist,
                    f"Cumulative {title_map[metric][0]} Percentage Distribution",
                    "Percentage (%)",
                    {"is_percentage": True},
                )
            )

        return figures

    def create_hourly_graphs(
        self, hourly_data: Dict[str, pd.DataFrame]
    ) -> List[go.Figure]:
        """
        Creates all hourly graphs from aggregated data.

        This method generates hourly distribution visualizations:
        1. Hourly values for each metric (stacked bar charts)
        2. Cumulative totals for activity (showing patterns across hours)

        Example:
            hourly_graphs = visualizer.create_hourly_graphs(hourly_data)
            # Returns list of figures showing hourly patterns
        """
        figures = []

        title_map = {
            "activity": ("Hourly Video Count per Device", "Count"),
            "duration": (
                "Hourly Video Duration per Device (Hours)",
                "Duration (Hours)",
            ),
            "filesize": ("Hourly Video File Size per Device (MB)", "File Size (MB)"),
        }

        for metric, df in hourly_data.items():
            figures.append(
                self.graph_creator.create_figure(
                    df, title_map[metric][0], title_map[metric][1], {"is_hourly": True}
                )
            )

            if metric == "activity":  # Only create cumulative for activity
                cumulative = df.set_index("Hour").cumsum().reset_index()
                figures.append(
                    self.graph_creator.create_figure(
                        cumulative,
                        "Cumulative Hourly Video Count per Device",
                        "Cumulative Count",
                        {"is_cumulative": True, "is_hourly": True},
                    )
                )

        return figures

    @staticmethod
    def save_graphs_to_html(figures: List[go.Figure], output_file: str):
        """
        Saves all graphs to a single HTML file.

        Creates an interactive HTML page with all graphs stacked vertically.
        Uses plotly's modular approach to minimize file size by including
        the plotly.js library only once.

        Example:
            visualizer.save_graphs_to_html(
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
            visualizer = VideoVisualizer()
            visualizer.create_graphs_file(videos, 'video_analytics.html')
        """
        # Get aggregated data
        daily_data = self.data_aggregator.get_daily_aggregates(videos)
        hourly_data = self.data_aggregator.get_hourly_aggregates(videos)

        # Create all graphs
        daily_graphs = self.create_daily_graphs(daily_data)
        hourly_graphs = self.create_hourly_graphs(hourly_data)

        # Save to file
        self.save_graphs_to_html(daily_graphs + hourly_graphs, output_file)
