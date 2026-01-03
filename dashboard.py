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

from typing import Dict, List, Optional, cast

import pandas as pd
import plotly.graph_objects as go
from dashboard_config import DashboardConfig
from logging_config import create_logger
from video_data_aggregator import VideoDataAggregator
from video_graph_creator import VideoGraphCreator
from video_metadata import VideoMetadata  # Main interface point

logger = create_logger(__name__)


class Dashboard:
    """
    Main class for generating and saving video analytics visualizations.

    This class orchestrates the entire visualization process:
    1. Takes a list of VideoMetadata objects as input
    2. Uses VideoDataAggregator to process the metadata into aggregated datasets
    3. Uses VideoGraphCreator to generate visualizations
    4. Saves the results to an interactive HTML file
    """

    def __init__(
        self,
        metrics: Optional[List[str]] = None,
        config: Optional[Dict[str, bool | int]] = None,
    ):
        """Initialize dashboard optionally specifying metrics of interest"""
        if metrics is None:
            # By default, aggregate across all metrics
            self.metrics = ["activity", "duration", "filesize"]
        else:
            # Only aggregate across a subset of specified metrics
            self.metrics = metrics
        # By default, set time interval to 5 minutes = 1 hour / 12 bin_per_hour
        if config is None:
            self.config = {"bins_per_hour": 12}
        else:
            self.config = config
        bins_per_hour = self.config.get("bins_per_hour", 12)
        self.config["bins_per_hour"] = bins_per_hour
        self.data_aggregator = VideoDataAggregator(
            metrics=self.metrics, config=self.config
        )

    def create_graphs(
        self,
        daily_data: Dict[str, pd.DataFrame],
        hourly_data: Dict[str, pd.DataFrame],
    ) -> List[go.Figure]:
        """
        Creates daily and hourly activity graphs from aggregated data.
        """
        # By default, set time interval to 5 minutes = 1 hour / 12 bin_per_hour
        bins_per_hour = self.config.get("bins_per_hour", 12)

        figs = []
        for metric in self.metrics:
            figs.append(
                VideoGraphCreator.create_figure(
                    daily_data[metric],
                    title="Daily Video " + metric.capitalize(),
                    config={"time_key": "date", "bins_per_hour": bins_per_hour},
                )
            )
            figs.append(
                VideoGraphCreator.create_figure(
                    hourly_data[metric],
                    title="Hourly Video " + metric.capitalize(),
                    config={"time_key": "hour", "bins_per_hour": bins_per_hour},
                )
            )
            figs.append(
                VideoGraphCreator.create_figure(
                    daily_data[metric].set_index("Date").cumsum().reset_index(),
                    title="Cumulative Daily Video " + metric.capitalize(),
                    config={"time_key": "date", "bins_per_hour": bins_per_hour},
                )
            )
        return figs

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
                <title>Video Analytics</title>
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
                    <h2 class="dashboard-title">Video Analytics Dashboard</h2>
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

    def create_graphs_file(
        self,
        videos: List[VideoMetadata],
        output_file: str,
    ):
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
        daily_data, hourly_data = self.data_aggregator.run(videos)
        graphs = self.create_graphs(daily_data, hourly_data)
        self.save_graphs_to_html(graphs, output_file)


if __name__ == "__main__":

    # Testing code for the module.
    import logging
    import os
    import sys

    from config import Config
    from logging_config import set_logger_level_and_format
    from video_database import VideoDatabase, VideoDatabaseList
    from video_filter import DateRange, TimeRange, VideoFilter, VideoSelector

    set_logger_level_and_format(logger, level=logging.DEBUG, extended_format=False)

    # Load video database
    root_database = "/Users/jbouguet/Documents/EufySecurityVideos/record/"
    metadata_files = [
        os.path.join(root_database, "archive_000.csv"),
        os.path.join(root_database, "archive_001.csv"),
        os.path.join(root_database, "archive_002.csv"),
        os.path.join(root_database, "backup.csv"),
        # Add more metadata files as needed
    ]
    out_dir: str = "/Users/jbouguet/Documents/EufySecurityVideos/stories"

    database_list = VideoDatabaseList(
        [
            VideoDatabase(video_directories=None, video_metadata_file=file)
            for file in metadata_files
        ]
    )

    database_list.to_file(os.path.join(out_dir, "video_database.yaml"))

    def validate_and_get_bounds(
        videos: Optional[object],
    ) -> tuple[List[VideoMetadata], VideoMetadata, VideoMetadata]:
        """Get the first and last videos from a validated video list."""
        if videos is None:
            logger.error("Failed to load video database")
            sys.exit(1)
        if not isinstance(videos, list) or not videos:
            logger.error("video_database is not a non-empty list")
            sys.exit(1)
        if not all(isinstance(v, VideoMetadata) for v in videos):
            logger.error("Not all elements are VideoMetadata objects")
            sys.exit(1)

        videos_typed = cast(List[VideoMetadata], videos)
        return videos_typed, videos_typed[0], videos_typed[-1]

    # Load and validate the videos
    raw_videos = database_list.load_videos()
    validated_videos, first_video, last_video = validate_and_get_bounds(raw_videos)
    min_date = first_video.date_str
    max_date = last_video.date_str

    bins_per_hour = 12

    start_date = min_date
    end_date = max_date
    start_time = "00:00:00"
    end_time = "23:59:59"
    devices = Config.get_all_devices()  # ["Backyard"]

    logger.debug(f"Date range: {start_date} to {end_date}")
    logger.debug(f"Time range: {start_time} to {end_time}")
    logger.debug(f"Time bin size: {60 / bins_per_hour} minutes")

    selector = VideoSelector(
        date_range=DateRange(start=start_date, end=end_date),
        time_range=TimeRange(start=start_time, end=end_time),
        devices=devices,
    )

    # Filter the database
    videos = VideoFilter.by_selectors(validated_videos, selector)

    logger.debug(f"Number of videos: {len(videos)}")

    VideoMetadata.export_videos_to_metadata_file(
        videos, os.path.join(out_dir, "filtered_videos.csv")
    )

    # Get aggregated data
    dashboard = Dashboard(config={"bins_per_hour": bins_per_hour})
    data_aggregator = VideoDataAggregator(config={"bins_per_hour": bins_per_hour})
    daily_data, hourly_data = data_aggregator.run(videos)
    graphs = dashboard.create_graphs(
        daily_data,
        hourly_data,
    )
    dashboard.save_graphs_to_html(graphs, os.path.join(out_dir, "video_analytics.html"))
