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

from typing import Dict, List

import pandas as pd
import plotly.graph_objects as go

from config import Config
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
        # Create figures
        daily_fig = self.graph_creator.create_figure(
            daily_data["activity"], "Daily Video Count per Device", "Count"
        )
        hourly_fig = self.graph_creator.create_figure(
            hourly_data["activity"],
            "Hourly Video Count per Device",
            "Count",
            {"is_hourly": True},
        )
        cumulative_fig = self.graph_creator.create_figure(
            daily_data["activity"].set_index("Date").cumsum().reset_index(),
            "Cumulative Daily Video Count per Device",
            "Cumulative Count",
            {"is_cumulative": True},
        )

        return [daily_fig, hourly_fig, cumulative_fig]

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
            <h2 style="text-align: center;">Video Analytics Dashboard</h2>
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

    # Load video database
    root_database = (
        "/Users/jeanyves.bouguet/Documents/EufySecurityVideos/EufyVideos/record/"
    )
    metadata_files = [
        os.path.join(root_database, "videos_in_batches.csv"),
        os.path.join(root_database, "videos_in_backup.csv"),
        # Add more metadata files as needed
    ]
    video_database = VideoDatabaseList(
        [
            VideoDatabase(video_directories=None, video_metadata_file=file)
            for file in metadata_files
        ]
    ).load_videos()

    min_date = video_database[0].date_str
    max_date = video_database[-1].date_str

    # Filter the database
    videos = VideoFilter.by_selectors(
        video_database,
        [
            VideoSelector(
                date_range=DateRange(start=min_date, end=max_date),
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

    # Save graph to file
    out_dir: str = "/Users/jeanyves.bouguet/Documents/EufySecurityVideos/stories"
    dashboard.save_graphs_to_html(graphs, os.path.join(out_dir, "video_analytics.html"))

    sys.exit()
