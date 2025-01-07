#!/usr/bin/env python3

"""
Interactive dashboard module using Dash to visualize video metadata statistics.

This module extends dashboard.py by providing an interactive web interface for:
- Date range selection via calendar
- Time range selection
- Device filtering
- Weekday selection

The graphs update dynamically based on user selections.
"""

from typing import List

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html
from dash.dependencies import Input, Output

from config import Config
from dashboard_config import DashboardConfig
from logging_config import create_logger
from video_data_aggregator import VideoDataAggregator
from video_filter import DateRange, TimeRange, VideoFilter, VideoSelector
from video_graph_creator import VideoGraphCreator
from video_metadata import VideoMetadata

logger = create_logger(__name__)


class InteractiveDashboard:
    """
    Interactive dashboard for video metadata visualization using Dash.

    Provides UI components for:
    - Date range selection
    - Time range selection
    - Device filtering
    - Weekday selection

    Updates graphs dynamically based on user selections.
    """

    def __init__(self, videos: List[VideoMetadata]):
        """Initialize dashboard with video data and create Dash app."""
        self.videos = videos

        # Get date range from videos
        dates = [v.date for v in videos]
        self.min_date = min(dates).strftime("%Y-%m-%d")
        self.max_date = max(dates).strftime("%Y-%m-%d")

        # Get available devices
        self.all_devices = Config.get_all_devices()

        # All weekdays
        self.all_weekdays = [
            "monday",
            "tuesday",
            "wednesday",
            "thursday",
            "friday",
            "saturday",
            "sunday",
        ]

        # Create Dash app
        self.app = dash.Dash(
            __name__,
            external_stylesheets=[dbc.themes.BOOTSTRAP],
            suppress_callback_exceptions=True,
        )
        self.setup_layout()
        self.setup_callbacks()

    def setup_layout(self):
        """Create the dashboard layout using shared configuration."""
        styles = DashboardConfig.get_dash_styles()

        """Create the dashboard layout with all UI components."""
        controls = dbc.Card(
            dbc.CardBody(
                [
                    # First row with date range, time bins, metric, devices, and weekdays
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.Label(
                                        "Date Range:",
                                        **styles["controls_labels"],
                                    ),
                                    dcc.DatePickerRange(
                                        id="date-range",
                                        min_date_allowed=self.min_date,
                                        max_date_allowed=self.max_date,
                                        start_date=self.min_date,
                                        end_date=self.max_date,
                                        display_format="YYYY-MM-DD",
                                        day_size=30,
                                        calendar_orientation="vertical",
                                    ),
                                ],
                                width=3,
                            ),
                            dbc.Col(
                                [
                                    html.Label(
                                        "Devices:",
                                        **styles["controls_labels"],
                                    ),
                                    dcc.Dropdown(
                                        id="device-selector",
                                        options=[
                                            {"label": d, "value": d}
                                            for d in self.all_devices
                                        ],
                                        value=self.all_devices.copy(),
                                        multi=True,
                                        **styles["controls_items"],
                                    ),
                                ],
                                width=3,
                            ),
                            dbc.Col(
                                [
                                    html.Label(
                                        "Week Days:",
                                        **styles["controls_labels"],
                                    ),
                                    dcc.Dropdown(
                                        id="weekday-selector",
                                        options=[
                                            {"label": d.capitalize(), "value": d}
                                            for d in self.all_weekdays
                                        ],
                                        value=self.all_weekdays.copy(),
                                        multi=True,
                                        **styles["controls_items"],
                                    ),
                                ],
                                width=3,
                            ),
                            dbc.Col(
                                [
                                    html.Label(
                                        "Time Bins:",
                                        **styles["controls_labels"],
                                    ),
                                    dcc.Dropdown(
                                        id="bin-size-selector",
                                        options=[
                                            {"label": "60 mins", "value": 1},
                                            {"label": "30 mins", "value": 2},
                                            {"label": "15 mins", "value": 4},
                                            {"label": "10 mins", "value": 6},
                                            {"label": "5 mins", "value": 12},
                                            {"label": "2 mins", "value": 30},
                                        ],
                                        value=4,
                                        **styles["controls_items"],
                                    ),
                                ],
                                width=1,
                            ),
                            dbc.Col(
                                [
                                    html.Label(
                                        "Metric:",
                                        **styles["controls_labels"],
                                    ),
                                    dcc.Dropdown(
                                        id="metric-selector",
                                        options=[
                                            {"label": "Activity", "value": "activity"},
                                            {
                                                "label": "Duration (in minutes)",
                                                "value": "duration",
                                            },
                                            {
                                                "label": "File Size (in MB)",
                                                "value": "filesize",
                                            },
                                        ],
                                        value="activity",
                                        **styles["controls_items"],
                                    ),
                                ],
                                width=2,
                            ),
                        ],
                        className="mb-3",  # Add margin bottom for spacing
                    ),
                    # Second row with start time
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.Label(
                                        "Start Time:",
                                        **styles["controls_labels"],
                                    )
                                ],
                                width=1,
                            ),
                            dbc.Col(
                                [
                                    dcc.Slider(
                                        id="start-time",
                                        min=0,
                                        max=24,
                                        step=1 / 12,  # 5-minute intervals
                                        value=0,
                                        marks={
                                            i: f"{i:02d}:00" for i in range(0, 25, 1)
                                        },
                                        updatemode="mouseup",
                                    ),
                                ],
                                width=11,
                            ),
                        ],
                        className="mb-3",  # Add margin bottom for spacing
                    ),
                    # Third row with end time
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.Label(
                                        "End Time:",
                                        **styles["controls_labels"],
                                    )
                                ],
                                width=1,
                            ),
                            dbc.Col(
                                [
                                    dcc.Slider(
                                        id="end-time",
                                        min=0,
                                        max=24,
                                        step=1 / 12,  # 5-minute intervals
                                        value=24,
                                        marks={
                                            i: f"{i:02d}:00" for i in range(0, 25, 1)
                                        },
                                        updatemode="mouseup",
                                    ),
                                ],
                                width=11,
                            ),
                        ],
                    ),
                ]
            ),
            **styles["controls_card"],
        )

        self.app.layout = dbc.Container(
            [
                controls,  # Add the card containing all controls
                html.H2("Video Analytics Dashboard", **styles["title"]),
                html.Div(
                    [
                        dcc.Graph(
                            id="daily-count-graph",
                            config={"displayModeBar": True},
                            **styles["graph"],
                        ),
                        dcc.Graph(
                            id="hourly-count-graph",
                            config={"displayModeBar": True},
                            **styles["graph"],
                        ),
                        dcc.Graph(
                            id="cumulative-count-graph",
                            config={"displayModeBar": True},
                            **styles["graph"],
                        ),
                    ],
                    **styles["graph"],
                ),
            ],
            fluid=True,
            **styles["container"],
        )

    def setup_callbacks(self):
        """Setup all dashboard callbacks for interactivity."""

        # Graphs update callback
        @self.app.callback(
            [
                Output("daily-count-graph", "figure"),
                Output("hourly-count-graph", "figure"),
                Output("cumulative-count-graph", "figure"),
            ],
            [
                Input("date-range", "start_date"),
                Input("date-range", "end_date"),
                Input("start-time", "value"),
                Input("end-time", "value"),
                Input("device-selector", "value"),
                Input("weekday-selector", "value"),
                Input("bin-size-selector", "value"),
                Input("metric-selector", "value"),
            ],
        )
        def update_graphs(
            start_date,
            end_date,
            start_time,
            end_time,
            selected_devices,
            weekdays,
            bins_per_hour,
            metric_to_graph,
        ):
            # Ensure we have valid inputs
            if not start_date or not end_date or start_time is None or end_time is None:
                logger.debug("Missing required date/time inputs")
                return dash.no_update, dash.no_update, dash.no_update

            # Handle device selection
            if selected_devices is None or len(selected_devices) == 0:
                logger.debug("No devices selected, using all devices")
                selected_devices = self.all_devices.copy()

            # Handle weekday selection
            if weekdays is None or len(weekdays) == 0:
                logger.debug("No weekdays selected, using all weekdays")
                weekdays = self.all_weekdays.copy()

            start_time_str = self.format_time(start_time)
            end_time_str = self.format_time(end_time)

            logger.debug("Filters:")
            logger.debug(f"  Date range: {start_date} to {end_date}")
            logger.debug(f"  Devices: {selected_devices}")
            logger.debug(f"  Weekdays: {weekdays}")
            logger.debug(f"  Time range: {start_time_str} to {end_time_str}")

            selector = VideoSelector(
                devices=selected_devices,
                date_range=DateRange(start=start_date, end=end_date),
                time_range=TimeRange(start=start_time_str, end=end_time_str),
                weekdays=weekdays,
            )
            filtered_videos = VideoFilter.by_selectors(self.videos, selector)

            # Get aggregated data
            data_aggregator = VideoDataAggregator(
                metrics=[metric_to_graph], config={"bins_per_hour": bins_per_hour}
            )

            daily_data = data_aggregator.get_daily_aggregates(filtered_videos)
            hourly_data = data_aggregator.get_hourly_aggregates(filtered_videos)

            # Create figures
            figs = VideoGraphCreator.create_graphs(
                daily_data, hourly_data, metric_to_graph, bins_per_hour
            )

            return figs[0], figs[1], figs[2]

    def format_time(self, t):
        """Format time value from slider, rounding to nearest 5 minutes"""
        if t == 24:
            return "23:59:59"

        hours = int(t)
        minutes = int((t % 1) * 60)
        # Round to nearest 5 minutes
        minutes = round(minutes / 5) * 5
        if minutes == 60:
            hours += 1
            minutes = 0
        return f"{hours:02d}:{minutes:02d}:00"

    def run(self, debug=False, port=8050):
        """Run the dashboard server."""
        self.app.run_server(debug=debug, port=port)


if __name__ == "__main__":
    import logging
    import os

    from logging_config import set_logger_level_and_format
    from video_database import VideoDatabase, VideoDatabaseList

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

    # Create and run dashboard
    dashboard = InteractiveDashboard(video_database)
    dashboard.run(debug=True)
