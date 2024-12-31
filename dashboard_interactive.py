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
from dash import dcc, html
from dash.dependencies import Input, Output, State

from config import Config
from dashboard import VideoDataAggregator, VideoGraphCreator
from logging_config import create_logger
from video_filter import DateRange, TimeRange, VideoFilter, VideoSelector
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
        self.data_aggregator = VideoDataAggregator()
        self.graph_creator = VideoGraphCreator()

        # Get date range from videos
        dates = [v.date for v in videos]
        self.min_date = min(dates).strftime("%Y-%m-%d")
        self.max_date = max(dates).strftime("%Y-%m-%d")

        # Get available devices
        self.devices = Config.get_all_devices()

        # Create Dash app
        self.app = dash.Dash(__name__, suppress_callback_exceptions=True)
        self.setup_layout()
        self.setup_callbacks()

    def setup_layout(self):
        """Create the dashboard layout with all UI components."""
        self.app.layout = html.Div(
            [
                html.H1(
                    "Video Analytics Dashboard",
                    style={"textAlign": "center", "marginBottom": "0px"},
                ),
                # Filters section
                html.Div(
                    [
                        # Date Range Selection
                        html.Div(
                            [
                                html.H3("Date Range"),
                                dcc.DatePickerRange(
                                    id="date-range",
                                    min_date_allowed=self.min_date,
                                    max_date_allowed=self.max_date,
                                    start_date=self.min_date,
                                    end_date=self.max_date,
                                    display_format="YYYY-MM-DD",
                                    day_size=30,
                                ),
                            ],
                            style={"marginBottom": "0px"},
                        ),
                        # Device Selection
                        html.Div(
                            [
                                html.H3("Devices"),
                                dcc.Checklist(
                                    id="device-selector",
                                    options=[
                                        {"label": device, "value": device}
                                        for device in self.devices
                                    ],
                                    value=self.devices.copy(),
                                    inline=True,
                                ),
                            ],
                            style={"marginBottom": "0px"},
                        ),
                        # Weekday Selection
                        html.Div(
                            [
                                html.H3("Weekdays"),
                                dcc.Checklist(
                                    id="weekday-selector",
                                    options=[
                                        {"label": day, "value": day.lower()}
                                        for day in [
                                            "Monday",
                                            "Tuesday",
                                            "Wednesday",
                                            "Thursday",
                                            "Friday",
                                            "Saturday",
                                            "Sunday",
                                        ]
                                    ],
                                    value=[
                                        "monday",
                                        "tuesday",
                                        "wednesday",
                                        "thursday",
                                        "friday",
                                        "saturday",
                                        "sunday",
                                    ],
                                    inline=True,
                                ),
                            ],
                            style={"marginBottom": "0px"},
                        ),
                        # Time Range Selection
                        html.Div(
                            [
                                html.H3("Time Range"),
                                # Start Time Slider
                                html.Div(
                                    [
                                        html.Label("Start Time:"),
                                        dcc.Slider(
                                            id="start-time",
                                            min=0,
                                            max=24,
                                            step=1 / 12,  # 5-minute intervals
                                            value=0,
                                            marks={
                                                i: f"{i:02d}:00"
                                                for i in range(0, 25, 1)
                                            },
                                            updatemode="drag",
                                        ),
                                        html.Div(id="start-time-display"),
                                    ],
                                    style={"marginBottom": "0px"},
                                ),
                                # End Time Slider
                                html.Div(
                                    [
                                        html.Label("End Time:"),
                                        dcc.Slider(
                                            id="end-time",
                                            min=0,
                                            max=24,
                                            step=1 / 12,  # 5-minute intervals
                                            value=24,
                                            marks={
                                                i: f"{i:02d}:00"
                                                for i in range(0, 25, 1)
                                            },
                                            updatemode="mouseup",
                                        ),
                                        html.Div(id="end-time-display"),
                                    ],
                                    style={"marginBottom": "0px"},
                                ),
                                # Apply Time Range Button
                                html.Button(
                                    "Apply Time Range",
                                    id="apply-time-range",
                                    n_clicks=0,
                                    style={
                                        "backgroundColor": "#007bff",
                                        "color": "white",
                                        "border": "none",
                                        "padding": "10px 20px",
                                        "borderRadius": "5px",
                                        "cursor": "pointer",
                                        "marginTop": "10px",
                                    },
                                ),
                            ],
                            style={"marginBottom": "0px"},
                        ),
                    ],
                    style={
                        "padding": "0px",
                        "backgroundColor": "#ffffff",
                        "borderRadius": "0px",
                        "marginBottom": "0px",
                    },
                ),
                # Graphs section
                html.Div(
                    [
                        dcc.Graph(id="daily-count-graph"),
                        dcc.Graph(id="hourly-count-graph"),
                        dcc.Graph(id="cumulative-count-graph"),
                    ]
                ),
            ],
            style={"padding": "10px"},
        )

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

    def format_display_time(self, t):
        """Format time for display"""
        if t == 24:
            return "23:59"

        hours = int(t)
        minutes = int((t % 1) * 60)
        # Round to nearest 5 minutes
        minutes = round(minutes / 5) * 5
        if minutes == 60:
            hours += 1
            minutes = 0
        return f"{hours:02d}:{minutes:02d}"

    def setup_callbacks(self):
        """Setup all dashboard callbacks for interactivity."""

        # Time display callbacks
        @self.app.callback(
            Output("start-time-display", "children"), Input("start-time", "value")
        )
        def update_start_time_display(value):
            # time = self.format_display_time(value)
            # return f"Start: {time}"
            return ""

        @self.app.callback(
            Output("end-time-display", "children"), Input("end-time", "value")
        )
        def update_end_time_display(value):
            # time = self.format_display_time(value)
            # return f"End: {time}"
            return ""

        # Store current time values
        @self.app.callback(
            Output("start-time", "value"),
            Output("end-time", "value"),
            Input("apply-time-range", "n_clicks"),
            State("start-time", "value"),
            State("end-time", "value"),
            prevent_initial_call=True,
        )
        def store_time_values(n_clicks, start_time, end_time):
            return start_time, end_time

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
                Input("apply-time-range", "n_clicks"),
                Input("device-selector", "value"),
                Input("weekday-selector", "value"),
            ],
            [State("start-time", "value"), State("end-time", "value")],
        )
        def update_graphs(
            start_date,
            end_date,
            n_clicks,
            selected_devices,
            weekdays,
            start_time,
            end_time,
        ):
            logger.debug("Callback triggered with:")
            logger.debug(f"  - Date range: {start_date} to {end_date}")
            logger.debug(f"  - Start time: {start_time}")
            logger.debug(f"  - End time: {end_time}")
            logger.debug(f"  - Selected devices: {selected_devices}")
            logger.debug(f"  - Weekdays: {weekdays}")

            # Ensure we have valid inputs
            if not start_date or not end_date or start_time is None or end_time is None:
                logger.debug("Missing required date/time inputs")
                return dash.no_update, dash.no_update, dash.no_update

            # Handle device selection
            if selected_devices is None or len(selected_devices) == 0:
                logger.debug("No devices selected, using all devices")
                selected_devices = self.devices.copy()
            else:
                logger.debug(f"Using selected devices: {selected_devices}")

            # Handle weekday selection
            all_weekdays = [
                "monday",
                "tuesday",
                "wednesday",
                "thursday",
                "friday",
                "saturday",
                "sunday",
            ]
            if weekdays is None or len(weekdays) == 0:
                logger.debug("No weekdays selected, using all weekdays")
                weekdays = all_weekdays.copy()
            else:
                logger.debug(f"Using selected weekdays: {weekdays}")

            logger.debug("Using filters:")
            logger.debug(f"  Devices: {selected_devices}")
            logger.debug(f"  Weekdays: {weekdays}")

            start_time_str = self.format_time(start_time)
            end_time_str = self.format_time(end_time)

            logger.debug(f"Time range: {start_time_str} to {end_time_str}")
            selector = VideoSelector(
                devices=selected_devices,
                date_range=DateRange(start=start_date, end=end_date),
                time_range=TimeRange(start=start_time_str, end=end_time_str),
                weekdays=weekdays,
            )
            filtered_videos = VideoFilter.by_selectors(self.videos, selector)

            # Get aggregated data
            daily_data = self.data_aggregator.get_daily_aggregates(filtered_videos)
            hourly_data = self.data_aggregator.get_hourly_aggregates(filtered_videos)

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

            return daily_fig, hourly_fig, cumulative_fig

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
