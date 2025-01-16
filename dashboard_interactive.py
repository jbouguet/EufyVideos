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

from datetime import datetime
from typing import List

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, dcc, html

from config import Config
from dashboard_config import DashboardConfig
from logging_config import create_logger
from story_creator import Story
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

    def __init__(self, videos: List[VideoMetadata], stories_output: str = ""):
        """Initialize dashboard with video data and the output directory and create Dash app."""
        self.videos = videos
        self.stories_output = stories_output

        # Add predefined directory options
        self.directory_options = [
            {"label": self.stories_output, "value": self.stories_output},
            {
                "label": os.path.join(self.stories_output, "dashboards"),
                "value": os.path.join(self.stories_output, "dashboards"),
            },
            {
                "label": os.path.join(self.stories_output, "daily"),
                "value": os.path.join(self.stories_output, "daily"),
            },
            {
                "label": os.path.join(self.stories_output, "weekly"),
                "value": os.path.join(self.stories_output, "weekly"),
            },
            {
                "label": os.path.join(self.stories_output, "monthly"),
                "value": os.path.join(self.stories_output, "monthly"),
            },
        ]

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

        self.num_videos = len(self.videos)
        self.num_days = (self.videos[-1].date - self.videos[0].date).days + 1

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
        main_controls = dbc.Card(
            dbc.CardBody(
                [
                    # First row with date range, devices, weekdays, time bins and metric.
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
                                        show_outside_days=True,  # Show days from adjacent months
                                    ),
                                ],
                                width=2,
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
                                        optionHeight=16,
                                        **styles["controls_items"],
                                    ),
                                ],
                                width=4,
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
                                        optionHeight=16,  # Control height of each option
                                        **styles["controls_items"],
                                    ),
                                ],
                                width=4,
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
                                            {"label": "60 minutes", "value": 1},
                                            {"label": "30 minutes", "value": 2},
                                            {"label": "15 minutes", "value": 4},
                                            {"label": "10 minutes", "value": 6},
                                            {"label": "5 minutes", "value": 12},
                                            {"label": "2 minutes", "value": 30},
                                        ],
                                        value=4,
                                        optionHeight=16,
                                        clearable=False,
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
                                        optionHeight=16,
                                        clearable=False,
                                        **styles["controls_items"],
                                    ),
                                ],
                                width=1,
                            ),
                        ],
                    ),
                    # Second row with start time.
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.Div(
                                        [
                                            html.Span(
                                                "Start Time: ",
                                                **styles["controls_labels"],
                                            ),
                                            html.Span(
                                                id="start-time-display",
                                                **styles["controls_text"],
                                            ),
                                        ]
                                    ),
                                ],
                                width=2,
                            ),
                            dbc.Col(
                                [
                                    dcc.Slider(
                                        id="start-time",
                                        min=0,
                                        max=24,
                                        step=1 / 60,  # 1-minute intervals
                                        value=0,
                                        marks={
                                            i: f"{i:02d}:00" for i in range(0, 25, 1)
                                        },
                                        updatemode="mouseup",
                                        included=True,
                                    ),
                                ],
                                width=10,
                            ),
                        ],
                    ),
                    # Third row with end time.
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.Div(
                                        [
                                            html.Span(
                                                "End Time: ",
                                                **styles["controls_labels"],
                                            ),
                                            html.Span(
                                                id="end-time-display",
                                                **styles["controls_text"],
                                            ),
                                        ]
                                    ),
                                ],
                                width=2,
                            ),
                            dbc.Col(
                                [
                                    dcc.Slider(
                                        id="end-time",
                                        min=0,
                                        max=24,
                                        step=1 / 60,  # 1-minute intervals
                                        value=24,
                                        marks={
                                            i: f"{i:02d}:00" for i in range(0, 25, 1)
                                        },
                                        updatemode="mouseup",
                                        included=True,
                                    ),
                                ],
                                width=10,
                            ),
                        ],
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.Label(
                                        "Story:",
                                        **styles["controls_labels"],
                                    ),
                                ],
                                width=1,
                            ),
                            dbc.Col(
                                [
                                    dcc.Dropdown(
                                        id="story-dir-input",
                                        options=self.directory_options,
                                        value=self.stories_output,
                                        clearable=False,
                                        optionHeight=16,
                                        **styles["controls_items"],
                                    ),
                                ],
                                width=4,
                            ),
                            dbc.Col(
                                [
                                    dbc.Input(
                                        id="story-name-input",
                                        type="text",
                                        placeholder="Story name",
                                        style=styles["controls_items"]["style"],
                                    ),
                                ],
                                width=2,
                            ),
                            dbc.Col(
                                [
                                    dbc.Button(
                                        "Save",
                                        id="save-button",
                                        color="primary",
                                        disabled=True,
                                        style={
                                            **styles["controls_items"]["style"],
                                            **styles["controls_spacing"]["style"],
                                        },
                                    ),
                                    dbc.Button(
                                        "Load",
                                        id="load-button",
                                        color="primary",
                                        disabled=True,
                                        style={
                                            **styles["controls_items"]["style"],
                                            **styles["controls_spacing"]["style"],
                                        },
                                    ),
                                ],
                                width=2,
                            ),
                            dbc.Col(
                                [
                                    html.Div(
                                        [
                                            html.Span(
                                                "Videos: ",
                                                **styles["controls_labels"],
                                            ),
                                            html.Span(
                                                id="num-videos-display",
                                                **styles["controls_text"],
                                            ),
                                        ],
                                        style={"marginBottom": "0px"},
                                    ),
                                    html.Div(
                                        [
                                            html.Span(
                                                "Days: ",
                                                **styles["controls_labels"],
                                            ),
                                            html.Span(
                                                id="num-days-display",
                                                **styles["controls_text"],
                                            ),
                                        ],
                                    ),
                                ],
                                width=2,
                                className="d-flex flex-column align-items-start",
                            ),
                        ],
                    ),
                ]
            ),
            **styles["controls_card"],
        )

        self.app.layout = dbc.Container(
            [
                dcc.Store(id="num-videos-store", data=self.num_videos),
                dcc.Store(id="num-days-store", data=self.num_days),
                main_controls,
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

        @self.app.callback(
            Output("start-time-display", "children"), Input("start-time", "value")
        )
        def update_start_time_display(value):
            if value is None:
                return ""
            return self.slider_to_time(value)

        @self.app.callback(
            Output("end-time-display", "children"), Input("end-time", "value")
        )
        def update_end_time_display(value):
            if value is None:
                return ""
            return self.slider_to_time(value)

        # load button enable/disable callback
        @self.app.callback(
            Output("load-button", "disabled"), Input("story-name-input", "value")
        )
        def update_load_button_state(input_value):
            return not (input_value and input_value.strip())

        # First callback updates the store when load button is clicked
        @self.app.callback(
            [
                Output("date-range", "start_date"),
                Output("date-range", "end_date"),
                Output("start-time", "value"),
                Output("end-time", "value"),
                Output("device-selector", "value"),
                Output("weekday-selector", "value"),
            ],
            Input("load-button", "n_clicks"),
            State("story-name-input", "value"),
            State("story-dir-input", "value"),
            prevent_initial_call=True,
        )
        def load_story(n_clicks, story_name, story_dir):
            if n_clicks is None:
                return dash.no_update

            story_config_filename = os.path.join(
                story_dir, f"{story_name}{Config.CONFIG}"
            )
            if not os.path.exists(story_config_filename):
                return dash.no_update

            story = Story.from_file(story_config_filename)
            if not story.selectors:
                return dash.no_update

            selector = story.selectors[0]
            devices = selector.devices or Config.get_all_devices()
            date_range = selector.date_range or DateRange(
                start=self.min_date, end=self.max_date
            )
            time_range = selector.time_range or TimeRange(
                start="00:00:00", end="23:59:59"
            )
            weekdays = selector.weekdays or self.all_weekdays

            return (
                date_range.start,
                date_range.end,
                self.time_to_slider(time_range.start),
                self.time_to_slider(time_range.end),
                devices,
                weekdays,
            )

        # Second callback updates graphs based on all inputs including the trigger
        @self.app.callback(
            [
                Output("daily-count-graph", "figure"),
                Output("hourly-count-graph", "figure"),
                Output("cumulative-count-graph", "figure"),
                Output("num-videos-store", "data"),
                Output("num-days-store", "data"),
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

            start_time_str = self.slider_to_time(start_time)
            end_time_str = self.slider_to_time(end_time)

            logger.debug("----------------------------------------------------")
            logger.debug(f"Date range: {start_date} to {end_date}")
            logger.debug(f"Devices: {selected_devices}")
            logger.debug(f"Weekdays: {weekdays}")
            logger.debug(f"Time range: {start_time_str} to {end_time_str}")
            logger.debug(f"Time bin size: {60 / bins_per_hour} minutes")
            logger.debug(f"Metric: {metric_to_graph}")

            selector = VideoSelector(
                devices=selected_devices,
                date_range=DateRange(start=start_date, end=end_date),
                time_range=TimeRange(start=start_time_str, end=end_time_str),
                weekdays=weekdays,
            )
            filtered_videos = VideoFilter.by_selectors(self.videos, selector)
            self.num_videos = len(filtered_videos)
            logger.debug(f"Number of videos: {self.num_videos:,}")
            self.num_days = 0
            if self.num_videos > 0:
                self.num_days = (
                    filtered_videos[-1].date - filtered_videos[0].date
                ).days + 1
            logger.debug(f"Number of days: {self.num_days :,}")
            if self.num_days > 0:
                logger.debug(
                    f"Average number of videos per day: {self.num_videos / self.num_days :.2f}"
                )

            # Get aggregated data
            data_aggregator = VideoDataAggregator(
                metrics=[metric_to_graph], config={"bins_per_hour": bins_per_hour}
            )

            daily_data, hourly_data = data_aggregator.run(filtered_videos)

            # Create figures
            figs = VideoGraphCreator.create_graphs(
                daily_data,
                hourly_data,
                metrics=[metric_to_graph],
                bins_per_hour=bins_per_hour,
            )

            return figs[0], figs[1], figs[2], self.num_videos, self.num_days

        # Update num-videos-display from store
        @self.app.callback(
            Output("num-videos-display", "children"), Input("num-videos-store", "data")
        )
        def update_num_videos_display(num_videos):
            return f"{num_videos:,}" if num_videos is not None else "0"

        # Update num-days-display from store
        @self.app.callback(
            Output("num-days-display", "children"), Input("num-days-store", "data")
        )
        def update_num_days_display(num_days):
            return f"{num_days:,}" if num_days is not None else "0"

        # save button enable/disable callback
        @self.app.callback(
            Output("save-button", "disabled"), Input("story-name-input", "value")
        )
        def update_save_button_state(input_value):
            return not (input_value and input_value.strip())

        # save button click callback
        @self.app.callback(
            Output("story-name-input", "value"),
            Input("save-button", "n_clicks"),
            State("story-name-input", "value"),
            State("story-dir-input", "value"),
            State("date-range", "start_date"),
            State("date-range", "end_date"),
            State("start-time", "value"),
            State("end-time", "value"),
            State("device-selector", "value"),
            State("weekday-selector", "value"),
            prevent_initial_call=True,
        )
        def save_story(
            n_clicks,
            story_name,
            story_dir,
            start_date,
            end_date,
            start_time,
            end_time,
            selected_devices,
            weekdays,
        ):
            if n_clicks is None:
                return dash.no_update

            os.makedirs(story_dir, exist_ok=True)

            Story(
                name=story_name,
                selectors=[
                    VideoSelector(
                        devices=selected_devices,
                        date_range=DateRange(start=start_date, end=end_date),
                        time_range=TimeRange(
                            start=self.slider_to_time(start_time),
                            end=self.slider_to_time(end_time),
                        ),
                        weekdays=weekdays,
                    )
                ],
            ).process(videos_database=self.videos, output_directory=story_dir)

            # Clear the input after save
            return ""

    def slider_to_time(self, t):
        """Format time value from slider, rounding to nearest minute and turning 24:00:00 to 23:59:59"""
        if t == 24:
            return "23:59:59"

        hours = int(t)
        # Round to nearest integer minute
        minutes = round((t - hours) * 60)  # int((t % 1) * 60)
        if minutes == 60:
            hours += 1
            minutes = 0
        return f"{hours:02d}:{minutes:02d}:00"

    def time_to_slider(self, time_str: str):
        _time = datetime.strptime(time_str, "%H:%M:%S").time()
        hour = _time.hour
        minute = _time.minute
        second = _time.second
        # The following code successfully map 13:59:45 to 14:00:00
        # Increment minute by one if second is larger than 30, then set second to 0
        if second > 30.0:
            minute += 1
        second = 0
        # If minute get to 60, increment hour by one and reset minute to 0.
        if minute == 60:
            hour += 1
            minute = minute - 60
        # The case hour = 24, minute = 0, second = 0 is dealt with by slider_to_time.
        return hour + (minute / 60)

    def run(self, debug=False, port=8050):
        """Run the dashboard server."""
        self.app.run_server(debug=debug, port=port)


if __name__ == "__main__":
    import logging
    import os

    from logging_config import set_logger_level_and_format
    from video_database import VideoDatabase, VideoDatabaseList

    set_logger_level_and_format(logger, level=logging.DEBUG, extended_format=True)

    # Refer to the configuration file analysis_config.yaml for following directory and file settings.
    # Video database location:
    root_database = (
        "/Users/jeanyves.bouguet/Documents/EufySecurityVideos/EufyVideos/record/"
    )
    # Output directory to save Stories:
    stories_output = "/Users/jeanyves.bouguet/Documents/EufySecurityVideos/stories"

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
    dashboard = InteractiveDashboard(video_database, stories_output)
    dashboard.run(debug=True)
