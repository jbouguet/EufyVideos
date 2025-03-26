#!/usr/bin/env python3

"""
Security module for determining property occupancy status based on video activity data.

This module provides functionality to analyze video activity data and determine
whether a property is occupied, not occupied, or has an unknown occupancy status
for specific dates.

Key components:
1. OccupancyStatus: Enum defining possible occupancy states
2. Occupancy: Class that analyzes daily video activity to determine occupancy

Example usage:
    from video_data_aggregator import VideoDataAggregator
    from video_database import VideoDatabase, VideoDatabaseList
    from security import Occupancy, OccupancyStatus

    # Load video database
    video_database = VideoDatabaseList([...]).load_videos()

    # Get aggregated data
    data_aggregator = VideoDataAggregator(metrics=["activity"])
    daily_data, _ = data_aggregator.run(video_database)

    # Create occupancy analyzer
    occupancy = Occupancy(daily_data["activity"])

    # Check occupancy for a specific date
    status = occupancy.status("2025-01-01")
    if status == OccupancyStatus.OCCUPIED:
        print("Property was occupied on 2025-01-01")
"""

import enum
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from logging_config import create_logger
from video_data_aggregator import VideoDataAggregator

logger = create_logger(__name__)


class OccupancyStatus(enum.Enum):
    """
    Enum representing the possible occupancy statuses of a property.

    Values:
        NOT_OCCUPIED: Property is determined to be unoccupied
        OCCUPIED: Property is determined to be occupied
        UNKNOWN: Occupancy status cannot be determined with confidence
    """

    NOT_OCCUPIED = "NOT_OCCUPIED"
    OCCUPIED = "OCCUPIED"
    UNKNOWN = "UNKNOWN"


class Occupancy:
    """
    Class for determining property occupancy status based on video activity data.

    This class analyzes daily video activity data to determine whether a property
    is occupied, not occupied, or has an unknown occupancy status for specific dates.
    It uses a set of heuristics to make this determination and caches the results
    for efficient lookup.

    Attributes:
        daily_data (pd.DataFrame): Daily aggregated video activity data
        occupancy_cache (Dict[str, OccupancyStatus]): Cache of date to occupancy status
    """

    def __init__(self, daily_data: Optional[pd.DataFrame] = None):
        """
        Initialize the Occupancy class with daily aggregated video activity data.

        Args:
            daily_data: DataFrame containing daily aggregated video activity data
                        from VideoDataAggregator.run()
        """
        self.occupancy_cache: Dict[str, OccupancyStatus] = {}
        if daily_data is not None:
            self.set_occupancy_status_from_daily_acivity(daily_data)
        self.set_occupancy_status_from_calendar()

    def set_occupancy_status_from_calendar(self):
        """
        Set occupancy status cache from calendar
        """
        calendar = [
            ("2024-02-27", "2024-03-03", OccupancyStatus.OCCUPIED),
            ("2024-03-04", "2024-03-07", OccupancyStatus.NOT_OCCUPIED),
            ("2024-03-08", "2024-03-09", OccupancyStatus.OCCUPIED),
            ("2024-03-10", "2024-03-10", OccupancyStatus.NOT_OCCUPIED),
            ("2024-03-11", "2024-03-11", OccupancyStatus.OCCUPIED),
            ("2024-03-12", "2024-03-12", OccupancyStatus.NOT_OCCUPIED),
            ("2024-03-13", "2024-03-14", OccupancyStatus.OCCUPIED),
            ("2024-03-15", "2024-03-17", OccupancyStatus.NOT_OCCUPIED),
            ("2024-03-18", "2024-03-18", OccupancyStatus.OCCUPIED),
            ("2024-03-19", "2024-03-21", OccupancyStatus.NOT_OCCUPIED),
            ("2024-03-22", "2024-03-22", OccupancyStatus.OCCUPIED),
            ("2024-03-23", "2024-03-23", OccupancyStatus.NOT_OCCUPIED),
            ("2024-03-24", "2024-04-13", OccupancyStatus.OCCUPIED),
            ("2024-04-14", "2024-04-17", OccupancyStatus.NOT_OCCUPIED),
            ("2024-04-18", "2024-04-19", OccupancyStatus.OCCUPIED),
            ("2024-04-20", "2024-04-20", OccupancyStatus.NOT_OCCUPIED),
            ("2024-04-21", "2024-04-21", OccupancyStatus.OCCUPIED),
            ("2024-04-22", "2024-04-23", OccupancyStatus.NOT_OCCUPIED),
            ("2024-04-24", "2024-04-24", OccupancyStatus.OCCUPIED),
            ("2024-04-25", "2024-04-27", OccupancyStatus.NOT_OCCUPIED),
            ("2024-04-28", "2024-04-28", OccupancyStatus.OCCUPIED),
            ("2024-04-29", "2024-05-01", OccupancyStatus.NOT_OCCUPIED),
            ("2024-05-02", "2024-05-02", OccupancyStatus.OCCUPIED),
            ("2024-05-03", "2024-05-10", OccupancyStatus.NOT_OCCUPIED),
            ("2024-05-11", "2024-05-22", OccupancyStatus.OCCUPIED),
            ("2024-05-23", "2024-05-31", OccupancyStatus.NOT_OCCUPIED),
            ("2024-06-01", "2024-06-01", OccupancyStatus.OCCUPIED),
            ("2024-06-02", "2024-06-09", OccupancyStatus.NOT_OCCUPIED),
            ("2024-06-10", "2024-07-19", OccupancyStatus.OCCUPIED),
            ("2024-07-20", "2024-08-29", OccupancyStatus.NOT_OCCUPIED),
            ("2024-08-30", "2024-09-09", OccupancyStatus.OCCUPIED),
            ("2024-09-10", "2024-09-21", OccupancyStatus.NOT_OCCUPIED),
            ("2024-09-23", "2024-10-05", OccupancyStatus.OCCUPIED),
            ("2024-10-06", "2024-10-08", OccupancyStatus.NOT_OCCUPIED),
            ("2024-10-09", "2024-10-12", OccupancyStatus.OCCUPIED),
            ("2024-10-13", "2024-11-01", OccupancyStatus.NOT_OCCUPIED),
            ("2024-11-02", "2024-11-02", OccupancyStatus.OCCUPIED),
            ("2024-11-03", "2024-11-15", OccupancyStatus.NOT_OCCUPIED),
            ("2024-11-16", "2024-11-20", OccupancyStatus.OCCUPIED),
            ("2024-11-21", "2024-12-09", OccupancyStatus.NOT_OCCUPIED),
            ("2024-12-10", "2024-12-13", OccupancyStatus.OCCUPIED),
            ("2024-12-14", "2024-12-25", OccupancyStatus.NOT_OCCUPIED),
            ("2024-12-26", "2025-01-08", OccupancyStatus.OCCUPIED),
            ("2025-01-09", "2025-03-05", OccupancyStatus.NOT_OCCUPIED),
            ("2025-03-06", "2025-03-10", OccupancyStatus.OCCUPIED),
            ("2025-03-11", "2025-03-24", OccupancyStatus.NOT_OCCUPIED),
            ("2025-03-25", "2025-04-01", OccupancyStatus.OCCUPIED),
        ]

        for start_date_str, end_date_str, status in calendar:
            start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
            end_date = datetime.strptime(end_date_str, "%Y-%m-%d")

            current_date = start_date
            while current_date <= end_date:
                date_str = current_date.strftime("%Y-%m-%d")
                self.occupancy_cache[date_str] = status
                current_date = datetime(
                    current_date.year, current_date.month, current_date.day
                ) + pd.Timedelta(days=1)

    def set_occupancy_status_from_daily_acivity(self, daily_data: pd.DataFrame):
        """
        Analyze daily video activity data to determine occupancy status for each date.

        This method implements heuristics to determine occupancy status based on
        video activity metrics. The current implementation uses a naive approach:
        - OCCUPIED if Backyard activity >= 50
        - NOT_OCCUPIED if Backyard activity == 0
        - UNKNOWN otherwise

        The results are stored in the occupancy_cache dictionary for efficient lookup.
        """
        for _, row in daily_data.iterrows():
            date_str = row["Date"].strftime("%Y-%m-%d")
            self.occupancy_cache[date_str] = OccupancyStatus.UNKNOWN

            # Check if 'Backyard' column exists in the data
            if "Backyard" in row:
                backyard_activity = row["Backyard"]

                # Apply naive heuristic
                if backyard_activity >= 50:
                    self.occupancy_cache[date_str] = OccupancyStatus.OCCUPIED
                elif backyard_activity == 0:
                    self.occupancy_cache[date_str] = OccupancyStatus.NOT_OCCUPIED

            # Check if 'Front Door' column exists in the data
            if "Front Door" in row:
                front_door_activity = row["Front Door"]

                # Apply naive heuristic
                if front_door_activity >= 5:
                    self.occupancy_cache[date_str] = OccupancyStatus.OCCUPIED

    def status(self, date_str: str) -> OccupancyStatus:
        """
        Get the occupancy status for a specific date.

        Args:
            date_str: Date string in 'YYYY-MM-DD' format

        Returns:
            OccupancyStatus enum value representing the occupancy status
        """
        # Validate date format
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            logger.error(
                f"Invalid date format: {date_str}. Expected format: YYYY-MM-DD"
            )
            return OccupancyStatus.UNKNOWN

        # Return cached status if available, otherwise UNKNOWN
        return self.occupancy_cache.get(date_str, OccupancyStatus.UNKNOWN)

    def get_all_dates_with_status(self) -> List[Dict[str, str]]:
        """
        Get a list of all dates with their corresponding occupancy status.

        Returns:
            List of dictionaries with 'date' and 'status' keys
        """
        return [
            {"date": date_str, "status": status.value}
            for date_str, status in sorted(self.occupancy_cache.items())
        ]


if __name__ == "__main__":
    # Testing code for the module
    import csv
    import logging
    import os
    import sys

    from config import Config
    from logging_config import set_logger_level_and_format
    from video_database import VideoDatabase, VideoDatabaseList
    from video_filter import DateRange, TimeRange, VideoFilter, VideoSelector

    set_logger_level_and_format(logger, level=logging.DEBUG, extended_format=True)

    # Load video database
    root_database = "/Users/jbouguet/Documents/EufySecurityVideos/record/"
    metadata_files = [
        os.path.join(root_database, "videos_in_batches.csv"),
        os.path.join(root_database, "videos_in_backup.csv"),
    ]
    out_dir: str = "/Users/jbouguet/Documents/EufySecurityVideos/stories"

    logger.info("Loading video database...")
    video_database = VideoDatabaseList(
        [
            VideoDatabase(video_directories=None, video_metadata_file=file)
            for file in metadata_files
        ]
    ).load_videos()

    # Define date range, time range, devices, and weekdays for filtering
    start_date = "2024-02-27"
    end_date = "2025-03-10"
    start_time = "00:00:00"
    end_time = "23:59:59"
    devices = Config.get_all_devices()
    weekdays = [
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
    ]

    logger.debug(f"Date range: {start_date} to {end_date}")
    logger.debug(f"Time range: {start_time} to {end_time}")
    logger.debug(f"Devices: {devices}")
    logger.debug(f"Weekdays: {weekdays}")

    # Create selector and filter videos
    selector = VideoSelector(
        date_range=DateRange(start=start_date, end=end_date),
        time_range=TimeRange(start=start_time, end=end_time),
        devices=devices,
        weekdays=weekdays,
    )

    if video_database is None:
        logger.error("Failed to load video database")
        sys.exit(1)

    logger.info("Filtering videos...")
    videos = VideoFilter.by_selectors(video_database, selector)
    logger.debug(f"Number of videos: {len(videos)}")

    # Get aggregated data
    logger.info("Aggregating video data...")
    data_aggregator = VideoDataAggregator(metrics=["activity"])
    daily_data, _ = data_aggregator.run(videos)

    # Create occupancy analyzer
    logger.info("Analyzing occupancy...")
    # occupancy = Occupancy(daily_data["activity"])
    occupancy = Occupancy()

    # Get all dates with occupancy status
    all_occupancy_data = occupancy.get_all_dates_with_status()

    # Save to CSV file
    output_file = os.path.join(out_dir, "daily_occupancies.csv")
    logger.info(f"Saving occupancy data to {output_file}")

    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        # Add a space after the comma to make it more readable
        writer.writerow(["date", "occupancy_status"])
        for item in all_occupancy_data:
            writer.writerow([item["date"], item["status"]])

    logger.info(
        f"Successfully saved occupancy data for {len(all_occupancy_data)} dates"
    )

    # Print the content of the CSV file for verification
    logger.info(f"Content of {output_file}:")
    with open(output_file, "r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            logger.info(f"Row: {row}")

    # Print sample of the results
    logger.info("Sample of occupancy results:")
    for item in all_occupancy_data[:5]:
        logger.info(f"Date: {item['date']}, Status: {item['status']}")
