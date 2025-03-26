#!/usr/bin/env python3

"""
Test script for the occupancy status filtering functionality.

This script creates a test story with an occupancy status filter and runs it
to demonstrate the new functionality.
"""

import os
import sys

from logging_config import create_logger, set_logger_level_and_format
from story_creator import Story
from video_database import VideoDatabase, VideoDatabaseList

logger = create_logger(__name__)


def main():
    # Set up logging
    import logging

    set_logger_level_and_format(logger, level=logging.INFO, extended_format=True)

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

    if video_database is None:
        logger.error("Failed to load video database")
        sys.exit(1)

    # Create a test story with occupancy status filter
    test_story = Story(
        name="Occupancy_Filter_Test",
        occupancy_status=[
            "OCCUPIED",
            "UNKNOWN",
        ],  # Only include OCCUPIED and UNKNOWN days
    )

    # Process the story
    logger.info("Processing story with occupancy status filter...")
    test_story.process(video_database, out_dir)

    logger.info("Test completed successfully")


if __name__ == "__main__":
    main()
