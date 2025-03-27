#!/usr/bin/env python3

"""
Test script for the machine learning functionality in the Occupancy class.

This script demonstrates how to:
1. Load video data
2. Create an Occupancy instance
3. Train a model using daily activity data
4. Save the model to a file
5. Load the model from a file
6. Apply the model to predict occupancy status
7. Compare the results with the original occupancy status from the calendar
"""

import csv
import logging
import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import plot_tree

from config import Config
from logging_config import create_logger, set_logger_level_and_format
from occupancy import Occupancy, OccupancyStatus
from video_data_aggregator import VideoDataAggregator
from video_database import VideoDatabase, VideoDatabaseList
from video_filter import DateRange, TimeRange, VideoFilter, VideoSelector

# Set up logging
logger = create_logger(__name__)
set_logger_level_and_format(logger, level=logging.DEBUG, extended_format=True)


def load_video_data():
    """
    Load and filter video data from the video database.

    Returns:
        Dictionary containing daily activity data
    """
    # Load video database
    root_database = "/Users/jbouguet/Documents/EufySecurityVideos/record/"
    metadata_files = [
        os.path.join(root_database, "videos_in_batches.csv"),
        os.path.join(root_database, "videos_in_backup.csv"),
    ]

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

    # Get aggregated data
    logger.info("Aggregating video data...")
    data_aggregator = VideoDataAggregator(metrics=["activity"])
    daily_data, _ = data_aggregator.run(video_database)

    return daily_data


def compare_occupancy_methods(calendar_occupancy, heuristic_occupancy, ml_occupancy):
    """
    Compare the occupancy status determined by different methods.

    Args:
        calendar_occupancy: Occupancy status from the calendar
        heuristic_occupancy: Occupancy status from the heuristic method
        ml_occupancy: Occupancy status from the machine learning model

    Returns:
        DataFrame with comparison results
    """
    # Get all unique dates across all methods
    all_dates = set()
    for occupancy in [calendar_occupancy, heuristic_occupancy, ml_occupancy]:
        all_dates.update(occupancy.keys())

    # Create comparison data
    comparison_data = []
    for date_str in sorted(all_dates):
        comparison_data.append(
            {
                "date": date_str,
                "calendar": calendar_occupancy.get(
                    date_str, OccupancyStatus.UNKNOWN
                ).value,
                "heuristic": heuristic_occupancy.get(
                    date_str, OccupancyStatus.UNKNOWN
                ).value,
                "ml_model": ml_occupancy.get(date_str, OccupancyStatus.UNKNOWN).value,
            }
        )

    return pd.DataFrame(comparison_data)


def calculate_accuracy(comparison_df):
    """
    Calculate accuracy metrics for the heuristic and ML methods compared to the calendar.

    Args:
        comparison_df: DataFrame with comparison results

    Returns:
        Dictionary with accuracy metrics
    """
    # Filter out rows where calendar status is UNKNOWN
    filtered_df = comparison_df[comparison_df["calendar"] != "UNKNOWN"].copy()

    # Calculate accuracy for heuristic method
    heuristic_correct = (filtered_df["calendar"] == filtered_df["heuristic"]).sum()
    heuristic_accuracy = (
        heuristic_correct / len(filtered_df) if len(filtered_df) > 0 else 0
    )

    # Calculate accuracy for ML method
    ml_correct = (filtered_df["calendar"] == filtered_df["ml_model"]).sum()
    ml_accuracy = ml_correct / len(filtered_df) if len(filtered_df) > 0 else 0

    return {
        "heuristic_accuracy": heuristic_accuracy,
        "ml_accuracy": ml_accuracy,
        "total_samples": len(filtered_df),
        "heuristic_correct": heuristic_correct,
        "ml_correct": ml_correct,
    }


def main():

    # Load video data
    daily_data = load_video_data()

    # 1. Create an Occupancy instance with calendar data only
    logger.info("Creating calendar-based occupancy...")
    calendar_occupancy = Occupancy()
    calendar_status = calendar_occupancy.occupancy_cache.copy()

    # 2. Create an Occupancy instance with heuristic method
    logger.info("Creating heuristic-based occupancy...")
    heuristic_occupancy = Occupancy()
    heuristic_occupancy.set_occupancy_status_from_daily_activity(daily_data["activity"])
    heuristic_status = heuristic_occupancy.occupancy_cache.copy()

    # 3. Train a machine learning model
    logger.info("Training machine learning model...")
    ml_occupancy = Occupancy()
    try:
        # Train the model using daily activity data
        model_metrics = ml_occupancy.train_occupancy_model(daily_data["activity"])

        # Print model evaluation metrics
        logger.info(f"Model accuracy: {model_metrics['accuracy']:.2f}")
        logger.info("Feature importances:")
        for feature, importance in sorted(
            model_metrics["feature_importances"].items(),
            key=lambda x: x[1],
            reverse=True,
        ):
            logger.info(f"  {feature}: {importance:.4f}")

        # Visualize the decision tree
        if ml_occupancy.model is not None:
            logger.info("Creating decision tree visualization...")
            plt.figure(figsize=(20, 10))
            plot_tree(
                ml_occupancy.model,
                feature_names=ml_occupancy.feature_names,
                class_names=["NOT_OCCUPIED", "OCCUPIED"],
                filled=True,
                rounded=True,
                fontsize=10,
            )
            plt.title("Occupancy Decision Tree", fontsize=14)

            # Save the visualization
            tree_viz_file = "occupancy_decision_tree.png"
            plt.savefig(tree_viz_file, dpi=300, bbox_inches="tight")
            logger.info(f"Decision tree visualization saved to {tree_viz_file}")
            plt.close()
        else:
            logger.error("Cannot visualize decision tree: model is None")

        # 4. Save the model to a file
        model_file = "occupancy_model.txt"
        logger.info(f"Saving model to {model_file}...")
        ml_occupancy.save_occupancy_model(model_file)

        # 5. Load the model from the file
        logger.info(f"Loading model from {model_file}...")
        new_occupancy = Occupancy()
        new_occupancy.load_occupancy_model(model_file)

        # 6. Apply the model to predict occupancy status
        logger.info("Applying model to predict occupancy status...")
        ml_occupancy.set_occupancy_status_from_model(daily_data["activity"])
        ml_status = ml_occupancy.occupancy_cache.copy()

        # 7. Compare the results
        logger.info("Comparing occupancy methods...")
        comparison_df = compare_occupancy_methods(
            calendar_status, heuristic_status, ml_status
        )

        # Save comparison results to CSV
        comparison_file = "occupancy_comparison.csv"
        logger.info(f"Saving comparison results to {comparison_file}...")
        comparison_df.to_csv(comparison_file, index=False)

        # Calculate and print accuracy metrics
        accuracy_metrics = calculate_accuracy(comparison_df)
        logger.info("Accuracy metrics:")
        logger.info(
            f"  Heuristic method: {accuracy_metrics['heuristic_accuracy']:.2f} "
            f"({accuracy_metrics['heuristic_correct']}/{accuracy_metrics['total_samples']})"
        )
        logger.info(
            f"  ML model: {accuracy_metrics['ml_accuracy']:.2f} "
            f"({accuracy_metrics['ml_correct']}/{accuracy_metrics['total_samples']})"
        )

        # Print sample of comparison results
        logger.info("Sample of comparison results:")
        print(comparison_df.head(10))

    except Exception as e:
        logger.error(f"Error in machine learning process: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
