#!/usr/bin/env python3

"""
Test script for the machine learning functionality in the Occupancy class.

This script demonstrates how to:
1. Load video data
2. Create an Occupancy instance
3. Train a model using daily activity data with different methods
4. Save the model to a file
5. Load the model from a file
6. Apply the model to predict occupancy status
7. Compare the results with the original occupancy status from the calendar
"""

import csv
import logging
import os
import sys

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

from logging_config import (
    create_logger,
    set_all_loggers_level_and_format,
    set_logger_level_and_format,
)
from occupancy import Occupancy, OccupancyMode, OccupancyStatus
from video_data_aggregator import VideoDataAggregator
from video_database import VideoDatabase, VideoDatabaseList

# Set up logging
logger = create_logger(__name__)
set_logger_level_and_format(logger, level=logging.DEBUG, extended_format=True)
set_all_loggers_level_and_format(level=logging.DEBUG, extended_format=True)


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
    metric = "activity"
    data_aggregator = VideoDataAggregator(metrics=[metric])
    daily_data, _ = data_aggregator.run(video_database)
    daily_activity_data = daily_data[metric]

    # Filter to keep only the important columns

    # important_columns = [
    #    "Date",
    #    "Front Door",
    #    "Walkway",
    #    "Backyard",
    #    "Back Entrance",
    # ]  # Perfect model

    # important_columns = [
    #    "Date",
    #    "Front Door",
    #    "Walkway",
    #    "Backyard",
    # ]  # 2 errors: 2024-10-12 and 2024-12-11

    # important_columns = [
    #    "Date",
    #    "Front Door",
    #    "Walkway",
    #    "Back Entrance",
    # ]  # 1 error: 2024-12-10

    important_columns = [
        "Date",
        "Front Door",
        "Walkway",
    ]  # 5 errors: 2024-12-28, 2024-12-26, 2024-12-11, 2024-10-12, 2024-03-19

    # 2024-12-11 and 2024-10-12: seem to directly depend on Back Entrance. Both identified as NOT OCCUPIED.
    # 2024-12-10 is strange. Not strictly depending on Backyard, but Back Entrance not enough.
    # 2024-12-28, 2024-12-26 are also strange, both identified as NOT OCCUPIED. 2024-03-19 is thought to be OCCUPIED

    filtered_columns = [
        col for col in daily_activity_data.columns if col in important_columns
    ]
    daily_activity_data = daily_activity_data[filtered_columns]

    logger.info(f"Filtered daily activity data to columns: {filtered_columns}")

    return daily_activity_data


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


def train_and_evaluate_model(daily_data, method, params=None):
    """
    Train a model using the specified method and evaluate its performance.

    Args:
        daily_data: Dictionary containing daily activity data
        method: Training method to use ('simple', 'cross_validation', 'leave_one_out', or 'grid_search')
        params: Dictionary of additional parameters for the training method

    Returns:
        Tuple containing:
            - Trained Occupancy instance
            - Model metrics
            - Model file path
    """
    if params is None:
        params = {}

    # Create an Occupancy instance
    ml_occupancy = Occupancy(mode=OccupancyMode.CALENDAR)  # Start in CALENDAR mode

    # Generate a model file name based on the method and parameters
    model_file = f"occupancy_model_{method}"
    if "max_depth" in params:
        model_file += f"_depth{params['max_depth']}"
    if "random_state" in params:
        model_file += f"_seed{params['random_state']}"
    model_file += ".txt"

    # Train the model using the specified method
    logger.info(f"Training model using {method} method...")
    model_metrics = ml_occupancy.train_occupancy_model(
        daily_data, method=method, **params
    )

    # Print model evaluation metrics
    logger.info(f"Model accuracy: {model_metrics['accuracy']:.2f}")

    # Print cross-validation scores if available
    if "cv_scores" in model_metrics:
        logger.info(f"Cross-validation scores: {model_metrics['cv_scores']}")
        logger.info(
            f"CV mean: {model_metrics['cv_mean']:.2f}, CV std: {model_metrics['cv_std']:.2f}"
        )

    # Print best parameters if available
    if "best_params" in model_metrics:
        logger.info(f"Best parameters: {model_metrics['best_params']}")

    # Print feature importances
    logger.info("Feature importances:")
    for feature, importance in sorted(
        model_metrics["feature_importances"].items(),
        key=lambda x: x[1],
        reverse=True,
    ):
        logger.info(f"  {feature}: {importance:.4f}")

    # Save the model to a file
    logger.info(f"Saving model to {model_file}...")
    ml_occupancy.save_occupancy_model(model_file)

    return ml_occupancy, model_metrics, model_file


def main():
    # Load video data
    daily_data = load_video_data()

    # 1. Create an Occupancy instance with calendar data only (default mode)
    logger.info("Creating calendar-based occupancy...")
    calendar_occupancy = Occupancy(mode=OccupancyMode.CALENDAR)
    calendar_status = calendar_occupancy.occupancy_cache.copy()

    # 2. Create an Occupancy instance with heuristic method
    logger.info("Creating heuristic-based occupancy...")
    heuristic_occupancy = Occupancy(mode=OccupancyMode.HEURISTIC, daily_data=daily_data)
    heuristic_status = heuristic_occupancy.occupancy_cache.copy()

    try:
        # 3. Train models using different methods

        # Simple train-test split (original method)
        simple_params = {
            "random_state": 50,  # Try different random seeds to see the effect
            "test_size": 0.2,
            "max_depth": 3,  # Limit tree depth for simplicity
        }
        simple_occupancy, simple_metrics, simple_model_file = train_and_evaluate_model(
            daily_data, "simple", simple_params
        )

        # Cross-validation
        cv_params = {
            "random_state": 50,
            "cv_folds": 5,
            "max_depth": 3,
        }
        cv_occupancy, cv_metrics, cv_model_file = train_and_evaluate_model(
            daily_data, "cross_validation", cv_params
        )

        # Leave-one-out cross-validation
        loo_params = {
            "random_state": 50,
            "max_depth": 3,
        }
        loo_occupancy, loo_metrics, loo_model_file = train_and_evaluate_model(
            daily_data, "leave_one_out", loo_params
        )

        # Grid search with cross-validation
        grid_params = {
            "random_state": 50,
            "cv_folds": 5,
        }
        grid_occupancy, grid_metrics, grid_model_file = train_and_evaluate_model(
            daily_data, "grid_search", grid_params
        )

        # 4. Compare the results of different methods
        logger.info("\nComparing model accuracies:")
        logger.info(f"  Simple train-test split: {simple_metrics['accuracy']:.2f}")
        logger.info(f"  Cross-validation: {cv_metrics['accuracy']:.2f}")
        logger.info(f"  Leave-one-out: {loo_metrics['accuracy']:.2f}")
        logger.info(f"  Grid search: {grid_metrics['accuracy']:.2f}")

        # 5. Use the best model for prediction (in this case, we'll use the grid search model)
        logger.info("\nUsing grid search model for prediction...")

        # Load the model from the file
        logger.info(f"Loading model from {grid_model_file}...")
        best_occupancy = Occupancy(
            mode=OccupancyMode.ML_MODEL,
            daily_data=daily_data,
            model_filepath=grid_model_file,
        )

        # Apply the model to predict occupancy status
        logger.info("Applying model to predict occupancy status...")
        best_occupancy.set_occupancy_status_from_daily_activity(daily_data)
        ml_status = best_occupancy.occupancy_cache.copy()

        # 6. Compare with calendar and heuristic methods
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
