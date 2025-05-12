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
8. Visualize the decision boundaries for 2D models
"""

import json
import logging
import os
import re
import sys

import pandas as pd
import plotly.graph_objects as go

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


def load_video_data(metric: str = "activity"):
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
    data_aggregator = VideoDataAggregator(metrics=[metric])
    daily_data, _ = data_aggregator.run(video_database)
    daily_activity_data = daily_data[metric]

    # Filter to keep only important columns (make sure that "Date" is always included)

    important_columns = [
        "Date",
        "Front Door",  # 404/414
        "Backyard",  # 409/414
        "Gateway",  # 409/414
        "Walkway",  # 409/414
        "Back Entrance",  # 410/414
    ]  # Best model

    filtered_columns = [
        col for col in daily_activity_data.columns if col in important_columns
    ]
    daily_activity_data = daily_activity_data[filtered_columns]

    logger.info(f"Filtered daily activity data to columns: {filtered_columns}")

    return daily_activity_data


def compare_occupancy_methods(
    calendar_occupancy, heuristic_occupancy, ml_occupancy, daily_data=None
):
    """
    Compare the occupancy status determined by different methods.

    Args:
        calendar_occupancy: Occupancy status from the calendar
        heuristic_occupancy: Occupancy status from the heuristic method
        ml_occupancy: Occupancy status from the machine learning model
        daily_data: DataFrame containing daily activity data

    Returns:
        DataFrame with comparison results
    """
    # Get all unique dates across all methods
    all_dates = set()
    for occupancy in [calendar_occupancy, heuristic_occupancy, ml_occupancy]:
        all_dates.update(occupancy.keys())

    # Create comparison data
    comparison_data = []

    # Create a date lookup dictionary from daily_data if provided
    daily_data_dict = {}
    if daily_data is not None:
        for _, row in daily_data.iterrows():
            date_str = row["Date"].strftime("%Y-%m-%d")
            daily_data_dict[date_str] = row

    for date_str in sorted(all_dates):
        data_entry = {
            "date": date_str,
            "calendar": calendar_occupancy.get(date_str, OccupancyStatus.UNKNOWN).value,
            "heuristic": heuristic_occupancy.get(
                date_str, OccupancyStatus.UNKNOWN
            ).value,
            "ml_model": ml_occupancy.get(date_str, OccupancyStatus.UNKNOWN).value,
        }

        # Add daily activity data if provided and date exists in daily_data
        if daily_data is not None and date_str in daily_data_dict:
            # Add all columns except 'Date' from daily_data
            for col in daily_data.columns:
                if col != "Date":
                    data_entry[col] = daily_data_dict[date_str][col]

        comparison_data.append(data_entry)

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


def visualize_decision_boundaries(
    daily_data, occupancy_model, calendar_status, output_file
):
    """
    Create a 2D visualization of the decision boundaries for a model with Front Door and Walkway features.

    Args:
        daily_data: DataFrame containing daily activity data
        occupancy_model: Trained Occupancy instance with a model
        calendar_status: Dictionary mapping dates to OccupancyStatus from the calendar
        output_file: Path to save the HTML visualization
    """
    # Check if we have a 2D model with Front Door and Walkway
    if not (
        set(occupancy_model.feature_names) == {"Front Door", "Walkway"}
        or (
            len(occupancy_model.feature_names) == 2
            and "Front Door" in occupancy_model.feature_names
            and "Walkway" in occupancy_model.feature_names
        )
    ):
        logger.info(
            "Skipping visualization: Model does not use exactly Front Door and Walkway features"
        )
        return

    # Extract the decision tree thresholds
    # For a simple decision tree with Front Door and Walkway, we expect thresholds like:
    # Front Door <= 3.5 and Walkway <= 3.0
    # These values might be different based on your trained model
    front_door_threshold = 3.5  # Default value
    walkway_threshold = 3.0  # Default value

    # Try to extract thresholds from the model
    if hasattr(occupancy_model, "model") and occupancy_model.model is not None:
        # Extract the tree structure
        tree = occupancy_model.model.tree_

        # Find the root node (node 0) and its primary threshold
        if tree.feature[0] >= 0:  # Not a leaf node
            root_feature_name = occupancy_model.feature_names[tree.feature[0]]
            root_threshold = tree.threshold[0]

            if root_feature_name == "Front Door":
                front_door_threshold = root_threshold

                # For a Front Door root, find the Walkway threshold in the right branch
                # The right child of node 0 is at index 2
                right_child_idx = 2

                # Check if the right child exists and is a decision node
                if (
                    right_child_idx < tree.node_count
                    and tree.feature[right_child_idx] >= 0
                ):
                    right_child_feature = occupancy_model.feature_names[
                        tree.feature[right_child_idx]
                    ]
                    if right_child_feature == "Walkway":
                        walkway_threshold = tree.threshold[right_child_idx]

                # If we didn't find Walkway at the immediate right child, look at the tree structure string
                # This is more reliable than trying to navigate the tree structure directly
                if hasattr(occupancy_model, "model") and hasattr(
                    occupancy_model, "model_filepath"
                ):
                    try:
                        # Load the model information from the JSON file
                        with open(occupancy_model.model_filepath, "r") as f:
                            model_info = json.load(f)

                        # Extract the tree structure from the model information
                        tree_text = model_info.get("tree_text", "")

                        # Find the section after "Front Door > 3.50"
                        right_branch_pattern = (
                            r"\|--- Front Door > +3\.50.*?\n(.*?)(?=\n\|---|$)"
                        )
                        right_branch_match = re.search(
                            right_branch_pattern, tree_text, re.DOTALL
                        )

                        if right_branch_match:
                            right_branch = right_branch_match.group(1)
                            # Find Walkway threshold in this branch
                            walkway_pattern = r"\|--- Walkway <= +([0-9.]+)"
                            walkway_match = re.search(walkway_pattern, right_branch)
                            if walkway_match:
                                walkway_threshold = float(walkway_match.group(1))
                                logger.debug(
                                    f"Found Walkway threshold {walkway_threshold} in right branch"
                                )
                    except Exception as e:
                        logger.error(f"Error parsing tree structure: {e}")

            elif root_feature_name == "Walkway":
                walkway_threshold = root_threshold

                # Similar logic for Walkway root
                right_child_idx = 2
                if (
                    right_child_idx < tree.node_count
                    and tree.feature[right_child_idx] >= 0
                ):
                    right_child_feature = occupancy_model.feature_names[
                        tree.feature[right_child_idx]
                    ]
                    if right_child_feature == "Front Door":
                        front_door_threshold = tree.threshold[right_child_idx]

    logger.info(
        f"Decision boundaries: Front Door = {front_door_threshold}, Walkway = {walkway_threshold}"
    )

    # Prepare data for plotting
    raw_data = []
    for _, row in daily_data.iterrows():
        date_str = row["Date"].strftime("%Y-%m-%d")
        if date_str in calendar_status:
            front_door = row.get("Front Door", 0)
            walkway = row.get("Walkway", 0)
            calendar_status_value = calendar_status[date_str].value

            # Get the model's prediction for this date
            model_status = occupancy_model.status(date_str)
            model_status_value = model_status.value

            # Check if this point is misclassified by the model
            is_misclassified = model_status != calendar_status[date_str]

            raw_data.append(
                {
                    "date": date_str,
                    "Front Door": front_door,
                    "Walkway": walkway,
                    "calendar_status": calendar_status_value,
                    "model_status": model_status_value,
                    "misclassified": is_misclassified,
                }
            )

    # Convert to DataFrame for easier processing
    raw_df = pd.DataFrame(raw_data)

    # Group by coordinates and model prediction to count points and check for misclassifications
    grouped_data = []
    for (front_door, walkway, model_status), group in raw_df.groupby(
        ["Front Door", "Walkway", "model_status"]
    ):
        # Count correctly classified and misclassified dates
        misclassified_dates = group[group["misclassified"]]["date"].tolist()
        correct_count = len(group) - len(misclassified_dates)
        misclassified_count = len(misclassified_dates)

        # A point has misclassifications if any date at this coordinate is misclassified
        has_misclassifications = len(misclassified_dates) > 0

        # Total count of dates at this coordinate
        total_count = len(group)

        grouped_data.append(
            {
                "Front Door": front_door,
                "Walkway": walkway,
                "model_status": model_status,
                "total_count": total_count,
                "correct_count": correct_count,
                "misclassified_count": misclassified_count,
                "misclassified_dates": misclassified_dates,
                "has_misclassifications": has_misclassifications,
            }
        )

    # Convert to DataFrame for plotting
    plot_df = pd.DataFrame(grouped_data)

    # Create figure
    fig = go.Figure()

    # Add scatter points for model's OCCUPIED prediction
    occupied_df = plot_df[plot_df["model_status"] == "OCCUPIED"]
    misclassified_occupied = occupied_df[occupied_df["has_misclassifications"]]
    correctly_classified_occupied = occupied_df[~occupied_df["has_misclassifications"]]

    # Correctly classified OCCUPIED points (no misclassifications)
    if not correctly_classified_occupied.empty:
        fig.add_trace(
            go.Scatter(
                x=correctly_classified_occupied["Front Door"],
                y=correctly_classified_occupied["Walkway"],
                mode="markers",
                name="Model: OCCUPIED (all correctly classified)",
                marker=dict(
                    color="green",
                    size=correctly_classified_occupied["total_count"].apply(
                        lambda x: min(10 + x * 2, 30)
                    ),
                    opacity=0.7,
                ),
                text=[
                    f"Correctly classified: {correct_count}<br>Misclassified: 0"
                    for correct_count in correctly_classified_occupied["correct_count"]
                ],
                hovertemplate="Coordinates: (%{x}, %{y})<br>%{text}<extra></extra>",
            )
        )

    # OCCUPIED points with some misclassifications
    if not misclassified_occupied.empty:
        fig.add_trace(
            go.Scatter(
                x=misclassified_occupied["Front Door"],
                y=misclassified_occupied["Walkway"],
                mode="markers",
                name="Model: OCCUPIED (some misclassified)",
                marker=dict(
                    color="green",
                    size=misclassified_occupied["total_count"].apply(
                        lambda x: min(10 + x * 2, 30)
                    ),
                    opacity=0.7,
                    line=dict(color="red", width=2),
                ),
                text=[
                    f"Correctly classified: {correct_count}<br>Misclassified: {misclassified_count}<br>Misclassified dates: {', '.join(misclassified_dates)}"
                    for correct_count, misclassified_count, misclassified_dates in zip(
                        misclassified_occupied["correct_count"],
                        misclassified_occupied["misclassified_count"],
                        misclassified_occupied["misclassified_dates"],
                    )
                ],
                hovertemplate="Coordinates: (%{x}, %{y})<br>%{text}<extra></extra>",
            )
        )

    # Add scatter points for model's NOT_OCCUPIED prediction
    not_occupied_df = plot_df[plot_df["model_status"] == "NOT_OCCUPIED"]
    misclassified_not_occupied = not_occupied_df[
        not_occupied_df["has_misclassifications"]
    ]
    correctly_classified_not_occupied = not_occupied_df[
        ~not_occupied_df["has_misclassifications"]
    ]

    # Correctly classified NOT_OCCUPIED points (no misclassifications)
    if not correctly_classified_not_occupied.empty:
        fig.add_trace(
            go.Scatter(
                x=correctly_classified_not_occupied["Front Door"],
                y=correctly_classified_not_occupied["Walkway"],
                mode="markers",
                name="Model: NOT_OCCUPIED (all correctly classified)",
                marker=dict(
                    color="blue",
                    size=correctly_classified_not_occupied["total_count"].apply(
                        lambda x: min(10 + x * 2, 30)
                    ),
                    opacity=0.7,
                ),
                text=[
                    f"Correctly classified: {correct_count}<br>Misclassified: 0"
                    for correct_count in correctly_classified_not_occupied[
                        "correct_count"
                    ]
                ],
                hovertemplate="Coordinates: (%{x}, %{y})<br>%{text}<extra></extra>",
            )
        )

    # NOT_OCCUPIED points with some misclassifications
    if not misclassified_not_occupied.empty:
        fig.add_trace(
            go.Scatter(
                x=misclassified_not_occupied["Front Door"],
                y=misclassified_not_occupied["Walkway"],
                mode="markers",
                name="Model: NOT_OCCUPIED (some misclassified)",
                marker=dict(
                    color="blue",
                    size=misclassified_not_occupied["total_count"].apply(
                        lambda x: min(10 + x * 2, 30)
                    ),
                    opacity=0.7,
                    line=dict(color="red", width=2),
                ),
                text=[
                    f"Correctly classified: {correct_count}<br>Misclassified: {misclassified_count}<br>Misclassified dates: {', '.join(misclassified_dates)}"
                    for correct_count, misclassified_count, misclassified_dates in zip(
                        misclassified_not_occupied["correct_count"],
                        misclassified_not_occupied["misclassified_count"],
                        misclassified_not_occupied["misclassified_dates"],
                    )
                ],
                hovertemplate="Coordinates: (%{x}, %{y})<br>%{text}<extra></extra>",
            )
        )

    # Add decision boundaries
    # Vertical line for Front Door threshold
    fig.add_shape(
        type="line",
        x0=front_door_threshold,
        y0=0,  # Start at 0 for linear scale
        x1=front_door_threshold,
        y1=plot_df["Walkway"].max() * 1.1,
        line=dict(color="red", width=2, dash="dash"),
        name="Front Door Threshold",
    )

    # Horizontal line for Walkway threshold
    fig.add_shape(
        type="line",
        x0=0,  # Start at 0 for linear scale
        y0=walkway_threshold,
        x1=plot_df["Front Door"].max() * 1.1,
        y1=walkway_threshold,
        line=dict(color="red", width=2, dash="dash"),
        name="Walkway Threshold",
    )

    # Add annotations for the thresholds
    fig.add_annotation(
        x=front_door_threshold,
        y=plot_df["Walkway"].max() * 0.9,
        text=f"Front Door = {front_door_threshold}",
        showarrow=True,
        arrowhead=1,
        ax=50,
        ay=0,
    )

    fig.add_annotation(
        x=plot_df["Front Door"].max() * 0.9,
        y=walkway_threshold,
        text=f"Walkway = {walkway_threshold}",
        showarrow=True,
        arrowhead=1,
        ax=0,
        ay=-50,
    )

    # Update layout with linear scales
    fig.update_layout(
        title="Decision Boundaries for Occupancy Model",
        xaxis_title="Front Door Activity",
        yaxis_title="Walkway Activity",
        xaxis=dict(
            range=[
                -0.5,  # Start slightly below 0 for linear scale
                plot_df["Front Door"].max() * 1.1,
            ]
        ),
        yaxis=dict(
            range=[
                -0.5,  # Start slightly below 0 for linear scale
                plot_df["Walkway"].max() * 1.1,
            ]
        ),
        legend=dict(
            x=0.01,
            y=0.99,
            bgcolor="rgba(255, 255, 255, 0.5)",
            bordercolor="rgba(0, 0, 0, 0.5)",
            borderwidth=1,
        ),
        width=1000,
        height=800,
        hovermode="closest",
    )

    # Add quadrant labels
    # Top-right quadrant
    fig.add_annotation(
        x=plot_df["Front Door"].max() * 0.7,
        y=plot_df["Walkway"].max() * 0.7,
        text="OCCUPIED Region",
        showarrow=False,
        font=dict(size=14, color="green"),
    )

    # Bottom-left quadrant
    fig.add_annotation(
        x=2,  # Fixed position for linear scale
        y=1,  # Fixed position for linear scale
        text="NOT_OCCUPIED Region",
        showarrow=False,
        font=dict(size=14, color="blue"),
    )

    # Count total points and misclassified points
    total_dates = len(raw_df)
    misclassified_dates = raw_df["misclassified"].sum()
    accuracy = (
        (total_dates - misclassified_dates) / total_dates if total_dates > 0 else 0
    )

    # Add accuracy information
    fig.add_annotation(
        x=0.5,
        y=1.05,
        xref="paper",
        yref="paper",
        text=f"Model Accuracy: {accuracy:.2%} ({total_dates - misclassified_dates}/{total_dates})",
        showarrow=False,
        font=dict(size=14),
    )

    # Save the figure
    fig.write_html(output_file)
    logger.info(f"Decision boundary visualization saved to {output_file}")


def train_and_evaluate_model(daily_data, out_dir="", method="simple", params=None):
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
    model_file = os.path.join(out_dir, model_file)

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

    # Path to the output directory for model files, comparison, graphs, etc...
    out_dir: str = "/Users/jbouguet/Documents/EufySecurityVideos/stories"

    # Load video data
    metric: str = "activity"  # "duration"
    daily_data = load_video_data(metric)

    # 1. Create an Occupancy instance with calendar data only (default mode)
    logger.info("Creating calendar-based occupancy...")
    calendar_occupancy = Occupancy(mode=OccupancyMode.CALENDAR)
    calendar_status = calendar_occupancy.occupancy_cache.copy()

    # 2. Create an Occupancy instance with heuristic method
    logger.info("Creating heuristic-based occupancy...")
    heuristic_occupancy = Occupancy(mode=OccupancyMode.HEURISTIC, daily_data=daily_data)
    heuristic_status = heuristic_occupancy.occupancy_cache.copy()

    random_state = 500
    max_depth = 3

    try:
        # 3. Train models using different methods

        # Simple train-test split (original method)
        simple_params = {
            "random_state": random_state,  # Try different random seeds to see the effect
            "test_size": 0.2,
            "max_depth": max_depth,  # Limit tree depth for simplicity
        }
        simple_occupancy, simple_metrics, simple_model_file = train_and_evaluate_model(
            daily_data,
            out_dir=out_dir,
            method="simple",
            params=simple_params,
        )

        # Cross-validation
        cv_params = {
            "random_state": random_state,
            "cv_folds": 5,
            "max_depth": max_depth,
        }
        cv_occupancy, cv_metrics, cv_model_file = train_and_evaluate_model(
            daily_data,
            out_dir=out_dir,
            method="cross_validation",
            params=cv_params,
        )

        # Leave-one-out cross-validation
        loo_params = {
            "random_state": random_state,
            "max_depth": max_depth,
        }
        loo_occupancy, loo_metrics, loo_model_file = train_and_evaluate_model(
            daily_data,
            out_dir=out_dir,
            method="leave_one_out",
            params=loo_params,
        )

        # 4. Compare the results of different methods
        logger.info("\nComparing model accuracies:")
        logger.info(f"  Simple train-test split: {simple_metrics['accuracy']:.2f}")
        logger.info(f"  Cross-validation: {cv_metrics['accuracy']:.2f}")
        logger.info(f"  Leave-one-out: {loo_metrics['accuracy']:.2f}")

        # 5. Use the best model for prediction (in this case, we'll use the Leave-one-out cross-validation)
        logger.info("\nUsing Leave-one-out cross-validation model for prediction...")

        # Load the model from the file
        logger.info(f"Loading model from {loo_model_file}...")
        best_occupancy = Occupancy(
            mode=OccupancyMode.ML_MODEL,
            daily_data=daily_data,
            model_filepath=loo_model_file,
        )

        # Apply the model to predict occupancy status
        logger.info("Applying model to predict occupancy status...")
        best_occupancy.set_occupancy_status_from_daily_activity(daily_data)
        ml_status = best_occupancy.occupancy_cache.copy()

        # 6. Compare with calendar and heuristic methods
        logger.info("Comparing occupancy methods...")
        comparison_df = compare_occupancy_methods(
            calendar_status, heuristic_status, ml_status, daily_data
        )

        # Save comparison results to CSV
        comparison_file = os.path.join(out_dir, "occupancy_comparison.csv")
        logger.info(f"Saving comparison results to {comparison_file}...")
        comparison_df.to_csv(comparison_file, index=False)

        # Create a filtered DataFrame for errors (where heuristic or ml_model differs from calendar)
        errors_df = comparison_df[
            (comparison_df["calendar"] != comparison_df["heuristic"])
            | (comparison_df["calendar"] != comparison_df["ml_model"])
        ]

        # Save errors to a separate CSV file
        errors_file = os.path.join(out_dir, "occupancy_comparison_errors.csv")
        logger.info(f"Saving error cases to {errors_file}...")
        errors_df.to_csv(errors_file, index=False)

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

        # Print sample of comparison results
        logger.info("Sample of comparison results for errors:")
        print(errors_df.head(10))

        # 7. Visualize decision boundaries if we have a 2D model
        logger.info("Creating decision boundary visualization...")
        visualize_decision_boundaries(
            daily_data,
            best_occupancy,
            calendar_status,
            os.path.join(out_dir, "occupancy_decision_boundaries.html"),
        )

    except Exception as e:
        logger.error(f"Error in machine learning process: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
