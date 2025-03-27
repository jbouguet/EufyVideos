#!/usr/bin/env python3

"""
Occupancy module for determining property occupancy status based on a manually maintained calendar and video activity data.

This module provides functionality to determine whether a property is occupied,
not occupied, or has an unknown occupancy status for specific dates. By default,
occupancy status is set via a manually maintained calendar, which can be augmented
or overwritten with data derived from video activity analysis.

Key components:
1. OccupancyStatus: Enum defining possible occupancy states
2. Occupancy: Class that manages occupancy status using both a calendar and video activity data
3. Machine learning capabilities to predict occupancy from daily activity data

Example usage:
    from video_data_aggregator import VideoDataAggregator
    from video_database import VideoDatabase, VideoDatabaseList
    from occupancy import Occupancy, OccupancyStatus

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

    # Train a machine learning model
    occupancy.train_occupancy_model(daily_data["activity"])

    # Save the model
    occupancy.save_occupancy_model("occupancy_model.txt")

    # Load a model
    occupancy.load_occupancy_model("occupancy_model.txt")

    # Apply the model to daily activity data
    occupancy.set_occupancy_status_from_model(daily_data["activity"])
"""

import enum
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

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
    Class for determining property occupancy status based on a calendar and video activity data.

    This class manages occupancy status using three primary sources:
    1. A manually maintained calendar (default source) - This calendar needs to be kept up to date
       by the user as it contains the expected occupancy schedule.
    2. Video activity data analysis (optional) - This can augment or be overwritten by the calendar data.
    3. Machine learning model (optional) - A trained model that predicts occupancy from daily activity data.

    The class uses a set of heuristics to determine occupancy from video activity and
    caches all results for efficient lookup. It also provides methods to train, save, load,
    and apply machine learning models for improved occupancy prediction.

    Attributes:
        daily_data (pd.DataFrame): Daily aggregated video activity data
        occupancy_cache (Dict[str, OccupancyStatus]): Cache of date to occupancy status
        model (DecisionTreeClassifier): Trained decision tree model for occupancy prediction
    """

    # Occupancy Calendar to be kept up to date by the user specified as a list of calendar
    # date intervals with occupancy status. Any date not specified in the calendar will be
    # assumed to be of occupancy status UNKNOWN.
    # Format: (start_date, end_date, occupancy_status)
    OCCUPANCY_CALENDAR = [
        ("2024-02-27", "2024-03-03", OccupancyStatus.OCCUPIED),
        ("2024-03-04", "2024-03-07", OccupancyStatus.NOT_OCCUPIED),
        ("2024-03-08", "2024-03-09", OccupancyStatus.OCCUPIED),
        ("2024-03-10", "2024-03-10", OccupancyStatus.NOT_OCCUPIED),
        ("2024-03-11", "2024-03-11", OccupancyStatus.OCCUPIED),
        ("2024-03-12", "2024-03-12", OccupancyStatus.NOT_OCCUPIED),
        ("2024-03-13", "2024-03-13", OccupancyStatus.OCCUPIED),
        ("2024-03-14", "2024-03-17", OccupancyStatus.NOT_OCCUPIED),
        ("2024-03-18", "2024-03-18", OccupancyStatus.OCCUPIED),
        ("2024-03-19", "2024-03-21", OccupancyStatus.NOT_OCCUPIED),
        ("2024-03-22", "2024-03-22", OccupancyStatus.OCCUPIED),
        ("2024-03-23", "2024-03-23", OccupancyStatus.NOT_OCCUPIED),
        ("2024-03-24", "2024-04-12", OccupancyStatus.OCCUPIED),
        ("2024-04-13", "2024-04-17", OccupancyStatus.NOT_OCCUPIED),
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
        ("2024-09-22", "2024-10-05", OccupancyStatus.OCCUPIED),
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
        # ("2025-03-11", "2025-03-24", OccupancyStatus.NOT_OCCUPIED),
        # ("2025-03-25", "2025-04-15", OccupancyStatus.OCCUPIED),
    ]

    def __init__(self, daily_data: Optional[pd.DataFrame] = None):
        """
        Initialize the Occupancy class with optionally provided daily aggregated video activity data.

        Args:
            daily_data: [optional] DataFrame containing daily aggregated video activity data
                        from VideoDataAggregator.run(). If provided, occupancy status
                        will first be set from this data, then potentially overwritten
                        by the calendar data OCCUPANCY_CALENDAR.
        """
        self.occupancy_cache: Dict[str, OccupancyStatus] = {}
        self.model: Optional[DecisionTreeClassifier] = None
        self.feature_names: List[str] = []

        if daily_data is not None:
            self.set_occupancy_status_from_daily_activity(daily_data)
        self.set_occupancy_status_from_calendar()

    def set_occupancy_status_from_calendar(self):
        """
        Set occupancy status cache from the manually maintained calendar.

        This method populates the occupancy_cache with statuses from the OCCUPANCY_CALENDAR.
        Note that these results may be overwritten by the status set from the daily
        video activity data when set_occupancy_status_from_daily_activity() is called.

        Note: The OCCUPANCY_CALENDAR should be kept up to date by the user to reflect
        the expected occupancy schedule.
        """

        for start_date_str, end_date_str, status in self.OCCUPANCY_CALENDAR:
            start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
            end_date = datetime.strptime(end_date_str, "%Y-%m-%d")

            current_date = start_date
            while current_date <= end_date:
                date_str = current_date.strftime("%Y-%m-%d")
                self.occupancy_cache[date_str] = status
                current_date = datetime(
                    current_date.year, current_date.month, current_date.day
                ) + pd.Timedelta(days=1)

    def set_occupancy_status_from_daily_activity(self, daily_data: pd.DataFrame):
        """
        Analyze daily video activity data to determine occupancy status for each date.

        This method implements heuristics to determine occupancy status based on
        video activity metrics derived from the decision tree model.

        The results are stored in the occupancy_cache dictionary for efficient lookup.
        Note that these results may be overwritten by the calendar data when
        set_occupancy_status_from_calendar() is called.

        This method will be deprecated in favor of set_occupancy_status_from_model
        once the machine learning model is fully tested and validated.
        """

        for _, row in daily_data.iterrows():
            date_str = row["Date"].strftime("%Y-%m-%d")

            # Extract activity values with default of 0 if not present
            fd = row.get("Front Door", 0)
            be = row.get("Back Entrance", 0)
            ww = row.get("Walkway", 0)
            by = row.get("Backyard", 0)

            # Set occupancy status using the decision function
            if (
                (fd > 3.5 and ww > 3.0 and (by > 1.5 or ww > 7.5))
                or (fd <= 3.5 and be > 2.0)
                or (fd <= 3.5 and be <= 2.0 and not (ww <= 4.5 or by <= 9.0))
            ):
                self.occupancy_cache[date_str] = OccupancyStatus.OCCUPIED
            else:
                self.occupancy_cache[date_str] = OccupancyStatus.NOT_OCCUPIED

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

    def _prepare_training_data(
        self, daily_data: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare training data from daily activity data and occupancy calendar.

        Args:
            daily_data: DataFrame containing daily aggregated video activity data

        Returns:
            Tuple containing:
                - X: Feature matrix (device activity counts)
                - y: Target vector (occupancy status)
                - feature_names: List of feature names (device names)
        """
        # Get all dates from the calendar with known occupancy status (not UNKNOWN)
        calendar_dates = {}
        for start_date_str, end_date_str, status in self.OCCUPANCY_CALENDAR:
            if status != OccupancyStatus.UNKNOWN:
                start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
                end_date = datetime.strptime(end_date_str, "%Y-%m-%d")

                current_date = start_date
                while current_date <= end_date:
                    date_str = current_date.strftime("%Y-%m-%d")
                    calendar_dates[date_str] = status
                    current_date = datetime(
                        current_date.year, current_date.month, current_date.day
                    ) + pd.Timedelta(days=1)

        # Filter daily_data to include only dates with known occupancy status
        filtered_data = []
        for _, row in daily_data.iterrows():
            date_str = row["Date"].strftime("%Y-%m-%d")
            if date_str in calendar_dates:
                # Create a dictionary with device activity counts
                row_dict = {"Date": date_str}
                for col in daily_data.columns:
                    if col != "Date":
                        row_dict[col] = row[col]
                row_dict["OccupancyStatus"] = calendar_dates[date_str]
                filtered_data.append(row_dict)

        if not filtered_data:
            raise ValueError(
                "No matching dates found between daily activity data and occupancy calendar"
            )

        # Convert to DataFrame
        df = pd.DataFrame(filtered_data)

        # Extract features (device activity counts) and target (occupancy status)
        feature_cols = [
            col for col in df.columns if col not in ["Date", "OccupancyStatus"]
        ]
        X = df[feature_cols].values

        # Convert occupancy status to binary target (1 for OCCUPIED, 0 for NOT_OCCUPIED)
        y = np.array(
            [
                1 if status == OccupancyStatus.OCCUPIED else 0
                for status in df["OccupancyStatus"]
            ]
        )

        return X, y, feature_cols

    def train_occupancy_model(
        self, daily_data: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
    ) -> Dict[str, Any]:
        """
        Train a decision tree model to predict occupancy status from daily activity data.

        Args:
            daily_data: DataFrame containing daily aggregated video activity data
            test_size: Proportion of data to use for testing (default: 0.2)
            random_state: Random seed for reproducibility (default: 42)

        Returns:
            Dictionary containing model evaluation metrics
        """
        logger.info("Training occupancy model...")

        # Prepare training data
        X, y, feature_names = self._prepare_training_data(daily_data)
        self.feature_names = feature_names

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Train decision tree model
        self.model = DecisionTreeClassifier(random_state=random_state)
        self.model.fit(X_train, y_train)

        # Evaluate model
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(
            y_test, y_pred, target_names=["NOT_OCCUPIED", "OCCUPIED"], output_dict=True
        )

        logger.info(f"Model trained with accuracy: {accuracy:.2f}")

        # Return evaluation metrics
        return {
            "accuracy": accuracy,
            "classification_report": report,
            "feature_names": feature_names,
            "feature_importances": dict(
                zip(feature_names, self.model.feature_importances_)
            ),
        }

    def save_occupancy_model(self, filepath: str) -> None:
        """
        Save the trained model to a human-readable file.

        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            raise ValueError(
                "No model has been trained. Call train_occupancy_model() first."
            )

        # Create directory if it doesn't exist
        os.makedirs(
            os.path.dirname(filepath) if os.path.dirname(filepath) else ".",
            exist_ok=True,
        )

        # Export decision tree as text
        tree_text = export_text(self.model, feature_names=self.feature_names)

        # Create a dictionary with model information
        model_info = {
            "feature_names": self.feature_names,
            "feature_importances": dict(
                zip(self.feature_names, self.model.feature_importances_.tolist())
            ),
            "tree_text": tree_text,
            "model_params": self.model.get_params(),
            "classes": ["NOT_OCCUPIED", "OCCUPIED"],
        }

        # Save to file
        with open(filepath, "w") as f:
            json.dump(model_info, f, indent=2)

        logger.info(f"Model saved to {filepath}")

    def load_occupancy_model(self, filepath: str) -> None:
        """
        Load a model from a file.

        Args:
            filepath: Path to the model file
        """
        # Load model information
        with open(filepath, "r") as f:
            model_info = json.load(f)

        # Extract feature names
        self.feature_names = model_info["feature_names"]

        # Create a new model with the same parameters
        self.model = DecisionTreeClassifier(
            **{
                k: v
                for k, v in model_info["model_params"].items()
                if k not in ["random_state"]
            }
        )

        logger.info(f"Model loaded from {filepath}")
        logger.info(f"Feature importances: {model_info['feature_importances']}")
        logger.info(f"Decision tree structure:\n{model_info['tree_text']}")

        # Note: This doesn't actually load the trained model parameters,
        # as scikit-learn doesn't provide a way to directly set the tree structure from text.
        # For a complete solution, we would need to use pickle or joblib to save/load the model.
        # However, this approach provides human-readable model information and can be used
        # as a reference for manual implementation of the decision rules.
        logger.warning(
            "Note: This is a human-readable representation of the model. "
            "For actual prediction, you'll need to retrain the model or use "
            "the decision rules manually."
        )

    def set_occupancy_status_from_model(self, daily_data: pd.DataFrame) -> None:
        """
        Apply the trained model to daily activity data to determine occupancy status.

        Args:
            daily_data: DataFrame containing daily aggregated video activity data
        """
        if self.model is None:
            raise ValueError(
                "No model has been trained. Call train_occupancy_model() first."
            )

        logger.info("Applying model to determine occupancy status...")

        for _, row in daily_data.iterrows():
            date_str = row["Date"].strftime("%Y-%m-%d")

            # Extract features for this date
            features = []
            for feature in self.feature_names:
                if feature in row:
                    features.append(row[feature])
                else:
                    # If feature is missing, use 0
                    features.append(0)

            # Make prediction
            features_array = np.array(features).reshape(1, -1)
            try:
                prediction = self.model.predict(features_array)[0]

                # Convert prediction to OccupancyStatus
                if prediction == 1:
                    self.occupancy_cache[date_str] = OccupancyStatus.OCCUPIED
                else:
                    self.occupancy_cache[date_str] = OccupancyStatus.NOT_OCCUPIED
            except Exception as e:
                logger.error(f"Error predicting occupancy for {date_str}: {e}")
                self.occupancy_cache[date_str] = OccupancyStatus.UNKNOWN

        logger.info(f"Model applied to {len(daily_data)} dates")


if __name__ == "__main__":
    # Testing code for the module
    import csv
    import logging

    from logging_config import set_logger_level_and_format
    from video_database import VideoDatabase, VideoDatabaseList

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

    logger.debug(f"Number of videos: {len(video_database)}")

    # Get aggregated data
    logger.info("Aggregating video data...")
    data_aggregator = VideoDataAggregator(metrics=["activity"])
    daily_data, _ = data_aggregator.run(video_database)

    # Create occupancy analyzer
    logger.info("Analyzing occupancy...")

    occupancy = Occupancy(daily_data["activity"])
    # occupancy = Occupancy()

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
