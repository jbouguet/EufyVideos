#!/usr/bin/env python3

"""
Occupancy module for determining property occupancy status based on a manually maintained calendar and video activity data.

This module provides functionality to determine whether a property is occupied,
not occupied, or has an unknown occupancy status for specific dates. By default,
occupancy status is set via a manually maintained calendar, which can be augmented
or overwritten with data derived from video activity analysis.

Key components:
1. OccupancyStatus: Enum defining possible occupancy states
2. OccupancyMode: Enum defining different modes of operation
3. Occupancy: Class that manages occupancy status using different modes

Example usage:
    from video_data_aggregator import VideoDataAggregator
    from video_database import VideoDatabase, VideoDatabaseList
    from occupancy import Occupancy, OccupancyStatus, OccupancyMode

    # Load video database
    video_database = VideoDatabaseList([...]).load_videos()

    # Get aggregated data
    data_aggregator = VideoDataAggregator(metrics=["activity"])
    daily_data, _ = data_aggregator.run(video_database)

    # Create occupancy analyzer in CALENDAR mode (default)
    occupancy_calendar = Occupancy()

    # Create occupancy analyzer in HEURISTIC mode
    occupancy_heuristic = Occupancy(mode=OccupancyMode.HEURISTIC, daily_data=daily_data["activity"])

    # Create occupancy analyzer in ML_MODEL mode
    occupancy_ml = Occupancy(mode=OccupancyMode.ML_MODEL, daily_data=daily_data["activity"])

    # Check occupancy for a specific date
    status = occupancy_ml.status("2025-01-01")
    if status == OccupancyStatus.OCCUPIED:
        print("Property was occupied on 2025-01-01")

    # Train a new model
    occupancy_ml.train_occupancy_model(daily_data["activity"])

    # Save the model
    occupancy_ml.save_occupancy_model("occupancy_model.txt")
"""

import enum
import json
import os
import pickle
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    LeaveOneOut,
    cross_val_score,
    train_test_split,
)
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

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


class OccupancyMode(enum.Enum):
    """
    Enum representing the different modes for determining occupancy status.

    Values:
        CALENDAR: Use the manually maintained calendar
        HEURISTIC: Use heuristic rules based on daily activity
        ML_MODEL: Use a trained machine learning model
    """

    CALENDAR = "CALENDAR"
    HEURISTIC = "HEURISTIC"
    ML_MODEL = "ML_MODEL"


class Occupancy:
    """
    Class for determining property occupancy status based on a calendar and video activity data.

    This class manages occupancy status using three primary modes:
    1. CALENDAR (default): Uses a manually maintained calendar that needs to be kept up to date
       by the user as it contains the expected occupancy schedule.
    2. HEURISTIC: Uses heuristic rules based on daily activity data to determine occupancy.
    3. ML_MODEL: Uses a trained machine learning model to predict occupancy from daily activity.

    The class maintains its mode of operation internally and handles the appropriate behavior
    based on the selected mode. It also provides methods to train, save, and load machine
    learning models for improved occupancy prediction.

    The ML_MODEL mode uses a decision tree classifier to predict occupancy status based on
    daily activity data. The model is trained using the OCCUPANCY_CALENDAR as ground truth
    and can be saved to a file for later use. The model can also be loaded from a file.

    Attributes:
        mode (OccupancyMode): The current mode of operation
        occupancy_cache (Dict[str, OccupancyStatus]): Cache of date to occupancy status
        model (DecisionTreeClassifier): Trained decision tree model for occupancy prediction
        feature_names (List[str]): Names of features used by the model
        model_filepath (str): Path to the model file (used in ML_MODEL mode)
        tree_structure (str): Human-readable representation of the decision tree structure
    """

    # Occupancy Calendar to be kept up to date by the user specified as a list of calendar
    # date intervals with occupancy status. Any date not specified in the calendar will be
    # assumed to be of occupancy status UNKNOWN.
    # Format: (start_date, end_date, occupancy_status)
    OCCUPANCY_CALENDAR = [
        ("2024-02-27", "2024-02-27", OccupancyStatus.NOT_OCCUPIED),
        ("2024-02-29", "2024-03-03", OccupancyStatus.OCCUPIED),
        ("2024-03-04", "2024-03-07", OccupancyStatus.NOT_OCCUPIED),
        ("2024-03-08", "2024-03-09", OccupancyStatus.OCCUPIED),
        ("2024-03-10", "2024-03-10", OccupancyStatus.NOT_OCCUPIED),
        ("2024-03-11", "2024-03-11", OccupancyStatus.OCCUPIED),
        ("2024-03-12", "2024-03-12", OccupancyStatus.NOT_OCCUPIED),
        ("2024-03-13", "2024-03-14", OccupancyStatus.OCCUPIED),
        ("2024-03-15", "2024-03-17", OccupancyStatus.NOT_OCCUPIED),
        ("2024-03-18", "2024-03-18", OccupancyStatus.OCCUPIED),
        ("2024-03-19", "2024-03-23", OccupancyStatus.NOT_OCCUPIED),
        ("2024-03-24", "2024-04-12", OccupancyStatus.OCCUPIED),
        ("2024-04-13", "2024-05-01", OccupancyStatus.NOT_OCCUPIED),
        ("2024-05-02", "2024-05-02", OccupancyStatus.OCCUPIED),
        ("2024-05-03", "2024-05-10", OccupancyStatus.NOT_OCCUPIED),
        ("2024-05-11", "2024-05-22", OccupancyStatus.OCCUPIED),
        ("2024-05-23", "2024-06-09", OccupancyStatus.NOT_OCCUPIED),
        ("2024-06-10", "2024-07-19", OccupancyStatus.OCCUPIED),
        ("2024-07-20", "2024-07-21", OccupancyStatus.NOT_OCCUPIED),
        ("2024-07-22", "2024-07-22", OccupancyStatus.OCCUPIED),
        ("2024-07-23", "2024-08-06", OccupancyStatus.NOT_OCCUPIED),
        ("2024-08-07", "2024-08-13", OccupancyStatus.OCCUPIED),
        ("2024-08-14", "2024-08-15", OccupancyStatus.NOT_OCCUPIED),
        ("2024-08-16", "2024-08-17", OccupancyStatus.OCCUPIED),
        ("2024-08-18", "2024-08-18", OccupancyStatus.NOT_OCCUPIED),
        ("2024-08-19", "2024-08-19", OccupancyStatus.OCCUPIED),
        ("2024-08-20", "2024-08-20", OccupancyStatus.NOT_OCCUPIED),
        ("2024-08-21", "2024-08-21", OccupancyStatus.OCCUPIED),
        ("2024-08-22", "2024-08-22", OccupancyStatus.NOT_OCCUPIED),
        ("2024-08-23", "2024-08-24", OccupancyStatus.OCCUPIED),
        ("2024-08-25", "2024-08-27", OccupancyStatus.NOT_OCCUPIED),
        ("2024-08-28", "2024-09-09", OccupancyStatus.OCCUPIED),
        ("2024-09-10", "2024-09-12", OccupancyStatus.NOT_OCCUPIED),
        ("2024-09-13", "2024-09-13", OccupancyStatus.OCCUPIED),
        ("2024-09-14", "2024-09-21", OccupancyStatus.NOT_OCCUPIED),
        ("2024-09-22", "2024-10-05", OccupancyStatus.OCCUPIED),
        ("2024-10-06", "2024-10-08", OccupancyStatus.NOT_OCCUPIED),
        ("2024-10-09", "2024-10-11", OccupancyStatus.OCCUPIED),
        ("2024-10-12", "2024-11-15", OccupancyStatus.NOT_OCCUPIED),
        ("2024-11-16", "2024-11-20", OccupancyStatus.OCCUPIED),
        ("2024-11-21", "2024-12-10", OccupancyStatus.NOT_OCCUPIED),
        ("2024-12-11", "2024-12-13", OccupancyStatus.OCCUPIED),
        ("2024-12-14", "2024-12-25", OccupancyStatus.NOT_OCCUPIED),
        ("2024-12-26", "2025-01-08", OccupancyStatus.OCCUPIED),
        ("2025-01-09", "2025-03-06", OccupancyStatus.NOT_OCCUPIED),
        ("2025-03-07", "2025-03-10", OccupancyStatus.OCCUPIED),
        ("2025-03-11", "2025-03-24", OccupancyStatus.NOT_OCCUPIED),
        ("2025-03-25", "2025-03-25", OccupancyStatus.OCCUPIED),
        ("2025-03-26", "2025-03-26", OccupancyStatus.NOT_OCCUPIED),
        ("2025-03-27", "2025-04-17", OccupancyStatus.OCCUPIED),
        ("2025-04-18", "2025-04-20", OccupancyStatus.NOT_OCCUPIED),
        ("2025-04-21", "2025-04-23", OccupancyStatus.OCCUPIED),
        ("2025-04-24", "2025-06-20", OccupancyStatus.NOT_OCCUPIED),
        ("2025-06-21", "2025-07-27", OccupancyStatus.OCCUPIED),
        ("2025-07-28", "2025-09-24", OccupancyStatus.NOT_OCCUPIED),
        ("2025-09-25", "2025-09-29", OccupancyStatus.OCCUPIED),
    ]

    def __init__(
        self,
        mode: OccupancyMode = OccupancyMode.CALENDAR,
        daily_data: Optional[pd.DataFrame] = None,
        model_filepath: str = "occupancy_model.txt",
    ):
        """
        Initialize the Occupancy class with the specified mode and optional data.

        Args:
            mode: The mode of operation (CALENDAR, HEURISTIC, or ML_MODEL)
            daily_data: [optional] DataFrame containing daily aggregated video activity data
                        from VideoDataAggregator.run(). Required for HEURISTIC and ML_MODEL modes.
            model_filepath: Path to the model file (used in ML_MODEL mode)
        """
        self.occupancy_cache: Dict[str, OccupancyStatus] = {}
        self.model: Optional[DecisionTreeClassifier] = None
        self.feature_names: List[str] = []
        self.mode = mode
        self.model_filepath = model_filepath

        # Initialize based on mode
        if mode == OccupancyMode.CALENDAR:
            self.set_occupancy_status_from_calendar()
        elif mode == OccupancyMode.HEURISTIC:
            if daily_data is None:
                raise ValueError("daily_data is required for HEURISTIC mode")
            self.set_occupancy_status_from_daily_activity(daily_data)
        elif mode == OccupancyMode.ML_MODEL:
            if daily_data is None:
                raise ValueError("daily_data is required for ML_MODEL mode")
            try:
                self.load_occupancy_model(model_filepath)
                self.set_occupancy_status_from_daily_activity(daily_data)
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                logger.info("Falling back to CALENDAR mode")
                self.mode = OccupancyMode.CALENDAR
                self.set_occupancy_status_from_calendar()

    def set_occupancy_status_from_calendar(self):
        """
        Set occupancy status cache from the manually maintained calendar.

        This method clears the occupancy_cache before populating it with statuses from
        the OCCUPANCY_CALENDAR, ensuring that the occupancy status is determined solely
        by the calendar data.

        Note: The OCCUPANCY_CALENDAR should be kept up to date by the user to reflect
        the expected occupancy schedule.
        """
        # Clear the cache before populating with new statuses
        self.occupancy_cache.clear()

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

        This method dispatches to the appropriate implementation based on the current mode.
        - In HEURISTIC mode, it uses heuristic rules based on activity metrics.
        - In ML_MODEL mode, it uses the trained machine learning model.
        - In CALENDAR mode, it does nothing (calendar data is used instead).

        Args:
            daily_data: DataFrame containing daily aggregated video activity data
        """
        if self.mode == OccupancyMode.HEURISTIC:
            self._set_occupancy_status_from_heuristic(daily_data)
        elif self.mode == OccupancyMode.ML_MODEL:
            self._set_occupancy_status_from_model(daily_data)
        else:
            logger.warning(
                "set_occupancy_status_from_daily_activity called in CALENDAR mode"
            )

    def _set_occupancy_status_from_heuristic(self, daily_data: pd.DataFrame):
        """
        Analyze daily video activity data using heuristic rules.

        This method implements heuristics to determine occupancy status based on
        video activity metrics derived from the decision tree model.

        The method clears the occupancy_cache before populating it with new statuses,
        ensuring that the occupancy status is determined solely by the heuristic logic.
        """
        # Clear the cache before populating with new statuses
        self.occupancy_cache.clear()

        for _, row in daily_data.iterrows():
            date_str = row["Date"].strftime("%Y-%m-%d")
            by = row.get("Backyard", 0)

            # Set occupancy status using the decision function
            if by > 15.5:  # 20:
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
        self,
        daily_data: pd.DataFrame,
        method: str = "cross_validation",
        test_size: float = 0.2,
        random_state: int = 1000,
        cv_folds: int = 5,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
    ) -> Dict[str, Any]:
        """
        Train a decision tree model to predict occupancy status from daily activity data.

        This method supports multiple training approaches to ensure model robustness:
        - 'simple': Simple train-test split (sensitive to random_state)
        - 'cross_validation': K-fold cross-validation (more robust)
        - 'leave_one_out': Leave-one-out cross-validation (thorough but computationally intensive)
        - 'grid_search': Grid search with cross-validation (finds optimal hyperparameters)

        Args:
            daily_data: DataFrame containing daily aggregated video activity data
            method: Training method to use ('simple', 'cross_validation', 'leave_one_out', or 'grid_search')
            test_size: Proportion of data to use for testing (default: 0.2, used only for 'simple' method)
            random_state: Random seed for reproducibility (default: 42)
            cv_folds: Number of folds for cross-validation (default: 5, used for 'cross_validation' and 'grid_search')
            max_depth: Maximum depth of the decision tree (default: None, used for all methods)
            min_samples_split: Minimum samples required to split a node (default: 2, used for all methods)

        Returns:
            Dictionary containing model evaluation metrics
        """
        logger.info(f"Training occupancy model using {method} method...")

        # Prepare training data
        X, y, feature_names = self._prepare_training_data(daily_data)
        self.feature_names = feature_names

        # Initialize model with base parameters
        base_model = DecisionTreeClassifier(
            random_state=random_state,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
        )

        # Train model using the specified method
        if method == "simple":
            # Simple train-test split (original method)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            self.model = base_model
            self.model.fit(X_train, y_train)

            # Evaluate model
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(
                y_test,
                y_pred,
                target_names=["NOT_OCCUPIED", "OCCUPIED"],
                output_dict=True,
            )

            cv_scores = None
            best_params = None

        elif method == "cross_validation":
            # K-fold cross-validation
            kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            cv_scores = cross_val_score(base_model, X, y, cv=kf)

            # Train final model on all data
            self.model = base_model
            self.model.fit(X, y)

            # For reporting purposes
            accuracy = cv_scores.mean()
            report = None  # No test set for classification report
            best_params = None

        elif method == "leave_one_out":
            # Leave-one-out cross-validation
            loo = LeaveOneOut()
            cv_scores = cross_val_score(base_model, X, y, cv=loo)

            # Train final model on all data
            self.model = base_model
            self.model.fit(X, y)

            # For reporting purposes
            accuracy = cv_scores.mean()
            report = None  # No test set for classification report
            best_params = None

        elif method == "grid_search":
            # Grid search with cross-validation
            param_grid = {
                "max_depth": [None, 3, 4, 5],
                "min_samples_split": [2, 3, 5],
                "min_samples_leaf": [1, 2, 4],
                "criterion": ["gini", "entropy"],
            }

            grid_search = GridSearchCV(
                base_model,
                param_grid,
                cv=cv_folds,
                scoring="accuracy",
                n_jobs=-1,  # Use all available cores
            )

            grid_search.fit(X, y)

            # Get best model
            self.model = grid_search.best_estimator_

            # For reporting purposes
            accuracy = grid_search.best_score_
            report = None  # No test set for classification report
            best_params = grid_search.best_params_
            cv_scores = grid_search.cv_results_["mean_test_score"]

        else:
            raise ValueError(
                f"Unknown method: {method}. Choose from 'simple', 'cross_validation', 'leave_one_out', or 'grid_search'."
            )

        logger.info(f"Model trained with accuracy: {accuracy:.2f}")

        if cv_scores is not None:
            logger.info(f"Cross-validation scores: {cv_scores}")
            logger.info(
                f"Cross-validation mean: {cv_scores.mean():.2f}, std: {cv_scores.std():.2f}"
            )

        if best_params is not None:
            logger.info(f"Best parameters: {best_params}")

        # Update mode to ML_MODEL since we now have a trained model
        self.mode = OccupancyMode.ML_MODEL

        # Return evaluation metrics
        result = {
            "accuracy": accuracy,
            "feature_names": feature_names,
            "feature_importances": dict(
                zip(feature_names, self.model.feature_importances_)
            ),
        }

        if report is not None:
            result["classification_report"] = report

        if cv_scores is not None:
            result["cv_scores"] = cv_scores.tolist()
            result["cv_mean"] = cv_scores.mean()
            result["cv_std"] = cv_scores.std()

        if best_params is not None:
            result["best_params"] = best_params

        return result

    def save_occupancy_model(self, filepath: Optional[str] = None) -> None:
        """
        Save the trained model to a file.

        This method saves:
        1. The trained model using pickle (for accurate predictions)
        2. A human-readable representation of the model as JSON (for inspection)
        3. A visualization of the decision tree as a PNG image

        Args:
            filepath: Path to save the model. If None, uses the model_filepath from initialization.
        """
        if filepath is not None:
            self.model_filepath = filepath

        if self.model is None:
            raise ValueError(
                "No model has been trained. Call train_occupancy_model() first."
            )

        # Create directory if it doesn't exist
        os.makedirs(
            (
                os.path.dirname(self.model_filepath)
                if os.path.dirname(self.model_filepath)
                else "."
            ),
            exist_ok=True,
        )

        # Save the actual trained model using pickle
        pickle_filepath = f"{os.path.splitext(self.model_filepath)[0]}.pkl"
        with open(pickle_filepath, "wb") as f:
            pickle.dump((self.model, self.feature_names), f)

        # Export decision tree as text for human readability
        tree_text = export_text(self.model, feature_names=self.feature_names)

        # Create a dictionary with model information for human readability
        model_info = {
            "feature_names": self.feature_names,
            "feature_importances": dict(
                zip(self.feature_names, self.model.feature_importances_.tolist())
            ),
            "tree_text": tree_text,
            "model_params": self.model.get_params(),
            "classes": ["NOT_OCCUPIED", "OCCUPIED"],
            "pickle_filepath": pickle_filepath,  # Reference to the pickle file
        }

        # Save human-readable info to file
        with open(self.model_filepath, "w") as f:
            json.dump(model_info, f, indent=2)

        # Save visualization of the decision tree
        logger.info("Creating decision tree visualization...")
        plt.figure(figsize=(20, 10))
        plot_tree(
            self.model,
            feature_names=self.feature_names,
            class_names=["NOT_OCCUPIED", "OCCUPIED"],
            filled=True,
            rounded=True,
            fontsize=10,
        )
        plt.title("Occupancy Decision Tree", fontsize=14)

        # Save the visualization with the same base name as the model file
        tree_viz_filepath = f"{os.path.splitext(self.model_filepath)[0]}.png"
        plt.savefig(tree_viz_filepath, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(
            f"Model saved to {self.model_filepath}, {pickle_filepath}, and {tree_viz_filepath}"
        )

    def load_occupancy_model(self, filepath: Optional[str] = None) -> None:
        """
        Load a model from a file.

        This method loads the trained model from a pickle file for accurate predictions,
        and also loads the human-readable representation for inspection.

        Args:
            filepath: Path to the model file. If None, uses the model_filepath from initialization.
        """
        if filepath is not None:
            self.model_filepath = filepath

        # Load model information
        with open(self.model_filepath, "r") as f:
            model_info = json.load(f)

        # Extract feature names from the JSON file
        self.feature_names = model_info["feature_names"]

        # Load the actual trained model from the pickle file
        pickle_filepath = model_info.get("pickle_filepath")

        # If pickle_filepath is not in the model_info, try to infer it
        if pickle_filepath is None:
            pickle_filepath = f"{os.path.splitext(self.model_filepath)[0]}.pkl"

        if os.path.exists(pickle_filepath):
            with open(pickle_filepath, "rb") as f:
                self.model, loaded_feature_names = pickle.load(f)

            # Ensure feature names match
            if loaded_feature_names != self.feature_names:
                logger.warning(
                    "Feature names in pickle file don't match those in JSON file. "
                    "Using feature names from pickle file."
                )
                self.feature_names = loaded_feature_names

            logger.info(f"Model loaded from {pickle_filepath}")

            # Update mode to ML_MODEL since we now have a loaded model
            self.mode = OccupancyMode.ML_MODEL
        else:
            # Fallback to creating a new model with the same parameters
            logger.warning(
                f"Pickle file {pickle_filepath} not found. "
                "Creating a new model with the same parameters, but it won't have the same trained weights."
            )
            self.model = DecisionTreeClassifier(
                **{
                    k: v
                    for k, v in model_info["model_params"].items()
                    if k not in ["random_state"]
                }
            )

        logger.info(f"Model information loaded from {self.model_filepath}")
        logger.info(f"Feature importances: {model_info['feature_importances']}")
        logger.info(f"Decision tree structure:\n{model_info['tree_text']}")

    def _set_occupancy_status_from_model(self, daily_data: pd.DataFrame) -> None:
        """
        Apply the trained model to daily activity data to determine occupancy status.

        This method clears the occupancy_cache before populating it with new statuses,
        ensuring that the occupancy status is determined solely by the model predictions.

        Args:
            daily_data: DataFrame containing daily aggregated video activity data
        """
        if self.model is None:
            raise ValueError(
                "No model has been trained or loaded. Call train_occupancy_model() or load_occupancy_model() first."
            )

        logger.info("Applying model to determine occupancy status...")

        # Clear the cache before populating with new statuses
        self.occupancy_cache.clear()

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
    from video_filter import DateRange, VideoFilter, VideoSelector

    def write_csv(all_occupancy_data, output_file):
        # Save to CSV file
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

    set_logger_level_and_format(logger, level=logging.DEBUG, extended_format=True)

    # Load video database
    root_database = "/Users/jbouguet/Documents/EufySecurityVideos/record/"
    metadata_files = [
        os.path.join(root_database, "archive_000.csv"),
        os.path.join(root_database, "archive_001.csv"),
        os.path.join(root_database, "backup.csv"),
    ]
    out_dir: str = "/Users/jbouguet/Documents/EufySecurityVideos/stories"

    logger.info("Loading video database...")
    video_database = VideoDatabaseList(
        [
            VideoDatabase(video_directories=None, video_metadata_file=file)
            for file in metadata_files
        ]
    ).load_videos()

    # selector = VideoSelector(
    #    date_range=DateRange(start="2024-08-25"),
    # )
    # video_database = VideoFilter.by_selectors(video_database, selector)

    # Check if database is empty
    if not video_database:
        logger.error("No videos found in database")
        sys.exit(1)
    logger.debug(f"Number of videos: {len(video_database)}")

    # Compute daily activity data
    logger.info("Computing daily activity data...")
    metric = "activity"
    data_aggregator = VideoDataAggregator(metrics=[metric])
    daily_data, _ = data_aggregator.run(video_database)
    daily_data = daily_data[metric]
    important_columns = [
        "Date",
        "Front Door",  # 404/414
        "Backyard",  # 409/414
        "Gateway",  # 409/414
        "Walkway",  # 409/414
    ]  # Best model
    filtered_columns = [col for col in daily_data.columns if col in important_columns]
    daily_data = daily_data[filtered_columns]

    # Create occupancy status using different modes
    logger.info("Analyzing occupancy...")

    # Manual Calendar
    write_csv(
        Occupancy(mode=OccupancyMode.CALENDAR).get_all_dates_with_status(),
        os.path.join(out_dir, "daily_occupancies_calendar.csv"),
    )

    # Heuristic computed from daily activity data
    write_csv(
        Occupancy(
            mode=OccupancyMode.HEURISTIC, daily_data=daily_data
        ).get_all_dates_with_status(),
        os.path.join(out_dir, "daily_occupancies_heuristic.csv"),
    )

    # ML Model:
    model_file = "occupancy_model.txt"

    # Train and save a new model from daily activity data
    occupancy_ml_train = Occupancy()
    # Leave-one-out cross-validation
    params = {
        "random_state": 1000,
        "max_depth": 3,
    }
    occupancy_ml_train.train_occupancy_model(
        daily_data, method="leave_one_out", **params
    )
    occupancy_ml_train.save_occupancy_model(model_file)
    occupancy_ml_train.set_occupancy_status_from_daily_activity(daily_data)
    write_csv(
        occupancy_ml_train.get_all_dates_with_status(),
        os.path.join(out_dir, "daily_occupancies_ml_train.csv"),
    )

    # Load an existing ML model and evalaute it on the daily activity data
    write_csv(
        Occupancy(
            mode=OccupancyMode.ML_MODEL,
            daily_data=daily_data,
            model_filepath=model_file,
        ).get_all_dates_with_status(),
        os.path.join(out_dir, "daily_occupancies_ml_eval.csv"),
    )
