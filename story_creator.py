#!/usr/bin/env python3

"""
Story Creator Module

This module provides the Story class, which serves as a sophisticated container for managing
and processing video slices from a larger video database. It enables systematic selection,
analysis, visualization, and processing of video segments based on various criteria.

Key Features:
- Video Selection: Filter videos based on devices, filenames, dates, etc.
- Tag Processing: Analyze video content and generate tags
- Video Generation: Create composite videos from selected segments
- Statistics Generation: Produce detailed analytics and visualizations
- Tag Visualization: Generate visual representations of tagged frames

Example Usage:
    # Create a story from a YAML configuration
    story = Story.from_file("story_config.yaml")

    # Process the story with a video database
    story.process(videos_database, "output_directory")

    # Example YAML configuration:
    '''
    name: "Morning Activity"
    selectors:
      - devices: ["camera1", "camera2"]
        start_time: "07:00"
        end_time: "09:00"
    video_generation: true
    tag_processing: true
    '''
"""

import os
from dataclasses import dataclass, field
from math import ceil
from typing import Any, Dict, List, Optional

import yaml
from termcolor import colored

from config import Config
from dashboard import Dashboard
from logging_config import create_logger
from occupancy import Occupancy
from tag_processor import TaggerConfig, TagProcessor, VideoTags
from tag_visualizer import TagVisualizer, TagVisualizerConfig
from video_filter import VideoFilter, VideoSelector
from video_generator import VideoGenerationConfig, VideoGenerator
from video_metadata import VideoMetadata
from video_scatter_plots_creator import PlotCreator
from yaml_utils import save_to_yaml, set_exclude_from_dict

logger = create_logger(__name__)


@dataclass
class Story:
    """
    A container class for managing and processing video segments.

    Attributes:
        name (str): Unique identifier for the story
        skip (bool): Flag to skip processing this story
        selectors (List[VideoSelector]): Criteria for selecting videos
        occupancy_status (Optional[List[str]]): List of occupancy statuses to include
            (e.g., ["OCCUPIED", "NOT_OCCUPIED", "UNKNOWN"])
        video_generation (bool): Whether to create a composite video
        video_generation_config (VideoGenerationConfig): Settings for video generation
        tag_processing (bool): Whether to analyze and generate tags
        tag_processing_config (TaggerConfig): Settings for tag processing
        tag_video_generation (bool): Whether to create a visualization of tagged frames
        tag_video_generation_config (TagVisualizerConfig): Settings for tag visualization
    """

    name: str
    skip: bool = False
    selectors: Optional[List[VideoSelector]] = None
    occupancy_status: Optional[List[str]] = field(default=None)
    video_generation: bool = False
    video_generation_config: VideoGenerationConfig = field(
        default_factory=VideoGenerationConfig
    )
    tag_processing: bool = False
    tag_processing_config: TaggerConfig = field(default_factory=TaggerConfig)
    tag_video_generation: bool = False
    tag_video_generation_config: TagVisualizerConfig = field(
        default_factory=TagVisualizerConfig
    )

    @property
    def devices(self) -> List[str]:
        """
        Get a unique list of devices from all selectors.

        Returns:
            List[str]: Unique list of device identifiers
        """
        if not self.selectors:
            return []
        devices = set()
        for selector in self.selectors:
            if selector.devices:
                devices.update(selector.devices)
        return list(devices)

    @classmethod
    def from_dict(cls, story_dict: Dict[str, Any]) -> "Story":
        """
        Create a Story instance from a dictionary configuration.

        Args:
            story_dict: Dictionary containing story configuration

        Returns:
            Story: New Story instance

        Raises:
            ValueError: If required fields are missing or invalid
        """
        Story.validate_story_dict(story_dict)
        selectors = None
        if "selectors" in story_dict and story_dict["selectors"]:
            selectors = []
            for selector_dict in story_dict["selectors"]:
                selectors.append(VideoSelector.from_dict(selector_dict))

        # Handle occupancy_status field
        occupancy_status = story_dict.get("occupancy_status")

        return cls(
            name=story_dict["name"],
            selectors=selectors,
            occupancy_status=occupancy_status,
            video_generation=story_dict.get("video_generation", False),
            video_generation_config=VideoGenerationConfig.from_dict(
                story_dict.get("video_generation_config", {})
            ),
            tag_video_generation_config=TagVisualizerConfig.from_dict(
                story_dict.get("tag_video_generation_config", {})
            ),
            skip=story_dict.get("skip", False),
            tag_processing=story_dict.get("tag_processing", False),
            tag_video_generation=story_dict.get("tag_video_generation", False),
            tag_processing_config=cls._create_tag_processing_config(
                story_dict.get("tag_processing_config", {})
            ),
        )

    @classmethod
    def _create_tag_processing_config(cls, config_dict: Dict[str, Any]) -> TaggerConfig:
        """
        Create tag processing configuration with person recognition support.
        
        Args:
            config_dict: Dictionary containing tag processing configuration
            
        Returns:
            TaggerConfig with person recognition settings if enabled
        """
        if config_dict.get("enable_person_recognition", False):
            logger.info("Person recognition enabled in config - configuring TaggerConfig with person recognition")
        # Note: Standard tag processing doesn't need logging for every story
        return TaggerConfig.from_dict(config_dict)

    @classmethod
    def from_file(cls, story_filename: str) -> "Story":
        """
        Create a Story instance from a YAML file.

        Args:
            story_filename: Path to YAML configuration file

        Returns:
            Story: New Story instance
        """
        with open(story_filename, "r") as f:
            return cls.from_dict(yaml.safe_load(f))

    def to_file(self, story_filename: str) -> None:
        """
        Save the story configuration to a YAML file.

        Args:
            story_filename: Path where to save the configuration
        """
        set_exclude_from_dict(
            self, "video_generation_config", not self.video_generation
        )
        set_exclude_from_dict(self, "tag_processing_config", not self.tag_processing)
        set_exclude_from_dict(
            self, "tag_video_generation_config", not self.tag_video_generation
        )
        save_to_yaml(self, story_filename)

    @staticmethod
    def validate_story_dict(story_dict: Dict[str, Any]) -> None:
        """
        Validate story configuration dictionary.

        Performs comprehensive validation of all story settings to ensure
        the configuration is complete and consistent.

        Args:
            story_dict: Dictionary containing story configuration

        Raises:
            ValueError: If validation fails
        """
        if not story_dict.get("skip", False):
            required_story_fields = ["name"]
            for field in required_story_fields:
                if field not in story_dict:
                    raise ValueError(
                        f"Missing required field in story configuration: {field}"
                    )

        # Validate name
        if not story_dict["name"].strip():
            raise ValueError("Story name cannot be empty or just whitespace")

        # Validate selectors
        if "selectors" in story_dict:
            if not isinstance(story_dict["selectors"], list):
                raise ValueError(
                    f"'selectors' must be a list in story: {story_dict['name']}"
                )

            for selector in story_dict["selectors"]:
                if "devices" in selector and not isinstance(
                    selector["devices"], (list, str)
                ):
                    raise ValueError(
                        f"'devices' must be a list or a string in a selector of story: {story_dict['name']}"
                    )
                if "filenames" in selector and not isinstance(
                    selector["filenames"], (list, str)
                ):
                    raise ValueError(
                        f"'filenames' must be a list or a string in a selector of story: {story_dict['name']}"
                    )

        # Validate occupancy_status
        if "occupancy_status" in story_dict:
            occupancy_status = story_dict["occupancy_status"]
            if not isinstance(occupancy_status, list):
                raise ValueError(
                    f"'occupancy_status' must be a list in story: {story_dict['name']}"
                )
            for status in occupancy_status:
                if not isinstance(status, str):
                    raise ValueError(
                        f"Each occupancy status must be a string in story: {story_dict['name']}"
                    )

        # Validate tag_processing
        if "tag_processing" in story_dict and not isinstance(
            story_dict["tag_processing"], bool
        ):
            raise ValueError(
                f"'tag_processing' must be a boolean value in story: {story_dict['name']}"
            )

        # Validate video generation settings
        if story_dict.get("video_generation", False):
            all_devices = set()
            if "selectors" in story_dict:
                for selector in story_dict["selectors"]:
                    devices = selector.get("devices", Config.get_all_devices())
                    if isinstance(devices, str):
                        devices = [devices]
                    all_devices.update(devices)
            else:
                all_devices.update(Config.get_all_devices())

            if len(all_devices) > 1:
                video_generation_config = story_dict.get("video_generation_config", {})
                input_fragments = video_generation_config.get("input_fragments", {})
                enforce_16_9_aspect_ratio = input_fragments.get(
                    "enforce_16_9_aspect_ratio", False
                )

                if not enforce_16_9_aspect_ratio:
                    logger.warning(
                        f"Story '{story_dict['name']}' has multiple devices but does not explicitly "
                        f"enforce a 16:9 aspect ratio for all raw video input streams. This may cause "
                        f"devices with inconsistent aspect ratio image sources to be accidentally "
                        f"skipped when generating combined videos. It is strongly recommended to set "
                        f"enforce_16_9_aspect_ratio to true for consistent image sizes in composite "
                        f"video creation."
                    )

    def select_videos(
        self, videos_database: List[VideoMetadata]
    ) -> List[VideoMetadata]:
        """
        Select videos from the database based on story selectors and occupancy status.

        Also generates and logs comprehensive statistics about the selected videos.

        Args:
            videos_database: List of all available videos

        Returns:
            List[VideoMetadata]: Selected videos matching criteria
        """
        videos = VideoFilter.by_selectors(videos_database, self.selectors)

        # Filter by occupancy status if specified
        if self.occupancy_status is not None and videos:
            logger.info(f"Filtering by occupancy status: {self.occupancy_status}")
            occupancy = Occupancy()

            # Filter videos based on occupancy status
            filtered_videos = []
            for video in videos:
                date_str = video.date.strftime("%Y-%m-%d")
                status = occupancy.status(date_str)
                if status.value in self.occupancy_status:
                    filtered_videos.append(video)

            logger.info(
                f"Filtered from {len(videos)} to {len(filtered_videos)} videos based on occupancy status"
            )
            videos = filtered_videos

        video_tags = VideoTags.from_videos(videos)

        # Calculate statistics
        num_videos = len(videos)
        num_frames = sum(video.frame_count for video in videos)
        total_duration_seconds = sum(video.duration.total_seconds() for video in videos)
        total_size_mb: float = sum(video.file_size for video in videos)
        average_fps = (
            num_frames / total_duration_seconds if total_duration_seconds > 0 else 0
        )
        tags_stats = video_tags.stats
        num_tagged_videos = tags_stats["num_tagged_videos"]
        num_tagged_frames = tags_stats["num_tagged_frames"]
        num_tags = tags_stats["num_tags"]

        # Log statistics
        logger.info(f"{colored("Selectors:","light_cyan")}")
        VideoSelector.log(self.selectors)

        # Log occupancy status if specified
        if self.occupancy_status is not None:
            logger.info(
                f"{colored("Occupancy Status:","light_cyan")} {', '.join(self.occupancy_status)}"
            )

        logger.info(f"{colored("Statistics:","light_cyan")}")

        logger.info(f"  - {'Number of videos':<23} = {num_videos:,}")
        logger.info(f"  - {'Number of frames':<23} = {num_frames:,}")
        logger.info(f"  - {'Size':<23} = {total_size_mb:,.3f} MB")
        logger.info(
            f"  - {'Duration':<23} = {total_duration_seconds / 60:,.3f} minutes"
        )
        logger.info(f"  - {'Average FPS':<23} = {average_fps:.3f}")
        logger.info(f"  - {'Number of tagged videos':<23} = {num_tagged_videos:,}")
        logger.info(f"  - {'Number of tagged frames':<23} = {num_tagged_frames:,}")
        logger.info(f"  - {'Number of tags in total':<23} = {num_tags:,}")

        return videos

    def generate_statistics_files(
        self, videos: List[VideoMetadata], output_directory: str
    ) -> None:
        """
        Generate and save various statistics and analytics files.

        Creates configuration files, metadata exports, playlists, and visualization graphs.

        Args:
            videos: List of videos to analyze
            output_directory: Directory to save output files
        """
        os.makedirs(output_directory, exist_ok=True)

        # Define output files
        config_filename = os.path.join(output_directory, f"{self.name}{Config.CONFIG}")
        video_metadata_file = os.path.join(
            output_directory, f"{self.name}{Config.METADATA}"
        )
        playlist_filename = os.path.join(
            output_directory, f"{self.name}{Config.PLAYLIST}"
        )
        graphs_filename = os.path.join(output_directory, f"{self.name}{Config.GRAPHS}")
        scatter_plots_filename = os.path.join(
            output_directory, f"{self.name}{Config.SCATTER_PLOTS}"
        )
        # Generate and save files
        self.to_file(config_filename)
        VideoMetadata.export_videos_to_metadata_file(videos, video_metadata_file)
        VideoMetadata.export_videos_to_playlist_file(videos, playlist_filename)

        logger.info(f"{colored("Output Files:","light_cyan")}")
        logger.info(f"  - config file:        {config_filename}")
        logger.info(f"  - metadata file:      {video_metadata_file}")
        logger.info(f"  - playlist file:      {playlist_filename}")

        if len(videos) > 0:
            Dashboard().create_graphs_file(videos, graphs_filename)
            PlotCreator.create_graphs_file(videos, scatter_plots_filename)
            logger.info(f"  - graphs file:        {graphs_filename}")
            logger.info(f"  - scatter plots file: {scatter_plots_filename}")

    def process_new_tags(
        self, videos: List[VideoMetadata], output_directory: str
    ) -> None:
        """
        Process and manage video tags.

        Handles tag generation, loading existing tags, and tag export operations.

        Args:
            videos: List of videos to process
            output_directory: Directory to save tag files
        """

        logger.info(f"{colored("Tag processing:","light_cyan")}")

        tag_filename = os.path.join(
            output_directory,
            f"{self.name}_{self.tag_processing_config.get_identifier()}{Config.TAGS}",
        )
        video_tags = VideoTags()

        if self.tag_processing:
            # Process new tags
            tagging_frame_rate = self.tag_processing_config.num_frames_per_second
            num_tagged_frames = sum(
                ceil(video.duration.total_seconds() * tagging_frame_rate)
                for video in videos
            )
            logger.info(
                f"Processing tags using {self.tag_processing_config.model} in {self.tag_processing_config.task} mode "
                f"at {tagging_frame_rate}fps with a confidence threshold of {self.tag_processing_config.conf_threshold}"
            )
            logger.info(f"Number of frames to be tagged: {num_tagged_frames:,}")
            
            # Use TagProcessor for all tag processing (includes person recognition if enabled)
            logger.info("Using TagProcessor for video analysis")
            tag_processor = TagProcessor(self.tag_processing_config)
            video_tags = tag_processor.run(videos).to_file(tag_filename)
            logger.info(
                f"{video_tags.stats} newly computed tags are saved to {tag_filename}."
            )

        elif os.path.exists(tag_filename):
            # Load existing tags
            video_tags = VideoTags.from_file(tag_filename)
            logger.info(f"{video_tags.stats} loaded from {tag_filename}")

        else:
            logger.info("No new tag computed or loaded")

        if video_tags.stats["num_tags"] > 0:
            # Export tags to videos by first creating a complete set of merged and deduped tags.
            merged_tags = video_tags.merge(VideoTags.from_videos(videos))
            merged_tags.dedupe().to_videos(videos, "REPLACE")
            tags_stats = VideoTags.from_videos(videos).stats
            logger.info(f"Tags statistics after merge of new tags: {tags_stats}")

    def process(
        self, videos_database: List[VideoMetadata], output_directory: str
    ) -> None:
        """
        Main processing method for the story.

        Orchestrates the entire story processing workflow including:
        - Video selection
        - Statistics generation
        - Tag processing
        - Video generation
        - Tag visualization

        Args:
            videos_database: Complete database of available videos
            output_directory: Directory for all output files
        """
        if self.skip:
            return

        logger.info("")
        logger.info(
            f"{colored("Story:", "light_yellow")} {colored(self.name, "light_yellow")}"
        )

        # Select relevant videos
        videos = self.select_videos(videos_database)

        # Generate statistics and analytics
        self.generate_statistics_files(videos, output_directory)

        if len(videos) == 0:
            return

        # Process video tags
        self.process_new_tags(videos, output_directory)

        # Generate tag visualization video if requested
        if self.tag_video_generation:
            video_tag_file = os.path.join(
                output_directory, f"{self.name}{Config.MOVIE_TAGS}"
            )
            logger.info(
                f"Generating the video {video_tag_file} of all of the tagged frames"
            )
            tag_visualizer = TagVisualizer(self.tag_video_generation_config)
            tag_visualizer.run(videos, video_tag_file)
            logger.info(f"Video saved to {video_tag_file}")

        # Generate composite video if requested
        if self.video_generation:
            video_file = os.path.join(output_directory, f"{self.name}{Config.MOVIE}")
            fragment_duration = (
                self.video_generation_config.input_fragments.duration_in_seconds
            )
            total_video_duration = fragment_duration * len(videos)
            logger.info(
                f"Generating a composite video combining {fragment_duration} seconds long snippets "
                f"of all of the videos. Approximate total video duration: {total_video_duration} seconds."
            )
            video_generator = VideoGenerator(self.video_generation_config)
            video_generator.run(videos, video_file)
            logger.info(f"Video saved to {video_file}")
