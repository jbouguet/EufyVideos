#!/usr/bin/env python3

import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Union

import yaml
from termcolor import colored

from config import Config
from logging_config import create_logger
from story_creator import Story
from tag_processor import VideoTags
from video_metadata import VideoDatabase, VideoDatabaseList, VideoMetadata
from video_visualizer import VideoVisualizer

logger = create_logger(__name__)


class VariableSubstitutionLoader(yaml.SafeLoader):
    def __init__(self, stream):
        super().__init__(stream)
        self.variables = {}

    def construct_scalar(self, node):
        value = super().construct_scalar(node)
        if isinstance(value, str):
            return self.substitute_variables(value)
        return value

    def substitute_variables(self, value):
        if "${" in value:
            for var_name, var_value in self.variables.items():
                value = value.replace(f"${{{var_name}}}", var_value)
        return value

    def construct_mapping(self, node, deep=False):
        mapping = super().construct_mapping(node, deep=deep)
        for key, value in mapping.items():
            if isinstance(value, str) and value.startswith("&"):
                self.variables[value[1:]] = mapping[key]
        return mapping


def load_yaml_with_substitution(file_path):
    with open(file_path, "r") as file:
        loader = VariableSubstitutionLoader(file)
        try:
            return yaml.load(file, Loader=loader)
        except yaml.YAMLError as exc:
            print(f"Error in configuration file: {exc}")


class JoinPathsLoader(yaml.SafeLoader):
    """Custom YAML Loader that handles path joining with variable resolution"""

    def __init__(self, stream):
        super().__init__(stream)
        self.add_constructor("!join", self._join_paths)

    def _join_paths(self, loader, node):
        """Join paths while resolving aliases"""
        logger.debug(f"Starting path join operation with node type: {type(node)}")
        try:
            # For sequence nodes (multiple path components)
            if isinstance(node, yaml.SequenceNode):
                logger.debug(
                    f"Processing sequence node with {len(node.value)} elements"
                )
                resolved_paths = []
                for i, path_node in enumerate(node.value):
                    path = loader.construct_object(path_node)
                    logger.debug(f"Raw path component {i}: {path}")

                    # Convert to string and ensure proper formatting
                    path_str = str(path)
                    if i > 0 and path_str.startswith("/"):
                        # Remove leading slash from all but first component
                        path_str = path_str.lstrip("/")
                        logger.debug(f"Formatted path component {i}: {path_str}")

                    resolved_paths.append(path_str)

                result = os.path.join(*resolved_paths)
                logger.debug(f"Final joined path: {result}")
                return result

            # For scalar nodes (single path)
            elif isinstance(node, yaml.ScalarNode):
                result = loader.construct_scalar(node)
                logger.debug(f"Processing scalar node, value: {result}")
                return result
            else:
                error_msg = f"Invalid node type for !join tag: {type(node)}"
                logger.error(error_msg)
                raise yaml.constructor.ConstructorError(
                    None, None, error_msg, node.start_mark
                )
        except Exception as e:
            logger.error(f"Error in _join_paths: {str(e)}")
            raise


@dataclass
class AnalysisConfig:
    video_database_list: VideoDatabaseList
    output_directory: str
    tag_database_files: List[str] = field(default_factory=list)
    process_stories: bool = True
    stories: List[Story] = field(default_factory=list)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "AnalysisConfig":
        required_elements = ["video_database_list", "output_directory"]
        for element in required_elements:
            if element not in config_dict:
                raise ValueError(
                    f"Missing required element in configuration: {element}"
                )
        if "tag_database_files" in config_dict:
            if not isinstance(config_dict["tag_database_files"], list):
                raise ValueError("'tag_database_files' must be a list of file paths")
            for file_path in config_dict["tag_database_files"]:
                if not os.path.isfile(file_path):
                    logger.warning(f"Tag file not found: {file_path}")
        stories = []
        for story_data in config_dict.get("stories", []):
            stories.append(Story.from_dict(story_data))
        return cls(
            video_database_list=VideoDatabaseList.from_dict(
                config_dict["video_database_list"]
            ),
            output_directory=config_dict["output_directory"],
            tag_database_files=config_dict.get("tag_database_files", []),
            process_stories=config_dict.get("process_stories", True),
            stories=stories,
        )

    @classmethod
    def from_file(cls, config_file: str) -> "AnalysisConfig":
        try:
            logger.debug(f"Loading configuration from file: {config_file}")
            config_dict = load_yaml_with_substitution(config_file)

            # with open(config_file, "r") as f:
            #     config_dict = yaml.load(f, Loader=JoinPathsLoader)

            # Enhanced logging for debugging paths
            logger.debug("Raw loaded configuration:")
            if "video_database_list" in config_dict:
                for idx, db in enumerate(config_dict["video_database_list"]):
                    logger.debug(f"Database {idx + 1}:")
                    if "video_metadata_file" in db:
                        logger.debug(f"  metadata_file: {db['video_metadata_file']}")
                        logger.debug(
                            f"  metadata_file exists: {os.path.exists(db['video_metadata_file'])}"
                        )
                    if "video_directories" in db:
                        if isinstance(db["video_directories"], list):
                            logger.debug("  directories:")
                            for dir in db["video_directories"]:
                                logger.debug(f"    - {dir}")
                                logger.debug(f"    - exists: {os.path.exists(dir)}")
                        else:
                            logger.debug(f"  directory: {db['video_directories']}")
                            logger.debug(
                                f"  directory exists: {os.path.exists(db['video_directories'])}"
                            )

                if "directories" in config_dict:
                    del config_dict["directories"]

                return cls.from_dict(config_dict)
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML configuration: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise

    def to_file(self, config_file: str) -> None:
        with open(config_file, "w") as f:
            yaml.dump(asdict(self), f)


class VideoAnalyzer:
    def __init__(self, config: AnalysisConfig) -> None:
        self.config: AnalysisConfig = config
        self.videos_database: List[VideoMetadata] = []
        self.tags_database: VideoTags = None

    @staticmethod
    def run(config_filename: str) -> None:
        config: AnalysisConfig = AnalysisConfig.from_file(config_filename)
        video_analyzer = VideoAnalyzer(config)
        video_analyzer._load_all_databases()
        video_analyzer._log_statistics()
        video_analyzer._export_files()
        video_analyzer._process_stories()

    def _load_all_databases(self) -> None:
        logger.info(f"{colored("Loading Databases", "light_yellow")}")
        self.videos_database = VideoAnalyzer._load_videos_database(
            self.config.video_database_list
        )
        self.tags_database = (
            VideoAnalyzer._load_tags_database(self.config.tag_database_files)
            if self.config.tag_database_files
            else None
        )
        if self.tags_database:
            self.tags_database.to_videos(self.videos_database)
            logger.info(f"Unique tags exported to videos: {self.tags_database.stats}")

    @staticmethod
    def _load_videos_database(
        video_database_list: Union[VideoDatabase, VideoDatabaseList]
    ) -> List[VideoMetadata]:
        corrupted_files: List[str] = []
        videos_database = video_database_list.load_videos(corrupted_files)
        for file in corrupted_files:
            logger.info(f"    - {file}")
        return videos_database

    @staticmethod
    def _load_tags_database(tag_database_files: Union[str, List[str]]) -> VideoTags:
        tags_database = VideoTags.from_tags(tags={})
        tag_database_files = (
            [tag_database_files]
            if isinstance(tag_database_files, str)
            else tag_database_files
        )
        for tag_file in tag_database_files:
            tags: VideoTags = VideoTags.from_file(tag_file)
            logger.info(f"{tags.stats} loaded from {tag_file}")
            tags_database.merge(tags)

        tags_database.remove_duplicates()

        return tags_database

    def _log_statistics(self) -> None:
        num_videos: int = len(self.videos_database)
        num_frames: int = sum(video.frame_count for video in self.videos_database)
        total_duration_seconds: float = sum(
            video.duration.total_seconds() for video in self.videos_database
        )
        average_fps = num_frames / total_duration_seconds
        total_size_gb: float = (
            sum(video.file_size for video in self.videos_database) / 1024
        )

        tags_stats = self.tags_database.stats
        num_tagged_videos = tags_stats["num_tagged_videos"]
        num_tagged_frames = tags_stats["num_tagged_frames"]
        num_tags = tags_stats["num_tags"]

        logger.info(f"{colored("Database Statistics:","light_cyan")}")
        logger.info(f"  - Number of videos = {num_videos}")
        logger.info(f"  - Number of frames = {num_frames}")
        logger.info(f"  - Size = {total_size_gb:.3f}GB")
        logger.info(f"  - Duration = {total_duration_seconds} seconds")
        logger.info(f"  - Average FPS = {average_fps:.3f}")
        logger.info(f"  - Number of tagged videos = {num_tagged_videos}")
        logger.info(f"  - Number of tagged frames = {num_tagged_frames}")
        logger.info(f"  - Number of tags in total = {num_tags}")

    def _export_files(self) -> None:
        os.makedirs(self.config.output_directory, exist_ok=True)

        config_filename: str = os.path.join(
            self.config.output_directory, f"{Config.CONFIG}"
        )
        video_metadata_file: str = os.path.join(
            self.config.output_directory, Config.METADATA
        )
        playlist_filename: str = os.path.join(
            self.config.output_directory, f"{Config.PLAYLIST}"
        )
        graphs_filename: str = os.path.join(
            self.config.output_directory,
            f"{Config.GRAPHS}",
        )

        logger.info(f"{colored("Output Files:","light_cyan")}")
        logger.info(f"  - config file:   {config_filename}")
        self.config.to_file(config_filename)
        logger.info(f"  - metadata file: {video_metadata_file}")
        VideoMetadata.export_videos_to_metadata_file(
            self.videos_database, video_metadata_file
        )
        logger.info(f"  - playlist file: {playlist_filename}")
        VideoMetadata.export_videos_to_playlist_file(
            self.videos_database, playlist_filename
        )
        logger.info(f"  - graphs file:   {graphs_filename}")
        visualizer = VideoVisualizer()
        visualizer.create_graphs_file(self.videos_database, graphs_filename)

    def _process_stories(self) -> None:
        if self.config.process_stories and self.config.stories:
            for story in self.config.stories:
                story.process(self.videos_database, self.config.output_directory)


if __name__ == "__main__":
    import argparse
    import logging

    parser = argparse.ArgumentParser(description="Run video analysis")
    parser.add_argument(
        "--config",
        default="analysis_config.yaml",
        help="Path to the configuration file",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)
        for handler in logger.handlers:
            handler.setLevel(logging.DEBUG)

    if not os.path.exists(args.config):
        logger.error(f"Configuration file {args.config} not found.")
        exit(1)

    VideoAnalyzer.run(args.config)