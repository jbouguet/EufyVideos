#!/usr/bin/env python3

import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Union

import yaml
from termcolor import colored

from config import Config
from dashboard import Dashboard
from logging_config import create_logger
from story_creator import Story
from tag_processor import VideoTags
from video_database import VideoDatabase, VideoDatabaseList
from video_metadata import VideoMetadata

logger = create_logger(__name__)


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


def load_yaml_with_substitution(file_path):
    with open(file_path, "r") as f:
        return yaml.load(f, Loader=JoinPathsLoader)


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
        logger.info(f"Loading configuration from file: {config_file}")
        config_dict = load_yaml_with_substitution(config_file)

        if "directories" in config_dict:
            logger.debug(f"directories: {config_dict["directories"]}")

        if "subdirs" in config_dict:
            logger.debug(f"subdirs: {config_dict["subdirs"]}")

        if "video_database_list" in config_dict:
            for idx, db in enumerate(config_dict["video_database_list"]):
                logger.debug(f"Database {idx + 1}:")
                if "video_metadata_file" in db:
                    logger.debug(f"  metadata_file: {db['video_metadata_file']}")
                    logger.debug(
                        f"  file exists: {os.path.exists(db['video_metadata_file'])}"
                    )
                if "video_directories" in db:
                    if isinstance(db["video_directories"], list):
                        logger.debug("  directories:")
                        for dir in db["video_directories"]:
                            logger.debug(f"    - {dir}")
                            logger.debug(
                                f"    - directory exists: {os.path.exists(dir)}"
                            )
                    else:
                        logger.debug(f"  directory: {db['video_directories']}")
                        logger.debug(
                            f"  directory exists: {os.path.exists(db['video_directories'])}"
                        )
        if "tag_database_files" in config_dict:
            logger.debug("Tag Database:")
            for tag_file in config_dict["tag_database_files"]:
                logger.debug(f"    - {tag_file}")
                logger.debug(f"    - exists: {os.path.exists(tag_file)}")

        if "output_directory" in config_dict:
            logger.debug(f"Output Dir: {config_dict["output_directory"]}")
            logger.debug(
                f"  directory exists: {os.path.exists(config_dict["output_directory"])}"
            )

        return cls.from_dict(config_dict)

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
            logger.info(f"Tag Database: {self.tags_database.stats}")

    @staticmethod
    def _load_videos_database(
        video_database_list: Union[VideoDatabase, VideoDatabaseList]
    ) -> List[VideoMetadata]:
        corrupted_files: List[str] = []
        videos_database = video_database_list.load_videos(corrupted_files)
        for file in corrupted_files:
            logger.warning(f"    - {file}")
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
            logger.debug(f"{tags.stats} loaded from {tag_file}")
            tags_database.merge(tags)

        logger.debug(f"{tags_database.stats} before duplicates removal")
        tags_database.dedupe()
        logger.debug(f"{tags_database.stats} after duplicates removal")

        return tags_database

    def _log_statistics(self) -> None:
        num_videos: int = len(self.videos_database)
        num_frames: int = sum(video.frame_count for video in self.videos_database)
        total_duration_seconds: float = sum(
            video.duration.total_seconds() for video in self.videos_database
        )
        average_fps = num_frames / total_duration_seconds
        total_size_mb: float = sum(video.file_size for video in self.videos_database)

        tags_stats = self.tags_database.stats
        num_tagged_videos = tags_stats["num_tagged_videos"]
        num_tagged_frames = tags_stats["num_tagged_frames"]
        num_tags = tags_stats["num_tags"]

        logger.info(f"{colored("Database Statistics:","light_cyan")}")
        logger.info(f"  - Number of videos        = {num_videos:,}")
        logger.info(f"  - Number of frames        = {num_frames:,}")
        logger.info(f"  - Size                    = {total_size_mb:,.3f} MB")
        logger.info(
            f"  - Duration                = {total_duration_seconds / 60:,.3f} minutes"
        )
        logger.info(f"  - Average FPS             = {average_fps:.3f}")
        logger.info(f"  - Number of tagged videos = {num_tagged_videos:,}")
        logger.info(f"  - Number of tagged frames = {num_tagged_frames:,}")
        logger.info(f"  - Number of tags in total = {num_tags:,}")

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
        dashboard = Dashboard(config={"bins_per_hour": 4})
        dashboard.create_graphs_file(self.videos_database, graphs_filename)

    def _process_stories(self) -> None:
        if self.config.process_stories and self.config.stories:
            for story in self.config.stories:
                story.process(self.videos_database, self.config.output_directory)


if __name__ == "__main__":
    import argparse
    import logging

    from logging_config import set_all_loggers_level_and_format

    parser = argparse.ArgumentParser(description="Run video analysis")
    parser.add_argument(
        "--config",
        default="analysis_config.yaml",
        help="Path to the configuration file",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--extended_format", action="store_true", help="Enable extended logging format"
    )
    args = parser.parse_args()

    set_all_loggers_level_and_format(
        logging.DEBUG if args.debug else logging.INFO,
        args.debug or args.extended_format,
    )

    if not os.path.exists(args.config):
        logger.error(f"Configuration file {args.config} not found.")
        exit(1)

    VideoAnalyzer.run(args.config)
