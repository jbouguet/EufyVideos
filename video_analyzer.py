#!/usr/bin/env python3

import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Union

import yaml
from termcolor import colored

from config import Config
from logging_config import setup_logger
from story_creator import Story
from tag_processor import VideoTags
from video_metadata import VideoDatabase, VideoDatabaseList, VideoMetadata
from video_visualizer import VideoVisualizer

logger = setup_logger(__name__)


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
        with open(config_file, "r") as f:
            return cls.from_dict(yaml.safe_load(f))

    def to_file(self, config_file: str) -> None:
        with open(config_file, "w") as f:
            yaml.dump(asdict(self), f)


class VideoAnalyzer:
    def __init__(self, config: AnalysisConfig) -> None:
        self.config: AnalysisConfig = config
        self.videos_database: List[VideoMetadata] = []
        self.tags_database: List[VideoTags] = []
        self.merged_tags_database: VideoTags = None

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
            else []
        )
        num_tags_loaded = (
            sum(tags.stats["num_tags"] for tags in self.tags_database)
            if self.tags_database
            else 0
        )

        # Create a merged tag database removing any duplicate.
        self.merged_tags_database = VideoTags.from_tags(tags={})
        for video_tags in self.tags_database:
            self.merged_tags_database.merge(video_tags)
        VideoTags.export_video_tags_to_videos(
            self.merged_tags_database, self.videos_database
        )
        logger.info(
            f"Unique tags exported to videos: {self.merged_tags_database.stats}  out of a total of {num_tags_loaded} tags loaded from files"
        )

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
    def _load_tags_database(
        tag_database_files: Union[str, List[str]]
    ) -> List[VideoTags]:
        tags_database: List[VideoTags] = []
        tag_database_files = (
            [tag_database_files]
            if isinstance(tag_database_files, str)
            else tag_database_files
        )
        for tag_file in tag_database_files:
            tags: VideoTags = VideoTags.from_file(tag_file)
            logger.info(f"{tags.stats} loaded from {tag_file}")
            tags_database.append(tags)

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

        tags_stats = self.merged_tags_database.stats
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
