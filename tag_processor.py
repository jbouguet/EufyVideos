"""
Tag Processor Library

This module provides a comprehensive interface for computing, representing, storing, and loading
tags from batches of videos. It serves as a high-level abstraction layer that:

1. Manages video object detection and tracking through configurable models
2. Provides lossless serialization of tags and configuration
3. Supports multiple object detection backends (YOLO, TensorFlow, Florence-2)
4. Handles both detection and tracking tasks
"""

import json
import os
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from math import ceil
from typing import Any, Dict, List, Union

from tqdm import tqdm

from logging_config import create_logger
from object_detector_base import ObjectDetectorFactory
from video_metadata import VideoMetadata

logger = create_logger(__name__)


class Model(Enum):
    """
    Supported object detection models.

    Each model has different characteristics:
    - TENSORFLOW: Google's TensorFlow-based EfficientDet model
    - FLORENCE2: Microsoft's Florence-2 vision model
    - YOLO11*: Various sizes of YOLO v11 models (N=Nano to X=Extra Large)
    """

    TENSORFLOW = "TensorFlow"
    FLORENCE2 = "Florence-2"
    YOLO11N = "Yolo11n"
    YOLO11S = "Yolo11s"
    YOLO11M = "Yolo11m"
    YOLO11L = "Yolo11l"
    YOLO11X = "Yolo11x"


class Task(Enum):
    """
    Supported video analysis tasks.

    DETECT: Frame-by-frame object detection
    TRACK: Object detection with tracking across frames
    """

    DETECT = "Detect"
    TRACK = "Track"


@dataclass
class TaggerConfig:
    """Configuration for video tagging operations."""

    model: str = Model.YOLO11X.value
    task: str = Task.TRACK.value
    num_frames_per_second: float = 1
    conf_threshold: float = 0.2

    def get_identifier(self) -> str:
        """Generate a unique identifier for this configuration."""
        return f"{self.model}_{self.task}_{round(self.num_frames_per_second,3)}fps"

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "TaggerConfig":
        """Create configuration from dictionary representation."""
        if "conf_threshold" in config_dict:
            if not 0 <= config_dict["conf_threshold"] <= 1:
                raise ValueError("conf_threshold must be between 0 and 1")

        if "num_frames_per_second" in config_dict:
            if config_dict["num_frames_per_second"] <= 0:
                raise ValueError("num_frames_per_second must be positive")

        return cls(
            model=config_dict.get("model", cls.model),
            task=config_dict.get("task", cls.task),
            num_frames_per_second=config_dict.get(
                "num_frames_per_second", cls.num_frames_per_second
            ),
            conf_threshold=config_dict.get("conf_threshold", cls.conf_threshold),
        )


@dataclass
class VideoTags:
    """
    In-memory representation of video tags with I/O capabilities.

    This class serves as the core data structure for managing video tags with
    built-in deduplication support through hash-based uniqueness checking.
    """

    timestamp: str = None
    tagger_config: TaggerConfig = None
    tags: Dict[str, Dict[int, Dict[int, Dict[str, Any]]]] = field(default_factory=list)

    @property
    def stats(self) -> Dict[str, int]:
        if self.tags is None or not self.tags:
            return {"num_tags": 0, "num_tagged_frames": 0, "num_tagged_videos": 0}
        return {
            "num_tags": sum(
                sum(len(tags) for tags in frames.values())
                for frames in self.tags.values()
            ),
            "num_tagged_frames": sum(len(frames) for frames in self.tags.values()),
            "num_tagged_videos": len(self.tags),
        }

    @classmethod
    def from_tags(
        cls,
        tags: Dict[str, Dict[int, Dict[int, Dict[str, Any]]]],
        tagger_config: TaggerConfig = None,
    ) -> "VideoTags":
        """Create VideoTags instance from raw tag data."""
        return cls(
            timestamp=datetime.now().isoformat(),
            tagger_config=tagger_config,
            tags=tags,
        )

    def to_file(self, tag_file: str) -> None:
        """Save tags to file with lossless serialization."""
        tagger_config_dict = None
        if self.tagger_config is not None:
            tagger_config_dict = asdict(self.tagger_config)
        batch_data = {
            "timestamp": self.timestamp,
            "tagger_config": tagger_config_dict,
            "tags": self.tags,
        }
        with open(tag_file, "w") as jsonfile:
            json.dump(batch_data, jsonfile, indent=2)

    @classmethod
    def from_file(cls, tag_file: str) -> "VideoTags":
        """Load tags from file with lossless restoration."""
        timestamp = None
        tagger_config = None
        tags = {}
        if os.path.exists(tag_file):
            with open(tag_file, "r") as f:
                tag_data = json.load(f)
                timestamp = tag_data.get("timestamp")
                if tag_data.get("tagger_config"):
                    tagger_config = TaggerConfig.from_dict(tag_data["tagger_config"])
                raw_tags = tag_data.get("tags", {})

                # Apply type casting and sorting to the loaded tags
                for filename, frames in raw_tags.items():
                    tags[filename] = {
                        int(frame): {
                            int(hash_key): tag_value
                            for hash_key, tag_value in frame_tags.items()
                        }
                        for frame, frame_tags in frames.items()
                    }
                    # Sort the frames for each filename
                    tags[filename] = dict(
                        sorted(tags[filename].items(), key=lambda x: int(x[0]))
                    )

        return cls(
            timestamp=timestamp,
            tagger_config=tagger_config,
            tags=tags,
        )

    def merge(self, other: "VideoTags") -> "VideoTags":
        # Merge other in self, and output the merged self.
        if self.timestamp is None and other.timestamp is not None:
            self.timestamp = other.timestamp
        if not other.tags:
            return self
        if self.tags is None:
            self.tags = {}
        for filename, frames in other.tags.items():
            if filename not in self.tags:
                self.tags[filename] = {}
            frames_added = False
            for frame_number, hashes in frames.items():
                if frame_number not in self.tags[filename]:
                    frames_added = True
                self.tags[filename].setdefault(frame_number, {}).update(hashes)
            if frames_added:
                self.tags[filename] = {
                    int(frame): frame_tags
                    for frame, frame_tags in sorted(
                        self.tags[filename].items(), key=lambda x: int(x[0])
                    )
                }
        return self

    def to_videos(
        self, videos: Union[VideoMetadata, List[VideoMetadata]]
    ) -> List[VideoMetadata]:
        """Export tags to VideoMetadata objects."""
        if not self.timestamp or not self.tags:
            return 0
        videos = [videos] if isinstance(videos, VideoMetadata) else videos
        for video in videos:
            if video.filename in self.tags:
                video.merge_new_tags(self.tags[video.filename])
        return videos

    @classmethod
    def from_videos(
        cls,
        videos: Union[VideoMetadata, List[VideoMetadata]],
        tagger_config: TaggerConfig = None,
    ) -> "VideoTags":
        """Create VideoTags from VideoMetadata objects."""
        videos = [videos] if isinstance(videos, VideoMetadata) else videos
        tags = {}
        for video in videos:
            if hasattr(video, "tags") and video.tags:
                tags[video.filename] = {
                    int(frame): frame_tags
                    for frame, frame_tags in sorted(
                        video.tags.items(), key=lambda x: int(x[0])
                    )
                }
        return cls.from_tags(tags=tags, tagger_config=tagger_config)

    @staticmethod
    def clear_tags(
        videos: Union[VideoMetadata, List[VideoMetadata]],
    ) -> List[VideoMetadata]:
        """Clear all tags from VideoMetadata objects."""
        videos = [videos] if isinstance(videos, VideoMetadata) else videos
        for video in videos:
            if hasattr(video, "tags") and video.tags:
                video.tags.clear()
        return videos

    @staticmethod
    def compute_tag_hash(filename: str, tag: Dict[str, Any]) -> int:
        """Compute a unique hash for a tag based on its key properties."""
        hash_filename = hash(filename)
        frame_number: int = tag["frame_number"]
        value_hash: int = hash(tag["value"])
        bbox = tag["bounding_box"]
        x1: int = bbox["x1"]
        x2: int = bbox["x2"]
        y1: int = bbox["y1"]
        y2: int = bbox["y2"]
        return int(hash((hash_filename, frame_number, value_hash, x1, x2, y1, y2)))

    @staticmethod
    def compute_iou(tag1: Dict[str, Any], tag2: Dict[str, Any]) -> float:
        """
        Compute Intersection over Union (IoU) between two bounding boxes.
        Boxes are dictionaries with keys 'x1', 'y1', 'x2', 'y2'.
        Returns a value between 0 (no overlap) and 1 (perfect overlap).
        """
        if tag1["value"] != tag2["value"]:
            return 0.0

        bbox1 = tag1["bounding_box"]
        bbox2 = tag2["bounding_box"]
        # Calculate intersection coordinates
        x_left = max(bbox1["x1"], bbox2["x1"])
        y_top = max(bbox1["y1"], bbox2["y1"])
        x_right = min(bbox1["x2"], bbox2["x2"])
        y_bottom = min(bbox1["y2"], bbox2["y2"])

        # If there's no intersection, return 0
        if x_right < x_left or y_bottom < y_top:
            return 0.0

        # Calculate intersection area
        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # Calculate union area
        bbox1_area = (bbox1["x2"] - bbox1["x1"]) * (bbox1["y2"] - bbox1["y1"])
        bbox2_area = (bbox2["x2"] - bbox2["x1"]) * (bbox2["y2"] - bbox2["y1"])
        union_area = bbox1_area + bbox2_area - intersection_area

        # Calculate IoU
        iou = intersection_area / union_area if union_area > 0 else 0.0

        return iou

    def remove_duplicates(self, iou_thresh: float = 0.8) -> "VideoTags":
        track_lengths = defaultdict(int)
        for file_tags in self.tags.values():
            for frame_tags in file_tags.values():
                for tag in frame_tags.values():
                    if "track_id" in tag:
                        track_lengths[int(tag["track_id"])] += 1

        tags_new = defaultdict(lambda: defaultdict(dict))

        for filename, file_tags in self.tags.items():
            for frame_number, frame_tags in file_tags.items():
                # Group tags by value
                value_groups = defaultdict(list)
                for hash_key, tag in frame_tags.items():
                    value_groups[tag["value"]].append((hash_key, tag))

                # Process each value group
                for group in value_groups.values():
                    while group:
                        hash_key, tag = group.pop(0)
                        track_len = track_lengths[tag["track_id"]]
                        best_tag = (hash_key, tag, track_len)

                        # Compare with remaining tags in the group
                        i = 0
                        while i < len(group):
                            other_hash, other_tag = group[i]
                            if VideoTags.compute_iou(tag, other_tag) > iou_thresh:
                                other_track_len = track_lengths[other_tag["track_id"]]
                                if other_track_len > best_tag[2]:
                                    best_tag = (other_hash, other_tag, other_track_len)
                                group.pop(i)
                            else:
                                i += 1

                        # Add the best tag to the new tags
                        tags_new[filename][frame_number][best_tag[0]] = best_tag[1]

        self.tags = tags_new
        return self


class TagProcessor:
    """Main interface for computing video tags using object detection models."""

    def __init__(self, tagger_config: TaggerConfig = None):
        """Initialize processor with configuration."""
        self.tagger_config = tagger_config or TaggerConfig()
        self.object_detector = ObjectDetectorFactory.create_detector(
            model=self.tagger_config.model,
            conf_threshold=self.tagger_config.conf_threshold,
        )

    def run(self, videos: Union[VideoMetadata, List[VideoMetadata]]) -> VideoTags:
        """Process videos to generate tags."""
        videos = [videos] if isinstance(videos, VideoMetadata) else videos
        return VideoTags.from_tags(
            tags=self._compute_video_tags(videos),
            tagger_config=self.tagger_config,
        )

    def _compute_video_tags(
        self, videos: Union[VideoMetadata, List[VideoMetadata]]
    ) -> Dict[str, Dict[int, Dict[int, Dict[str, Any]]]]:
        """Internal method to compute tags for all videos."""
        videos = [videos] if isinstance(videos, VideoMetadata) else videos
        tags = {}
        unique_run_key = hash(datetime.now().isoformat())
        for video in tqdm(
            videos,
            desc=f"Tagging {len(videos)} videos",
            unit="video",
            colour="green",
            position=0,
            leave=False,
        ):
            raw_tags = self._compute_task_specific_tags(video)

            if raw_tags:
                filename = video.filename
                unique_filename_key = hash(filename)
                if filename not in tags:
                    tags[filename] = {}
                for tag in raw_tags:
                    frame_number = int(tag["frame_number"])
                    if frame_number not in tags[filename]:
                        tags[filename][frame_number] = {}
                    hash_key = int(VideoTags.compute_tag_hash(video.filename, tag))
                    del tag["frame_number"]
                    # Make track_id unique to every run, filename to avoid collisions when
                    # merging tags from different runs. Also ensures consistent tag values.
                    unique_value_key = hash(tag["value"])
                    if "track_id" in tag:
                        tag["track_id"] = hash(
                            (
                                unique_run_key,
                                unique_filename_key,
                                unique_value_key,
                                int(tag["track_id"]),
                            )
                        )
                    if hash_key not in tags[filename][frame_number]:
                        tags[filename][frame_number][hash_key] = tag

        # Sort tags by frame number for each video
        for filename in tags:
            tags[filename] = dict(
                sorted(tags[filename].items(), key=lambda x: int(x[0]))
            )

        return tags

    def _compute_task_specific_tags(self, video: VideoMetadata) -> List[Dict[str, Any]]:
        """Compute tags based on configured task (detect or track)."""
        num_frames = ceil(
            self.tagger_config.num_frames_per_second * video.duration.total_seconds()
        )
        match self.tagger_config.task:
            case Task.DETECT.value:
                return self.object_detector.detect_objects(
                    video.full_path, num_frames=num_frames
                )
            case Task.TRACK.value:
                return self.object_detector.track_objects(
                    video.full_path, num_frames=num_frames
                )
            case _:
                raise ValueError(
                    f"Invalid task: {self.tagger_config.task}. Must be one of {[dt.value for dt in Task]}"
                )


if __name__ == "__main__":
    # Testing code for the module.
    import logging
    import sys

    from logging_config import set_logger_level_and_format
    from tag_visualizer import TagVisualizer, TagVisualizerConfig

    # Set extended logging for this module only.
    set_logger_level_and_format(logger, level=logging.DEBUG, extended_format=True)
    # Set extended logging for all modules.
    # set_all_loggers_level_and_format(level=logging.DEBUG, extended_format=True)

    video_files: List[str] = [
        "/Users/jeanyves.bouguet/Documents/EufySecurityVideos/EufyVideos/record/Batch022/T8600P102338033E_20240930085536.mp4",
        "/Users/jeanyves.bouguet/Documents/EufySecurityVideos/EufyVideos/record/Batch010/T8600P1024260D5E_20241119181809.mp4",
    ]
    out_dir: str = (
        "/Users/jeanyves.bouguet/Documents/EufySecurityVideos/stories/tag_processor_test"
    )
    story_name: str = "T8600P102338033E_20240930085536-T8600P1024260D5E_20241119181809"
    videos = VideoMetadata.clean_and_sort(
        [VideoMetadata.from_video_file(file) for file in video_files]
    )

    # Tracking configurations
    # temporal_subsamplings: List[float] = [100, 50, 25, 15, 10, 5, 2, 1]
    temporal_subsamplings: List[float] = [100, 50, 25, 15, 10, 5, 2, 1]
    tracker_configs: List[TaggerConfig] = [
        TaggerConfig(
            model="Yolo11x",
            task="Track",
            num_frames_per_second=round(videos[0].fps / subsampling, 2),
            conf_threshold=0.2,
        )
        for subsampling in temporal_subsamplings
    ]

    tag_files: List[str] = [
        os.path.join(out_dir, f"{story_name}_{config.get_identifier()}_tags.json")
        for config in tracker_configs
    ]

    # Process tags
    force_recompute: bool = False
    for config, tag_file in zip(tracker_configs, tag_files):
        logger.info(f"Processing tag file {tag_file}")
        if not os.path.exists(tag_file) or force_recompute:
            TagProcessor(config).run(videos).to_file(tag_file)

    tag_visualizer = TagVisualizer(
        TagVisualizerConfig(output_size={"width": 1600, "height": 900})
    )

    # Merge and export tags
    merged_tags = VideoTags.from_tags(tags={})
    logger.info(f"Initial forward merge tags: {merged_tags.stats}")
    for tag_file in tag_files:
        merged_tags.merge(VideoTags.from_file(tag_file))

    logger.info(f"Merged tags before duplicates removal: {merged_tags.stats}")
    merged_tags.remove_duplicates()
    logger.info(f"Merged tags after duplicates removal:  {merged_tags.stats}")

    tag_video_file = os.path.join(out_dir, "merged_tags.mp4")
    logger.info(f"Generating video tag file {tag_video_file}")

    tag_visualizer.run(merged_tags.to_videos(videos), tag_video_file)

    sys.exit()
