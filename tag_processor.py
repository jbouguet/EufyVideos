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
from typing import Any, Dict, List, Optional, Union

from tqdm import tqdm

from logging_config import create_logger
from object_detector_base import ObjectDetectorFactory
from video_metadata import VideoMetadata

logger = create_logger(__name__)

TagDict = Dict[str, Dict[int, Dict[int, Dict[str, Any]]]]


class Model(Enum):
    """
    Supported object detection models.

    Each model has different characteristics:
    - FLORENCE2: Microsoft's Florence-2 vision model
    - YOLO10*: Various sizes of YOLO v10 models (N=Nano to X=Extra Large)
    - YOLO11*: Various sizes of YOLO v11 models (N=Nano to X=Extra Large)
    - YOLO10*_OPTIMIZED: GPU-accelerated of YOLO v10 models with batch processing
    - YOLO11*_OPTIMIZED: GPU-accelerated of YOLO v11 models with batch processing
    - TENSORFLOW: Google's TensorFlow-based EfficientDet model [no longer supported]

    """

    # TENSORFLOW = "TensorFlow"
    FLORENCE2 = "Florence-2"
    YOLO10N = "Yolo10n"
    YOLO10S = "Yolo10s"
    YOLO10M = "Yolo10m"
    YOLO10L = "Yolo10l"
    YOLO10X = "Yolo10x"
    YOLO11N = "Yolo11n"
    YOLO11S = "Yolo11s"
    YOLO11M = "Yolo11m"
    YOLO11L = "Yolo11l"
    YOLO11X = "Yolo11x"

    # Optimized versions with GPU acceleration and batch processing
    YOLO10N_OPTIMIZED = "Yolo10n_Optimized"
    YOLO10S_OPTIMIZED = "Yolo10s_Optimized"
    YOLO10M_OPTIMIZED = "Yolo10m_Optimized"
    YOLO10L_OPTIMIZED = "Yolo10l_Optimized"
    YOLO10X_OPTIMIZED = "Yolo10x_Optimized"
    YOLO11N_OPTIMIZED = "Yolo11n_Optimized"
    YOLO11S_OPTIMIZED = "Yolo11s_Optimized"
    YOLO11M_OPTIMIZED = "Yolo11m_Optimized"
    YOLO11L_OPTIMIZED = "Yolo11l_Optimized"
    YOLO11X_OPTIMIZED = "Yolo11x_Optimized"


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
    """Configuration for video tagging operations with optional person recognition."""

    model: str = Model.YOLO11X.value
    task: str = Task.TRACK.value
    num_frames_per_second: float = 1
    conf_threshold: float = 0.2
    batch_size: int = 8  # For optimized detectors
    enable_gpu: bool = False  # Enable GPU acceleration for non-optimized YOLO models
    
    # Person recognition settings
    enable_person_recognition: bool = False
    person_database_file: Optional[str] = None
    person_embeddings_file: Optional[str] = None
    person_crops_dir: Optional[str] = None
    
    # Person detection parameters
    person_crop_size: List[int] = field(default_factory=lambda: [224, 224])
    person_min_confidence: float = 0.6
    person_min_bbox_area: int = 2000
    max_crops_per_track: int = 10
    
    # Embedding generation parameters
    embedding_device: str = "mps"
    embedding_dim: int = 512
    clip_weight: float = 0.7
    reid_weight: float = 0.3
    
    # Person identification parameters
    similarity_threshold: float = 0.75
    auto_label_confidence: float = 0.8
    enable_auto_labeling: bool = True

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

        if "batch_size" in config_dict:
            if config_dict["batch_size"] <= 0:
                raise ValueError("batch_size must be positive")

        return cls(
            model=config_dict.get("model", Model.YOLO11X.value),
            task=config_dict.get("task", Task.TRACK.value),
            num_frames_per_second=config_dict.get("num_frames_per_second", 1),
            conf_threshold=config_dict.get("conf_threshold", 0.2),
            batch_size=config_dict.get("batch_size", 8),
            enable_gpu=config_dict.get("enable_gpu", False),
            
            # Person recognition settings
            enable_person_recognition=config_dict.get("enable_person_recognition", False),
            person_database_file=config_dict.get("person_database_file", None),
            person_embeddings_file=config_dict.get("person_embeddings_file", None),
            person_crops_dir=config_dict.get("person_crops_dir", None),
            
            # Person detection parameters
            person_crop_size=config_dict.get("person_crop_size", [224, 224]),
            person_min_confidence=config_dict.get("person_min_confidence", 0.6),
            person_min_bbox_area=config_dict.get("person_min_bbox_area", 2000),
            max_crops_per_track=config_dict.get("max_crops_per_track", 10),
            
            # Embedding generation parameters
            embedding_device=config_dict.get("embedding_device", "mps"),
            embedding_dim=config_dict.get("embedding_dim", 512),
            clip_weight=config_dict.get("clip_weight", 0.7),
            reid_weight=config_dict.get("reid_weight", 0.3),
            
            # Person identification parameters
            similarity_threshold=config_dict.get("similarity_threshold", 0.75),
            auto_label_confidence=config_dict.get("auto_label_confidence", 0.8),
            enable_auto_labeling=config_dict.get("enable_auto_labeling", True),
        )


@dataclass
class VideoTags:
    """
    In-memory representation of video tags with I/O capabilities.

    This class serves as the core data structure for managing video tags with
    built-in deduplication support through hash-based uniqueness checking.
    """

    timestamp: Optional[str] = None
    tag_processing_config: Optional[TaggerConfig] = None
    tags: TagDict = field(default_factory=dict)

    @property
    def stats(self) -> Dict[str, int]:
        """Return the total number of tags, tagged frames and tagged videos."""
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

    @property
    def filenames(self) -> List[str]:
        """Returns the list of filenames of tagged videos."""
        if self.tags is None or not self.tags:
            return []
        return list(self.tags.keys())

    @classmethod
    def from_tags(
        cls,
        tags: TagDict,
        tag_processing_config: Optional[TaggerConfig] = None,
    ) -> "VideoTags":
        """Create VideoTags instance from raw tag data."""
        return cls(
            timestamp=datetime.now().isoformat(),
            tag_processing_config=tag_processing_config,
            tags=tags,
        )

    def to_file(self, tag_file: str) -> "VideoTags":
        """Save tags to file with lossless serialization."""
        tag_processing_config_dict = None
        if self.tag_processing_config is not None:
            tag_processing_config_dict = asdict(self.tag_processing_config)
        batch_data = {
            "timestamp": self.timestamp,
            "tag_processing_config": tag_processing_config_dict,
            "tags": self.tags,
        }
        with open(tag_file, "w") as jsonfile:
            json.dump(batch_data, jsonfile, indent=2)
        return self

    @classmethod
    def from_file(cls, tag_file: str) -> "VideoTags":
        """Load tags from file with lossless restoration."""
        timestamp = None
        tag_processing_config = None
        tags: TagDict = {}
        if os.path.exists(tag_file):
            with open(tag_file, "r") as f:
                tag_data = json.load(f)
                timestamp = tag_data.get("timestamp")
                if tag_data.get("tag_processing_config"):
                    tag_processing_config = TaggerConfig.from_dict(
                        tag_data["tag_processing_config"]
                    )
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
            tag_processing_config=tag_processing_config,
            tags=tags,
        )

    def merge(self, other: "VideoTags") -> "VideoTags":
        """Merge in another VideoTags instance."""
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
        self, videos: Union[VideoMetadata, List[VideoMetadata]], mode: str = "ADD"
    ) -> List[VideoMetadata]:
        """
        Export tags to VideoMetadata objects.

        Args:
            videos: VideoMetadata object or list of VideoMetadata objects.
            mode: Export mode, either "ADD" (default) or "REPLACE".

        Returns:
            List of VideoMetadata objects with updated tags.
        """
        if mode.upper() == "ADD":
            return self.to_videos_add(videos)
        elif mode.upper() == "REPLACE":
            return self.to_videos_replace(videos)
        else:
            raise ValueError("Invalid mode. Use 'ADD' or 'REPLACE'.")

    def to_videos_add(
        self, videos: Union[VideoMetadata, List[VideoMetadata]]
    ) -> List[VideoMetadata]:
        """Export tags to VideoMetadata objects with addition."""
        if not self.timestamp or not self.tags:
            return [] if isinstance(videos, list) else [videos]
        videos_list = [videos] if isinstance(videos, VideoMetadata) else videos
        for video in videos_list:
            if video.filename in self.tags:
                video.merge_new_tags(self.tags[video.filename])
        return videos_list

    def to_videos_replace(
        self, videos: Union[VideoMetadata, List[VideoMetadata]]
    ) -> List[VideoMetadata]:
        """Export tags to VideoMetadata objects with full replacement."""
        if not self.timestamp or not self.tags:
            return [] if isinstance(videos, list) else [videos]
        videos_list = [videos] if isinstance(videos, VideoMetadata) else videos
        for video in videos_list:
            if video.filename in self.tags:
                if hasattr(video, "tags") and video.tags:
                    video.tags.clear()
                video.merge_new_tags(self.tags[video.filename])
        return videos_list

    @classmethod
    def from_videos(
        cls,
        videos: Union[VideoMetadata, List[VideoMetadata]],
        tag_processing_config: Optional[TaggerConfig] = None,
    ) -> "VideoTags":
        """Create VideoTags instance from existing tags in VideoMetadata objects."""
        videos_list = [videos] if isinstance(videos, VideoMetadata) else videos
        tags: TagDict = {}
        for video in videos_list:
            if hasattr(video, "tags") and video.tags:
                tags[video.filename] = {
                    int(frame): frame_tags
                    for frame, frame_tags in sorted(
                        video.tags.items(), key=lambda x: int(x[0])
                    )
                }
        return cls.from_tags(tags=tags, tag_processing_config=tag_processing_config)

    @staticmethod
    def clear_tags_in_videos(
        videos: Union[VideoMetadata, List[VideoMetadata]],
    ) -> List[VideoMetadata]:
        """Clear all tags from VideoMetadata objects."""
        videos_list = [videos] if isinstance(videos, VideoMetadata) else videos
        for video in videos_list:
            if hasattr(video, "tags") and video.tags:
                video.tags.clear()
        return videos_list

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

    def dedupe(self, iou_thresh: float = 0.8) -> "VideoTags":
        """
        Remove near duplicate tags. Two tags are declares near duplicates
        if they have the same "value" and their Intersecton over Union score
        (IoU) is larger than the input threshold iou_thresh.
        Within a group of near duplicate tags, the one belonging to the
        longest track will be kept as representative.
        
        Note: If track_id is missing (detection mode), deduplication is skipped
        and original tags are returned unchanged.
        """
        # Check if any tags have track_id - if not, skip deduplication
        has_track_ids = False
        for file_tags in self.tags.values():
            for frame_tags in file_tags.values():
                for tag in frame_tags.values():
                    if "track_id" in tag:
                        has_track_ids = True
                        break
                if has_track_ids:
                    break
            if has_track_ids:
                break
        
        # Skip deduplication if no track_ids found (detection mode)
        if not has_track_ids:
            logger.info("No track_id found in tags - skipping deduplication (detection mode)")
            return self
        
        track_lengths = defaultdict(int)
        for file_tags in self.tags.values():
            for frame_tags in file_tags.values():
                for tag in frame_tags.values():
                    if "track_id" in tag:
                        track_lengths[int(tag["track_id"])] += 1

        tags_new: TagDict = {}

        for filename, file_tags in self.tags.items():
            if filename not in tags_new:
                tags_new[filename] = {}
            for frame_number, frame_tags in file_tags.items():
                if frame_number not in tags_new[filename]:
                    tags_new[filename][frame_number] = {}
                # Group tags by value
                value_groups = defaultdict(list)
                for hash_key, tag in frame_tags.items():
                    value_groups[tag["value"]].append((hash_key, tag))

                # Process each value group
                for group in value_groups.values():
                    while group:
                        hash_key, tag = group.pop(0)
                        # Safe access to track_id with fallback
                        track_len = track_lengths.get(tag.get("track_id", 0), 0)
                        best_tag = (hash_key, tag, track_len)

                        # Compare with remaining tags in the group
                        i = 0
                        while i < len(group):
                            other_hash, other_tag = group[i]
                            if VideoTags.compute_iou(tag, other_tag) > iou_thresh:
                                other_track_len = track_lengths.get(other_tag.get("track_id", 0), 0)
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

    def __init__(self, tag_processing_config: Optional[TaggerConfig] = None):
        """Initialize processor with configuration."""
        self.tag_processing_config = tag_processing_config or TaggerConfig()
        
        # Initialize object detector
        self.object_detector = ObjectDetectorFactory.create_detector(
            model=self.tag_processing_config.model,
            conf_threshold=self.tag_processing_config.conf_threshold,
            batch_size=self.tag_processing_config.batch_size,
            enable_gpu=self.tag_processing_config.enable_gpu,
        )
        
        # Check for person recognition configuration
        if self.tag_processing_config.enable_person_recognition:
            logger.info("Person recognition is enabled in configuration")
            # TODO: Initialize person recognition components
            # This would integrate person_detector.py, person_embedding.py, etc.
            logger.warning("Person recognition functionality not yet integrated into TagProcessor")
            logger.info("Person recognition features will be processed in future updates")
        else:
            logger.info("Person recognition is disabled - standard object detection only")

    def run(self, videos: Union[VideoMetadata, List[VideoMetadata]]) -> VideoTags:
        """Process videos to generate tags."""
        videos_list = [videos] if isinstance(videos, VideoMetadata) else videos
        return VideoTags.from_tags(
            tags=self._compute_video_tags(videos_list),
            tag_processing_config=self.tag_processing_config,
        )

    def _compute_video_tags(self, videos: List[VideoMetadata]) -> TagDict:
        """Internal method to compute tags for all videos."""
        import time
        
        start_time = time.time()
        total_frames = 0
        total_results = 0
        
        logger.info(f"Starting batch processing of {len(videos)} videos")
        logger.info(f"Configuration: {self.tag_processing_config.task} mode with {self.tag_processing_config.model}")
        
        tags: TagDict = {}
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
            
            # Track stats
            video_frames = ceil(
                self.tag_processing_config.num_frames_per_second
                * video.duration.total_seconds()
            )
            total_frames += video_frames
            total_results += len(raw_tags) if raw_tags else 0

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

        # Log overall processing statistics
        total_time = time.time() - start_time
        overall_fps = total_frames / total_time if total_time > 0 else 0
        
        logger.info("=" * 60)
        logger.info("BATCH PROCESSING COMPLETED")
        logger.info("=" * 60)
        logger.info(f"Videos processed: {len(videos)}")
        logger.info(f"Total frames: {total_frames}")
        logger.info(f"Total execution time: {total_time:.2f}s")
        logger.info(f"Overall processing speed: {overall_fps:.1f} FPS")
        logger.info(f"Total results: {total_results} {self.tag_processing_config.task.lower()}ions")
        logger.info(f"Average time per video: {total_time/len(videos):.2f}s")
        logger.info("=" * 60)

        return tags

    def _compute_task_specific_tags(self, video: VideoMetadata) -> List[Dict[str, Any]]:
        """Compute tags based on configured task (detect or track)."""
        import time
        
        num_frames = ceil(
            self.tag_processing_config.num_frames_per_second
            * video.duration.total_seconds()
        )
        
        start_time = time.time()
        task_name = self.tag_processing_config.task
        
        match self.tag_processing_config.task:
            case Task.DETECT.value:
                result = self.object_detector.detect_objects(
                    video.full_path, num_frames=num_frames
                )
            case Task.TRACK.value:
                result = self.object_detector.track_objects(
                    video.full_path, num_frames=num_frames
                )
            case _:
                raise ValueError(
                    f"Invalid task: {self.tag_processing_config.task}. Must be one of {[dt.value for dt in Task]}"
                )
        
        execution_time = time.time() - start_time
        fps = num_frames / execution_time if execution_time > 0 else 0
        
        logger.info("Detector execution completed:")
        logger.info(f"  Video: {video.filename}")
        logger.info(f"  Task: {task_name}")
        logger.info(f"  Model: {self.tag_processing_config.model}")
        logger.info(f"  Frames processed: {num_frames}")
        logger.info(f"  Total execution time: {execution_time:.2f}s")
        logger.info(f"  Processing speed: {fps:.1f} FPS")
        logger.info(f"  Results: {len(result)} {task_name.lower()}ions")
        
        return result


if __name__ == "__main__":
    # Testing code for the module.
    import logging
    import sys

    from logging_config import set_logger_level_and_format
    from tag_visualizer import TagVisualizer, TagVisualizerConfig
    from video_database import VideoDatabase
    from video_filter import VideoFilter, VideoSelector

    # Set extended logging for this module only.
    set_logger_level_and_format(logger, level=logging.DEBUG, extended_format=True)

    root_database: str = "/Users/jbouguet/Documents/EufySecurityVideos/record/"
    out_dir: str = (
        "/Users/jbouguet/Documents/EufySecurityVideos/stories/tag_processor_test"
    )
    video_metadata_file: str = os.path.join(root_database, "videos_in_batches.csv")
    video_database = VideoDatabase(
        video_directories=None, video_metadata_file=video_metadata_file
    ).load_videos()

    if video_database is None:
        logger.error("Failed to load video database")
        sys.exit(1)

    story_name: str = "T8600P102338033E_20240930085536-T8600P1024260D5E_20241119181809"
    filenames = [
        "T8600P102338033E_20240930085536.mp4",
        "T8600P1024260D5E_20241119181809.mp4",
    ]
    videos = VideoFilter.by_selectors(
        video_database, VideoSelector(filenames=filenames)
    )

    if not videos:
        logger.error("No videos found matching the specified filenames")
        sys.exit(1)

    # Tracking configurations
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
            video_tags = TagProcessor(config).run(videos).to_file(tag_file)
        else:
            video_tags = VideoTags.from_file(tag_file)
        logger.info(f"Tags for config {config}: {video_tags.stats}")

    tag_visualizer = TagVisualizer(
        TagVisualizerConfig(output_size={"width": 1600, "height": 900})
    )

    # Merge and export tags
    merged_tags = VideoTags.from_tags(tags={})
    logger.info(f"Initial merge tags: {merged_tags.stats}")
    for tag_file in tag_files:
        merged_tags.merge(VideoTags.from_file(tag_file))

    logger.info(f"Merged tags before duplicates removal: {merged_tags.stats}")
    merged_tags.dedupe()
    logger.info(f"Merged tags after duplicates removal:  {merged_tags.stats}")

    tag_video_file = os.path.join(out_dir, "merged_tags.mp4")
    logger.info(f"Generating video tag file {tag_video_file}")

    tag_visualizer.run(merged_tags.to_videos(videos), tag_video_file)

    sys.exit()
