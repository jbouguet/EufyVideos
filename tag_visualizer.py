"""
Tag Visualization Library

This module provides functionality for creating composite videos that visualize object detection
and tracking results. It processes videos with their associated tags to create visual representations
of detections and tracking paths.

Key Components:
- TagVisualizerConfig: Configuration class for output video parameters
- TagVisualizer: Main class for generating composite videos with visualized tags

Example Usage:

1. Basic usage with default configuration:
    ```python
    from tag_visualizer import TagVisualizer
    from video_metadata import VideoMetadata

    # Assume videos is a List[VideoMetadata] with tags already exported
    visualizer = TagVisualizer()
    visualizer.generate_video(videos, "output_visualization.mp4")
    ```

2. Custom configuration:
    ```python
    from tag_visualizer import TagVisualizer, TagVisualizerConfig
    
    config = TagVisualizerConfig(
        output_size={"width": 1920, "height": 1080}
    )
    visualizer = TagVisualizer(config)
    visualizer.generate_video(videos, "output_visualization.mp4")
    ```
"""

import contextlib
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from tqdm import tqdm

from logging_config import create_logger
from video_metadata import VideoMetadata

logger = create_logger(__name__)


@contextlib.contextmanager
def suppress_stdout_stderr():
    """Context manager to suppress stdout and stderr."""
    try:
        null_fd = os.open(os.devnull, os.O_RDWR)
        save_fds = [os.dup(2), os.dup(1)]
        os.dup2(null_fd, 2)
        os.dup2(null_fd, 1)
        yield
    finally:
        os.dup2(save_fds[0], 2)
        os.dup2(save_fds[1], 1)
        os.close(null_fd)
        for fd in save_fds:
            os.close(fd)


def color_from_hash(number: int) -> tuple[int, int, int]:
    # Predefined list of contrasting colors in BGR format
    colors = [
        (255, 0, 0),  # Blue
        (0, 255, 0),  # Green
        (0, 0, 255),  # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
        (128, 0, 0),  # Dark Blue
        (0, 128, 0),  # Dark Green
        (0, 0, 128),  # Dark Red
        (128, 128, 0),  # Teal
        (128, 0, 128),  # Purple
        (0, 128, 128),  # Orange
        (255, 128, 0),  # Light Blue
        (128, 255, 0),  # Light Green
        (0, 128, 255),  # Light Red
        (192, 192, 192),  # Silver
        (128, 128, 128),  # Gray
        (128, 0, 255),  # Pink
        (0, 128, 255),  # Orange-Red
        (255, 128, 128),  # Light Purple
        (128, 255, 128),  # Light Teal
        (128, 128, 255),  # Salmon
        (0, 69, 255),  # Orange-Yellow
        (255, 64, 64),  # Steel Blue
        (64, 128, 128),  # Dark Teal
        (0, 165, 255),  # Deep Orange
        (211, 0, 148),  # Purple-Pink
        (130, 0, 75),  # Deep Purple
        (0, 215, 255),  # Gold
        (208, 224, 64),  # Turquoise
    ]

    # Use modulo to wrap around if the number is larger than the color list
    index = abs(number) % len(colors)
    return colors[index]


@dataclass
class TagVisualizerConfig:
    """
    Configuration for tag visualization parameters.

    Attributes:
        output_size: Dictionary containing width and height for output video
        max_track_history: Maximum number of points to keep in track history
        bbox_color: Color for bounding boxes (BGR format)
        track_color: Color for tracking lines (BGR format)
        text_color: Color for text labels (BGR format)
        info_color: Color for video info text (BGR format)
        font_scale: Scale factor for text size
        line_thickness: Thickness of lines for boxes and tracks
        use_color_from_track_id: If true, color individual track uniquely
    """

    output_size: Dict[str, int] = field(
        default_factory=lambda: {"width": 1600, "height": 900}
    )
    max_track_history: int = field(default=1000)
    bbox_color: Tuple[int, int, int] = field(default=(0, 255, 0))  # Green in BGR
    track_color: Tuple[int, int, int] = field(default=(230, 230, 230))  # Light gray
    text_color: Tuple[int, int, int] = field(default=(0, 255, 0))  # Green
    info_color: Tuple[int, int, int] = field(default=(255, 255, 255))  # White
    font_scale: float = field(default=0.6)
    line_thickness: int = field(default=2)
    use_color_from_track_id: bool = field(default=True)

    @classmethod
    def from_dict(cls, config_dict: Optional[Dict[str, Any]]) -> "TagVisualizerConfig":
        """Create configuration from dictionary representation."""
        if config_dict is None:
            return cls()

        output_size = config_dict.get("output_size", {"width": 1600, "height": 900})
        return cls(
            output_size=output_size,
            max_track_history=config_dict.get("max_track_history", 1000),
            bbox_color=tuple(config_dict.get("bbox_color", (0, 255, 0))),
            track_color=tuple(config_dict.get("track_color", (230, 230, 230))),
            text_color=tuple(config_dict.get("text_color", (0, 255, 0))),
            info_color=tuple(config_dict.get("info_color", (255, 255, 255))),
            font_scale=config_dict.get("font_scale", 0.6),
            line_thickness=config_dict.get("line_thickness", 2),
            use_color_from_track_id=config_dict.get("use_color_from_track_id", True),
        )


class TagVisualizer:
    """
    Creates composite videos with visualized detection and tracking results.

    This class processes videos and their associated tags to create visual
    representations of object detections and tracking paths. It supports:
    - Visualization of bounding boxes
    - Object tracking paths
    - Frame information overlay
    - Configurable output parameters
    """

    def __init__(self, config: Optional[TagVisualizerConfig] = None, **kwargs):
        """
        Initialize visualizer with configuration.

        Args:
            config: TagVisualizerConfig object for visualization settings
            **kwargs: Backward compatibility for tag_visualizer_config parameter
        """
        if config is None and "tag_visualizer_config" in kwargs:
            config = kwargs["tag_visualizer_config"]
        self.config = config if config else TagVisualizerConfig()

    def run(
        self, videos: Union[VideoMetadata, List[VideoMetadata]], output_file: str
    ) -> None:
        """
        Generate a composite video with visualized tags.

        Args:
            videos: List of VideoMetadata objects with tags
            output_file: Path for the output video file
        """
        if not videos:
            raise ValueError("No videos provided for visualization")
        if isinstance(videos, VideoMetadata):
            videos = [videos]
        output_size = (
            self.config.output_size["width"],
            self.config.output_size["height"],
        )
        self._create_visualization(videos, output_file, output_size)

    def _create_visualization(
        self,
        videos: List[VideoMetadata],
        output_file: str,
        output_size: Tuple[int, int],
    ) -> None:
        """Create the visualization video from tagged frames."""
        with suppress_stdout_stderr():
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = None

        for video in tqdm(
            videos,
            desc=f"Generating video from {len(videos)} tagged videos",
            unit="video",
            colour="green",
            position=0,
            leave=False,
        ):
            if not hasattr(video, "tags") or not video.tags:
                continue

            cap = self._open_video_capture(video)
            if cap is None:
                continue

            fps = self._get_video_fps(cap)
            if out is None:
                out = cv2.VideoWriter(output_file, fourcc, fps, output_size)

            self._process_video_frames(video, cap, out, output_size)
            cap.release()

        if out:
            with suppress_stdout_stderr():
                out.release()

    def _open_video_capture(self, video: VideoMetadata) -> Optional[cv2.VideoCapture]:
        """Open video capture for reading frames."""
        with suppress_stdout_stderr():
            cap = cv2.VideoCapture(video.full_path)
            if not cap.isOpened():
                print(f"Error opening video file: {video.full_path}")
                return None
        return cap

    def _get_video_fps(self, cap: cv2.VideoCapture) -> int:
        """Get video FPS from capture object."""
        with suppress_stdout_stderr():
            return int(cap.get(cv2.CAP_PROP_FPS))

    def _process_video_frames(
        self,
        video: VideoMetadata,
        cap: cv2.VideoCapture,
        out: cv2.VideoWriter,
        output_size: Tuple[int, int],
    ) -> None:
        """Process and write frames for a single video."""
        track_history = self._initialize_track_history()

        for frame_number in tqdm(
            video.tags.keys(),  # Assumed to be sorted numerically.
            desc=f"Rendering {len(video.tags)} tagged frames in {video.filename}",
            unit="frame",
            colour="green",
            position=1,
            leave=False,
        ):
            frame_number_int = int(frame_number)
            frame = self._read_frame(cap, frame_number_int)
            if frame is None:
                continue

            frame = self._draw_visualizations(
                frame,
                video,
                frame_number_int,
                video.tags[frame_number],
                track_history,
            )
            frame = self._resize_frame(frame, output_size)

            with suppress_stdout_stderr():
                out.write(frame)

    def _initialize_track_history(self) -> Dict[int, List[Tuple[int, int]]]:
        """Initialize empty track history dictionary."""
        return {}

    def _read_frame(
        self, cap: cv2.VideoCapture, frame_number: int
    ) -> Optional[np.ndarray]:
        """Read a specific frame from video capture."""
        with suppress_stdout_stderr():
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            if not ret:
                return None
        return frame

    def _draw_visualizations(
        self,
        frame: np.ndarray,
        video: VideoMetadata,
        frame_number: int,
        frame_tags: Dict[int, Dict[str, Any]],
        track_history: Dict[int, List[Tuple[int, int]]],
    ) -> np.ndarray:
        """Draw all visualizations on the frame."""
        for tag in frame_tags.values():
            frame = self._draw_detection(frame, tag, track_history)

        return self._draw_frame_info(frame, video, frame_number, len(frame_tags))

    def _draw_detection(
        self,
        frame: np.ndarray,
        tag: Dict[str, Any],
        track_history: Dict[int, List[Tuple[int, int]]],
    ) -> np.ndarray:
        """Draw detection box, label, and tracking path if applicable."""
        bbox_color = self.config.bbox_color
        if "track_id" in tag and self.config.use_color_from_track_id:
            bbox_color = color_from_hash(int(tag["track_id"]))
        bbox = tag["bounding_box"]
        cv2.rectangle(
            frame,
            (bbox["x1"], bbox["y1"]),
            (bbox["x2"], bbox["y2"]),
            bbox_color,
            self.config.line_thickness,
        )

        if "track_id" in tag:
            frame = self._draw_tracking_visualization(frame, tag, bbox, track_history)
        else:
            label = f"{tag['value']} ({tag['confidence']})"
            self._draw_label(frame, label, bbox, bbox_color)

        return frame

    def _draw_tracking_visualization(
        self,
        frame: np.ndarray,
        tag: Dict[str, Any],
        bbox: Dict[str, int],
        track_history: Dict[int, List[Tuple[int, int]]],
    ) -> np.ndarray:
        """Draw tracking-specific visualizations."""
        track_id = tag["track_id"]
        track_color = (
            color_from_hash(int(track_id))
            if self.config.use_color_from_track_id
            else self.config.track_color
        )

        # If center_x and center_y are computed here, why do we need to store them in the tag object?
        center_x = (bbox["x1"] + bbox["x2"]) // 2
        center_y = (bbox["y1"] + bbox["y2"]) // 2

        if track_id not in track_history:
            track_history[track_id] = []
        track_history[track_id].append((center_x, center_y))
        if len(track_history[track_id]) > self.config.max_track_history:
            track_history[track_id].pop(0)

        # Selection of tag labels for debug purposes:
        # label = f"{tag['value']} ({tag['confidence']})"
        label = f"{tag['value']} ({len(track_history[track_id])})"
        # label = f"{track_id}"

        self._draw_label(frame, label, bbox, track_color)

        if len(track_history[track_id]) > 1:
            points = np.array(track_history[track_id], dtype=np.int32).reshape(
                (-1, 1, 2)
            )
            cv2.polylines(
                frame,
                [points],
                isClosed=False,
                color=track_color,
                thickness=self.config.line_thickness,
            )

        return frame

    def _draw_label(
        self,
        frame: np.ndarray,
        label: str,
        bbox: Dict[str, int],
        text_color: tuple[int, int, int] = None,
    ) -> None:
        """Draw label text above bounding box."""
        if text_color is None:
            text_color = self.config.text_color
        cv2.putText(
            frame,
            label,
            (bbox["x1"], bbox["y1"] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.config.font_scale,
            text_color,
            self.config.line_thickness,
        )

    def _draw_frame_info(
        self, frame: np.ndarray, video: VideoMetadata, frame_number: int, num_tags: int
    ) -> np.ndarray:
        """Draw video filename and frame number information."""

        header_text = [
            f"Video: {video.filename}",
            f"Frame number: {frame_number}",
            f"Number of tags: {num_tags}",
        ]
        positions = [(20, 80), (20, 120), (20, 160)]
        colors = [(0, 0, 0), (255, 255, 255)]
        thicknesses = [6, 2]

        for text, pos in zip(header_text, positions):
            for color, thickness in zip(colors, thicknesses):
                cv2.putText(
                    frame,
                    text,
                    pos,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    color,
                    thickness,
                )

        return frame

    def _resize_frame(
        self, frame: np.ndarray, output_size: Tuple[int, int]
    ) -> np.ndarray:
        """Resize frame to match output dimensions if needed."""
        if frame.shape[:2][::-1] != output_size:
            frame = cv2.resize(frame, output_size)
        return frame


if __name__ == "__main__":
    # Testing code for the module.
    from tag_processor import TaggerConfig, TagProcessor

    video_file: str = (
        "/Users/jeanyves.bouguet/Documents/EufySecurityVideos/EufyVideos/record/Batch022/T8600P102338033E_20240930085536.mp4"
    )
    out_dir: str = "/Users/jeanyves.bouguet/Documents/EufySecurityVideos/stories"
    tag_video: str = os.path.join(out_dir, "T8600P102338033E_20240930085536_tags.mp4")
    videos = [VideoMetadata.from_video_file(video_file)]
    TagVisualizer(TagVisualizerConfig(output_size={"width": 1600, "height": 900})).run(
        TagProcessor(
            TaggerConfig(
                model="Yolo11x",
                task="Track",
                num_frames_per_second=1,
                conf_threshold=0.2,
            )
        )
        .run(videos)
        .to_videos(videos),
        tag_video,
    )
