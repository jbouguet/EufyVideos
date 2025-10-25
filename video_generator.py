#!/usr/bin/env python3

"""
Video Generator Module

This module provides functionality for creating composite summary videos from multiple input videos.
It handles video trimming, scaling, and assembly of video fragments into a single output video.

Key Features:
- Trim input videos to specific time segments
- Apply ROI cropping with normalized coordinates
- Scale videos to consistent dimensions
- Add datetime labels to video frames
- Support for both H264 and H265 video codecs
- Maintain audio synchronization
- Handle aspect ratio enforcement (16:9)

Example Usage:
    ```python
    # Create configuration
    config = VideoGenerationConfig(
        input_fragments=InputFragments(
            offset_in_seconds=2.0,
            duration_in_seconds=5.0,
            normalized_crop_roi=(0.2, 0.2, 0.8, 0.8)
        ),
        output_video=OutputVideo(
            width=1280,
            vcodec="h264"
        )
    )

    # Initialize generator and create video
    generator = VideoGenerator(config=config)
    generator.run(
        videos=[video1, video2, video3],
        output_file="summary.mp4"
    )
    ```
"""

import os
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

from logging_config import create_logger
from video_metadata import VideoMetadata

logger = create_logger(__name__)


@dataclass
class InputFragments:
    """
    Configuration for input video fragment processing.

    Attributes:
        offset_in_seconds: Start time offset for video trimming
        duration_in_seconds: Duration of video segment to extract. If None, extracts until end of video
        normalized_crop_roi: Region of interest for cropping (left, top, right, bottom)
                           in normalized coordinates [0,1]
        enforce_16_9_aspect_ratio: Whether to force 16:9 aspect ratio

    Example:
        ```python
        fragments = InputFragments(
            offset_in_seconds=1.5,
            duration_in_seconds=3.0,
            normalized_crop_roi=(0.1, 0.1, 0.9, 0.9),
            enforce_16_9_aspect_ratio=True
        )
        ```
    """

    offset_in_seconds: float = 0.0
    duration_in_seconds: Optional[float] = None
    normalized_crop_roi: Optional[Tuple[float, float, float, float]] = None
    enforce_16_9_aspect_ratio: bool = False

    @classmethod
    def from_dict(cls, input_fragments_dict: Dict[str, Any]) -> "InputFragments":
        """Creates an InputFragments instance from a dictionary configuration."""
        normalized_crop_roi = input_fragments_dict.get("normalized_crop_roi")
        if normalized_crop_roi is not None:
            try:
                normalized_crop_roi = tuple(float(x) for x in normalized_crop_roi)
                if len(normalized_crop_roi) != 4:
                    raise ValueError(
                        f"normalized_crop_roi must have exactly 4 values, got {len(normalized_crop_roi)}"
                    )
            except ValueError as e:
                logger.error(f"Error parsing normalized_crop_roi': {str(e)}")
                normalized_crop_roi = None
        return cls(
            offset_in_seconds=input_fragments_dict.get("offset_in_seconds", 0.0),
            duration_in_seconds=input_fragments_dict.get("duration_in_seconds", None),
            enforce_16_9_aspect_ratio=input_fragments_dict.get(
                "enforce_16_9_aspect_ratio",
                False,
            ),
            normalized_crop_roi=normalized_crop_roi,
        )


@dataclass
class DateTimeLabel:
    """
    Configuration for datetime label rendering on videos.

    Attributes:
        draw: Boolean controling the datetime rendering
        fontsize: Size of the font for the datetime label
        left_margin: Left margin in pixels for label placement
        top_margin: Top margin in pixels for label placement
        border_width: Width of the text border/outline

    Example:
        ```python
        label = DateTimeLabel(
            draw=True
            fontsize=24,
            left_margin=10,
            top_margin=10,
            border_width=2
        )
        ```
    """

    draw: bool = False
    fontsize: int = 20
    left_margin: int = 10
    top_margin: int = 30
    border_width: int = 1

    @classmethod
    def from_dict(cls, date_time_label_dict: Dict[str, Any]) -> "DateTimeLabel":
        """Creates a DateTimeLabel instance from a dictionary configuration."""
        return cls(
            draw=date_time_label_dict.get("draw", False),
            fontsize=date_time_label_dict.get("fontsize", 20),
            left_margin=date_time_label_dict.get("left_margin", 10),
            top_margin=date_time_label_dict.get("top_margin", 30),
            border_width=date_time_label_dict.get("border_width", 1),
        )


@dataclass
class OutputVideo:
    """
    Configuration for output video generation.

    Attributes:
        width: Target width for the output video (height auto-calculated)
        date_time_label: Configuration for datetime label rendering
        vcodec: Video codec to use ('h264' or 'h265')

    Example:
        ```python
        output = OutputVideo(
            width=1920,
            date_time_label=DateTimeLabel(),
            vcodec="h264"
        )
        ```
    """

    width: Optional[int] = None
    date_time_label: DateTimeLabel = field(
        default_factory=DateTimeLabel, metadata={"exclude_from_dict": True}
    )
    vcodec: str = field(default="h264")  # Default to h264, never None

    def __post_init__(self):
        """Ensure vcodec is a valid string."""
        try:
            if not isinstance(self.vcodec, str):
                self.vcodec = str(self.vcodec)
            self.vcodec = self.vcodec.strip().lower()
            if self.vcodec not in ["h264", "h265", "hevc", "libx265", "libx264"]:
                logger.warning(f"Invalid codec: {self.vcodec}, defaulting to h264")
                self.vcodec = "h264"
        except (AttributeError, TypeError, ValueError) as e:
            logger.warning(f"Error validating codec: {str(e)}, defaulting to h264")
            self.vcodec = "h264"

    @classmethod
    def from_dict(cls, output_video_dict: Dict[str, Any]) -> "OutputVideo":
        """Creates an OutputVideo instance from a dictionary configuration."""
        return cls(
            width=output_video_dict.get("width"),
            date_time_label=DateTimeLabel.from_dict(
                date_time_label_dict=output_video_dict.get("date_time_label", {})
            ),
            vcodec=str(output_video_dict.get("vcodec", "h264")),
        )


@dataclass
class VideoGenerationConfig:
    """
    Main configuration for video generation process.

    Attributes:
        input_fragments: Configuration for processing input video fragments
        output_video: Configuration for the final output video

    Example:
        ```python
        config = VideoGenerationConfig(
            input_fragments=InputFragments(duration_in_seconds=5.0),
            output_video=OutputVideo(width=1280)
        )
        ```
    """

    input_fragments: InputFragments = field(default_factory=InputFragments)
    output_video: OutputVideo = field(default_factory=OutputVideo)

    @classmethod
    def from_dict(
        cls, video_generation_config_dict: Dict[str, Any]
    ) -> "VideoGenerationConfig":
        """Creates a VideoGenerationConfig instance from a dictionary configuration."""
        return cls(
            input_fragments=InputFragments.from_dict(
                input_fragments_dict=video_generation_config_dict.get(
                    "input_fragments", {}
                )
            ),
            output_video=OutputVideo.from_dict(
                output_video_dict=video_generation_config_dict.get("output_video", {})
            ),
        )


def _extract_video_and_audio_streams(
    video: VideoMetadata, start_time: float, duration: Optional[float]
) -> Optional[Tuple[Any, Optional[Any]]]:
    """Get video and audio streams from input file."""
    import ffmpeg

    if not video or not video.full_path:
        logger.error("Invalid video metadata")
        return None

    try:
        # Probe video file
        try:
            probe = ffmpeg.probe(filename=video.full_path)
        except ffmpeg.Error as e:
            logger.error(f"Error probing video {video.filename}: {str(e)}")
            return None

        # Get streams from probe
        streams = probe.get("streams")
        if not streams:
            logger.error("No streams found in probe data")
            return None

        # Find stream indices
        video_index = None
        audio_index = None
        for i, stream in enumerate(streams):
            if not isinstance(stream, dict):
                continue
            stream_type = stream.get("codec_type", "")
            if stream_type == "video":
                video_index = i
            elif stream_type == "audio":
                audio_index = i

        if video_index is None:
            logger.warning(f"No video stream found in {video.full_path}")
            return None

        # Create input stream
        try:
            if duration is None:
                # Extract from start_time to end of video
                input_stream = ffmpeg.input(filename=video.full_path, ss=start_time)
            else:
                # Extract specific duration
                input_stream = ffmpeg.input(
                    filename=video.full_path, ss=start_time, t=duration
                )
        except ffmpeg.Error as e:
            logger.error(f"Error creating input stream: {str(e)}")
            return None

        # Process video stream
        try:
            video_stream = input_stream.video.filter("setpts", "PTS-STARTPTS")
        except (AttributeError, TypeError) as e:
            logger.error(f"Error processing video stream: {str(e)}")
            return None

        # Process audio stream if available
        audio_stream = None
        if audio_index is not None:
            try:
                audio_stream = (
                    input_stream["a"]
                    .filter("asetpts", "PTS-STARTPTS")
                    .filter("aresample", 48000)
                )
            except (AttributeError, TypeError) as e:
                logger.warning(f"Error processing audio stream: {str(e)}")

        return (video_stream, audio_stream)

    except (AttributeError, TypeError) as e:
        logger.error(f"Unexpected error processing streams: {str(e)}")
        return None


def _crop_to_enforce_16_9_aspect_ratio(video_stream, width: int, height: int):
    """Enforce 16:9 aspect ratio through cropping."""
    if 9 * width != 16 * height:
        if 9 * width < 16 * height:
            new_height = (width * 9) // 16
            video_stream = video_stream.filter("crop", w="iw", h=new_height, x=0, y=0)
            height = new_height
        else:
            new_width = (height * 16) // 9
            video_stream = video_stream.filter("crop", w=new_width, h="ih", x=0, y=0)
            width = new_width
    return video_stream, width, height


def _crop_region_of_interest(
    video_stream,
    video: VideoMetadata,
    normalized_crop_roi: Tuple[float, float, float, float],
):
    """Apply region of interest cropping."""
    left, top, right, bottom = normalized_crop_roi
    if not (0 <= left < right <= 1) or not (0 <= top < bottom <= 1):
        raise ValueError(
            "normalized_crop_roi values must be between 0 and 1, and left < right, top < bottom"
        )

    crop_w = max(1, min(int((right - left) * video.width), video.width))
    crop_h = max(1, min(int((bottom - top) * video.height), video.height))
    crop_x = max(0, min(int(left * video.width), video.width - crop_w))
    crop_y = max(0, min(int(top * video.height), video.height - crop_h))

    if crop_w <= 0 or crop_h <= 0:
        raise ValueError("Crop dimensions must be positive")

    video_stream = video_stream.filter("crop", w=crop_w, h=crop_h, x=crop_x, y=crop_y)
    return video_stream, crop_w, crop_h


def _add_text_label(video_stream, label: str, date_time_label: DateTimeLabel):
    """Add datetime label to video stream."""
    # Add black outline
    video_stream = video_stream.drawtext(
        text=label,
        fontsize=date_time_label.fontsize,
        fontcolor="black",
        x=f"{date_time_label.left_margin}",
        y=f"{date_time_label.top_margin}",
        borderw=date_time_label.border_width,
    )
    # Add white text
    video_stream = video_stream.drawtext(
        text=label,
        fontsize=date_time_label.fontsize,
        fontcolor="white",
        x=f"{date_time_label.left_margin}",
        y=f"{date_time_label.top_margin}",
        borderw=0,
    )
    return video_stream


def _output_fragment(
    video_stream,
    audio_stream,
    fragment_file: str,
    vcodec: str = "libx264",
):
    """Output processed fragment with video and optional audio."""
    import ffmpeg

    try:
        if audio_stream is not None:
            output = ffmpeg.output(
                video_stream,
                audio_stream,
                fragment_file,
                vcodec=vcodec,
                acodec="aac",
                strict="experimental",
                ac=1,  # mono
                ar=48000,  # sample rate
                ab="128k",  # audio bitrate
                map_metadata=-1,  # clear metadata
            )
        else:
            output = ffmpeg.output(
                video_stream,
                fragment_file,
                vcodec=vcodec,
                strict="experimental",
                map_metadata=-1,  # clear metadata
            )

        ffmpeg.run(
            stream_spec=output,
            overwrite_output=True,
            capture_stderr=True,
        )

    except ffmpeg.Error:
        if audio_stream is not None:
            logger.warning(
                f"Audio processing failed. Falling back to video-only output for fragment {fragment_file}"
            )
            _output_fragment(
                video_stream=video_stream,
                audio_stream=None,
                fragment_file=fragment_file,
                vcodec=vcodec,
            )
        else:
            raise


def _apply_video_transformations(
    video_stream,
    video: VideoMetadata,
    output_width: int,
    normalized_crop_roi: Optional[Tuple[float, float, float, float]] = None,
    enforce_16_9_aspect_ratio: bool = False,
):
    """Apply various video transformations (aspect ratio, cropping, scaling)."""
    video_stream_width = video.width
    video_stream_height = video.height

    # Apply 16:9 aspect ratio if needed
    if enforce_16_9_aspect_ratio:
        video_stream, video_stream_width, video_stream_height = (
            _crop_to_enforce_16_9_aspect_ratio(
                video_stream=video_stream,
                width=video_stream_width,
                height=video_stream_height,
            )
        )

    # Apply ROI cropping if specified
    if normalized_crop_roi:
        video_stream, video_stream_width, video_stream_height = (
            _crop_region_of_interest(
                video_stream=video_stream,
                video=video,
                normalized_crop_roi=normalized_crop_roi,
            )
        )

    # Scale to target dimensions
    target_height = int(output_width * video_stream_height / video_stream_width)
    target_height = (target_height // 2) * 2  # Ensure even height
    video_stream = video_stream.filter(
        "scale", width=output_width, height=target_height
    )

    return video_stream


def _create_one_fragment(
    video: VideoMetadata,
    input_fragments: InputFragments,
    fragment_directory: str,
    output_width: int,
    date_time_label: DateTimeLabel,
    vcodec: str = "libx264",
) -> Optional[str]:
    """Process a single video fragment with all transformations."""

    # Get video/audio streams
    streams = _extract_video_and_audio_streams(
        video=video,
        start_time=input_fragments.offset_in_seconds,
        duration=input_fragments.duration_in_seconds,
    )

    if not streams:
        return None

    video_stream, audio_stream = streams

    # Apply transformations
    video_stream = _apply_video_transformations(
        video_stream=video_stream,
        video=video,
        output_width=output_width,
        normalized_crop_roi=input_fragments.normalized_crop_roi,
        enforce_16_9_aspect_ratio=input_fragments.enforce_16_9_aspect_ratio,
    )

    # Add datetime label
    if date_time_label.draw:
        label: str = f"{video.date_str}    {video.time_str}"
        video_stream = _add_text_label(
            video_stream=video_stream, label=label, date_time_label=date_time_label
        )

    # Output processed fragment
    fragment_file = os.path.join(
        fragment_directory,
        f"{video.date_str} {video.time_str} {video.device}.mp4",
    )
    _output_fragment(
        video_stream=video_stream,
        audio_stream=audio_stream,
        fragment_file=fragment_file,
        vcodec=vcodec,
    )

    return fragment_file


def _create_fragments(
    videos: List[VideoMetadata | None],
    input_fragments: InputFragments,
    fragment_directory: str,
    output_width: int,
    date_time_label: DateTimeLabel,
    vcodec: str = "libx264",
) -> List[str]:
    """Process individual video fragments with transformations."""
    import ffmpeg
    from tqdm import tqdm

    fragment_files = []
    for video in tqdm(
        iterable=videos,
        desc=f"Generating composite video from {len(videos)} videos",
        unit="video",
        colour="green",
        position=0,
        leave=False,
    ):
        if video is None:
            continue
        try:
            fragment_file = _create_one_fragment(
                video=video,
                input_fragments=input_fragments,
                fragment_directory=fragment_directory,
                output_width=output_width,
                date_time_label=date_time_label,
                vcodec=vcodec,
            )
            if fragment_file is not None:
                fragment_files.append(fragment_file)
            else:
                logger.warning(f"Failed to process video fragment for {video.filename}")
        except ffmpeg.Error as e:
            logger.error(f"FFmpeg error processing video {video.filename}:")
            logger.error(e.stderr.decode())
            raise

    return fragment_files


def _concatenate_fragments(
    fragment_files: List[str],
    fragment_directory: str,
    output_file: str,
    target_fps: int,
    vcodec: str = "libx264",
) -> VideoMetadata | None:
    """Concatenate processed fragments into final output video."""
    import ffmpeg

    if not fragment_files:
        logger.warning("No valid video fragments to concatenate")
        return None

    logger.debug(f"Concatenating {len(fragment_files)} video fragments")
    concat_file = os.path.join(fragment_directory, "concat_list.txt")

    with open(concat_file, "w") as f:
        for fragment in fragment_files:
            if fragment is not None:
                f.write(f"file '{fragment}'\n")

    try:
        concat_output = ffmpeg.input(filename=concat_file, format="concat", safe=0)
        output = ffmpeg.output(
            concat_output,
            output_file,
            vcodec=vcodec,
            acodec="aac",
            r=target_fps,
        )
        ffmpeg.run(stream_spec=output, overwrite_output=True, capture_stderr=True)
        logger.debug(f"Successfully concatenated all fragments into {output_file}")

        return VideoMetadata(full_path=output_file)

    except ffmpeg.Error as e:
        logger.error("FFmpeg error during concatenation:")
        logger.error(e.stderr.decode())
        raise


class VideoGenerator:
    """
    Main class for generating composite videos from multiple input videos.

    This class handles the entire video generation process including:
    - Video fragment extraction and processing
    - Applying transformations (cropping, scaling)
    - Adding datetime labels
    - Concatenating fragments into final output

    Example:
        ```python
        generator = VideoGenerator(config=VideoGenerationConfig())
        generator.run(
            videos=[video1, video2],
            output_file="output.mp4"
        )
        ```
    """

    def __init__(self, config: Optional[VideoGenerationConfig] = None) -> None:
        """Initialize generator with optional configuration."""
        self.config = VideoGenerationConfig() if config is None else config

    def run(
        self,
        videos: Union[VideoMetadata, List[VideoMetadata | None]] | None,
        output_file: str,
    ) -> VideoMetadata | None:
        """
        Generate a composite video from multiple input videos.

        Args:
            videos: List of VideoMetadata objects for input videos
            output_file: Path for the output video file
        """
        if videos is None:
            return None

        return self._create_from_fragments(videos=videos, output_file=output_file)

    def _create_from_fragments(
        self,
        videos: Union[VideoMetadata, List[VideoMetadata | None]] | None,
        output_file: str,
    ) -> VideoMetadata | None:
        """
        Internal method to create video from fragments.

        This method handles the core video processing including:
        - Fragment extraction and processing
        - Video/audio stream handling
        - Applying transformations
        - Fragment concatenation
        """
        if videos is None:
            return None
        videos = [videos] if isinstance(videos, VideoMetadata) else videos

        target_fps = compute_default_fps(videos=videos)
        default_width = compute_default_width(
            videos=videos,
            normalized_crop_roi=self.config.input_fragments.normalized_crop_roi,
        )

        output_width = self.config.output_video.width or default_width
        output_width = output_width - (output_width % 2)  # Ensure even width

        # Determine video codec (h264 is default)
        vcodec = (
            "libx265"
            if self.config.output_video.vcodec.lower() in {"h265", "hevc", "libx265"}
            else "libx264"
        )

        fragment_directory: str = os.path.join(
            os.path.dirname(p=output_file), f"video_fragments_{uuid.uuid4().hex[:8]}"
        )
        os.makedirs(name=fragment_directory, exist_ok=True)

        fragment_files: List[str] = _create_fragments(
            videos=videos,
            input_fragments=self.config.input_fragments,
            fragment_directory=fragment_directory,
            output_width=output_width,
            date_time_label=self.config.output_video.date_time_label,
            vcodec=vcodec,
        )

        video: VideoMetadata | None = _concatenate_fragments(
            fragment_files=fragment_files,
            fragment_directory=fragment_directory,
            output_file=output_file,
            target_fps=target_fps,
            vcodec=vcodec,
        )
        delete_directory(directory=fragment_directory)
        return video


def delete_directory(directory: str) -> None:
    """Clean up temporary fragment directory and its contents."""
    try:
        for root, dirs, files in os.walk(top=directory, topdown=False):
            for name in files:
                try:
                    os.remove(path=os.path.join(root, name))
                except (OSError, PermissionError) as e:
                    logger.error(f"Error removing file {name}: {str(e)}")
            for name in dirs:
                try:
                    os.rmdir(path=os.path.join(root, name))
                except (OSError, PermissionError) as e:
                    logger.error(f"Error removing directory {name}: {str(e)}")
        os.rmdir(path=directory)
        logger.debug(f"Fragment directory {directory} removed")
    except (OSError, PermissionError) as e:
        logger.error(
            f"Error during cleanup of fragment directory {directory}: {str(e)}"
        )


def robust_average_iqr(data: List[float]) -> Optional[float]:
    """
    Calculate robust average of a list of floats using IQR method.

    Returns None if no valid FPS values found.
    """

    if not data:
        return None

    # For small lists, return simple average
    n = len(data)
    if n < 4:  # Need at least 4 values for quartile calculation
        return sum(data) / n

    # Sort for quartile calculation
    data.sort()
    q1_idx = n // 4
    q3_idx = (3 * n) // 4

    # Calculate quartiles and IQR
    q1 = data[q1_idx]
    q3 = data[q3_idx]
    iqr = q3 - q1

    # If all values are the same or very close
    if iqr < 0.001:  # Use small epsilon for float comparison
        return data[0]

    # Calculate bounds and filter outliers
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    filtered_data = [v for v in data if lower_bound <= v <= upper_bound]

    if not filtered_data:
        # If all values were filtered out, return average of original values
        return sum(data) / n

    return sum(filtered_data) / len(filtered_data)


def compute_default_fps(videos: List[VideoMetadata | None]) -> int:
    """
    Compute default FPS based on input videos.
    """
    # Compute target FPS
    fps_robust_ave = robust_average_iqr(
        [video.fps for video in videos if video is not None and video.fps > 0]
    )
    if fps_robust_ave is None:
        return 15

    return max(1, round(fps_robust_ave))


def compute_default_width(
    videos: List[VideoMetadata | None],
    normalized_crop_roi: Optional[Tuple[float, float, float, float]] = None,
) -> int:
    """
    Compute default width values based on input videos.
    """
    default_width = 854  # Default width
    width_values = [
        video.width for video in videos if video is not None and video.width > 0
    ]
    if width_values:
        default_width = max(width_values)

    # Adjust width for ROI if specified
    if normalized_crop_roi is not None:
        left, _, right, _ = normalized_crop_roi
        if not 0 <= left < right <= 1:
            raise ValueError(
                "normalized_crop_roi values must be between 0 and 1, and left < right"
            )
        default_width = max(1, min(int((right - left) * default_width), default_width))

    return default_width


if __name__ == "__main__":
    # Testing code for the module.

    def run_example_1():
        logger.info(
            "EXAMPLE 1: Generate a merged videos from fragments picked from videos using identical offsets and "
            "durations. If offsets and durations are identical for all fragments, VideoGenerator can be used in a "
            "single step to generate the final concatenated video. Fragments are generated and concatenated internally."
        )

        # Using one config for all videos does not allow to specify fragment specific offsets, durations and rois
        video_list: List[VideoMetadata | None] = [
            VideoMetadata(
                full_path="/Users/jbouguet/Documents/EufySecurityVideos/record/Batch043/T8162T1024354A8B_20251018085049.mp4"
            ),
            VideoMetadata(
                full_path="/Users/jbouguet/Documents/EufySecurityVideos/record/backup/T8162T1024354A8B_20251022130727.mp4"
                # full_path="/Users/jbouguet/Documents/EufySecurityVideos/record/Batch044/T8162T1024354A8B_20251022130727.mp4"
            ),
        ]
        video_merged_sc: str = (
            "/Users/jbouguet/Documents/EufySecurityVideos/stories/video_merged_sc.mp4"
        )
        # Single call of VideoGenerator.run() does fragment creation and concatenation.
        video_sc = VideoGenerator(
            VideoGenerationConfig(
                input_fragments=InputFragments(
                    offset_in_seconds=14,
                    duration_in_seconds=8.3,
                    normalized_crop_roi=[0.65, 0.25, 0.8, 0.9],
                ),
                output_video=OutputVideo(date_time_label=DateTimeLabel(draw=True)),
            )
        ).run(video_list, video_merged_sc)
        logger.info("Merged video with single config:")
        logger.info(video_sc)

    def run_example_2():
        logger.info(
            "*** EXAMPLE 2: Generate a merged videos from fragments picked from videos using different offsets and "
            "durations. This example shows how VideoGenerator can be used in steps to create custom fragments prior to "
            "concatenation."
        )

        # Define video fragments with specific offsets, durations and rois.
        video_fragments_config = [
            {
                "video_in": "/Users/jbouguet/Documents/EufySecurityVideos/record/Batch041/T8600P1023450AFB_20250923085351.mp4",
                "video_out": "/Users/jbouguet/Documents/EufySecurityVideos/stories/video1.mp4",
                "offset": 8.0,
                "duration": 14.0,
                "roi": [0.08, 0.30, 0.19, 0.57],
            },
            {
                "video_in": "/Users/jbouguet/Documents/EufySecurityVideos/record/Batch041/T8600P1023450AFB_20250923085416.mp4",
                "video_out": "/Users/jbouguet/Documents/EufySecurityVideos/stories/video2.mp4",
                "offset": 0.0,
                "duration": 7.0,
                "roi": [0.08, 0.30, 0.19, 0.57],
            },
        ]

        # Individual video fragments are first created with different configurations.
        video_fragments: List[VideoMetadata | None] = [
            VideoGenerator(
                VideoGenerationConfig(
                    input_fragments=InputFragments(
                        offset_in_seconds=config["offset"],
                        duration_in_seconds=config["duration"],
                        normalized_crop_roi=config["roi"],
                    )
                )
            ).run(
                VideoMetadata(full_path=config["video_in"]),
                config["video_out"],
            )
            for config in video_fragments_config
        ]

        # The final concatenated videos is then created from the list of video fragments.
        video_merged_mc: str = (
            "/Users/jbouguet/Documents/EufySecurityVideos/stories/video_merged_mc.mp4"
        )
        video_mc = VideoGenerator().run(video_fragments, video_merged_mc)
        logger.info("Merged video with multiple configs:")
        logger.info(video_mc)

    def run_example_3():
        logger.info(
            "*** EXAMPLE 3: Re-encoding a video using h264 and h265 video codecs."
        )
        video_hevc: str = (
            "/Users/jbouguet/Documents/EufySecurityVideos/record/Batch043/T8162T1024354A8B_20251003021936.mp4"
        )
        video_h264: str = (
            "/Users/jbouguet/Documents/EufySecurityVideos/stories/T8162T1024354A8B_20251003021936_h264.mp4"
        )
        video_h265: str = (
            "/Users/jbouguet/Documents/EufySecurityVideos/stories/T8162T1024354A8B_20251003021936_h265.mp4"
        )
        video_hevc_meta = VideoMetadata(full_path=video_hevc)
        h264_encoder = VideoGenerator()
        h265_encoder = VideoGenerator(
            VideoGenerationConfig(output_video=OutputVideo(vcodec="hevc"))
        )
        video_h264_meta = h264_encoder.run(video_hevc_meta, video_h264)
        video_h265_meta = h265_encoder.run(video_hevc_meta, video_h265)

        logger.info("Video original:")
        logger.info(video_hevc_meta)
        logger.info("Video h264 encoded:")
        logger.info(video_h264_meta)
        logger.info("Video h265 encoded:")
        logger.info(video_h265_meta)

    # Execute test examples
    run_example_1()
    run_example_2()
    run_example_3()
