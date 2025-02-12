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
            output_video_codec="h264"
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

from config import Config
from logging_config import create_logger
from video_metadata import VideoMetadata

logger = create_logger(__name__)


@dataclass
class InputFragments:
    """
    Configuration for input video fragment processing.

    Attributes:
        offset_in_seconds: Start time offset for video trimming
        duration_in_seconds: Duration of video segment to extract
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
    duration_in_seconds: float = 1.0
    normalized_crop_roi: Optional[Tuple[float, float, float, float]] = None
    enforce_16_9_aspect_ratio: bool = False

    @classmethod
    def from_dict(cls, input_fragments_dict: Dict[str, Any]) -> "InputFragments":
        """Creates an InputFragments instance from a dictionary configuration."""
        normalized_crop_roi = input_fragments_dict.get("normalized_crop_roi")
        if normalized_crop_roi is not None:
            try:
                normalized_crop_roi = [float(x) for x in normalized_crop_roi]
                if len(normalized_crop_roi) != 4:
                    raise ValueError(
                        f"normalized_crop_roi must have exactly 4 values, got {len(normalized_crop_roi)}"
                    )
            except ValueError as e:
                logger.error(f"Error parsing normalized_crop_roi': {str(e)}")
                normalized_crop_roi = None
        return cls(
            offset_in_seconds=input_fragments_dict.get("offset_in_seconds", 0.0),
            duration_in_seconds=input_fragments_dict.get("duration_in_seconds", 1.0),
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

    draw: bool = True
    fontsize: int = Config.DATE_TIME_LABEL_FONTSIZE
    left_margin: int = Config.DATE_TIME_LABEL_LEFT_MARGIN
    top_margin: int = Config.DATE_TIME_LABEL_TOP_MARGIN
    border_width: int = Config.DATE_TIME_LABEL_BORDER_WIDTH

    @classmethod
    def from_dict(cls, date_time_label_dict: Dict[str, Any]) -> "DateTimeLabel":
        """Creates a DateTimeLabel instance from a dictionary configuration."""
        return cls(
            draw=date_time_label_dict.get("draw", True),
            fontsize=date_time_label_dict.get(
                "fontsize", Config.DATE_TIME_LABEL_FONTSIZE
            ),
            left_margin=date_time_label_dict.get(
                "left_margin", Config.DATE_TIME_LABEL_LEFT_MARGIN
            ),
            top_margin=date_time_label_dict.get(
                "top_margin", Config.DATE_TIME_LABEL_TOP_MARGIN
            ),
            border_width=date_time_label_dict.get(
                "border_width", Config.DATE_TIME_LABEL_BORDER_WIDTH
            ),
        )


@dataclass
class OutputVideo:
    """
    Configuration for output video generation.

    Attributes:
        width: Target width for the output video (height auto-calculated)
        date_time_label: Configuration for datetime label rendering
        output_video_codec: Video codec to use ('h264' or 'h265')

    Example:
        ```python
        output = OutputVideo(
            width=1920,
            date_time_label=DateTimeLabel(),
            output_video_codec="h264"
        )
        ```
    """

    width: Optional[int] = None
    date_time_label: DateTimeLabel = field(default_factory=DateTimeLabel)
    output_video_codec: Optional[str] = "h264"

    @classmethod
    def from_dict(cls, output_video_dict: Dict[str, Any]) -> "OutputVideo":
        """Creates an OutputVideo instance from a dictionary configuration."""
        return cls(
            width=output_video_dict.get("width"),
            date_time_label=DateTimeLabel.from_dict(
                date_time_label_dict=output_video_dict.get("date_time_label", {})
            ),
            output_video_codec=output_video_dict.get("output_video_codec", "h264"),
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

    def __init__(self, config: VideoGenerationConfig = None) -> None:
        """Initialize generator with optional configuration."""
        self.config = config or VideoGenerationConfig()

    def run(
        self, videos: Union[VideoMetadata, List[VideoMetadata]], output_file: str
    ) -> None:
        """
        Generate a composite video from multiple input videos.

        Args:
            videos: List of VideoMetadata objects for input videos
            output_file: Path for the output video file
        """
        temp_dir = os.path.join(
            os.path.dirname(p=output_file), f"video_fragments_{uuid.uuid4().hex[:8]}"
        )
        self._create_from_video_fragments(
            videos=videos, output_file=output_file, fragment_directory=temp_dir
        )
        cleanup_fragment_directory(fragment_directory=temp_dir)

    def _create_from_video_fragments(
        self,
        videos: Union[VideoMetadata, List[VideoMetadata]],
        output_file: str,
        fragment_directory: str,
    ) -> None:
        """
        Internal method to create video from fragments.

        This method handles the core video processing including:
        - Fragment extraction and processing
        - Video/audio stream handling
        - Applying transformations
        - Fragment concatenation
        """
        videos = [videos] if isinstance(videos, VideoMetadata) else videos
        os.makedirs(name=fragment_directory, exist_ok=True)

        default_width, target_fps = compute_default_fps_and_width(
            videos=videos,
            normalized_crop_roi=self.config.input_fragments.normalized_crop_roi,
        )

        output_width = self.config.output_video.width or default_width
        output_width = output_width - (output_width % 2)  # Ensure even width

        ffmpeg_vcodec = (
            "libx265"
            if self.config.output_video.output_video_codec.lower() == "h265"
            else "libx264"
        )

        try:
            fragment_files = self._process_video_fragments(
                videos=videos,
                fragment_directory=fragment_directory,
                output_width=output_width,
                ffmpeg_vcodec=ffmpeg_vcodec,
            )

            self._concatenate_fragments(
                fragment_files=fragment_files,
                fragment_directory=fragment_directory,
                output_file=output_file,
                ffmpeg_vcodec=ffmpeg_vcodec,
                target_fps=target_fps,
            )

        except Exception as e:
            logger.error(f"Error in create_video_from_fragments: {str(e)}")
            raise

    def _process_video_fragments(
        self,
        videos: List[VideoMetadata],
        fragment_directory: str,
        output_width: int,
        ffmpeg_vcodec: str,
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
            try:
                fragment_file = self._process_single_fragment(
                    video=video,
                    fragment_directory=fragment_directory,
                    output_width=output_width,
                    ffmpeg_vcodec=ffmpeg_vcodec,
                )
                fragment_files.append(fragment_file)
            except ffmpeg.Error as e:
                logger.error(f"FFmpeg error processing video {video.filename}:")
                logger.error(e.stderr.decode())
                raise

        return fragment_files

    def _process_single_fragment(
        self,
        video: VideoMetadata,
        fragment_directory: str,
        output_width: int,
        ffmpeg_vcodec: str,
    ) -> str:
        """Process a single video fragment with all transformations."""

        fragment_file = os.path.join(
            fragment_directory,
            f"{video.date_str} {video.time_str} {video.device}.mp4",
        )

        # Get video/audio streams
        streams = self._get_video_audio_streams(
            video=video,
            start_time=self.config.input_fragments.offset_in_seconds,
            duration=self.config.input_fragments.duration_in_seconds,
        )

        if not streams:
            return None

        video_stream, audio_stream = streams

        # Apply transformations
        video_stream = self._apply_video_transformations(
            video_stream=video_stream, video=video, output_width=output_width
        )

        # Add datetime label
        video_stream = self._add_datetime_label(video_stream=video_stream, video=video)

        # Output processed fragment
        self._output_processed_fragment(
            video_stream=video_stream,
            audio_stream=audio_stream,
            fragment_file=fragment_file,
            ffmpeg_vcodec=ffmpeg_vcodec,
        )

        return fragment_file

    def _get_video_audio_streams(
        self, video: VideoMetadata, start_time: float, duration: float
    ):
        """Get video and audio streams from input file."""
        import ffmpeg

        try:
            probe = ffmpeg.probe(filename=video.full_path)

            video_index = None
            audio_index = None
            for i, stream in enumerate(iterable=probe["streams"]):
                if stream["codec_type"] == "video":
                    video_index = i
                elif stream["codec_type"] == "audio":
                    audio_index = i

            if video_index is None:
                logger.warning(msg=f"No video stream found in {video.full_path}.")
                return None

            # Input with trimming
            input_stream = ffmpeg.input(
                filename=video.full_path, ss=start_time, t=duration
            )

            video_stream = input_stream.video.filter("setpts", "PTS-STARTPTS")

            audio_stream = None
            if audio_index is not None:
                audio_stream = (
                    input_stream["a"]
                    .filter("asetpts", "PTS-STARTPTS")
                    .filter("aresample", 48000)
                )

            return video_stream, audio_stream

        except ffmpeg.Error as e:
            logger.error(f"Error probing video {video.filename}: {str(e)}")
            return None

    def _apply_video_transformations(
        self, video_stream, video: VideoMetadata, output_width: int
    ):
        """Apply various video transformations (aspect ratio, cropping, scaling)."""
        video_stream_width = video.width
        video_stream_height = video.height

        # Apply 16:9 aspect ratio if needed
        if self.config.input_fragments.enforce_16_9_aspect_ratio:
            video_stream, video_stream_width, video_stream_height = (
                self._enforce_aspect_ratio(
                    video_stream=video_stream,
                    width=video_stream_width,
                    height=video_stream_height,
                )
            )

        # Apply ROI cropping if specified
        if self.config.input_fragments.normalized_crop_roi:
            video_stream, video_stream_width, video_stream_height = (
                self._apply_roi_crop(
                    video_stream=video_stream,
                    video=video,
                    normalized_crop_roi=self.config.input_fragments.normalized_crop_roi,
                )
            )

        # Scale to target dimensions
        target_height = int(output_width * video_stream_height / video_stream_width)
        target_height = (target_height // 2) * 2  # Ensure even height
        video_stream = video_stream.filter(
            "scale", width=output_width, height=target_height
        )

        return video_stream

    def _enforce_aspect_ratio(self, video_stream, width: int, height: int):
        """Enforce 16:9 aspect ratio through cropping."""
        if 9 * width != 16 * height:
            if 9 * width < 16 * height:
                new_height = (width * 9) // 16
                video_stream = video_stream.filter(
                    "crop", w="iw", h=new_height, x=0, y=0
                )
                height = new_height
            else:
                new_width = (height * 16) // 9
                video_stream = video_stream.filter(
                    "crop", w=new_width, h="ih", x=0, y=0
                )
                width = new_width
        return video_stream, width, height

    def _apply_roi_crop(
        self, video_stream, video: VideoMetadata, normalized_crop_roi: tuple
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

        video_stream = video_stream.filter(
            "crop", w=crop_w, h=crop_h, x=crop_x, y=crop_y
        )
        return video_stream, crop_w, crop_h

    def _add_datetime_label(self, video_stream, video: VideoMetadata):
        """Add datetime label to video stream."""
        date_time_label = self.config.output_video.date_time_label

        if date_time_label.draw:
            text = f"{video.date_str}    {video.time_str}"

            # Add black outline
            video_stream = video_stream.drawtext(
                text=text,
                fontsize=date_time_label.fontsize,
                fontcolor="black",
                x=f"{date_time_label.left_margin}",
                y=f"{date_time_label.top_margin}",
                borderw=date_time_label.border_width,
            )
            # Add white text
            video_stream = video_stream.drawtext(
                text=text,
                fontsize=date_time_label.fontsize,
                fontcolor="white",
                x=f"{date_time_label.left_margin}",
                y=f"{date_time_label.top_margin}",
                borderw=0,
            )
        return video_stream

    def _output_processed_fragment(
        self, video_stream, audio_stream, fragment_file: str, ffmpeg_vcodec: str
    ):
        """Output processed fragment with video and optional audio."""
        import ffmpeg

        try:
            if audio_stream is not None:
                output = ffmpeg.output(
                    video_stream,
                    audio_stream,
                    fragment_file,
                    vcodec=ffmpeg_vcodec,
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
                    vcodec=ffmpeg_vcodec,
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
                self._output_processed_fragment(
                    video_stream=video_stream,
                    audio_stream=None,
                    fragment_file=fragment_file,
                    ffmpeg_vcodec=ffmpeg_vcodec,
                )
            else:
                raise

    def _concatenate_fragments(
        self,
        fragment_files: List[str],
        fragment_directory: str,
        output_file: str,
        ffmpeg_vcodec: str,
        target_fps: int,
    ):
        """Concatenate processed fragments into final output video."""
        import ffmpeg

        logger.debug("Concatenating video fragments")
        concat_file = os.path.join(fragment_directory, "concat_list.txt")

        with open(concat_file, "w") as f:
            for fragment in fragment_files:
                f.write(f"file '{fragment}'\n")

        try:
            concat_output = ffmpeg.input(filename=concat_file, format="concat", safe=0)
            output = ffmpeg.output(
                concat_output,
                output_file,
                vcodec=ffmpeg_vcodec,
                acodec="aac",
                r=target_fps,
            )
            ffmpeg.run(stream_spec=output, overwrite_output=True, capture_stderr=True)
            logger.debug(f"Successfully concatenated all fragments into {output_file}")

        except ffmpeg.Error as e:
            logger.error("FFmpeg error during concatenation:")
            logger.error(e.stderr.decode())
            raise


def cleanup_fragment_directory(fragment_directory: str) -> None:
    """Clean up temporary fragment directory and its contents."""
    try:
        for root, dirs, files in os.walk(top=fragment_directory, topdown=False):
            for name in files:
                try:
                    os.remove(path=os.path.join(root, name))
                except Exception as e:
                    logger.error(f"Error removing file {name}: {str(e)}")
            for name in dirs:
                try:
                    os.rmdir(path=os.path.join(root, name))
                except Exception as e:
                    logger.error(f"Error removing directory {name}: {str(e)}")
        os.rmdir(path=fragment_directory)
        logger.debug(f"Fragment directory {fragment_directory} removed")
    except Exception as e:
        logger.error(
            f"Error during cleanup of fragment directory {fragment_directory}: {str(e)}"
        )


def fps_robust_average(videos: List[VideoMetadata]) -> None | float:
    """
    Calculate robust average FPS from video list using IQR method.

    Returns None if no valid FPS values found.
    """
    if not videos:
        return None
    fps_values = [video.fps for video in videos if video.fps > 0]
    if not fps_values:
        return None

    fps_values.sort()
    n = len(fps_values)
    q1 = fps_values[n // 4]
    q3 = fps_values[(3 * n) // 4]
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    filtered_fps = [fps for fps in fps_values if lower_bound <= fps <= upper_bound]
    return (
        sum(filtered_fps) / len(filtered_fps) if filtered_fps else sum(fps_values) / n
    )


def width_minimum(videos: List[VideoMetadata]) -> None | int:
    """Get minimum width from video list. Returns None if no valid widths found."""
    if not videos:
        return None
    width_values = [video.width for video in videos]
    if not width_values:
        return None
    return min(width_values)


def compute_default_fps_and_width(
    videos: List[VideoMetadata], normalized_crop_roi: tuple = None
) -> Tuple[int]:
    """
    Compute default FPS and width values based on input videos.

    Args:
        videos: List of input videos
        normalized_crop_roi: Optional ROI for width adjustment

    Returns:
        Tuple of (default_width, target_fps)
    """
    target_fps = 15  # Default FPS
    default_width = 854  # Default width

    # Compute target FPS
    fps_robust_ave = fps_robust_average(videos=videos)
    if fps_robust_ave is None:
        logger.warning(
            f"No valid FPS data available. Using default FPS of {target_fps}."
        )
    else:
        target_fps = round(number=fps_robust_ave)
        logger.debug(
            f"Using target FPS of {target_fps} based on input video statistics."
        )

    # Compute default width
    width_min = width_minimum(videos=videos)
    if width_min is None:
        logger.warning("No valid video width data available.")
        logger.debug(f"Default video width is {default_width}")
    else:
        default_width = width_min

    # Adjust width for ROI if specified
    if normalized_crop_roi:
        left, _, right, _ = normalized_crop_roi
        if not (0 <= left < right <= 1):
            raise ValueError(
                "normalized_crop_roi values must be between 0 and 1, and left < right"
            )
        default_width = max(1, min(int((right - left) * default_width), default_width))

    logger.debug(
        f"Default video width is {default_width} based on input video statistics"
    )

    return default_width, target_fps


if __name__ == "__main__":
    # Testing code for the module.
    import sys

    from tag_processor import TaggerConfig, TagProcessor
    from tag_visualizer import TagVisualizer, TagVisualizerConfig

    video_file: str = (
        "/Users/jbouguet/Documents/EufySecurityVideos/record/Batch022/T8600P102338033E_20240930085536.mp4"
    )
    out_dir: str = "/Users/jbouguet/Documents/EufySecurityVideos/stories"
    tag_video: str = os.path.join(
        out_dir, "T8600P102338033E_20240930085536_crops_tags.mp4"
    )

    # Define crop configurations
    crop_configs = [
        {"duration": 10.0, "width": 1600},
        {"duration": 10.0, "width": 1600},
        {"duration": 10.0, "width": 1600},
    ]

    # Initialize variables for crop generation
    videos = []
    current_offset = 0.0

    logger.info(f"Cropping source video {video_file}")
    video_in = [VideoMetadata.from_video_file(video_file)]

    # Generate crops
    for i, config in enumerate(crop_configs, 1):
        crop_config = VideoGenerationConfig(
            input_fragments=InputFragments(
                offset_in_seconds=current_offset, duration_in_seconds=config["duration"]
            ),
            output_video=OutputVideo(
                width=config["width"], date_time_label=DateTimeLabel(draw=False)
            ),
        )
        crop_output = os.path.join(
            out_dir, f"T8600P102338033E_20240930085536_crop{i}.mp4"
        )
        logger.info(f"Creating cropped video {crop_output}")
        VideoGenerator(crop_config).run(video_in, crop_output)
        videos.append(VideoMetadata.from_video_file(crop_output))

        # Update offset for the next crop
        current_offset += config["duration"]

    logger.info(
        f"Generating video {tag_video} showing extracted tags in the cropped videos."
    )
    viz = TagVisualizer(TagVisualizerConfig(output_size={"width": 1600, "height": 900}))
    proc = TagProcessor(
        TaggerConfig(
            model="Yolo11x", task="Track", num_frames_per_second=5, conf_threshold=0.2
        )
    )
    viz.run(proc.run(videos).dedupe().to_videos(videos), tag_video)

    sys.exit()
