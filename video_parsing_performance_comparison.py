#!/usr/bin/env python3

import argparse
import io
import os
import random
import sys
import time
from datetime import datetime, timedelta
from typing import Optional

import cv2
import ffmpeg
from tqdm import tqdm

from config import Config
from logging_config import create_logger
from video_analyzer import AnalysisConfig
from video_metadata import VideoMetadata

logger = create_logger(__name__)


class VideoParser:

    @staticmethod
    def get_video_properties(
        file_path: str, library: str = "LibAV"
    ) -> Optional[VideoMetadata]:
        try:
            match library.lower():
                case "opencv":
                    return VideoParser.get_video_properties_cv2(file_path)
                case "ffmpeg":
                    return VideoParser.get_video_properties_ffmpeg(file_path)
                case "libav":
                    return VideoParser.get_video_properties_libav(file_path)
                case _:
                    logger.warning(f"Invalid file parsing library method {library}")
                    return None
        except Exception as e:
            logger.warning(f"Error parsing file {file_path}: {str(e)}")
            return None

    @staticmethod
    def get_video_properties_cv2(file_path: str) -> Optional[VideoMetadata]:
        try:
            # Parse basic information from the filename only
            filename = os.path.basename(file_path)
            serial, datetime_part = filename.split("_")
            device = Config.get_device_dict().get(serial, serial)
            date = datetime.strptime(datetime_part[:8], "%Y%m%d")
            time_obj = datetime.strptime(datetime_part[8:14], "%H%M%S")

            # Redirect stderr to capture Cv2 error messages
            old_stderr = sys.stderr
            sys.stderr = io.StringIO()

            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                return None

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = frame_count / fps if fps > 0 else 0
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
            video_codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])

            cap.release()

            # Check for any error messages
            error_output = sys.stderr.getvalue()
            if error_output:
                logger.warning(
                    f"opencv warnings/errors for file {file_path}:\n{error_output}"
                )

            return VideoMetadata(
                filename=filename,
                full_path=file_path,
                device=device,
                date=date,
                time=time_obj,
                serial=serial,
                file_size=os.path.getsize(file_path) / (1024 * 1024),  # Size in MB,
                width=width,
                height=height,
                frame_count=frame_count,
                duration=timedelta(seconds=float(duration)),
                fps=fps,
                video_codec=video_codec,
            )

        except Exception as e:
            logger.error(
                f"Error in get_video_properties_cv2 for file {file_path}: {str(e)}"
            )
            return None

        finally:
            # Restore stderr
            sys.stderr = old_stderr

    @staticmethod
    def get_video_properties_libav(file_path: str) -> Optional[VideoMetadata]:
        return VideoMetadata.from_video_file(file_path)

    @staticmethod
    def get_video_properties_ffmpeg(file_path: str) -> Optional[VideoMetadata]:
        try:
            # Parse basic information from the filename only
            filename = os.path.basename(file_path)
            serial, datetime_part = filename.split("_")
            device = Config.get_device_dict().get(serial, serial)
            date = datetime.strptime(datetime_part[:8], "%Y%m%d")
            time_obj = datetime.strptime(datetime_part[8:14], "%H%M%S")

            # Redirect stderr to capture ffmpeg error messages
            old_stderr = sys.stderr
            sys.stderr = io.StringIO()

            probe = ffmpeg.probe(file_path)
            video_stream = next(
                (
                    stream
                    for stream in probe["streams"]
                    if stream["codec_type"] == "video"
                ),
                None,
            )
            # Check for any error messages

            error_output = sys.stderr.getvalue()
            if error_output:
                logger.warning(
                    f"ffmpeg warnings/errors for file {file_path}:\n{error_output}"
                )

            if video_stream is None:
                logger.warning(f"No video stream found in file: {file_path}")
                return None

            duration = float(probe["format"]["duration"])
            fps = eval(
                video_stream["r_frame_rate"]
            )  # This is usually a string like '30000/1001'
            video_codec = video_stream["codec_name"]

            return VideoMetadata(
                filename=filename,
                full_path=file_path,
                device=device,
                date=date,
                time=time_obj,
                serial=serial,
                file_size=os.path.getsize(file_path) / (1024 * 1024),  # Size in MB,
                width=int(video_stream["width"]),
                height=int(video_stream["height"]),
                frame_count=(
                    int(float(video_stream["nb_frames"]))
                    if "nb_frames" in video_stream
                    else int(duration * fps)
                ),
                duration=timedelta(seconds=float(duration)),
                fps=fps,
                video_codec=video_codec,
            )

        except Exception as e:
            logger.error(
                f"Error in get_video_properties_ffmpeg for file {file_path}: {str(e)}"
            )
            return None

        finally:
            # Restore stderr
            sys.stderr = old_stderr

    # OpenCV - Total Time: 243.4420 seconds, Processed Files: 23274, Average Time per File: 0.0105 seconds
    # LibAV - Total Time: 86.9873 seconds, Processed Files: 23274, Average Time per File: 0.0037 seconds
    # FFmpeg - Total Time: 1218.6489 seconds, Processed Files: 23274, Average Time per File: 0.0524 seconds
    @staticmethod
    def compare_performance(video_directory: str, max_files: int = 1000):
        methods = [
            ("OpenCV", VideoParser.get_video_properties_cv2),
            ("LibAV", VideoParser.get_video_properties_libav),
            ("FFmpeg", VideoParser.get_video_properties_ffmpeg),
        ]
        results = {name: {"time": 0, "count": 0} for name, _ in methods}

        # Get the list of MP4 files
        mp4_files = [f for f in os.listdir(video_directory) if f.endswith(".mp4")]
        total_files = len(mp4_files)

        # If max_files is specified and less than total files, randomly select a subset
        if max_files and max_files < total_files:
            selected_indices = sorted(random.sample(range(total_files), max_files))
            mp4_files = [mp4_files[i] for i in selected_indices]
            total_files = max_files

        logger.info(
            f"Processing {'randomly selected ' if max_files else ''}{total_files} mp4 files found in {video_directory}..."
        )

        with tqdm(
            total=total_files, desc="Percentage completed", unit="file", colour="red"
        ) as pbar:
            for filename in mp4_files:
                file_path = os.path.join(video_directory, filename)
                for name, method in methods:
                    start_time = time.time()
                    metadata = method(file_path)
                    end_time = time.time()
                    if metadata:
                        results[name]["time"] += end_time - start_time
                        results[name]["count"] += 1
                # Update the progress bar
                pbar.update(1)

        for name, data in results.items():
            if data["count"] > 0:
                avg_time = data["time"] / data["count"]
                logger.info(
                    f"{name} - Total Time: {data['time']:.4f} seconds, Processed Files: {data['count']}, Average Time per File: {avg_time:.4f} seconds"
                )
            else:
                logger.warning(f"{name} - Failed to process any files")

        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run video analysis")
    parser.add_argument(
        "--config",
        default="analysis_config.yaml",
        help="Path to the configuration file",
    )
    parser.add_argument(
        "--num_files", default=100, help="Max number of files to test on"
    )
    args = parser.parse_args()
    if not os.path.exists(args.config):
        logger.error(f"Configuration file {args.config} not found.")
        exit(1)
    try:
        config = AnalysisConfig.from_file(args.config)
        max_num_files: int = int(args.num_files)
        for video_directory in config.video_directories:
            VideoParser.compare_performance(video_directory, max_num_files)

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
