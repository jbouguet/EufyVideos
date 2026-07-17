#!/usr/bin/env python3
"""
Quick performance/detection comparison between YOLO11 and YOLO26.

Runs both model families (matched by size, e.g. 'n' or 'x') on the same
sample video(s) from the "2024-11-18 - Backyard Planning" story and reports
FPS and detection counts side by side.

Usage:
    python test_yolo11_vs_yolo26.py --size n --frames 30
    python test_yolo11_vs_yolo26.py --size x --frames 30 --task track
"""

import argparse
import logging

from logging_config import create_logger, set_all_loggers_level_and_format
from performance_benchmarks import PerformanceBenchmark
from test_optimization import find_sample_videos, find_video_paths

logger = create_logger(__name__)


def run_comparison(video_paths: list, size: str, num_frames: int, task: str) -> None:
    if not video_paths:
        logger.error("No video paths provided for testing")
        return

    models = [f"yolo11{size}.pt", f"yolo26{size}.pt"]
    benchmark = PerformanceBenchmark()

    for i, video_path in enumerate(video_paths, 1):
        logger.info(f"\n--- Video {i}/{len(video_paths)}: {video_path} ---")
        results = benchmark.benchmark_different_models(
            video_path=video_path, num_frames=num_frames, models=models, task=task
        )

        print(f"\n{'Model':<15}{'FPS':>10}{'Time (s)':>12}{'Detections':>14}")
        for model_name, result in results.items():
            print(
                f"{model_name:<15}{result.fps:>10.2f}{result.total_time:>12.2f}"
                f"{result.detections_found:>14}"
            )

    print("\n" + benchmark.generate_report())
    benchmark.save_results(f"yolo11_vs_yolo26_{size}_{task}_results.json")


def main():
    parser = argparse.ArgumentParser(description="Compare YOLO11 vs YOLO26 performance")
    parser.add_argument("--size", default="n", choices=["n", "s", "m", "l", "x"],
                         help="Model size to compare (default: n)")
    parser.add_argument("--frames", type=int, default=30, help="Number of frames to sample")
    parser.add_argument("--task", default="detect", choices=["detect", "track"])
    parser.add_argument("--video", help="Explicit video path (skips story lookup)")
    parser.add_argument("--num-videos", type=int, default=2,
                         help="Number of sample videos to use when --video is not given")
    args = parser.parse_args()

    set_all_loggers_level_and_format(level=logging.INFO, extended_format=False)

    if args.video:
        video_paths = [args.video]
    else:
        video_files = find_sample_videos()
        video_paths = find_video_paths(video_files)[: args.num_videos]

    if not video_paths:
        logger.error("No video paths found - aborting")
        return

    run_comparison(video_paths, args.size, args.frames, args.task)


if __name__ == "__main__":
    main()
