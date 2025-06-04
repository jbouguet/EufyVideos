#!/usr/bin/env python3
"""
Test script for YOLO optimization on the "2024-11-18 - Backyard Planning" story.

This script validates that our optimizations work correctly and measure performance
improvements on the specific story mentioned in the analysis_config.yaml file.
"""

import os
import sys
import logging

from logging_config import create_logger, set_all_loggers_level_and_format
from performance_benchmarks import PerformanceBenchmark
from object_detector_yolo import YoloObjectDetector
from object_detector_yolo_optimized import OptimizedYoloObjectDetector

logger = create_logger(__name__)


def find_sample_videos(config_file: str = "analysis_config.yaml") -> list:
    """
    Find the sample videos from the "2024-11-18 - Backyard Planning" story.
    
    Returns:
        List of video file paths from the story configuration
    """
    from video_analyzer import load_yaml_with_substitution
    
    if not os.path.exists(config_file):
        logger.error(f"Configuration file not found: {config_file}")
        return []
    
    config = load_yaml_with_substitution(config_file)
    
    # Find the specific story
    target_story = None
    for story in config.get('stories', []):
        if story.get('name') == '2024-11-18 - Backyard Planning - 5 videos':
            target_story = story
            break
    
    if not target_story:
        logger.error("Could not find '2024-11-18 - Backyard Planning - 5 videos' story")
        return []
    
    # Extract filenames from selectors
    video_files = []
    for selector in target_story.get('selectors', []):
        if 'filenames' in selector:
            video_files.extend(selector['filenames'])
    
    logger.info(f"Found {len(video_files)} videos in the sample story")
    for video in video_files:
        logger.info(f"  - {video}")
    
    return video_files


def find_video_paths(video_files: list, config_file: str = "analysis_config.yaml") -> list:
    """
    Find full paths to the video files by searching the configured directories.
    
    Args:
        video_files: List of video filenames
        config_file: Path to configuration file
        
    Returns:
        List of full paths to video files
    """
    from video_analyzer import load_yaml_with_substitution
    
    config = load_yaml_with_substitution(config_file)
    
    # Get video directories from database list
    video_directories = []
    for db_config in config.get('video_database_list', []):
        if 'video_directories' in db_config:
            dirs = db_config['video_directories']
            if isinstance(dirs, list):
                video_directories.extend(dirs)
            else:
                video_directories.append(dirs)
    
    logger.info(f"Searching in {len(video_directories)} directories...")
    
    # Search for video files
    found_videos = []
    for video_file in video_files:
        found = False
        for directory in video_directories:
            video_path = os.path.join(directory, video_file)
            if os.path.exists(video_path):
                found_videos.append(video_path)
                logger.info(f"Found: {video_path}")
                found = True
                break
        
        if not found:
            logger.warning(f"Video not found: {video_file}")
    
    return found_videos


def test_basic_functionality():
    """Test that both detectors work correctly."""
    logger.info("Testing basic detector functionality...")
    
    # Test original detector
    try:
        YoloObjectDetector(model_name="yolo11n.pt")  # Use smaller model for testing
        logger.info("✓ Original detector initialized successfully")
    except Exception as e:
        logger.error(f"✗ Original detector failed: {e}")
        return False
    
    # Test optimized detector
    try:
        optimized_detector = OptimizedYoloObjectDetector(model_name="yolo11n.pt", batch_size=4)
        logger.info("✓ Optimized detector initialized successfully")
        
        # Test GPU detection
        stats = optimized_detector.get_performance_stats()
        logger.info(f"Device: {stats['device']}")
        logger.info(f"GPU available: {stats['gpu_available']}")
        logger.info(f"MPS available: {stats['mps_available']}")
        
    except Exception as e:
        logger.error(f"✗ Optimized detector failed: {e}")
        return False
    
    return True


def run_optimization_test(video_paths: list):
    """
    Run optimization test on the sample videos.
    
    Args:
        video_paths: List of video file paths to test
    """
    if not video_paths:
        logger.error("No video paths provided for testing")
        return
    
    logger.info(f"Running optimization test on {len(video_paths)} videos")
    
    benchmark = PerformanceBenchmark()
    
    # Test on a subset of videos (to avoid long test times)
    test_videos = video_paths[:2]  # Test first 2 videos
    
    for i, video_path in enumerate(test_videos, 1):
        logger.info(f"\n--- Testing video {i}/{len(test_videos)}: {os.path.basename(video_path)} ---")
        
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            continue
        
        try:
            # Compare detectors on this video
            results = benchmark.compare_detectors(
                video_path=video_path,
                num_frames=30,  # Test with 30 frames
                model_name="yolo11n.pt"  # Use smaller model for faster testing
            )
            
            logger.info(f"Results for {os.path.basename(video_path)}:")
            for detector_type, result in results.items():
                logger.info(f"  {detector_type}: {result.fps:.1f} FPS, {result.detections_found} detections")
        
        except Exception as e:
            logger.error(f"Error testing {video_path}: {e}")
            continue
    
    # Generate and save report
    report = benchmark.generate_report()
    print("\n" + "="*60)
    print("OPTIMIZATION TEST REPORT")
    print("="*60)
    print(report)
    
    # Save detailed results
    output_file = "optimization_test_results.json"
    benchmark.save_results(output_file)
    logger.info(f"Detailed results saved to: {output_file}")


def main():
    """Main test function."""
    # Set up logging
    set_all_loggers_level_and_format(level=logging.INFO, extended_format=False)
    
    logger.info("Starting YOLO optimization test...")
    
    # Test 1: Basic functionality
    if not test_basic_functionality():
        logger.error("Basic functionality test failed - aborting")
        sys.exit(1)
    
    # Test 2: Find sample videos
    video_files = find_sample_videos()
    if not video_files:
        logger.error("No sample videos found - aborting")
        sys.exit(1)
    
    # Test 3: Find video paths
    video_paths = find_video_paths(video_files)
    if not video_paths:
        logger.error("No video paths found - aborting")
        sys.exit(1)
    
    # Test 4: Run optimization test
    run_optimization_test(video_paths)
    
    logger.info("Optimization test completed!")


if __name__ == "__main__":
    main()