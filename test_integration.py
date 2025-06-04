#!/usr/bin/env python3
"""
Integration test script for optimized YOLO detector with existing workflow.

This script demonstrates how to use the optimized detector within the existing
story processing framework, comparing performance between original and optimized
approaches.
"""

import logging
import os
import tempfile
import time
from pathlib import Path

from logging_config import create_logger, set_all_loggers_level_and_format
from story_creator import Story
from tag_processor import TaggerConfig, Model, Task
from video_analyzer import load_yaml_with_substitution

logger = create_logger(__name__)


def create_test_story_config() -> dict:
    """
    Create a test story configuration that uses optimized detection.
    
    Returns:
        Dictionary with story configuration for testing
    """
    return {
        "name": "Integration Test - Optimized Detection",
        "skip": False,
        "selectors": [
            {
                "filenames": [
                    "T8600P1024260D5E_20241118084615.mp4",
                    "T8600P1024260D5E_20241118084819.mp4"
                ]
            }
        ],
        "tag_processing": True,
        "tag_processing_config": {
            "model": "Yolo11x_Optimized",  # Use optimized version
            "task": "Track",
            "num_frames_per_second": 2.0,
            "conf_threshold": 0.2,
            "batch_size": 8
        },
        "tag_video_generation": False,
        "video_generation": False
    }


def create_original_story_config() -> dict:
    """
    Create a story configuration that uses the original detector for comparison.
    
    Returns:
        Dictionary with story configuration for testing
    """
    return {
        "name": "Integration Test - Original Detection",
        "skip": False,
        "selectors": [
            {
                "filenames": [
                    "T8600P1024260D5E_20241118084615.mp4",
                    "T8600P1024260D5E_20241118084819.mp4"
                ]
            }
        ],
        "tag_processing": True,
        "tag_processing_config": {
            "model": "Yolo11x",  # Use original version
            "task": "Track",
            "num_frames_per_second": 2.0,
            "conf_threshold": 0.2
        },
        "tag_video_generation": False,
        "video_generation": False
    }


def load_sample_videos(config_file: str = "analysis_config.yaml") -> list:
    """
    Load sample videos from the existing configuration.
    
    Args:
        config_file: Path to analysis configuration file
        
    Returns:
        List of VideoMetadata objects for testing
    """
    from video_analyzer import AnalysisConfig
    from video_database import VideoDatabaseList
    
    # Load the existing configuration
    analysis_config = AnalysisConfig.from_file(config_file)
    
    # Load video database
    corrupted_files = []
    videos_database = analysis_config.video_database_list.load_videos(corrupted_files)
    
    if corrupted_files:
        logger.warning(f"Found {len(corrupted_files)} corrupted files during loading")
    
    # Filter for our test videos
    test_filenames = [
        "T8600P1024260D5E_20241118084615.mp4",
        "T8600P1024260D5E_20241118084819.mp4"
    ]
    
    test_videos = [
        video for video in videos_database 
        if any(filename in video.file_path for filename in test_filenames)
    ]
    
    logger.info(f"Loaded {len(test_videos)} test videos")
    for video in test_videos:
        logger.info(f"  - {os.path.basename(video.file_path)}")
    
    return test_videos


def run_story_with_timing(story_config: dict, videos_database: list, output_dir: str) -> dict:
    """
    Run a story with timing measurements.
    
    Args:
        story_config: Story configuration dictionary
        videos_database: List of video metadata
        output_dir: Output directory for results
        
    Returns:
        Dictionary with timing results and statistics
    """
    logger.info(f"Running story: {story_config['name']}")
    logger.info(f"Model: {story_config['tag_processing_config']['model']}")
    
    # Create story from configuration
    story = Story.from_dict(story_config)
    
    # Measure processing time
    start_time = time.time()
    
    # Process the story
    story.process(videos_database, output_dir)
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    # Count generated tags (if tag files were created)
    tag_files = list(Path(output_dir).glob("*_tags.json"))
    total_tags = 0
    
    if tag_files:
        import json
        for tag_file in tag_files:
            with open(tag_file, 'r') as f:
                tag_data = json.load(f)
                # Count tags in the file
                if 'tags' in tag_data:
                    for video_tags in tag_data['tags'].values():
                        for frame_tags in video_tags.values():
                            total_tags += len(frame_tags)
    
    results = {
        "story_name": story_config['name'],
        "model": story_config['tag_processing_config']['model'],
        "processing_time": processing_time,
        "fps": len(videos_database) / processing_time if processing_time > 0 else 0,
        "total_tags": total_tags,
        "tag_files_created": len(tag_files)
    }
    
    logger.info(f"Processing completed:")
    logger.info(f"  Time: {processing_time:.2f}s")
    logger.info(f"  Videos processed: {len(videos_database)}")
    logger.info(f"  Tags generated: {total_tags}")
    logger.info(f"  Tag files: {len(tag_files)}")
    
    return results


def compare_detectors():
    """
    Compare optimized vs original detector performance in the existing workflow.
    """
    logger.info("Starting detector integration comparison...")
    
    # Load sample videos
    try:
        videos_database = load_sample_videos()
    except Exception as e:
        logger.error(f"Failed to load sample videos: {e}")
        return
    
    if not videos_database:
        logger.error("No test videos found - aborting")
        return
    
    # Create temporary directories for output
    with tempfile.TemporaryDirectory() as temp_dir:
        original_dir = os.path.join(temp_dir, "original")
        optimized_dir = os.path.join(temp_dir, "optimized")
        os.makedirs(original_dir, exist_ok=True)
        os.makedirs(optimized_dir, exist_ok=True)
        
        # Test original detector
        logger.info("\n" + "="*60)
        logger.info("TESTING ORIGINAL DETECTOR")
        logger.info("="*60)
        
        original_config = create_original_story_config()
        original_results = run_story_with_timing(original_config, videos_database, original_dir)
        
        # Test optimized detector
        logger.info("\n" + "="*60)
        logger.info("TESTING OPTIMIZED DETECTOR")
        logger.info("="*60)
        
        optimized_config = create_test_story_config()
        optimized_results = run_story_with_timing(optimized_config, videos_database, optimized_dir)
        
        # Compare results
        logger.info("\n" + "="*60)
        logger.info("INTEGRATION COMPARISON RESULTS")
        logger.info("="*60)
        
        speedup = (original_results['processing_time'] / optimized_results['processing_time'] 
                  if optimized_results['processing_time'] > 0 else 0)
        
        logger.info(f"Original Detector ({original_results['model']}):")
        logger.info(f"  Processing time: {original_results['processing_time']:.2f}s")
        logger.info(f"  Tags generated: {original_results['total_tags']}")
        logger.info("")
        logger.info(f"Optimized Detector ({optimized_results['model']}):")
        logger.info(f"  Processing time: {optimized_results['processing_time']:.2f}s")
        logger.info(f"  Tags generated: {optimized_results['total_tags']}")
        logger.info("")
        logger.info(f"Performance Improvement:")
        logger.info(f"  Speedup: {speedup:.1f}x")
        logger.info(f"  Time reduction: {((original_results['processing_time'] - optimized_results['processing_time']) / original_results['processing_time'] * 100):.1f}%")
        
        # Validate that both detectors produced similar results
        tag_difference = abs(original_results['total_tags'] - optimized_results['total_tags'])
        tag_difference_percent = (tag_difference / original_results['total_tags'] * 100 
                                 if original_results['total_tags'] > 0 else 0)
        
        logger.info(f"  Tag count difference: {tag_difference} ({tag_difference_percent:.1f}%)")
        
        if tag_difference_percent < 10:
            logger.info("✅ Integration successful - similar detection results")
        else:
            logger.warning("⚠️ Significant difference in detection results")
        
        return {
            "original": original_results,
            "optimized": optimized_results,
            "speedup": speedup,
            "integration_success": tag_difference_percent < 10
        }


def test_configuration_options():
    """
    Test different configuration options for the optimized detector.
    """
    logger.info("Testing different optimization configurations...")
    
    # Test different batch sizes
    batch_sizes = [4, 8, 16]
    configs = []
    
    for batch_size in batch_sizes:
        config = {
            "name": f"Batch Size Test - {batch_size}",
            "skip": False,
            "selectors": [{"filenames": ["T8600P1024260D5E_20241118084615.mp4"]}],
            "tag_processing": True,
            "tag_processing_config": {
                "model": "Yolo11x_Optimized",
                "task": "Track",
                "num_frames_per_second": 1.0,
                "conf_threshold": 0.2,
                "batch_size": batch_size
            },
            "tag_video_generation": False,
            "video_generation": False
        }
        configs.append((batch_size, config))
    
    logger.info(f"Testing {len(configs)} different batch size configurations")
    
    # Note: Would run tests here but keeping this as a demo
    # In practice, you would run each config and measure performance
    for batch_size, config in configs:
        logger.info(f"  - Batch size {batch_size}: {config['tag_processing_config']['model']}")


def main():
    """Main integration test function."""
    # Set up logging
    set_all_loggers_level_and_format(level=logging.INFO, extended_format=False)
    
    logger.info("Starting integration test for optimized YOLO detector...")
    
    # Test 1: Compare detectors in existing workflow
    try:
        results = compare_detectors()
        if results and results.get('integration_success'):
            logger.info("✅ Integration test passed!")
        else:
            logger.error("❌ Integration test failed!")
    except Exception as e:
        logger.error(f"Integration test failed with error: {e}")
        return 1
    
    # Test 2: Configuration options
    test_configuration_options()
    
    logger.info("Integration testing completed!")
    return 0


if __name__ == "__main__":
    exit(main())