#!/usr/bin/env python3
"""
Quick validation script for optimized detector integration.

This script validates that the optimized detector works correctly within
the existing workflow without running full story processing.
"""

import logging
import os
import tempfile
from logging_config import create_logger, set_all_loggers_level_and_format
from tag_processor import TaggerConfig, TagProcessor, Model, Task
from video_metadata import VideoMetadata

logger = create_logger(__name__)


def test_optimized_detector_instantiation():
    """Test that optimized detectors can be created correctly."""
    logger.info("Testing optimized detector instantiation...")
    
    models_to_test = [
        Model.YOLO11N_OPTIMIZED,
        Model.YOLO11S_OPTIMIZED, 
        Model.YOLO11M_OPTIMIZED,
        Model.YOLO11L_OPTIMIZED,
        Model.YOLO11X_OPTIMIZED
    ]
    
    for model in models_to_test:
        try:
            config = TaggerConfig(
                model=model.value,
                task=Task.TRACK.value,
                num_frames_per_second=1.0,
                conf_threshold=0.2,
                batch_size=4
            )
            
            processor = TagProcessor(config)
            logger.info(f"✅ {model.value}: Successfully created")
            
            # Get performance stats
            if hasattr(processor.object_detector, 'get_performance_stats'):
                stats = processor.object_detector.get_performance_stats()
                logger.info(f"   Device: {stats.get('device', 'unknown')}")
                logger.info(f"   GPU available: {stats.get('gpu_available', False)}")
        
        except Exception as e:
            logger.error(f"❌ {model.value}: Failed - {e}")
            return False
    
    return True


def test_configuration_parsing():
    """Test that configurations with batch_size are parsed correctly."""
    logger.info("Testing configuration parsing...")
    
    config_dict = {
        "model": "Yolo11x_Optimized",
        "task": "Track", 
        "num_frames_per_second": 2.0,
        "conf_threshold": 0.3,
        "batch_size": 16
    }
    
    try:
        config = TaggerConfig.from_dict(config_dict)
        logger.info(f"✅ Configuration parsed successfully:")
        logger.info(f"   Model: {config.model}")
        logger.info(f"   Task: {config.task}")
        logger.info(f"   FPS: {config.num_frames_per_second}")
        logger.info(f"   Confidence: {config.conf_threshold}")
        logger.info(f"   Batch size: {config.batch_size}")
        return True
    
    except Exception as e:
        logger.error(f"❌ Configuration parsing failed: {e}")
        return False


def test_video_processing_simulation():
    """Simulate video processing without actual video files."""
    logger.info("Testing video processing simulation...")
    
    # Create a mock video metadata object
    test_video_path = "/fake/path/test_video.mp4"
    
    # Test with optimized detector
    config = TaggerConfig(
        model=Model.YOLO11X_OPTIMIZED.value,
        task=Task.TRACK.value,
        num_frames_per_second=1.0,
        conf_threshold=0.2,
        batch_size=8
    )
    
    try:
        processor = TagProcessor(config)
        logger.info(f"✅ TagProcessor created with optimized detector")
        logger.info(f"   Detector type: {type(processor.object_detector).__name__}")
        logger.info(f"   Configuration ID: {config.get_identifier()}")
        
        # Test that the processor has the right configuration
        assert processor.tag_processing_config.model == config.model
        assert processor.tag_processing_config.batch_size == config.batch_size
        
        logger.info("✅ Configuration correctly passed to processor")
        return True
        
    except Exception as e:
        logger.error(f"❌ Video processing simulation failed: {e}")
        return False


def validate_backward_compatibility():
    """Ensure original detectors still work correctly."""
    logger.info("Testing backward compatibility...")
    
    # Test original YOLO detector
    config = TaggerConfig(
        model=Model.YOLO11X.value,  # Original, not optimized
        task=Task.TRACK.value,
        num_frames_per_second=1.0,
        conf_threshold=0.2
        # No batch_size for original detector
    )
    
    try:
        processor = TagProcessor(config)
        logger.info(f"✅ Original detector still works:")
        logger.info(f"   Detector type: {type(processor.object_detector).__name__}")
        logger.info(f"   Model: {processor.object_detector.model_name}")
        
        # Ensure it's not the optimized version
        assert "Optimized" not in type(processor.object_detector).__name__
        logger.info("✅ Backward compatibility maintained")
        return True
        
    except Exception as e:
        logger.error(f"❌ Backward compatibility test failed: {e}")
        return False


def main():
    """Run all integration validation tests."""
    set_all_loggers_level_and_format(level=logging.INFO, extended_format=False)
    
    logger.info("Starting integration validation...")
    logger.info("="*60)
    
    tests = [
        ("Optimized Detector Instantiation", test_optimized_detector_instantiation),
        ("Configuration Parsing", test_configuration_parsing),
        ("Video Processing Simulation", test_video_processing_simulation),
        ("Backward Compatibility", validate_backward_compatibility)
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} ---")
        try:
            result = test_func()
            results.append((test_name, result))
            if result:
                logger.info(f"✅ {test_name} PASSED")
            else:
                logger.error(f"❌ {test_name} FAILED")
        except Exception as e:
            logger.error(f"❌ {test_name} FAILED with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("VALIDATION SUMMARY")
    logger.info("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{test_name:<35} {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All integration tests PASSED!")
        logger.info("The optimized detector is ready for production use.")
        return 0
    else:
        logger.error(f"💥 {total - passed} tests FAILED!")
        logger.error("Integration issues need to be resolved.")
        return 1


if __name__ == "__main__":
    exit(main())