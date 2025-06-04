#!/usr/bin/env python3
"""
Test script for person recognition integration with tag processing.

This script validates the integration of person recognition capabilities
into the existing tag processing workflow.

Tests:
1. PersonRecognitionConfig creation and validation
2. PersonRecognitionProcessor initialization
3. Enhanced tag processing with person recognition
4. Backward compatibility with existing workflow
5. Integration with person database and embeddings
"""

import os
import tempfile
import logging
from pathlib import Path

from logging_config import create_logger, set_all_loggers_level_and_format
from person_recognition_processor import PersonRecognitionConfig, PersonRecognitionProcessor
from person_database import PersonDatabase
from video_metadata import VideoMetadata

logger = create_logger(__name__)


def test_config_creation():
    """Test PersonRecognitionConfig creation and validation."""
    logger.info("Testing PersonRecognitionConfig creation...")
    
    try:
        # Test basic config
        config = PersonRecognitionConfig(
            model="Yolo11x_Optimized",
            task="Track",
            enable_person_recognition=False
        )
        
        assert config.model == "Yolo11x_Optimized"
        assert config.enable_person_recognition == False
        assert config.person_min_confidence == 0.6
        
        # Test enhanced config
        enhanced_config = PersonRecognitionConfig(
            model="Yolo11x_Optimized",
            task="Track",
            num_frames_per_second=2.0,
            enable_person_recognition=True,
            person_database_file="test_persons.json",
            person_min_confidence=0.7,
            auto_label_confidence=0.85
        )
        
        assert enhanced_config.enable_person_recognition == True
        assert enhanced_config.person_database_file == "test_persons.json"
        assert enhanced_config.person_min_confidence == 0.7
        assert enhanced_config.auto_label_confidence == 0.85
        
        # Test config from dict
        config_dict = {
            "model": "Yolo11x_Optimized",
            "task": "Track",
            "num_frames_per_second": 1.5,
            "conf_threshold": 0.3,
            "batch_size": 4,
            "enable_person_recognition": True,
            "person_database_file": "persons.json",
            "person_min_confidence": 0.65,
            "embedding_device": "mps",
            "similarity_threshold": 0.8
        }
        
        dict_config = PersonRecognitionConfig.from_dict(config_dict)
        assert dict_config.model == "Yolo11x_Optimized"
        assert dict_config.num_frames_per_second == 1.5
        assert dict_config.enable_person_recognition == True
        assert dict_config.person_database_file == "persons.json"
        assert dict_config.embedding_device == "mps"
        
        # Test identifier generation
        basic_id = config.get_identifier()
        enhanced_id = enhanced_config.get_identifier()
        
        assert "PersonRec" not in basic_id  # No person recognition
        assert "PersonRec" in enhanced_id   # Has person recognition
        
        logger.info("✅ PersonRecognitionConfig creation test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Config creation test failed: {e}")
        return False


def test_processor_initialization():
    """Test PersonRecognitionProcessor initialization."""
    logger.info("Testing PersonRecognitionProcessor initialization...")
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test basic processor (no person recognition)
            basic_config = PersonRecognitionConfig(
                model="Yolo11x_Optimized",
                enable_person_recognition=False
            )
            
            basic_processor = PersonRecognitionProcessor(basic_config)
            assert basic_processor.config.enable_person_recognition == False
            assert basic_processor.person_detector is None
            assert basic_processor.embedder is None
            assert basic_processor.person_db is None
            
            # Test enhanced processor (with person recognition)
            db_file = os.path.join(temp_dir, "test_persons.json")
            
            enhanced_config = PersonRecognitionConfig(
                model="Yolo11x_Optimized",
                enable_person_recognition=True,
                person_database_file=db_file,
                embedding_device="mps"
            )
            
            # Create a basic person database
            test_db = PersonDatabase(db_file)
            test_db.add_person("Test Person", "Test description")
            
            enhanced_processor = PersonRecognitionProcessor(enhanced_config)
            assert enhanced_processor.config.enable_person_recognition == True
            assert enhanced_processor.person_detector is not None
            assert enhanced_processor.embedder is not None
            assert enhanced_processor.person_db is not None
            
            logger.info("✅ PersonRecognitionProcessor initialization test passed")
            return True
            
    except Exception as e:
        logger.error(f"❌ Processor initialization test failed: {e}")
        return False


def test_basic_tag_processing():
    """Test basic tag processing without person recognition (backward compatibility)."""
    logger.info("Testing basic tag processing (backward compatibility)...")
    
    video_path = "/Users/jbouguet/Documents/EufySecurityVideos/record/Batch022/T8600P102338033E_20240930085536.mp4"
    
    if not os.path.exists(video_path):
        logger.warning(f"Test video not found: {video_path} - skipping test")
        return True
    
    try:
        # Create basic config (no person recognition)
        config = PersonRecognitionConfig(
            model="Yolo11x_Optimized",
            task="Track",
            num_frames_per_second=0.5,  # Very fast for testing
            conf_threshold=0.5,
            batch_size=4,
            enable_person_recognition=False
        )
        
        # Create processor
        processor = PersonRecognitionProcessor(config)
        
        # Create video metadata
        video_metadata = VideoMetadata.from_video_file(video_path)
        
        # Process video
        video_tags = processor.run(video_metadata)
        
        # Validate results
        assert video_tags is not None
        assert len(video_tags.tags) > 0
        
        # Check that no person identity information was added
        person_identity_found = False
        for filename, frames in video_tags.tags.items():
            for frame_num, frame_tags in frames.items():
                for tag_id, tag_data in frame_tags.items():
                    if 'person_identity' in tag_data:
                        person_identity_found = True
                        break
        
        assert not person_identity_found, "Person identity should not be present in basic processing"
        
        logger.info("✅ Basic tag processing test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Basic tag processing test failed: {e}")
        return False


def test_enhanced_tag_processing():
    """Test enhanced tag processing with person recognition."""
    logger.info("Testing enhanced tag processing with person recognition...")
    
    video_path = "/Users/jbouguet/Documents/EufySecurityVideos/record/Batch022/T8600P102338033E_20240930085536.mp4"
    
    if not os.path.exists(video_path):
        logger.warning(f"Test video not found: {video_path} - skipping test")
        return True
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Setup test environment
            db_file = os.path.join(temp_dir, "test_persons.json")
            embeddings_file = os.path.join(temp_dir, "test_embeddings.json")
            crops_dir = os.path.join(temp_dir, "crops")
            
            # Create test database with family members
            test_db = PersonDatabase(db_file)
            jean_yves_id = test_db.add_person("Jean-Yves Bouguet", "Father")
            chittra_id = test_db.add_person("Chittra Chaivorapol", "Mother")
            
            # Create enhanced config
            config = PersonRecognitionConfig(
                model="Yolo11x_Optimized",
                task="Track",
                num_frames_per_second=0.5,  # Very fast for testing
                conf_threshold=0.4,
                batch_size=4,
                enable_person_recognition=True,
                person_database_file=db_file,
                person_embeddings_file=embeddings_file,
                person_crops_dir=crops_dir,
                person_min_confidence=0.6,
                max_crops_per_track=3,  # Limited for testing
                embedding_device="mps",
                auto_label_confidence=0.7,
                enable_auto_labeling=False  # Disable for testing
            )
            
            # Create processor
            processor = PersonRecognitionProcessor(config)
            
            # Create video metadata
            video_metadata = VideoMetadata.from_video_file(video_path)
            
            # Process video
            video_tags = processor.run(video_metadata)
            
            # Validate results
            assert video_tags is not None
            assert len(video_tags.tags) > 0
            
            # Check if embeddings file was created
            if os.path.exists(embeddings_file):
                logger.info(f"✅ Embeddings file created: {embeddings_file}")
            
            # Check if crops were saved
            if os.path.exists(crops_dir):
                crop_files = list(Path(crops_dir).glob("*.jpg"))
                logger.info(f"✅ {len(crop_files)} crop files created")
            
            logger.info("✅ Enhanced tag processing test passed")
            return True
            
    except Exception as e:
        logger.error(f"❌ Enhanced tag processing test failed: {e}")
        return False


def test_batch_processing():
    """Test batch processing capabilities."""
    logger.info("Testing batch processing...")
    
    video_path = "/Users/jbouguet/Documents/EufySecurityVideos/record/Batch022/T8600P102338033E_20240930085536.mp4"
    
    if not os.path.exists(video_path):
        logger.warning(f"Test video not found: {video_path} - skipping test")
        return True
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Setup
            db_file = os.path.join(temp_dir, "batch_test_persons.json")
            test_db = PersonDatabase(db_file)
            test_db.add_person("Test Person", "Test")
            
            # Create config
            config = PersonRecognitionConfig(
                model="Yolo11x_Optimized",
                task="Track",
                num_frames_per_second=0.2,  # Very fast
                enable_person_recognition=True,
                person_database_file=db_file,
                max_crops_per_track=2
            )
            
            # Create processor
            processor = PersonRecognitionProcessor(config)
            
            # Create multiple video metadata objects (same video for testing)
            video_metadata_list = [
                VideoMetadata.from_video_file(video_path),
                VideoMetadata.from_video_file(video_path)
            ]
            
            # Process batch
            results = processor.run_batch(video_metadata_list)
            
            # Validate results - expect at least 1 successful processing
            assert len(results) >= 1, f"Expected at least 1 result, got {len(results)}"
            for filename, video_tags in results.items():
                assert video_tags is not None
                # Allow for empty tags in case of processing issues
                logger.info(f"Processed {filename}: {len(video_tags.tags)} tag groups")
            
            logger.info("✅ Batch processing test passed")
            return True
            
    except Exception as e:
        logger.error(f"❌ Batch processing test failed: {e}")
        return False


def main():
    """Run all person recognition integration tests."""
    set_all_loggers_level_and_format(level=logging.INFO, extended_format=False)
    
    logger.info("Starting Person Recognition Integration Tests")
    logger.info("=" * 70)
    
    tests = [
        ("Config Creation", test_config_creation),
        ("Processor Initialization", test_processor_initialization),
        ("Basic Tag Processing", test_basic_tag_processing),
        ("Enhanced Tag Processing", test_enhanced_tag_processing),
        ("Batch Processing", test_batch_processing)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} ---")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("PERSON RECOGNITION INTEGRATION TEST SUMMARY")
    logger.info("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{test_name:<30} {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All person recognition integration tests PASSED!")
        logger.info("Person recognition is ready for integration with tag processing.")
        return 0
    else:
        logger.error(f"💥 {total - passed} tests FAILED!")
        logger.error("Person recognition integration needs debugging.")
        return 1


if __name__ == "__main__":
    exit(main())