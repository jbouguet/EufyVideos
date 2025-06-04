#!/usr/bin/env python3
"""
Test script for person detection and database functionality.

This script validates the person detection system by:
1. Testing person detection on a sample video
2. Extracting person crops
3. Creating and managing person database
4. Demonstrating the labeling workflow
"""

import os
import tempfile
import logging
from pathlib import Path

from logging_config import create_logger, set_all_loggers_level_and_format
from person_detector import PersonDetector, PersonTrack
from person_database import PersonDatabase
from tag_processor import TaggerConfig

logger = create_logger(__name__)


def test_person_detection():
    """Test person detection functionality."""
    logger.info("Testing person detection...")
    
    # Sample video path (use one from our previous tests)
    video_path = "/Users/jbouguet/Documents/EufySecurityVideos/record/Batch022/T8600P102338033E_20240930085536.mp4"
    
    if not os.path.exists(video_path):
        logger.error(f"Test video not found: {video_path}")
        return False
    
    try:
        # Create person detector with optimized YOLO
        person_detector = PersonDetector(
            base_detector_config=TaggerConfig(
                model="Yolo11x_Optimized",
                task="Track",
                num_frames_per_second=2.0,  # Sample every 0.5 seconds
                conf_threshold=0.2,
                batch_size=8
            ),
            crop_size=(224, 224),
            min_confidence=0.5,
            min_bbox_area=2000
        )
        
        # Detect persons in video
        person_tracks = person_detector.detect_persons_in_video(video_path)
        
        logger.info(f"✅ Person detection completed")
        logger.info(f"   Found {len(person_tracks)} person tracks")
        
        for track in person_tracks[:5]:  # Show first 5 tracks
            logger.info(f"   Track {track.track_id}: {len(track.crops)} detections, "
                       f"avg confidence: {track.confidence_avg:.2f}, "
                       f"duration: {track.duration_frames} frames")
        
        return person_tracks
        
    except Exception as e:
        logger.error(f"❌ Person detection failed: {e}")
        return None


def test_crop_extraction(person_tracks, video_path):
    """Test person crop extraction."""
    logger.info("Testing crop extraction...")
    
    if not person_tracks:
        logger.warning("No person tracks to extract crops from")
        return False
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            crop_dir = os.path.join(temp_dir, "person_crops")
            
            # Create person detector
            person_detector = PersonDetector()
            
            # Extract crops for top 3 tracks
            top_tracks = sorted(person_tracks, key=lambda t: len(t.crops), reverse=True)[:3]
            
            extracted_tracks = person_detector.extract_person_crops(
                video_path=video_path,
                person_tracks=top_tracks,
                output_dir=crop_dir,
                max_crops_per_track=5
            )
            
            # Count extracted crops
            total_crops = sum(len([c for c in track.crops if c.crop_image is not None]) 
                            for track in extracted_tracks)
            
            # Count saved files
            saved_files = len(list(Path(crop_dir).glob("*.jpg"))) if os.path.exists(crop_dir) else 0
            
            logger.info(f"✅ Crop extraction completed")
            logger.info(f"   Extracted {total_crops} crop images")
            logger.info(f"   Saved {saved_files} files to disk")
            
            return extracted_tracks
            
    except Exception as e:
        logger.error(f"❌ Crop extraction failed: {e}")
        return None


def test_person_database():
    """Test person database functionality."""
    logger.info("Testing person database...")
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_file = os.path.join(temp_dir, "test_persons.json")
            
            # Create database
            db = PersonDatabase(db_file)
            
            # Add family members
            jean_yves_id = db.add_person(
                "Jean-Yves Bouguet", 
                description="Father, homeowner",
                aliases=["JY", "Dad"]
            )
            
            chittra_id = db.add_person(
                "Chittra Chaivorapol",
                description="Mother", 
                aliases=["Mom", "Chittra"]
            )
            
            lucas_id = db.add_person(
                "Lucas Bouguet",
                description="Son",
                aliases=["Luke"]
            )
            
            logger.info(f"✅ Added 3 persons to database")
            
            # Test labeling tracks
            db.label_track(
                video_filename="T8600P102338033E_20240930085536.mp4",
                track_id=1,
                person_id=jean_yves_id,
                notes="Test label"
            )
            
            db.label_track(
                video_filename="T8600P102338033E_20240930085536.mp4", 
                track_id=2,
                person_name="Chittra Chaivorapol",
                confidence=0.9,
                labeled_by="automatic"
            )
            
            logger.info(f"✅ Added track labels")
            
            # Test retrieval
            jean_yves_tracks = db.get_person_tracks("Jean-Yves Bouguet")
            chittra_tracks = db.get_person_tracks(chittra_id)
            
            logger.info(f"   Jean-Yves tracks: {len(jean_yves_tracks)}")
            logger.info(f"   Chittra tracks: {len(chittra_tracks)}")
            
            # Test statistics
            stats = db.get_database_statistics()
            logger.info(f"✅ Database statistics:")
            logger.info(f"   Total persons: {stats['total_persons']}")
            logger.info(f"   Total tracks: {stats['total_tracks']}")
            logger.info(f"   Manual labels: {stats['manual_labels']}")
            logger.info(f"   Automatic labels: {stats['automatic_labels']}")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ Database test failed: {e}")
        return False


def test_integration():
    """Test integration of person detection with database."""
    logger.info("Testing person detection + database integration...")
    
    try:
        # Use a smaller test video if the full one is too large
        video_path = "/Users/jbouguet/Documents/EufySecurityVideos/record/Batch022/T8600P102338033E_20240930085536.mp4"
        
        if not os.path.exists(video_path):
            logger.error(f"Test video not found: {video_path}")
            return False
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Setup
            db_file = os.path.join(temp_dir, "persons.json")
            tracks_file = os.path.join(temp_dir, "person_tracks.json")
            
            # Create database and add persons
            db = PersonDatabase(db_file)
            jean_yves_id = db.add_person("Jean-Yves Bouguet", "Father")
            chittra_id = db.add_person("Chittra Chaivorapol", "Mother")
            
            # Detect persons (use faster settings for testing)
            person_detector = PersonDetector(
                base_detector_config=TaggerConfig(
                    model="Yolo11x_Optimized",
                    task="Track", 
                    num_frames_per_second=1.0,  # Faster sampling for testing
                    conf_threshold=0.3,
                    batch_size=8
                ),
                min_confidence=0.6,
                min_bbox_area=3000
            )
            
            person_tracks = person_detector.detect_persons_in_video(video_path)
            
            if person_tracks:
                # Save tracks to file
                person_detector.save_person_tracks(person_tracks, tracks_file)
                
                # Label some tracks manually (simulation)
                video_filename = os.path.basename(video_path)
                if len(person_tracks) >= 1:
                    db.label_track(video_filename, person_tracks[0].track_id, jean_yves_id)
                    logger.info(f"   Labeled track {person_tracks[0].track_id} as Jean-Yves")
                
                if len(person_tracks) >= 2:
                    db.label_track(video_filename, person_tracks[1].track_id, chittra_id)
                    logger.info(f"   Labeled track {person_tracks[1].track_id} as Chittra")
                
                # Get statistics
                stats = db.get_database_statistics()
                logger.info(f"✅ Integration test completed:")
                logger.info(f"   Detected {len(person_tracks)} person tracks")
                logger.info(f"   Labeled {stats['total_tracks']} tracks")
                logger.info(f"   Database has {stats['total_persons']} persons")
                
                return True
            else:
                logger.warning("No person tracks detected in test video")
                return False
            
    except Exception as e:
        logger.error(f"❌ Integration test failed: {e}")
        return False


def main():
    """Run all person detection tests."""
    set_all_loggers_level_and_format(level=logging.INFO, extended_format=False)
    
    logger.info("Starting Person Detection System Tests")
    logger.info("=" * 60)
    
    tests = [
        ("Person Detection", test_person_detection),
        ("Person Database", test_person_database),
        ("Integration Test", test_integration)
    ]
    
    results = []
    person_tracks = None
    
    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} ---")
        try:
            if test_name == "Person Detection":
                result = test_func()
                person_tracks = result if result else None
                results.append((test_name, result is not None))
            elif test_name == "Crop Extraction" and person_tracks:
                video_path = "/Users/jbouguet/Documents/EufySecurityVideos/record/Batch022/T8600P102338033E_20240930085536.mp4"
                result = test_crop_extraction(person_tracks, video_path)
                results.append((test_name, result is not None))
            else:
                result = test_func()
                results.append((test_name, result))
                
        except Exception as e:
            logger.error(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("PERSON DETECTION TEST SUMMARY")
    logger.info("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{test_name:<25} {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All person detection tests PASSED!")
        logger.info("Person detection system is ready for use.")
        return 0
    else:
        logger.error(f"💥 {total - passed} tests FAILED!")
        logger.error("Person detection system needs debugging.")
        return 1


if __name__ == "__main__":
    exit(main())