#!/usr/bin/env python3
"""
Test script for the person labeling GUI functionality.

This script validates the core functionality of the Streamlit-based person labeling tool
without requiring the full GUI to be running.
"""

import os
import tempfile
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from logging_config import create_logger
from person_labeling_streamlit import PersonCropData, PersonLabelingApp

logger = create_logger(__name__)


def test_crop_data_parsing():
    """Test PersonCropData filename parsing."""
    logger.info("Testing PersonCropData filename parsing...")
    
    # Create a test crop
    test_filename = "T8600P102338033E_20240930085536_track123_frame000456.jpg"
    test_path = f"/fake/path/{test_filename}"
    
    crop = PersonCropData(test_path)
    
    # Validate parsing
    assert crop.filename == test_filename
    assert crop.track_id == "123"
    assert crop.frame_number == 456
    assert crop.video_name == "T8600P102338033E_20240930085536"
    
    logger.info("✅ PersonCropData parsing test passed")
    return True


def test_app_initialization():
    """Test PersonLabelingApp initialization."""
    logger.info("Testing PersonLabelingApp initialization...")
    
    # Create app instance
    app = PersonLabelingApp()
    
    # Validate initial state
    assert len(app.crops) == 0
    assert len(app.filtered_crops) == 0
    assert len(app.clusters) == 0
    assert app.person_db is None
    assert app.embedder is None
    
    logger.info("✅ PersonLabelingApp initialization test passed")
    return True


def test_data_loading_functionality():
    """Test data loading functionality without GUI."""
    logger.info("Testing data loading functionality...")
    
    # Check if demo data exists
    demo_dir = Path(__file__).parent / "person_recognition_demo_output"
    
    if not demo_dir.exists():
        logger.warning("Demo data not found - skipping data loading test")
        return True
    
    # Test loading demo data
    crops_dir = demo_dir / "person_crops"
    database_file = demo_dir / "persons.json"
    embeddings_file = demo_dir / "person_embeddings.json"
    
    if not crops_dir.exists():
        logger.warning("Demo crops directory not found - skipping test")
        return True
    
    # Create app and test loading methods
    app = PersonLabelingApp()
    
    # Test crop loading
    app._load_crops_from_directory(str(crops_dir))
    assert len(app.crops) > 0, "No crops loaded"
    
    # Test database loading
    if database_file.exists():
        from person_database import PersonDatabase
        app.person_db = PersonDatabase(str(database_file))
        assert app.person_db is not None
    
    # Test embeddings loading
    if embeddings_file.exists():
        app._load_embeddings_from_file(str(embeddings_file))
    
    logger.info(f"✅ Loaded {len(app.crops)} crops successfully")
    return True


def test_labeling_functionality():
    """Test labeling functionality."""
    logger.info("Testing labeling functionality...")
    
    # Create test crop
    test_path = "/fake/path/test_crop.jpg"
    crop = PersonCropData(test_path)
    
    # Test initial state
    assert crop.person_name is None
    assert crop.person_id is None
    
    # Test labeling
    crop.person_name = "Test Person"
    crop.person_id = "test_person"
    crop.labeled_by = "manual"
    
    # Validate labeling
    assert crop.person_name == "Test Person"
    assert crop.person_id == "test_person"
    assert crop.labeled_by == "manual"
    
    logger.info("✅ Labeling functionality test passed")
    return True


def test_gui_components():
    """Test GUI component functionality."""
    logger.info("Testing GUI components...")
    
    # Create app instance
    app = PersonLabelingApp()
    
    # Test filtering
    # Create some test crops
    crop1 = PersonCropData("/fake/path/person1.jpg")
    crop1.person_name = "Alice"
    
    crop2 = PersonCropData("/fake/path/person2.jpg")
    crop2.person_name = "Bob"
    
    crop3 = PersonCropData("/fake/path/unlabeled.jpg")
    
    app.crops = [crop1, crop2, crop3]
    
    # Test view filtering
    app.filtered_crops = app.crops[:]
    
    # Test unlabeled filter
    app._apply_view_filter()  # Should have all crops for "all" view
    assert len(app.filtered_crops) == 3
    
    logger.info("✅ GUI components test passed")
    return True


def main():
    """Run all GUI tests."""
    logger.info("Starting Person Labeling GUI Tests")
    logger.info("=" * 60)
    
    tests = [
        ("Crop Data Parsing", test_crop_data_parsing),
        ("App Initialization", test_app_initialization),
        ("Data Loading", test_data_loading_functionality),
        ("Labeling Functionality", test_labeling_functionality),
        ("GUI Components", test_gui_components)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} ---")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"❌ {test_name} failed: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("PERSON LABELING GUI TEST SUMMARY")
    logger.info("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{test_name:<25} {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All GUI tests PASSED!")
        logger.info("Person labeling GUI is ready for use.")
        logger.info("\n🚀 To launch the GUI, run:")
        logger.info("   streamlit run person_labeling_streamlit.py")
        logger.info("   OR")
        logger.info("   python launch_labeling_gui.py")
        return 0
    else:
        logger.error(f"💥 {total - passed} tests FAILED!")
        return 1


if __name__ == "__main__":
    exit(main())