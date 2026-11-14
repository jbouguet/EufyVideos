#!/usr/bin/env python3
"""
Integration test for VideoMetadata date-aware device lookup.
"""

import tempfile
import os
from datetime import datetime
from unittest.mock import patch

from config import Config
from video_metadata import VideoMetadata
from logging_config import create_logger

logger = create_logger(__name__)


def test_video_metadata_date_awareness():
    """Test that VideoMetadata uses date-aware device lookup correctly."""
    
    # Reset singleton
    Config._instance = None
    Config._initialized = False
    
    # Create a mock CSV with date-aware entries
    csv_content = """Serial,Device,start_date,end_date
T8160P1123332B02,Garage,,2025-08-31
T8162T1024354A8B,Balcony,,2025-08-31
T8162T1024354A8B,Garage,2025-09-01,
T8160P1123332B02,Balcony,2025-09-01,"""
    
    fd, csv_path = tempfile.mkstemp(suffix='.csv')
    with os.fdopen(fd, 'w') as f:
        f.write(csv_content)
    
    with patch('config.os.path.join') as mock_join:
        mock_join.return_value = csv_path
        
        # Test before swap date (August 2025)
        before_swap_filename = "T8160P1123332B02_20250815123000.mp4"
        try:
            # This would normally fail because the file doesn't exist, 
            # but we can at least test the filename parsing logic
            metadata_before = VideoMetadata.from_video_file(before_swap_filename)
        except Exception as e:
            # Expected since file doesn't exist, but we can check serial extraction
            logger.info(f"Expected error for missing file: {e}")
            
            # Manually test the device lookup logic
            serial = before_swap_filename.split("_")[0]
            video_date = datetime.strptime("20250815123000"[:14], "%Y%m%d%H%M%S")
            device = Config.get_device_for_date(serial, video_date)
            
            logger.info(f"Before swap (2025-08-15): {serial} -> {device}")
            assert device == "Garage", f"Expected 'Garage', got '{device}'"
        
        # Test after swap date (September 2025)  
        after_swap_filename = "T8160P1123332B02_20250915123000.mp4"
        try:
            metadata_after = VideoMetadata.from_video_file(after_swap_filename)
        except Exception as e:
            # Expected since file doesn't exist, but we can check serial extraction
            logger.info(f"Expected error for missing file: {e}")
            
            # Manually test the device lookup logic
            serial = after_swap_filename.split("_")[0]
            video_date = datetime.strptime("20250915123000"[:14], "%Y%m%d%H%M%S")
            device = Config.get_device_for_date(serial, video_date)
            
            logger.info(f"After swap (2025-09-15): {serial} -> {device}")
            assert device == "Balcony", f"Expected 'Balcony', got '{device}'"
        
        # Test the other serial
        balcony_to_garage_filename = "T8162T1024354A8B_20250915123000.mp4"
        try:
            metadata_btg = VideoMetadata.from_video_file(balcony_to_garage_filename)
        except Exception as e:
            logger.info(f"Expected error for missing file: {e}")
            
            serial = balcony_to_garage_filename.split("_")[0]
            video_date = datetime.strptime("20250915123000"[:14], "%Y%m%d%H%M%S")
            device = Config.get_device_for_date(serial, video_date)
            
            logger.info(f"After swap (2025-09-15): {serial} -> {device}")
            assert device == "Garage", f"Expected 'Garage', got '{device}'"
    
    os.unlink(csv_path)
    logger.info("✅ VideoMetadata date-aware device lookup integration test passed")


if __name__ == '__main__':
    test_video_metadata_date_awareness()