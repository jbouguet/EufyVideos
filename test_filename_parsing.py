#!/usr/bin/env python3
"""
Test filename parsing and date-aware device lookup without video dependencies.
"""

import tempfile
import os
from datetime import datetime
from unittest.mock import patch

from config import Config
from logging_config import create_logger

logger = create_logger(__name__)


def test_filename_parsing_with_date_awareness():
    """Test filename parsing with date-aware device lookup."""
    
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
        config = Config()
        
        # Test various filename scenarios
        test_cases = [
            # Before swap (August 2025)
            ("T8160P1123332B02_20250815123000.mp4", "Garage"),
            ("T8162T1024354A8B_20250815123000.mp4", "Balcony"),
            
            # Boundary date (end of August)
            ("T8160P1123332B02_20250831235959.mp4", "Garage"),
            ("T8162T1024354A8B_20250831235959.mp4", "Balcony"),
            
            # Swap date (September 1st)
            ("T8160P1123332B02_20250901000000.mp4", "Balcony"),
            ("T8162T1024354A8B_20250901000000.mp4", "Garage"),
            
            # After swap (September 2025)
            ("T8160P1123332B02_20250915123000.mp4", "Balcony"),
            ("T8162T1024354A8B_20250915123000.mp4", "Garage"),
            
            # Far future (should still work with open-ended dates)
            ("T8160P1123332B02_20301231235959.mp4", "Balcony"),
            ("T8162T1024354A8B_20301231235959.mp4", "Garage"),
        ]
        
        for filename, expected_device in test_cases:
            # Parse filename (mimicking VideoMetadata.from_video_file logic)
            serial_and_datetime = filename.split("_")
            serial = serial_and_datetime[0]
            datetime_part = serial_and_datetime[1]
            video_date = datetime.strptime(datetime_part[:14], "%Y%m%d%H%M%S")
            
            # Get device using date-aware lookup
            device = Config.get_device_for_date(serial, video_date)
            
            logger.info(f"{filename} -> Serial: {serial}, Date: {video_date.strftime('%Y-%m-%d')}, Device: {device}")
            
            assert device == expected_device, f"For {filename}, expected '{expected_device}', got '{device}'"
        
        logger.info("✅ All filename parsing and date-aware device lookup tests passed")
    
    os.unlink(csv_path)


if __name__ == '__main__':
    test_filename_parsing_with_date_awareness()