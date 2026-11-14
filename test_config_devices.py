#!/usr/bin/env python3
"""
Test suite for date-aware device configuration functionality.

Tests the enhanced Config class that supports time-based device assignments
with start and end dates.
"""

import unittest
import tempfile
import os
import csv
from datetime import datetime
from unittest.mock import patch

from config import Config
from logging_config import create_logger

logger = create_logger(__name__)


class TestConfigDevices(unittest.TestCase):
    """Test cases for date-aware device configuration."""
    
    def setUp(self):
        """Set up test environment with temporary CSV files."""
        # Reset singleton instance
        Config._instance = None
        Config._initialized = False
    
    def create_test_csv(self, content: str) -> str:
        """Create a temporary CSV file with given content."""
        fd, path = tempfile.mkstemp(suffix='.csv')
        with os.fdopen(fd, 'w') as f:
            f.write(content)
        return path
    
    def test_basic_device_loading(self):
        """Test loading devices from CSV without date columns (backward compatibility)."""
        csv_content = """Serial,Device
T8160P1123332B02,Garage
T8162T1024354A8B,Balcony
T8600P102338033E,Backyard"""
        
        csv_path = self.create_test_csv(csv_content)
        
        with patch('config.os.path.join') as mock_join:
            mock_join.return_value = csv_path
            config = Config()
            
            # Test backward compatibility
            self.assertEqual(config.get_device_dict()["T8160P1123332B02"], "Garage")
            self.assertEqual(config.get_device_dict()["T8162T1024354A8B"], "Balcony")
            
            # Test date-aware lookup (should return first entry)
            test_date = datetime(2025, 6, 15)
            self.assertEqual(Config.get_device_for_date("T8160P1123332B02", test_date), "Garage")
        
        os.unlink(csv_path)
    
    def test_date_aware_device_loading(self):
        """Test loading devices with date columns."""
        csv_content = """Serial,Device,start_date,end_date
T8160P1123332B02,Garage,,2025-08-31
T8162T1024354A8B,Balcony,,2025-08-31
T8162T1024354A8B,Garage,2025-09-01,
T8160P1123332B02,Balcony,2025-09-01,"""
        
        csv_path = self.create_test_csv(csv_content)
        
        with patch('config.os.path.join') as mock_join:
            mock_join.return_value = csv_path
            config = Config()
            
            # Test device assignment before swap date
            before_swap = datetime(2025, 8, 15)
            self.assertEqual(Config.get_device_for_date("T8160P1123332B02", before_swap), "Garage")
            self.assertEqual(Config.get_device_for_date("T8162T1024354A8B", before_swap), "Balcony")
            
            # Test device assignment after swap date
            after_swap = datetime(2025, 9, 15)
            self.assertEqual(Config.get_device_for_date("T8160P1123332B02", after_swap), "Balcony")
            self.assertEqual(Config.get_device_for_date("T8162T1024354A8B", after_swap), "Garage")
            
            # Test exact boundary dates
            boundary_date = datetime(2025, 8, 31)
            self.assertEqual(Config.get_device_for_date("T8160P1123332B02", boundary_date), "Garage")
            
            swap_date = datetime(2025, 9, 1)
            self.assertEqual(Config.get_device_for_date("T8160P1123332B02", swap_date), "Balcony")
        
        os.unlink(csv_path)
    
    def test_overlapping_dates_validation(self):
        """Test validation of overlapping date ranges."""
        csv_content = """Serial,Device,start_date,end_date
T8160P1123332B02,Garage,2025-01-01,2025-08-31
T8160P1123332B02,Balcony,2025-08-30,2025-12-31"""
        
        csv_path = self.create_test_csv(csv_content)
        
        with patch('config.os.path.join') as mock_join:
            mock_join.return_value = csv_path
            with self.assertRaises(ValueError) as context:
                Config()
            self.assertIn("Overlapping date ranges", str(context.exception))
        
        os.unlink(csv_path)
    
    def test_invalid_date_format(self):
        """Test handling of invalid date formats."""
        csv_content = """Serial,Device,start_date,end_date
T8160P1123332B02,Garage,2025/01/01,2025-08-31
T8162T1024354A8B,Balcony,invalid-date,"""
        
        csv_path = self.create_test_csv(csv_content)
        
        with patch('config.os.path.join') as mock_join:
            mock_join.return_value = csv_path
            # Should not raise exception, but should log warnings
            config = Config()
            
            # Invalid dates should be treated as None
            test_date = datetime(2025, 6, 15)
            self.assertEqual(Config.get_device_for_date("T8160P1123332B02", test_date), "Garage")
            self.assertEqual(Config.get_device_for_date("T8162T1024354A8B", test_date), "Balcony")
        
        os.unlink(csv_path)
    
    def test_empty_date_fields(self):
        """Test handling of empty date fields (open-ended ranges)."""
        csv_content = """Serial,Device,start_date,end_date
T8160P1123332B02,Garage,,
T8162T1024354A8B,Balcony,2025-09-01,"""
        
        csv_path = self.create_test_csv(csv_content)
        
        with patch('config.os.path.join') as mock_join:
            mock_join.return_value = csv_path
            config = Config()
            
            # Open-ended start date (Garage should work for any date)
            early_date = datetime(2020, 1, 1)
            self.assertEqual(Config.get_device_for_date("T8160P1123332B02", early_date), "Garage")
            
            # Open-ended end date (Balcony should work from 2025-09-01 onwards)
            late_date = datetime(2030, 1, 1)
            self.assertEqual(Config.get_device_for_date("T8162T1024354A8B", late_date), "Balcony")
            
            # Before Balcony start date (should return device name from first entry)
            before_balcony = datetime(2025, 8, 15)
            self.assertEqual(Config.get_device_for_date("T8162T1024354A8B", before_balcony), "Balcony")
        
        os.unlink(csv_path)
    
    def test_unknown_serial(self):
        """Test handling of unknown serial numbers."""
        csv_content = """Serial,Device,start_date,end_date
T8160P1123332B02,Garage,,"""
        
        csv_path = self.create_test_csv(csv_content)
        
        with patch('config.os.path.join') as mock_join:
            mock_join.return_value = csv_path
            config = Config()
            
            # Unknown serial should return the serial itself
            test_date = datetime(2025, 6, 15)
            self.assertEqual(Config.get_device_for_date("UNKNOWN123", test_date), "UNKNOWN123")
        
        os.unlink(csv_path)
    
    def test_multiple_entries_same_serial(self):
        """Test proper handling of multiple entries for the same serial."""
        csv_content = """Serial,Device,start_date,end_date
T8160P1123332B02,Garage,2025-01-01,2025-06-30
T8160P1123332B02,Workshop,2025-07-01,2025-08-31
T8160P1123332B02,Balcony,2025-09-01,"""
        
        csv_path = self.create_test_csv(csv_content)
        
        with patch('config.os.path.join') as mock_join:
            mock_join.return_value = csv_path
            config = Config()
            
            # Test different time periods
            self.assertEqual(Config.get_device_for_date("T8160P1123332B02", datetime(2025, 3, 15)), "Garage")
            self.assertEqual(Config.get_device_for_date("T8160P1123332B02", datetime(2025, 7, 15)), "Workshop")
            self.assertEqual(Config.get_device_for_date("T8160P1123332B02", datetime(2025, 10, 15)), "Balcony")
        
        os.unlink(csv_path)
    
    def test_get_all_devices_with_dates(self):
        """Test that get_all_devices() works correctly with date-aware entries."""
        csv_content = """Serial,Device,start_date,end_date
T8160P1123332B02,Garage,,2025-08-31
T8162T1024354A8B,Balcony,,2025-08-31
T8600P102338033E,Backyard,,
T8162T1024354A8B,Garage,2025-09-01,
T8160P1123332B02,Balcony,2025-09-01,"""
        
        csv_path = self.create_test_csv(csv_content)
        
        with patch('config.os.path.join') as mock_join:
            mock_join.return_value = csv_path
            config = Config()
            devices = Config.get_all_devices()
            
            # Should return unique device names in order of first appearance
            expected = ["Garage", "Balcony", "Backyard"]
            self.assertEqual(devices, expected)
        
        os.unlink(csv_path)


def run_csv_validation_test():
    """Validate the actual devices.csv file for conflicts."""
    try:
        config = Config()
        logger.info("✅ devices.csv validation passed - no date conflicts detected")
        
        # Test specific swap scenario
        before_swap = datetime(2025, 8, 15)
        after_swap = datetime(2025, 9, 15)
        
        garage_before = Config.get_device_for_date("T8160P1123332B02", before_swap)
        balcony_before = Config.get_device_for_date("T8162T1024354A8B", before_swap)
        garage_after = Config.get_device_for_date("T8160P1123332B02", after_swap)
        balcony_after = Config.get_device_for_date("T8162T1024354A8B", after_swap)
        
        logger.info(f"Before swap (2025-08-15): T8160P1123332B02 -> {garage_before}, T8162T1024354A8B -> {balcony_before}")
        logger.info(f"After swap (2025-09-15): T8160P1123332B02 -> {garage_after}, T8162T1024354A8B -> {balcony_after}")
        
        # Verify the swap worked
        assert garage_before == "Garage" and balcony_before == "Balcony"
        assert garage_after == "Balcony" and balcony_after == "Garage"
        logger.info("✅ Device swap scenario working correctly")
        
    except Exception as e:
        logger.error(f"❌ devices.csv validation failed: {e}")
        raise


if __name__ == '__main__':
    # Run CSV validation first
    print("Validating actual devices.csv file...")
    run_csv_validation_test()
    print()
    
    # Run unit tests
    print("Running unit tests...")
    unittest.main()