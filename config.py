#!/usr/bin/env python3

import csv
import os
from datetime import datetime
from typing import Optional

from logging_config import create_logger

logger = create_logger(__name__)


class Config:
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self._devices = []
            self._device_dict = {}
            self._device_entries = []  # Store full device entries with dates
            self._load_devices()
            self._initialized = True

    # Load devices from the devices.csv file
    # Requirements for devices.csv:
    # 1. File format: CSV with columns "Serial", "Device", and optional "start_date", "end_date"
    # 2. File location: Same directory as this config.py file
    # 3. Order: The order of devices in the CSV file is preserved and used for plotting
    # 4. The header row must include Serial,Device at minimum
    # 5. Date format: YYYY-MM-DD or empty for open-ended ranges
    def _load_devices(self):
        csv_path = os.path.join(os.path.dirname(__file__), "devices.csv")
        with open(csv_path, "r") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                serial = row["Serial"]
                device = row["Device"]
                start_date = self._parse_date(row.get("start_date", ""))
                end_date = self._parse_date(row.get("end_date", ""))
                
                entry = {
                    "serial": serial,
                    "device": device,
                    "start_date": start_date,
                    "end_date": end_date
                }
                self._device_entries.append(entry)
                
                # Maintain backward compatibility with simple device list
                self._devices.append((serial, device))
            
            # Create device dict from first occurrence of each serial (backward compatibility)
            seen_serials = set()
            for serial, device in self._devices:
                if serial not in seen_serials:
                    self._device_dict[serial] = device
                    seen_serials.add(serial)
        
        # Validate for date conflicts
        self._validate_device_entries()
        
        logger.debug(f"{len(self._device_entries)} device entries loaded from {csv_path}:")
        for entry in self._device_entries:
            logger.debug(f"  Serial: {entry['serial']}, Device: {entry['device']}, Start: {entry['start_date']}, End: {entry['end_date']}")

    @classmethod
    def get_device_dict(cls):
        # Returns a dictionary mapping device serials to device names
        return cls()._device_dict

    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse date string in YYYY-MM-DD format, return None if empty or invalid."""
        if not date_str or not date_str.strip():
            return None
        try:
            return datetime.strptime(date_str.strip(), "%Y-%m-%d")
        except ValueError as e:
            logger.warning(f"Invalid date format '{date_str}': {e}")
            return None
    
    def _validate_device_entries(self):
        """Validate that there are no overlapping date ranges for the same serial."""
        serial_entries = {}
        for entry in self._device_entries:
            serial = entry["serial"]
            if serial not in serial_entries:
                serial_entries[serial] = []
            serial_entries[serial].append(entry)
        
        for serial, entries in serial_entries.items():
            # Sort entries by start date (None values go first)
            entries.sort(key=lambda x: x["start_date"] or datetime.min)
            
            for i in range(len(entries) - 1):
                current = entries[i]
                next_entry = entries[i + 1]
                
                # Check for overlap
                if current["end_date"] and next_entry["start_date"]:
                    if current["end_date"] >= next_entry["start_date"]:
                        logger.error(f"Date overlap for serial {serial}: {current['device']} ends {current['end_date'].strftime('%Y-%m-%d')}, {next_entry['device']} starts {next_entry['start_date'].strftime('%Y-%m-%d')}")
                        raise ValueError(f"Overlapping date ranges for serial {serial}")
    
    @classmethod
    def get_device_for_date(cls, serial: str, target_date: datetime) -> str:
        """Get device name for a given serial at a specific date.
        
        Args:
            serial: Device serial number
            target_date: Date to look up device assignment for
            
        Returns:
            Device name or serial if no match found
        """
        instance = cls()
        
        # Find matching entries for this serial
        matching_entries = [
            entry for entry in instance._device_entries
            if entry["serial"] == serial
        ]
        
        if not matching_entries:
            return serial  # Return serial if no device mapping found
        
        # Find the entry that covers the target date
        for entry in matching_entries:
            start_date = entry["start_date"]
            end_date = entry["end_date"]
            
            # Check if target_date falls within this entry's range
            if start_date and target_date < start_date:
                continue  # Too early
            if end_date and target_date > end_date:
                continue  # Too late
            
            return entry["device"]
        
        # If no date-specific entry found, but we have entries, return the device from the first entry
        # This handles cases where all entries have future start dates
        if matching_entries:
            return matching_entries[0]["device"]
        else:
            return serial
    
    @classmethod
    def get_all_devices(cls):
        # Returns the list of unique device names in the order they first appear in the CSV file
        seen = set()
        return [
            device
            for _, device in cls()._devices
            if not (device in seen or seen.add(device))
        ]

    # File naming conventions
    METADATA = "_videos.csv"
    PLAYLIST = "_videos.m3u"
    GRAPHS = "_videos.html"
    CONFIG = "_videos.yaml"
    MOVIE = "_video.mp4"
    FRAGMENTS = "_video_fragments"
    TAGS = "_tags.json"
    MOVIE_TAGS = "_tags.mp4"
    SCATTER_PLOTS = "_scatter_plots.html"

    # Label parameters for video timestamps
    DATE_TIME_LABEL_BORDER_WIDTH = 1
    DATE_TIME_LABEL_LEFT_MARGIN = 10
    DATE_TIME_LABEL_TOP_MARGIN = 30
    DATE_TIME_LABEL_FONTSIZE = 20
