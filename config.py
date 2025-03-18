#!/usr/bin/env python3

import csv
import os

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
            self._load_devices()
            self._initialized = True

    # Load devices from the devices.csv file
    # Requirements for devices.csv:
    # 1. File format: CSV with two columns - "Serial" and "Device"
    # 2. File location: Same directory as this config.py file
    # 3. Order: The order of devices in the CSV file is preserved and used for plotting
    # 4. The header row Serial,Device is expected in the CSV file
    def _load_devices(self):
        csv_path = os.path.join(os.path.dirname(__file__), "devices.csv")
        with open(csv_path, "r") as csvfile:
            reader = csv.DictReader(csvfile)
            self._devices = [(row["Serial"], row["Device"]) for row in reader]
            self._device_dict = dict(self._devices)
        logger.debug(f"{len(self._device_dict)} known devices loaded from {csv_path}:")
        for serial, device in self._devices:
            logger.debug(f"  Serial: {serial}, Device: {device}")

    @classmethod
    def get_device_dict(cls):
        # Returns a dictionary mapping device serials to device names
        return cls()._device_dict

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
