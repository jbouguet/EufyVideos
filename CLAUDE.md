# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Python library for managing, analyzing, and processing home security video collections from Eufy cameras. Covers video metadata management, YOLO-based object detection/tracking, person recognition (CLIP + ReID embeddings), occupancy detection, interactive dashboards, and composite video generation.

## Setup & Environment

- **Python 3.12** recommended (3.13 also supported; 3.11 and below not supported)
- **ffmpeg** required: `brew install ffmpeg`
- **Plotly 5.24.1** strictly required (6.x breaks graph orientations)
- Virtual env: `python -m venv myenv && source myenv/bin/activate`
- Install deps: `pip install -r requirements.txt`
- Verify: `python -c "from version_config import verify_environment; verify_environment(raise_on_error=True)"`

### Audio Enhancement Setup (DeepFilterNet)

Optional dependency used by `audio_enhancer.py` to isolate human speech and remove background noise from video audio tracks. Requires a separate install sequence — a plain `pip install -r requirements.txt` will not set this up correctly:

```bash
# 1. Install a Rust toolchain (one-time; needed to build DeepFilterNet's native extension)
brew install rust

# 2. Install DeepFilterNet without letting pip touch this project's numpy/torch versions.
#    Its packaging metadata pins numpy<2.0, which conflicts with the numpy>=2.5 this
#    project otherwise requires (scipy/opencv), even though it runs correctly against
#    numpy 2.x in practice.
pip install --no-deps deepfilternet deepfilterlib

# 3. Install torchcodec, the backend torchaudio's load()/save() now require.
pip install torchcodec
```

`audio_enhancer.py` also patches around two torchaudio APIs that DeepFilterNet 0.5.6 still targets but that were removed from modern torchaudio (`torchaudio.backend.common.AudioMetaData`, `torchaudio.info`). The patch is applied automatically on first use and is a no-op if those APIs are already present (e.g. with an older torchaudio).

## Running Tests

```bash
# All tests
pytest

# Single test file
pytest test_config_devices.py

# Single test function
pytest test_config_devices.py::TestConfigDevices::test_basic_device_loading

# Verbose
pytest -v
```

## Main Entry Points

All scripts run from the `EufyVideos/` directory:

```bash
# Main analysis pipeline (primary entry point)
python video_analyzer.py --config analysis_config.yaml [--debug]

# Interactive web dashboard (Dash on localhost:8050)
python dashboard_interactive.py

# Static dashboard graphs
python dashboard.py

# Occupancy model training/inference
python occupancy.py

# Tag visualization (annotated video output)
python tag_visualizer.py
```

## Architecture

### Configuration System
- **`config.py`**: Singleton `Config` class. Loads device serial-to-name mappings from `devices.csv`. Supports date-ranged device assignments (same serial can map to different device names over time). Also defines file naming conventions (METADATA, PLAYLIST, GRAPHS, etc.).
- **`analysis_config.yaml`**: Primary pipeline config. Uses custom YAML loader (`JoinPathsLoader` in `video_analyzer.py`) with `!join` tag for path composition and YAML anchors/aliases for DRY paths.
- **`devices.csv`**: CSV with columns `Serial,Device,start_date,end_date`. Date ranges allow reassigning physical cameras to different logical locations.

### Data Flow Pipeline

```
analysis_config.yaml
  → VideoAnalyzer loads VideoDatabaseList (multiple video sources)
    → VideoDatabase loads VideoMetadata from directories or cached CSV files
      → VideoFilter/VideoSelector applies date/time/device/duration filters
        → Story objects process filtered videos:
            ├─ Dashboard/InteractiveDashboard (visualization)
            ├─ TagProcessor → ObjectDetector (detection/tracking)
            ├─ VideoGenerator (composite video creation)
            └─ Occupancy (occupancy status detection)
```

### Key Module Groups

**Video Management**: `video_metadata.py` (core `VideoMetadata` dataclass), `video_database.py` (`VideoDatabase`/`VideoDatabaseList`), `video_filter.py` (`VideoFilter`, `VideoSelector`, `DateRange`, `TimeRange`)

**Detection/Tracking**: `object_detector_base.py` (abstract `ObjectDetector` + `ObjectDetectorFactory`), `object_detector_yolo.py`, `object_detector_yolo_optimized.py`, `object_detector_florence2.py`. Factory creates detectors from `TaggerConfig`. `tag_processor.py` orchestrates detection and produces `VideoTags` (JSON-serializable).

**Person Recognition**: `person_detector.py` (`PersonCrop`, `PersonTrack`) → `person_embedding.py` (CLIP + ReID embeddings) → `person_clustering.py` / `conservative_person_clustering.py` (DBSCAN/agglomerative clustering with quality metrics)

**Visualization**: `video_data_aggregator.py` (temporal aggregation into daily/hourly DataFrames), `video_graph_creator.py` / `video_scatter_plots_creator.py` (Plotly graphs), `dashboard.py` (static HTML), `dashboard_interactive.py` (Dash web app)

**Video Generation**: `video_generator.py` (`VideoGenerator` with `InputFragments`/`OutputVideo` configs for trimming, cropping, scaling, timestamp overlay), `story_creator.py` (`Story` class orchestrating the full workflow from YAML config), `audio_enhancer.py` (`AudioEnhancer` — optional DeepFilterNet-based post-process that takes a `VideoGenerator` output mp4 and produces a copy with the audio track denoised/voice-isolated; video stream is untouched)

**Occupancy**: `occupancy.py` — three modes: `CALENDAR` (manual), `HEURISTIC` (activity thresholds), `ML_MODEL` (Decision Tree trained on daily activity features, saved as `occupancy_model.pkl`)

### Design Patterns
- **Singleton**: `Config` class
- **Factory**: `ObjectDetectorFactory.create_detector()` — register new detectors here
- **Dataclass-heavy**: `VideoMetadata`, `InputFragments`, `OutputVideo`, `TaggerConfig`, `Story`, etc.
- **Custom YAML**: `!join` tag for path composition via `JoinPathsLoader`
- **Centralized logging**: `logging_config.py` → `create_logger(__name__)`

## Code Style

- Formatter: **black**
- Type checking: **mypy**
- Linting: **pylint**
