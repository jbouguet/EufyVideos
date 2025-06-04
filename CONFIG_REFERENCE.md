# Person Recognition Configuration Reference

## Overview

This document provides detailed explanations for all configuration parameters in the person recognition system, with examples and recommendations for different use cases.

## Configuration File Structure

```yaml
# analysis_config_with_person_recognition.yaml

# Base directory definitions
directories:
  root_database: &root_database "/path/to/videos/record"
  stories_output: &stories_output "/path/to/output/stories"
  tags_database: &tags_dir !join [*root_database, "/tags_database"]
  batch_dirs: &batch_dirs
    - !join [*root_database, "/Batch022"]

# Person Recognition Database Configuration  
person_recognition:
  database_file: &person_db_file !join [*root_database, "/person_recognition/persons.json"]
  embeddings_dir: &embeddings_dir !join [*root_database, "/person_recognition/embeddings"]
  crops_dir: &crops_dir !join [*root_database, "/person_recognition/crops"]

# Video database configuration
video_database_list:
- force_video_directories_scanning: false
  video_metadata_file: !join [*root_database, "/videos_in_batches.csv"]
  video_directories: *batch_dirs

# Tag database files (will include person recognition data)
tag_database_files:
- !join [*tags_dir, "/person_recognition_tags.json"]

output_directory: *stories_output
process_stories: true

stories:
  - name: 'Video Analysis with Person Recognition'
    selectors:
      # Video selection options...
    
    tag_processing: true
    tag_processing_config:
      # Configuration parameters detailed below...
    
    tag_video_generation: true
    tag_video_generation_config:
      # Video generation parameters...
```

## Directory Configuration

### Base Directories

| Parameter | Description | Example |
|-----------|-------------|---------|
| `root_database` | Root directory containing all video files and analysis data | `/Users/jbouguet/Documents/EufySecurityVideos/record` |
| `stories_output` | Output directory for generated videos and reports | `/Users/jbouguet/Documents/EufySecurityVideos/stories` |
| `tags_database` | Directory containing tag analysis results | `{root_database}/tags_database` |
| `batch_dirs` | List of directories containing video files to analyze | `["{root_database}/Batch022"]` |

### Person Recognition Directories

| Parameter | Description | Example |
|-----------|-------------|---------|
| `database_file` | Path to the JSON file storing person identities and track labels | `{root_database}/person_recognition/persons.json` |
| `embeddings_dir` | Directory to store face embedding files (one per video/analysis) | `{root_database}/person_recognition/embeddings` |
| `crops_dir` | Directory to store extracted face crop images | `{root_database}/person_recognition/crops` |

## Story Configuration

### Video Selection (`selectors`)

```yaml
selectors:
  # Option 1: Specific files
  - filenames: ['T8600P102338033E_20240930085536.mp4']
  
  # Option 2: Date and time range
  - date_range:
      start: '2024-11-18'
      end: '2024-11-18'
    time_range:
      start: '08:40:00'
      end: '09:10:00'
    devices: ['Backyard', 'FrontDoor']
  
  # Option 3: All files in batch directories
  - all_files: true
```

## Tag Processing Configuration

### Core Detection Parameters

#### Basic YOLO Settings

```yaml
tag_processing_config:
  model: "Yolo11x_Optimized"           # YOLO model variant
  task: "Track"                        # Detection task type
  num_frames_per_second: 2.0           # Frame sampling rate
  conf_threshold: 0.3                  # General object detection confidence
  batch_size: 8                        # Processing batch size
```

**Parameter Details:**

| Parameter | Options | Default | Description |
|-----------|---------|---------|-------------|
| `model` | `"Yolo11x_Optimized"`, `"Yolo11n"`, `"Yolo11s"`, `"Yolo11m"`, `"Yolo11l"` | `"Yolo11x_Optimized"` | YOLO model size. Larger models are more accurate but slower |
| `task` | `"Track"`, `"Detect"` | `"Track"` | `"Track"` enables object tracking across frames (required for person recognition) |
| `num_frames_per_second` | `0.1` - `30.0` | `2.0` | How many frames per second to analyze. Higher = more accurate but slower |
| `conf_threshold` | `0.1` - `0.9` | `0.3` | Minimum confidence for general object detection |
| `batch_size` | `1` - `32` | `8` | Number of frames processed simultaneously. Higher = faster but more memory |

#### Person Recognition Master Switch

```yaml
enable_person_recognition: true        # Enable/disable person recognition features
```

| Value | Description |
|-------|-------------|
| `true` | Enable full person recognition pipeline |
| `false` | Disable person recognition (backward compatibility mode) |

### Person Detection Parameters

#### Detection Thresholds

```yaml
person_min_confidence: 0.6             # Minimum confidence for person detection
person_min_bbox_area: 2000             # Minimum bounding box area in pixels²
```

**Parameter Details:**

| Parameter | Range | Default | Description | Use Cases |
|-----------|-------|---------|-------------|-----------|
| `person_min_confidence` | `0.1` - `0.9` | `0.6` | Minimum confidence specifically for person detection | High values (0.7-0.8) reduce false positives; low values (0.4-0.5) catch more distant people |
| `person_min_bbox_area` | `500` - `10000` | `2000` | Minimum area of person bounding box | Filters out very small/distant people. Adjust based on camera distance |

**Recommendations:**
- **Close-range cameras** (indoor): `person_min_confidence: 0.7`, `person_min_bbox_area: 3000`
- **Medium-range cameras** (backyard): `person_min_confidence: 0.6`, `person_min_bbox_area: 2000`  
- **Long-range cameras** (perimeter): `person_min_confidence: 0.4`, `person_min_bbox_area: 1000`

#### Face Crop Parameters

```yaml
person_crop_size: [224, 224]           # Size of extracted face images (width, height)
max_crops_per_track: 10                # Maximum face crops to save per person track
```

| Parameter | Options | Default | Description |
|-----------|---------|---------|-------------|
| `person_crop_size` | `[128, 128]` to `[512, 512]` | `[224, 224]` | Pixel dimensions for cropped face images. Larger = better quality but more storage |
| `max_crops_per_track` | `1` - `50` | `10` | Maximum face crops saved per person track. More crops = better recognition but more storage |

**Storage Impact:**
- `[128, 128]` crops: ~10KB each
- `[224, 224]` crops: ~30KB each  
- `[512, 512]` crops: ~100KB each

### File Path Configuration

```yaml
person_database_file: "/path/to/persons.json"
person_embeddings_file: "/path/to/video_embeddings.json"  
person_crops_dir: "/path/to/crops/"
```

| Parameter | Description | Example |
|-----------|-------------|---------|
| `person_database_file` | Path to person database JSON file | `persons.json` |
| `person_embeddings_file` | Output path for video-specific embeddings | `analysis_embeddings.json` |
| `person_crops_dir` | Directory to save face crop images | `./crops/analysis_name/` |

### AI Embedding Generation

#### Device Configuration

```yaml
embedding_device: "mps"               # Computing device for AI processing
```

| Option | Description | Performance | Requirements |
|--------|-------------|-------------|--------------|
| `"mps"` | Apple Metal Performance Shaders (Mac GPU) | Fast | macOS with Apple Silicon |
| `"cuda"` | NVIDIA GPU acceleration | Fastest | NVIDIA GPU with CUDA |
| `"cpu"` | CPU processing | Slowest | Any system |

#### Embedding Parameters

```yaml
embedding_dim: 512                    # Dimensionality of face embedding vectors
clip_weight: 0.7                      # Weight for CLIP-based embeddings (0.0-1.0)
reid_weight: 0.3                      # Weight for ReID-based embeddings (0.0-1.0)
```

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| `embedding_dim` | `256`, `512`, `1024` | `512` | Size of face embedding vectors. Higher = more precise but slower |
| `clip_weight` | `0.0` - `1.0` | `0.7` | Weight for CLIP-based features (good for general appearance) |
| `reid_weight` | `0.0` - `1.0` | `0.3` | Weight for ReID-based features (good for person re-identification) |

**Note:** `clip_weight + reid_weight` should equal `1.0`

### Auto-Identification Parameters

#### Recognition Thresholds

```yaml
similarity_threshold: 0.75            # Minimum similarity for clustering faces
auto_label_confidence: 0.8            # Minimum confidence for automatic labeling
enable_auto_labeling: true            # Enable/disable automatic person labeling
```

| Parameter | Range | Default | Description | Impact |
|-----------|-------|---------|-------------|--------|
| `similarity_threshold` | `0.5` - `0.95` | `0.75` | Minimum similarity score for clustering similar faces | Higher = tighter clusters, fewer false matches |
| `auto_label_confidence` | `0.5` - `0.95` | `0.8` | Minimum confidence for automatic person identification | Higher = fewer auto-labels but more accurate |
| `enable_auto_labeling` | `true`/`false` | `true` | Whether to automatically assign names to recognized faces | Disable for manual-only labeling |

**Tuning Guidelines:**
- **Conservative** (fewer false positives): `similarity_threshold: 0.8`, `auto_label_confidence: 0.85`
- **Balanced** (recommended): `similarity_threshold: 0.75`, `auto_label_confidence: 0.8`
- **Aggressive** (more auto-labels): `similarity_threshold: 0.7`, `auto_label_confidence: 0.7`

## Video Generation Configuration

### Enhanced Visualization

```yaml
tag_video_generation: true
tag_video_generation_config:
  output_size:
    width: 1600
    height: 900
  
  # Person recognition visualization options
  show_person_identities: true         # Display person names on video
  highlight_family_members: true      # Special highlighting for known family
  show_confidence_scores: false       # Show recognition confidence values
  
  # Standard options
  object_detection_threshold: 0.4
  show_object_labels: true
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `show_person_identities` | `false` | Display person names above bounding boxes |
| `highlight_family_members` | `false` | Use special colors/styling for family members |
| `show_confidence_scores` | `false` | Display confidence percentages with names |

## Example Configurations

### 1. Family Home Surveillance

```yaml
# Optimized for recognizing family members in home environment
tag_processing_config:
  model: "Yolo11x_Optimized"
  task: "Track"
  num_frames_per_second: 2.0
  conf_threshold: 0.2
  batch_size: 8
  
  enable_person_recognition: true
  person_min_confidence: 0.6           # Moderate threshold for family
  person_min_bbox_area: 2500           # Filter small distant figures
  max_crops_per_track: 15              # More crops for better recognition
  
  embedding_device: "mps"
  similarity_threshold: 0.75
  auto_label_confidence: 0.8
  enable_auto_labeling: true
  
tag_video_generation_config:
  show_person_identities: true
  highlight_family_members: true
```

### 2. Security Monitoring

```yaml
# Optimized for security with higher accuracy requirements
tag_processing_config:
  model: "Yolo11x_Optimized"
  task: "Track"
  num_frames_per_second: 3.0           # Higher frame rate for security
  conf_threshold: 0.3
  batch_size: 6
  
  enable_person_recognition: true
  person_min_confidence: 0.7           # Higher confidence for security
  person_min_bbox_area: 1500           # Catch smaller/distant people
  max_crops_per_track: 20              # More evidence per person
  
  embedding_device: "cuda"             # Use fastest processing
  similarity_threshold: 0.8            # Conservative clustering
  auto_label_confidence: 0.85          # High confidence auto-labeling
  enable_auto_labeling: true
  
tag_video_generation_config:
  show_person_identities: true
  show_confidence_scores: true         # Show confidence for security review
```

### 3. Research/Development

```yaml
# Optimized for research with maximum data collection
tag_processing_config:
  model: "Yolo11x_Optimized"
  task: "Track"
  num_frames_per_second: 1.0           # Slower for research budget
  conf_threshold: 0.2
  batch_size: 12
  
  enable_person_recognition: true
  person_min_confidence: 0.4           # Low threshold to catch all detections
  person_min_bbox_area: 1000           # Include very small detections
  max_crops_per_track: 30              # Maximum data collection
  
  embedding_device: "cpu"              # Use available hardware
  similarity_threshold: 0.7            # Loose clustering for exploration
  auto_label_confidence: 0.9           # Conservative auto-labeling
  enable_auto_labeling: false          # Manual labeling only for research
```

### 4. Performance Testing

```yaml
# Optimized for fast processing and testing
tag_processing_config:
  model: "Yolo11n"                     # Smallest/fastest model
  task: "Track"
  num_frames_per_second: 0.5           # Very low frame rate
  conf_threshold: 0.5
  batch_size: 16                       # Large batches
  
  enable_person_recognition: true
  person_min_confidence: 0.7
  person_min_bbox_area: 3000           # Only clear, large detections
  max_crops_per_track: 3               # Minimal crops
  
  embedding_device: "mps"
  similarity_threshold: 0.8
  auto_label_confidence: 0.85
  enable_auto_labeling: true
```

## Configuration Validation

### Validation Script

```bash
# Validate configuration before running analysis
python -c "
import yaml
from person_recognition_processor import PersonRecognitionConfig

# Load and validate configuration
with open('analysis_config_with_person_recognition.yaml', 'r') as f:
    config_data = yaml.safe_load(f)

# Extract tag processing config
tag_config = config_data['stories'][0]['tag_processing_config']

try:
    # Create config object (this validates parameters)
    config = PersonRecognitionConfig.from_dict(tag_config)
    print('✅ Configuration is valid')
    print(f'Person recognition enabled: {config.enable_person_recognition}')
    print(f'Processing device: {config.embedding_device}')
    print(f'Frame rate: {config.num_frames_per_second} FPS')
except Exception as e:
    print(f'❌ Configuration error: {e}')
"
```

### Common Validation Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `Invalid device: xyz` | Unsupported `embedding_device` | Use `"mps"`, `"cuda"`, or `"cpu"` |
| `Weights don't sum to 1.0` | `clip_weight + reid_weight ≠ 1.0` | Adjust weights to sum to 1.0 |
| `File not found` | Invalid path in `person_database_file` | Check file path exists |
| `Invalid threshold` | Threshold outside 0.0-1.0 range | Use values between 0.0 and 1.0 |

## Performance Optimization

### Memory Usage

| Configuration | Memory Impact | Recommendation |
|---------------|---------------|----------------|
| `batch_size: 16` | High | Reduce if out-of-memory errors |
| `person_crop_size: [512, 512]` | High | Use `[224, 224]` for most cases |
| `max_crops_per_track: 50` | High | Limit to 10-20 for most uses |
| `embedding_dim: 1024` | Medium | Use 512 unless high precision needed |

### Processing Speed

| Configuration | Speed Impact | Recommendation |
|---------------|--------------|----------------|
| `num_frames_per_second: 5.0` | Slow | Use 1.0-2.0 for most videos |
| `model: "Yolo11x_Optimized"` | Medium | Use `"Yolo11n"` for speed testing |
| `embedding_device: "cpu"` | Slow | Use GPU (`"mps"` or `"cuda"`) if available |
| `enable_auto_labeling: true` | Medium | Disable for manual-only workflows |

### Storage Usage

| Configuration | Storage Impact | Typical Usage |
|---------------|----------------|---------------|
| `max_crops_per_track: 30` | High | ~1MB per person track |
| `person_crop_size: [512, 512]` | High | ~100KB per crop |
| `person_crop_size: [224, 224]` | Medium | ~30KB per crop |
| `person_crop_size: [128, 128]` | Low | ~10KB per crop |

## Integration with Existing Workflow

### Backward Compatibility

To maintain compatibility with existing configurations:

```yaml
# Existing configuration without person recognition
tag_processing_config:
  model: "Yolo11x_Optimized"
  task: "Track"
  num_frames_per_second: 2.0
  conf_threshold: 0.3
  # enable_person_recognition: false  # Default behavior
```

### Gradual Migration

1. **Phase 1**: Add person recognition to one story with `enable_person_recognition: true`
2. **Phase 2**: Build person database with manual labeling
3. **Phase 3**: Enable auto-labeling: `enable_auto_labeling: true`
4. **Phase 4**: Apply to all stories in configuration

### Legacy Support

All existing analysis configurations continue to work unchanged. Person recognition features are only activated when explicitly enabled with `enable_person_recognition: true`.