# EufyVideos Person Recognition System - Complete Guide

## 📋 Table of Contents

1. [Overview](#overview)
2. [Architecture & Components](#architecture--components)
3. [Configuration Reference](#configuration-reference)
4. [Standalone Workflow](#standalone-workflow)
5. [Integrated Production Workflow](#integrated-production-workflow)
6. [Manual Labeling Tools](#manual-labeling-tools)
7. [Automatic Recognition](#automatic-recognition)
8. [Visualization & Export](#visualization--export)
9. [Best Practices](#best-practices)
10. [Troubleshooting](#troubleshooting)

---

## Overview

The EufyVideos Person Recognition System extends the existing video analysis pipeline with AI-powered person detection, identification, and tracking capabilities. It provides both standalone tools for manual labeling and seamless integration with the production analysis workflow.

### Key Capabilities

- **🎯 Person Detection**: Automatic detection of people in surveillance videos
- **🧠 Face Recognition**: AI-powered face embedding generation and similarity matching  
- **🏷️ Manual Labeling**: Web-based GUI for manual face annotation
- **🤖 Auto-Labeling**: Automatic identification based on existing labeled data
- **📊 Clustering**: Grouping similar faces for efficient batch labeling
- **🎬 Enhanced Visualization**: Person identities displayed in generated videos
- **💾 Database Integration**: Persistent storage with existing person database

---

## Architecture & Components

### Core Components

```
┌─────────────────────────────────────────────────────────────────┐
│                    EufyVideos Person Recognition                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐    ┌──────────────────┐    ┌─────────────┐ │
│  │   Video Input   │    │  Person Detection│    │ Embeddings  │ │
│  │   (.mp4 files)  │───▶│    (YOLO11x)     │───▶│ Generation  │ │
│  └─────────────────┘    └──────────────────┘    └─────────────┘ │
│                                                         │       │
│  ┌─────────────────┐    ┌──────────────────┐           │       │
│  │ Person Database │    │  Manual Labeling │           │       │
│  │  (persons.json) │◀──▶│   GUI (Streamlit)│◀──────────┘       │
│  └─────────────────┘    └──────────────────┘                   │
│            │                       │                           │
│            ▼                       ▼                           │
│  ┌─────────────────┐    ┌──────────────────┐                   │
│  │ Auto Recognition│    │   Crop Storage   │                   │
│  │   & Labeling    │    │  (image files)   │                   │
│  └─────────────────┘    └──────────────────┘                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### File Structure

```
/path/to/your/videos/
├── record/                              # Main video storage
│   ├── Batch022/                        # Video batch directory
│   │   └── T8600P102338033E_*.mp4      # Video files
│   ├── person_recognition/              # Person recognition data
│   │   ├── persons.json                 # Person database
│   │   ├── embeddings/                  # Face embeddings per video
│   │   │   ├── video1_embeddings.json
│   │   │   └── video2_embeddings.json
│   │   └── crops/                       # Extracted face images
│   │       ├── video1_track1_frame001.jpg
│   │       └── video2_track5_frame156.jpg
│   └── tags_database/                   # Enhanced tag data
│       └── person_recognition_tags.json
└── stories/                             # Generated output videos
    └── Person_Recognition_Demo/
        ├── tagged_video.mp4             # Video with person labels
        └── analysis_report.json
```

---

## Configuration Reference

### Basic Configuration Structure

The person recognition system uses YAML configuration files that extend the existing EufyVideos analysis configuration:

```yaml
# Enhanced Analysis Configuration with Person Recognition
directories:
  root_database: "/path/to/your/videos/record"
  stories_output: "/path/to/output/stories"
  
# Person Recognition Database Configuration
person_recognition:
  database_file: !join [*root_database, "/person_recognition/persons.json"]
  embeddings_dir: !join [*root_database, "/person_recognition/embeddings"] 
  crops_dir: !join [*root_database, "/person_recognition/crops"]

stories:
  - name: 'Video Analysis with Person Recognition'
    tag_processing_config:
      # Enable person recognition
      enable_person_recognition: true
      
      # Core detection settings
      model: "Yolo11x_Optimized"
      num_frames_per_second: 2.0
      conf_threshold: 0.3
      
      # Person recognition paths
      person_database_file: *person_db_file
      person_embeddings_file: "video_specific_embeddings.json"
      person_crops_dir: *crops_dir
      
      # Detection parameters
      person_crop_size: [224, 224]
      person_min_confidence: 0.6
      person_min_bbox_area: 2000
      max_crops_per_track: 10
      
      # AI processing
      embedding_device: "mps"  # or "cuda" or "cpu"
      embedding_dim: 512
      clip_weight: 0.7
      reid_weight: 0.3
      
      # Auto-identification
      similarity_threshold: 0.75
      auto_label_confidence: 0.8
      enable_auto_labeling: true
    
    # Enhanced video generation with person IDs
    tag_video_generation_config:
      show_person_identities: true
      highlight_family_members: true
```

### Configuration Parameters Explained

#### Core Detection Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enable_person_recognition` | `false` | Master switch for person recognition features |
| `model` | `"Yolo11x_Optimized"` | YOLO model for object/person detection |
| `num_frames_per_second` | `2.0` | Frame sampling rate for processing |
| `conf_threshold` | `0.3` | Minimum confidence for general object detection |

#### Person-Specific Detection

| Parameter | Default | Description |
|-----------|---------|-------------|
| `person_min_confidence` | `0.6` | Minimum confidence threshold specifically for person detection |
| `person_min_bbox_area` | `2000` | Minimum bounding box area (pixels²) to filter small detections |
| `person_crop_size` | `[224, 224]` | Size for cropped face images (width, height) |
| `max_crops_per_track` | `10` | Maximum face crops to save per person track |

#### AI Embedding Generation

| Parameter | Default | Description |
|-----------|---------|-------------|
| `embedding_device` | `"mps"` | Computing device: `"mps"` (Mac), `"cuda"` (NVIDIA), `"cpu"` |
| `embedding_dim` | `512` | Dimensionality of face embedding vectors |
| `clip_weight` | `0.7` | Weight for CLIP-based embeddings (0.0-1.0) |
| `reid_weight` | `0.3` | Weight for ReID-based embeddings (0.0-1.0) |

#### Auto-Identification

| Parameter | Default | Description |
|-----------|---------|-------------|
| `similarity_threshold` | `0.75` | Minimum similarity score for clustering faces (0.0-1.0) |
| `auto_label_confidence` | `0.8` | Minimum confidence for automatic person labeling (0.0-1.0) |
| `enable_auto_labeling` | `true` | Whether to automatically assign names to recognized faces |

#### File Paths

| Parameter | Description | Example |
|-----------|-------------|---------|
| `person_database_file` | Path to person database JSON file | `persons.json` |
| `person_embeddings_file` | Video-specific embeddings output file | `video_embeddings.json` |
| `person_crops_dir` | Directory to save extracted face crops | `./crops/` |

#### Visualization Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `show_person_identities` | `false` | Display person names on generated videos |
| `highlight_family_members` | `false` | Special highlighting for known family members |

---

## Standalone Workflow

The person recognition system can be used independently of the main EufyVideos pipeline for testing, development, or specialized analysis.

### 1. Person Detection Demo

Run basic person detection on a single video:

```bash
# Run person detection demo
python person_recognition_demo.py

# Or with custom parameters
python person_recognition_demo.py \
  --video /path/to/video.mp4 \
  --output ./demo_output \
  --device mps \
  --max_crops 20
```

**Output:**
- `demo_output/person_crops/` - Extracted face images
- `demo_output/person_embeddings.json` - Face embedding vectors  
- `demo_output/persons.json` - Person database with detected tracks
- `demo_output/analysis_summary.txt` - Processing statistics

### 2. Manual Labeling Session

Launch the web-based labeling tool:

```bash
# Quick launch with demo data
python launch_labeling_gui.py

# Or launch manually
streamlit run person_labeling_streamlit.py

# Or with custom paths
streamlit run person_labeling_streamlit.py -- \
  --crops_dir /path/to/crops \
  --database /path/to/persons.json \
  --embeddings /path/to/embeddings.json
```

**Web Interface Features:**
- **Browse crops** in thumbnail grid or cluster view
- **Individual labeling** by clicking on faces and entering names
- **Batch labeling** by selecting multiple faces and applying one name
- **Smart clustering** to group similar faces automatically
- **Search and filter** by person name or filename
- **Export labels** to JSON for integration with other tools

### 3. Batch Processing

Process multiple videos independently:

```bash
# Create custom processing script
python -c "
from person_recognition_processor import PersonRecognitionConfig, PersonRecognitionProcessor
from video_metadata import VideoMetadata

# Configure processing
config = PersonRecognitionConfig(
    model='Yolo11x_Optimized',
    enable_person_recognition=True,
    person_database_file='persons.json',
    num_frames_per_second=1.0,
    person_min_confidence=0.6
)

# Process videos
processor = PersonRecognitionProcessor(config)
videos = ['video1.mp4', 'video2.mp4', 'video3.mp4']

for video_path in videos:
    metadata = VideoMetadata.from_video_file(video_path)
    tags = processor.run(metadata)
    print(f'Processed {video_path}: {len(tags.tags)} tag groups')
"
```

### 4. Testing and Validation

Run comprehensive tests:

```bash
# Test person recognition integration
python test_person_recognition_integration.py

# Test GUI functionality  
python test_labeling_gui.py

# Test core components
python test_person_database.py
python test_person_embedding.py
```

---

## Integrated Production Workflow

### 1. Configuration Setup

Create or modify your analysis configuration file:

```yaml
# analysis_config_with_person_recognition.yaml
directories:
  root_database: "/Users/jbouguet/Documents/EufySecurityVideos/record"
  stories_output: "/Users/jbouguet/Documents/EufySecurityVideos/stories"

person_recognition:
  database_file: !join [*root_database, "/person_recognition/persons.json"]
  embeddings_dir: !join [*root_database, "/person_recognition/embeddings"]
  crops_dir: !join [*root_database, "/person_recognition/crops"]

stories:
  - name: 'Daily Family Activity Analysis'
    selectors:
      - date_range:
          start: '2024-11-18'
          end: '2024-11-18'
        devices: ['Backyard', 'FrontDoor']
    
    tag_processing: true
    tag_processing_config:
      # Base YOLO detection
      model: "Yolo11x_Optimized"
      task: "Track"
      num_frames_per_second: 2.0
      conf_threshold: 0.2
      batch_size: 8
      
      # Enable person recognition
      enable_person_recognition: true
      person_database_file: !join [*root_database, "/person_recognition/persons.json"]
      person_embeddings_file: !join [*embeddings_dir, "/daily_activity_embeddings.json"]
      person_crops_dir: !join [*crops_dir, "/daily_activity"]
      
      # Optimized for family recognition
      person_min_confidence: 0.5
      max_crops_per_track: 15
      auto_label_confidence: 0.8
      enable_auto_labeling: true
    
    tag_video_generation: true
    tag_video_generation_config:
      output_size:
        width: 1600
        height: 900
      show_person_identities: true
      highlight_family_members: true
```

### 2. Run Analysis Pipeline

Execute the enhanced analysis:

```bash
# Run with person recognition enabled
python story_creator.py analysis_config_with_person_recognition.yaml
```

### 3. Review and Label

After initial processing:

```bash
# Launch labeling GUI to review auto-labels and add manual labels
python launch_labeling_gui.py

# Or launch with specific data from the analysis
streamlit run person_labeling_streamlit.py -- \
  --crops_dir /path/to/record/person_recognition/crops/daily_activity \
  --database /path/to/record/person_recognition/persons.json \
  --embeddings /path/to/record/person_recognition/embeddings/daily_activity_embeddings.json
```

### 4. Re-run with Updated Labels

After manual labeling, re-run analysis to apply updated identifications:

```bash
# Re-run the same configuration - it will use updated person database
python story_creator.py analysis_config_with_person_recognition.yaml
```

### 5. Production Automation

For ongoing production use:

```bash
#!/bin/bash
# daily_analysis.sh - Automated daily person recognition analysis

DATE=$(date +%Y-%m-%d)
CONFIG="analysis_config_with_person_recognition.yaml"

echo "Running daily analysis for $DATE..."

# Run analysis with person recognition
python story_creator.py $CONFIG

# Check if new unknown faces were detected
UNKNOWN_COUNT=$(python -c "
from person_database import PersonDatabase
db = PersonDatabase('/path/to/persons.json')
# Count unlabeled tracks from today
print(len([t for t in db.get_all_track_labels() if not t.person_name]))
")

if [ "$UNKNOWN_COUNT" -gt 0 ]; then
    echo "Found $UNKNOWN_COUNT unlabeled faces. Manual review recommended."
    echo "Run: python launch_labeling_gui.py"
fi

echo "Analysis complete. Output in stories/$DATE/"
```

---

## Manual Labeling Tools

### Web-Based GUI (Recommended)

The Streamlit-based labeling tool provides an intuitive web interface:

#### Features
- **Visual browsing** with thumbnail grids
- **Smart clustering** groups similar faces automatically  
- **Batch operations** for efficient labeling of multiple faces
- **Real-time search** and filtering
- **Statistics dashboard** showing progress
- **Export capabilities** for training data

#### Launching
```bash
# Quick launch with demo data
python launch_labeling_gui.py

# Manual launch with custom data
streamlit run person_labeling_streamlit.py -- \
  --crops_dir /path/to/face/crops \
  --database /path/to/persons.json \
  --embeddings /path/to/embeddings.json
```

#### Workflow
1. **Load Data**: Click "Load Demo Data" or specify custom paths
2. **Browse Crops**: View faces in grid layout, switch between "All", "Unlabeled", "Labeled" views
3. **Generate Clusters**: Click "Generate Clusters" to group similar faces
4. **Label Faces**: 
   - **Individual**: Click on a face, enter name, click "Apply"
   - **Batch**: Select multiple faces, enter name, click "Apply to Selected"
   - **Cluster**: Switch to cluster view, label entire clusters at once
5. **Save Work**: Click "Save Database" to persist changes
6. **Export**: Use "Export Labels" to download JSON for external use

### Desktop GUI (Alternative)

For environments where web access is restricted:

```bash
# Launch tkinter-based desktop application
python person_labeling_gui.py \
  --crops_dir /path/to/crops \
  --database /path/to/persons.json \
  --embeddings /path/to/embeddings.json
```

#### Features
- Native desktop application
- Thumbnail grid with selection
- Manual labeling with autocomplete
- Cluster visualization
- Export functionality

### Command-Line Tools

For scripted or automated labeling:

```bash
# Add a person to the database
python -c "
from person_database import PersonDatabase
db = PersonDatabase('persons.json')
person_id = db.add_person('John Doe', 'Family member')
print(f'Added person: {person_id}')
"

# Label a specific track
python -c "
from person_database import PersonDatabase
db = PersonDatabase('persons.json')
db.label_track(
    video_filename='video.mp4',
    track_id='track_5',
    person_id='person_123',
    labeled_by='manual_script'
)
db.save_database()
print('Track labeled successfully')
"

# Export current labels
python -c "
from person_database import PersonDatabase
db = PersonDatabase('persons.json')
labels = db.get_all_track_labels()
print(f'Found {len(labels)} labeled tracks')
for label in labels[:5]:  # Show first 5
    print(f'  {label.video_filename} track {label.track_id}: {label.person_name}')
"
```

---

## Automatic Recognition

### How Auto-Recognition Works

1. **Embedding Generation**: Each detected face is converted to a 512-dimensional vector
2. **Similarity Calculation**: New faces are compared against known faces using cosine similarity
3. **Threshold Matching**: Faces above the similarity threshold are automatically labeled
4. **Confidence Scoring**: Each auto-label includes a confidence score

### Configuration for Auto-Recognition

```yaml
tag_processing_config:
  # Enable auto-recognition
  enable_auto_labeling: true
  auto_label_confidence: 0.8      # Minimum confidence for auto-labeling
  similarity_threshold: 0.75      # Minimum similarity for matching
  
  # Fine-tuning parameters
  embedding_device: "mps"         # Use GPU acceleration
  clip_weight: 0.7               # Balance between CLIP and ReID features
  reid_weight: 0.3
```

### Monitoring Auto-Recognition

Check auto-recognition performance:

```bash
# View auto-recognition statistics
python -c "
from person_database import PersonDatabase
db = PersonDatabase('persons.json')

auto_labels = [t for t in db.get_all_track_labels() if t.labeled_by.startswith('auto')]
manual_labels = [t for t in db.get_all_track_labels() if t.labeled_by == 'manual']

print(f'Auto-labeled tracks: {len(auto_labels)}')
print(f'Manual-labeled tracks: {len(manual_labels)}')
print(f'Auto-labeling rate: {len(auto_labels)/(len(auto_labels)+len(manual_labels)):.1%}')

# Show confidence distribution
confidences = [t.confidence for t in auto_labels if t.confidence]
if confidences:
    import statistics
    print(f'Auto-label confidence: min={min(confidences):.2f}, avg={statistics.mean(confidences):.2f}, max={max(confidences):.2f}')
"
```

### Improving Auto-Recognition Accuracy

1. **Increase Training Data**: Label more diverse examples of each person
2. **Adjust Thresholds**: Lower `auto_label_confidence` for more aggressive labeling
3. **Review and Correct**: Use GUI to review auto-labels and fix errors
4. **Quality Crops**: Ensure manual labels use high-quality, well-lit face crops

---

## Visualization & Export

### Enhanced Video Generation

Videos generated with person recognition include visual person identification:

```yaml
tag_video_generation_config:
  output_size:
    width: 1600
    height: 900
  
  # Person recognition visualization
  show_person_identities: true      # Display names above bounding boxes
  highlight_family_members: true    # Special colors for family members
  show_confidence_scores: false     # Optionally show recognition confidence
  
  # Standard tag video options
  object_detection_threshold: 0.4
  show_object_labels: true
```

### Export Formats

#### Person Database Export

```bash
# Export complete person database
python -c "
from person_database import PersonDatabase
import json

db = PersonDatabase('persons.json')
export_data = {
    'persons': [
        {
            'id': p.person_id,
            'name': p.name, 
            'description': p.description,
            'created_at': p.created_at
        } for p in db.list_persons()
    ],
    'track_labels': [
        {
            'video_filename': t.video_filename,
            'track_id': t.track_id,
            'person_name': t.person_name,
            'confidence': t.confidence,
            'labeled_by': t.labeled_by,
            'created_at': t.created_at
        } for t in db.get_all_track_labels()
    ]
}

with open('person_database_export.json', 'w') as f:
    json.dump(export_data, f, indent=2)
print('Database exported to person_database_export.json')
"
```

#### Training Data Export

Generate data suitable for training custom recognition models:

```bash
# Export labeled crops for ML training
python -c "
import json
import os
from pathlib import Path
from person_database import PersonDatabase

db = PersonDatabase('persons.json')
crops_dir = Path('person_crops')
export_dir = Path('training_data')
export_dir.mkdir(exist_ok=True)

# Group crops by person
person_crops = {}
for crop_file in crops_dir.glob('*.jpg'):
    # Parse filename to get track info
    parts = crop_file.stem.split('_')
    video_name = '_'.join(parts[:-2])  # Remove track and frame parts
    track_id = parts[-2].replace('track', '')
    
    # Find person label
    label = db.get_track_label(f'{video_name}.mp4', track_id)
    if label and label.person_name:
        person_name = label.person_name
        if person_name not in person_crops:
            person_crops[person_name] = []
        person_crops[person_name].append(str(crop_file))

# Save training manifest
training_data = {
    'dataset_info': {
        'total_persons': len(person_crops),
        'total_images': sum(len(crops) for crops in person_crops.values())
    },
    'persons': {
        name: {
            'image_count': len(crops),
            'image_paths': crops
        } for name, crops in person_crops.items()
    }
}

with open(export_dir / 'training_manifest.json', 'w') as f:
    json.dump(training_data, f, indent=2)

print(f'Training data exported to {export_dir}/training_manifest.json')
for name, crops in person_crops.items():
    print(f'  {name}: {len(crops)} images')
"
```

#### Analytics Export

Generate analytics reports:

```bash
# Generate person recognition analytics
python -c "
from person_database import PersonDatabase
from collections import defaultdict
import json

db = PersonDatabase('persons.json')
labels = db.get_all_track_labels()

# Analytics
analytics = {
    'summary': {
        'total_persons': len(db.list_persons()),
        'total_labeled_tracks': len(labels),
        'auto_labeled_tracks': len([l for l in labels if l.labeled_by.startswith('auto')]),
        'manual_labeled_tracks': len([l for l in labels if l.labeled_by == 'manual'])
    },
    'per_person_stats': {},
    'per_video_stats': defaultdict(lambda: {'tracks': 0, 'persons': set()})
}

# Per-person statistics
for person in db.list_persons():
    person_labels = [l for l in labels if l.person_name == person.name]
    analytics['per_person_stats'][person.name] = {
        'track_count': len(person_labels),
        'videos': list(set(l.video_filename for l in person_labels)),
        'avg_confidence': sum(l.confidence for l in person_labels if l.confidence) / len(person_labels) if person_labels else 0
    }

# Per-video statistics  
for label in labels:
    analytics['per_video_stats'][label.video_filename]['tracks'] += 1
    analytics['per_video_stats'][label.video_filename]['persons'].add(label.person_name)

# Convert sets to lists for JSON serialization
for video_stats in analytics['per_video_stats'].values():
    video_stats['persons'] = list(video_stats['persons'])
    video_stats['person_count'] = len(video_stats['persons'])

with open('person_recognition_analytics.json', 'w') as f:
    json.dump(analytics, f, indent=2)

print('Analytics exported to person_recognition_analytics.json')
print(f\"Summary: {analytics['summary']['total_persons']} persons, {analytics['summary']['total_labeled_tracks']} labeled tracks\")
"
```

---

## Best Practices

### 1. Data Organization

**Directory Structure**
```
person_recognition/
├── persons.json                 # Master person database
├── embeddings/                  # Video-specific embeddings
│   ├── daily_analysis_embeddings.json
│   └── weekly_review_embeddings.json
├── crops/                       # Organized face crops
│   ├── daily_analysis/
│   └── weekly_review/
└── backups/                     # Regular database backups
    ├── persons_2024_11_18.json
    └── persons_2024_11_17.json
```

**Naming Conventions**
- Person names: Use full names consistently ("Jean-Yves Bouguet", not "Jean-Yves" or "JY")
- Track IDs: Maintain system-generated IDs for consistency
- Embeddings files: Include video name or analysis purpose

### 2. Manual Labeling Workflow

**Initial Setup**
1. Run person detection on representative videos
2. Generate clusters to identify frequent faces
3. Label cluster representatives first (highest impact)
4. Use batch labeling for similar faces within clusters

**Quality Control**
1. Review auto-labels periodically for accuracy
2. Maintain consistent naming across family members
3. Remove or correct low-quality face crops
4. Document labeling decisions for edge cases

**Efficiency Tips**
1. Label in cluster view for similar faces
2. Use quick-select buttons for frequent persons
3. Save work frequently during long labeling sessions
4. Export labeled data as backup before major changes

### 3. Configuration Optimization

**Performance Tuning**
```yaml
# For fast preview/testing
num_frames_per_second: 0.5
batch_size: 16
max_crops_per_track: 3

# For high-quality analysis  
num_frames_per_second: 2.0
batch_size: 8
max_crops_per_track: 15

# For production deployment
num_frames_per_second: 1.0
batch_size: 12
max_crops_per_track: 8
```

**Accuracy Tuning**
```yaml
# Conservative (fewer false positives)
person_min_confidence: 0.7
auto_label_confidence: 0.85
similarity_threshold: 0.8

# Balanced (recommended for families)
person_min_confidence: 0.6
auto_label_confidence: 0.8
similarity_threshold: 0.75

# Aggressive (more detections, more false positives)
person_min_confidence: 0.4
auto_label_confidence: 0.7
similarity_threshold: 0.7
```

### 4. Database Management

**Regular Backups**
```bash
# Daily backup script
DATE=$(date +%Y_%m_%d)
cp persons.json "backups/persons_$DATE.json"
tar -czf "backups/embeddings_$DATE.tar.gz" embeddings/
```

**Database Maintenance**
```bash
# Remove orphaned data
python -c "
from person_database import PersonDatabase
import os

db = PersonDatabase('persons.json')

# Find persons with no labeled tracks
all_persons = {p.person_id: p for p in db.list_persons()}
labeled_person_ids = {l.person_id for l in db.get_all_track_labels() if l.person_id}
orphaned_persons = set(all_persons.keys()) - labeled_person_ids

print(f'Found {len(orphaned_persons)} persons with no labeled tracks')
for person_id in orphaned_persons:
    person = all_persons[person_id]
    print(f'  - {person.name} (ID: {person_id})')
    # Uncomment to remove: db.remove_person(person_id)

# db.save_database()
"
```

### 5. Integration Patterns

**Development Workflow**
1. Test with demo data using standalone tools
2. Create configuration for small video subset  
3. Validate results and tune parameters
4. Scale to full production dataset
5. Implement automated monitoring

**Production Deployment**
1. Use version-controlled configuration files
2. Implement automated backup procedures
3. Monitor auto-labeling accuracy over time
4. Set up alerts for processing failures
5. Document manual labeling procedures for new team members

---

## Troubleshooting

### Common Issues

#### GPU/Device Issues
```bash
# Error: "MPS device not available"
# Solution: Fall back to CPU processing
embedding_device: "cpu"

# Error: CUDA out of memory
# Solution: Reduce batch size
batch_size: 4
```

#### Performance Issues
```bash
# Slow processing
# Solution: Reduce frame sampling rate
num_frames_per_second: 1.0

# Large crop files
# Solution: Reduce crop size
person_crop_size: [128, 128]

# Memory usage
# Solution: Limit crops per track
max_crops_per_track: 5
```

#### Recognition Accuracy Issues
```bash
# Too many false positives
# Solution: Increase confidence thresholds
auto_label_confidence: 0.9
person_min_confidence: 0.7

# Missing detections
# Solution: Lower confidence thresholds and increase frame rate
auto_label_confidence: 0.6
person_min_confidence: 0.4
num_frames_per_second: 3.0
```

#### Database Issues
```bash
# Corrupted database
# Solution: Restore from backup
cp backups/persons_2024_11_17.json persons.json

# Missing embeddings file
# Solution: Regenerate embeddings
python person_recognition_demo.py --video video.mp4 --output ./regenerated
```

### Debug Commands

**Check System Health**
```bash
# Test person recognition components
python test_person_recognition_integration.py

# Validate database
python -c "
from person_database import PersonDatabase
try:
    db = PersonDatabase('persons.json')
    print(f'✅ Database loaded: {len(db.list_persons())} persons')
except Exception as e:
    print(f'❌ Database error: {e}')
"

# Check GPU availability
python -c "
import torch
print(f'MPS available: {torch.backends.mps.is_available()}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'Device count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}')
"
```

**Verbose Logging**
```python
# Add to any script for detailed logging
from logging_config import set_all_loggers_level_and_format
import logging
set_all_loggers_level_and_format(level=logging.DEBUG, extended_format=True)
```

### Support Resources

- **Configuration Examples**: `/path/to/analysis_config_with_person_recognition.yaml`
- **Test Scripts**: `test_person_recognition_integration.py`, `test_labeling_gui.py`
- **Demo Tools**: `person_recognition_demo.py`, `launch_labeling_gui.py`
- **Component Tests**: Individual test files for each module

---

## Conclusion

The EufyVideos Person Recognition System provides a comprehensive solution for automated person detection, manual labeling, and intelligent identification in surveillance videos. By combining powerful AI models with intuitive manual tools and seamless pipeline integration, it enables both casual users and production deployments to effectively manage and analyze person-centric video content.

The flexible architecture supports both standalone usage for testing and development, as well as full integration with existing video analysis workflows. With proper configuration and following the best practices outlined in this guide, users can achieve high-accuracy person recognition suitable for family surveillance, security applications, and research projects.