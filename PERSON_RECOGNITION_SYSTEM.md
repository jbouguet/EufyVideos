# Person Recognition System - Complete Documentation

This document provides a comprehensive overview of the person recognition system, including all components, files, and workflows.

## Table of Contents
1. [System Overview](#system-overview)
2. [Core Components](#core-components)
3. [File Documentation](#file-documentation)
4. [Complete Workflow](#complete-workflow)
5. [Configuration](#configuration)
6. [Troubleshooting](#troubleshooting)

---

## System Overview

The person recognition system implements a complete pipeline from person detection to auto-labeling:

```
Video → Detection → Crops + Embeddings → Conservative Clustering → 
Manual Labeling → Mega-Clusters → Refined Model → Auto-Labeling
```

### Key Features
- **Conservative Clustering**: Prefers over-segmentation to avoid mixing different people
- **Visual Inspection**: HTML interface for manual cluster review
- **Interactive Labeling**: Command-line tool for cluster labeling and grouping
- **Refined Learning**: Siamese neural network learns from manual labels
- **Auto-Labeling**: Real-time person identification for new detections

---

## Core Components

### 1. Detection & Embedding Generation
- **Person Detection**: YOLO-based person detection in video frames
- **Crop Extraction**: Extract person regions at 224x224 resolution
- **Embedding Generation**: CLIP + ReID features combined into 512-dim vectors
- **Quality Assessment**: Automatic quality scoring for embeddings

### 2. Conservative Clustering
- **Algorithm**: Hierarchical clustering with strict similarity thresholds
- **Parameters**: Optimized for high precision (similarity_threshold=0.88)
- **Output**: ~170 high-quality clusters from ~1,400 embeddings
- **Quality Control**: Filters low-quality embeddings and small clusters

### 3. Visual Inspection & Labeling
- **Grid Generation**: Visual grids showing all crops per cluster
- **HTML Interface**: Easy browsing and review of clusters
- **Interactive Labeling**: Command-line tool for manual labeling
- **Mega-Clustering**: Group related clusters into person identities

### 4. Refined Model & Auto-Labeling
- **Siamese Network**: Learns person similarity from labeled data
- **Person Profiles**: Stores learned characteristics and thresholds
- **Auto-Labeling**: Real-time identification of new person detections
- **Confidence Scoring**: Provides confidence estimates for predictions

---

## File Documentation

### Core System Files

#### `person_detector.py`
**Purpose**: Person detection and crop extraction from video frames
- **Classes**: `PersonDetector`, `PersonCrop`, `PersonTrack`
- **Key Functions**: 
  - `detect_persons_in_frame()`: YOLO-based person detection
  - `extract_person_crops()`: Extract and resize person regions
  - `track_persons()`: Associate detections across frames
- **Integration**: Used by `tag_processor.py` in main video pipeline
- **Configuration**: Crop size, confidence thresholds, bounding box filters

#### `person_embedding.py`
**Purpose**: Generate embeddings from person crops using CLIP + ReID
- **Classes**: `PersonEmbeddingGenerator`, `PersonEmbedding`
- **Key Functions**:
  - `generate_embedding()`: Create 512-dim vectors from crops
  - `compute_similarity()`: Calculate cosine similarity between embeddings
  - `save_embeddings()` / `load_embeddings()`: Persistence
- **Models**: CLIP ViT-B/32 + MobileNetV3 ReID
- **Output**: Normalized 512-dimensional embedding vectors

#### `person_clustering.py`
**Purpose**: Conservative clustering of person embeddings
- **Classes**: `PersonClusterer`, `PersonCluster`
- **Algorithm**: Hierarchical clustering with quality filtering
- **Parameters**: 
  - `similarity_threshold=0.88` (conservative)
  - `quality_threshold=0.5` (balanced)
  - `min_cluster_size=2`
- **Output**: ~170 high-quality clusters with detailed statistics
- **Reports**: Comprehensive analysis with quality metrics

#### `conservative_person_clustering.py`
**Purpose**: Ultra-conservative clustering for near-duplicates only
- **Classes**: `ConservativePersonClusterer`, `ConservativePersonCluster`
- **Strategy**: Track-based and temporal proximity clustering
- **Features**: Groups only near-duplicate detections from same tracks or close temporal windows
- **Use case**: When visual inspection shows too many mixed people in standard clustering

### Visual Inspection & Labeling

#### `cluster_visual_inspector.py`
**Purpose**: Create visual grids for manual cluster review
- **Classes**: `ClusterVisualInspector`
- **Output**:
  - Individual cluster grid images (PNG)
  - HTML inspection interface with navigation
  - Summary reports with cluster statistics
- **Features**: Automatic crop finding, quality indicators, responsive layout

#### `cluster_labeling_tool.py`
**Purpose**: Interactive tool for manual cluster labeling and grouping
- **Classes**: `ClusterLabelingManager`, `ClusterLabel`, `MegaCluster`
- **Commands**:
  - `label <id> <name>`: Assign labels to clusters
  - `create <name> <ids>`: Create mega-clusters
  - `show clusters/megas`: Display information
- **Output**: Labeled clusters, mega-clusters, training data
- **Features**: Progress saving, comprehensive reporting

### Refined Model & Auto-Labeling

#### `refined_person_model.py`
**Purpose**: Train and use Siamese network for person similarity learning
- **Classes**: `RefinedPersonRecognitionModel`, `SiameseNetwork`, `PersonProfile`
- **Training**: 
  - Siamese neural network architecture
  - Positive/negative pair generation from mega-clusters
  - Person-specific threshold learning
- **Prediction**: 
  - Real-time person identification
  - Confidence-based prediction with uncertainty
- **Integration**: Ready for deployment in video processing pipeline

### Utility/Testing Tools

#### `simple_resolution_test.py`
**Purpose**: Test impact of crop resolution on embedding quality and clustering
- **Function**: Compare clustering results at 224x224 vs 384x384 resolution
- **Output**: Side-by-side clustering analysis showing minimal improvement at higher resolution
- **Usage**: `python simple_resolution_test.py` (testing confirms 224x224 is optimal)

#### `crop_resolution_comparison.py`
**Purpose**: Comprehensive resolution testing framework
- **Function**: Detailed analysis of resolution impact on embedding generation
- **Features**: Multiple resolution testing, quality metrics, performance analysis

#### `test_higher_resolution.py`
**Purpose**: Specific testing for higher resolution crop processing
- **Function**: Validates higher resolution crop handling in embedding pipeline

### System Status

✅ **CLEANUP COMPLETED**: All obsolete and redundant files have been removed from the codebase for a clean, production-ready system.

⚠️ **INTEGRATION NOTE**: Person recognition functionality is currently being integrated into the unified `TagProcessor` system. The configuration supports person recognition parameters in `TaggerConfig`, but the actual person recognition processing is handled separately using the individual components listed above.

**Current Integration Status**:
- ✅ Person recognition components are functional independently
- ✅ Configuration is integrated into `TaggerConfig` 
- ⏳ Person recognition processing integration into `TagProcessor` is planned for future updates
- ✅ Manual workflow (clustering → labeling → training → auto-labeling) is fully operational

---

## Complete Workflow

### Phase 1: Setup and Detection
```bash
# Ensure person recognition is enabled in analysis_config.yaml
enable_person_recognition: true
person_crop_size: [224, 224]
person_min_confidence: 0.6

# Run person detection and embedding generation
python story_creator.py
```

### Phase 2: Conservative Clustering
```bash
# Generate ~170 conservative clusters
python person_clustering.py

# Expected output:
# - 170 clusters with quality >0.92
# - Results in /record/person_recognition/clustering/
```

### Phase 3: Visual Inspection
```bash
# Create visual grids for manual review
python cluster_visual_inspector.py

# Open HTML interface
open /record/person_recognition/visual_inspection/cluster_inspection_index.html
```

### Phase 4: Manual Labeling
```bash
# Interactive labeling session
python cluster_labeling_tool.py
# Choose option 1: Interactive labeling

# Example commands:
# > label 15 Person_A
# > label 18 Person_A  
# > create Person_A 15,18
# > save
```

### Phase 5: Model Training
```bash
# Generate training data (after labeling)
python cluster_labeling_tool.py
# Choose option 2: Generate training data

# Train refined model
python refined_person_model.py --mode train --epochs 50
```

### Phase 6: Auto-Labeling
```bash
# Load trained model for prediction
python refined_person_model.py --mode predict

# Integration example:
# predicted_person, confidence, _ = model.predict_person(embedding)
```

---

## Configuration

### Main Configuration (`analysis_config.yaml`)
```yaml
tag_processing_config:
  # Person recognition settings
  enable_person_recognition: true
  person_database_file: "/record/person_recognition/persons.json"
  person_embeddings_file: "/record/person_recognition/embeddings/person_embeddings.json"
  person_crops_dir: "/record/person_recognition/crops"
  
  # Detection parameters
  person_crop_size: [224, 224]      # Optimal crop resolution
  person_min_confidence: 0.6        # Detection confidence threshold
  person_min_bbox_area: 2000        # Minimum bounding box area
  max_crops_per_track: 10           # Max crops per person track
  
  # Embedding parameters
  embedding_device: "mps"           # GPU acceleration (Mac)
  embedding_dim: 512                # Output embedding dimension
  clip_weight: 0.7                  # CLIP model weight
  reid_weight: 0.3                  # ReID model weight
  
  # Identification parameters
  similarity_threshold: 0.75        # Initial similarity threshold
  auto_label_confidence: 0.8        # Auto-labeling confidence
  enable_auto_labeling: true        # Enable auto-labeling
```

### Clustering Parameters (`person_clustering.py`)
```python
# Conservative clustering settings
similarity_threshold = 0.88         # Conservative but allows granular clusters
quality_threshold = 0.5             # Balanced quality filter
use_dbscan = False                  # Use hierarchical for better control
min_cluster_size = 2                # Minimum embeddings per cluster
```

### File Locations
```
/record/person_recognition/
├── embeddings/                     # Generated embeddings (JSON files)
├── crops/                          # Extracted person crops (JPG files)
├── clustering/                     # Standard clustering results
│   ├── clusters.json               # Cluster definitions
│   ├── cluster_report.txt          # Detailed analysis
│   └── similarity_analysis.png     # Quality charts
├── conservative_clustering/        # Ultra-conservative clustering results
│   ├── conservative_clusters.json  # Conservative cluster definitions
│   └── conservative_cluster_report.txt # Conservative analysis
├── visual_inspection/              # Visual grids and HTML interface
│   ├── cluster_XXX_grid.png        # Individual cluster grids
│   ├── cluster_inspection_index.html # Main inspection interface
│   └── cluster_summary_report.txt  # Summary statistics
├── cluster_labeling/              # Manual labels and mega-clusters
│   ├── cluster_labels.json         # Individual cluster labels
│   ├── mega_clusters.json          # Mega-cluster definitions
│   ├── training_data.json          # Training data for refined model
│   └── labeling_report.txt         # Labeling progress report
└── refined_model/                 # Trained model files
    ├── siamese_model.pth           # Trained neural network
    ├── person_profiles.json        # Person-specific profiles
    └── training_log.json           # Training metrics and history
```

---

## Troubleshooting

### Common Issues

**Too Few Clusters (< 100)**
- Check `similarity_threshold` (should be 0.88, not higher)
- Ensure `use_dbscan=False` for hierarchical clustering
- Verify parameter consistency in main() function

**Mixed People in Clusters**
- Lower `similarity_threshold` (more conservative)
- Increase `quality_threshold` to filter poor embeddings
- Check crop quality and detection confidence

**Visual Inspection Errors**
- Verify crop file naming convention matches embedding metadata
- Check crops directory structure (video_name/crop_files)
- Ensure cluster JSON file exists and is valid

**Labeling Tool Issues**
- Check file permissions in output directories
- Ensure cluster and embedding files are accessible
- Verify JSON file format consistency

**Training/Prediction Errors**
- Ensure sufficient labeled data (>10 mega-clusters recommended)
- Check PyTorch installation and device compatibility
- Verify model file paths and permissions

### Performance Optimization

**Speed Improvements**
- Use MPS/CUDA acceleration for embedding generation
- Process embeddings in batches
- Cache computed similarity matrices
- Use hierarchical clustering (faster than DBSCAN)

**Quality Improvements**
- Increase person detection confidence threshold
- Filter crops by minimum size and quality
- Use conservative clustering parameters
- Manual review of high-quality clusters first

### System Cleanup Summary

✅ **COMPLETED CLEANUP**:
- **Removed 13 obsolete Python files**: test files, demos, alternative implementations, incomplete GUIs
- **Removed 6 redundant documentation files**: consolidated into single comprehensive guide  
- **Previously removed data files**: all *_embeddings.json files moved from code directory to proper data locations
- **Result**: Clean, production-ready system with organized code and consolidated documentation

---

## Summary

The person recognition system provides a complete, production-ready solution:

1. **Detection**: Robust YOLO-based person detection with quality filtering
2. **Embedding**: CLIP + ReID features for rich person representations  
3. **Clustering**: Conservative clustering optimized for high precision
4. **Inspection**: Visual tools for manual cluster review and validation
5. **Labeling**: Interactive tools for cluster labeling and mega-cluster creation
6. **Learning**: Siamese network training from manually labeled data
7. **Auto-Labeling**: Real-time person identification for new detections

**Current System Files** (9 files total):

**Core Pipeline**:
- `person_detector.py` - Person detection and crop extraction
- `person_embedding.py` - CLIP + ReID embedding generation
- `person_clustering.py` - Conservative clustering algorithm
- `cluster_visual_inspector.py` - Visual inspection interface with HTML grids
- `cluster_labeling_tool.py` - Interactive labeling and mega-cluster management
- `refined_person_model.py` - Siamese network training and auto-labeling

**Utility Tools**:
- `conservative_person_clustering.py` - Ultra-conservative track/temporal clustering
- `simple_resolution_test.py` - Resolution impact testing (confirms 224x224 optimal)
- `crop_resolution_comparison.py` - Comprehensive resolution analysis framework
- `test_higher_resolution.py` - Higher resolution validation testing

**Production-Ready Status**: The system is now fully cleaned up and ready for production use with the current ~170 conservative clusters providing excellent precision for manual labeling and subsequent model training.