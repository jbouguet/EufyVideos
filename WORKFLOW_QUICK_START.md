# Person Recognition Quick Start Workflow

## 🚀 Quick Start Guide

### Option 1: Demo Mode (Recommended for First Time)

```bash
# 1. Run demo to generate sample data
python person_recognition_demo.py

# 2. Launch labeling GUI to explore features
python launch_labeling_gui.py

# 3. Open web browser to http://localhost:8501
# 4. Click "Load Demo Data" in sidebar
# 5. Try labeling faces and generating clusters
```

### Option 2: Production Integration

```bash
# 1. Copy example configuration
cp analysis_config_with_person_recognition.yaml my_person_analysis.yaml

# 2. Edit paths in configuration file
# 3. Run analysis with person recognition
python story_creator.py my_person_analysis.yaml

# 4. Review results and label unknown faces
python launch_labeling_gui.py

# 5. Re-run analysis with updated labels
python story_creator.py my_person_analysis.yaml
```

## 🔄 Complete Workflow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            PERSON RECOGNITION WORKFLOW                       │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Input Videos  │    │  Configuration  │    │ Person Database │
│   (.mp4 files)  │    │     (.yaml)     │    │  (persons.json) │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          ▼                      ▼                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PERSON RECOGNITION PROCESSOR                 │
│                                                                 │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────────────┐ │
│  │    YOLO     │  │   Person     │  │      Embedding          │ │
│  │  Detection  │→ │  Filtering   │→ │     Generation          │ │
│  │             │  │              │  │                         │ │
│  └─────────────┘  └──────────────┘  └─────────────────────────┘ │
│                                                 │               │
│  ┌─────────────┐  ┌──────────────┐             │               │
│  │   Face      │  │   Auto       │             │               │
│  │   Crops     │← │  Labeling    │←────────────┘               │
│  │             │  │              │                             │
│  └─────────────┘  └──────────────┘                             │
└─────────────────────────────────────────────────────────────────┘
          │                      │
          ▼                      ▼
┌─────────────────┐    ┌─────────────────┐
│ Manual Labeling │    │  Enhanced Tags  │
│  GUI (Streamlit)│    │  with Person    │
│                 │    │   Identities    │
└─────────┬───────┘    └─────────┬───────┘
          │                      │
          ▼                      ▼
┌─────────────────┐    ┌─────────────────┐
│ Updated Person  │    │   Generated     │
│   Database      │    │  Videos with    │
│               │    │  Person Labels  │
└─────────────────┘    └─────────────────┘
```

## 📋 Step-by-Step Workflows

### Workflow A: New Project Setup

```bash
# Step 1: Initialize directory structure
mkdir -p /path/to/your/videos/record/person_recognition/{embeddings,crops}

# Step 2: Run initial demo to understand the system
python person_recognition_demo.py

# Step 3: Create configuration file
cat > my_analysis.yaml << 'EOF'
directories:
  root_database: "/path/to/your/videos/record"
  stories_output: "/path/to/output"

person_recognition:
  database_file: !join [*root_database, "/person_recognition/persons.json"]
  embeddings_dir: !join [*root_database, "/person_recognition/embeddings"]
  crops_dir: !join [*root_database, "/person_recognition/crops"]

stories:
  - name: 'Initial Person Recognition Test'
    selectors:
      - filenames: ['your_test_video.mp4']  # Replace with actual filename
    
    tag_processing: true
    tag_processing_config:
      model: "Yolo11x_Optimized"
      task: "Track"
      num_frames_per_second: 1.0
      enable_person_recognition: true
      person_database_file: !join [*root_database, "/person_recognition/persons.json"]
      person_embeddings_file: !join [*embeddings_dir, "/test_embeddings.json"]
      person_crops_dir: !join [*crops_dir, "/test_analysis"]
EOF

# Step 4: Run initial analysis
python story_creator.py my_analysis.yaml

# Step 5: Label detected faces
python launch_labeling_gui.py
# In GUI: Load data from crops directory created in step 4

# Step 6: Re-run analysis with labeled data
python story_creator.py my_analysis.yaml
```

### Workflow B: Daily Family Monitoring

```bash
# Step 1: Configure daily analysis
cat > daily_family.yaml << 'EOF'
person_recognition:
  database_file: "/path/to/persons.json"
  
stories:
  - name: 'Daily Family Activity'
    selectors:
      - date_range:
          start: '2024-11-20'
          end: '2024-11-20'
        devices: ['Backyard', 'FrontDoor']
    
    tag_processing_config:
      enable_person_recognition: true
      num_frames_per_second: 2.0
      person_min_confidence: 0.6
      auto_label_confidence: 0.8
      enable_auto_labeling: true
EOF

# Step 2: Run daily analysis
python story_creator.py daily_family.yaml

# Step 3: Check for new unknown faces
python -c "
from person_database import PersonDatabase
db = PersonDatabase('/path/to/persons.json')
unlabeled = [t for t in db.get_all_track_labels() if not t.person_name]
if unlabeled:
    print(f'Found {len(unlabeled)} unlabeled faces. Run labeling GUI.')
else:
    print('All faces are labeled!')
"

# Step 4: If needed, label new faces
python launch_labeling_gui.py
```

### Workflow C: Security Event Analysis

```bash
# Step 1: Analyze specific security event
cat > security_event.yaml << 'EOF'
stories:
  - name: 'Security Event Investigation'
    selectors:
      - date_range:
          start: '2024-11-20'
          end: '2024-11-20'
        time_range:
          start: '22:30:00'
          end: '23:30:00'
    
    tag_processing_config:
      enable_person_recognition: true
      num_frames_per_second: 3.0        # Higher frame rate for security
      person_min_confidence: 0.4        # Lower threshold to catch all people
      person_min_bbox_area: 1000        # Include distant figures
      max_crops_per_track: 20           # More evidence per person
      auto_label_confidence: 0.9        # Very conservative auto-labeling
      
    tag_video_generation_config:
      show_person_identities: true
      show_confidence_scores: true      # Show confidence for security review
EOF

# Step 2: Run security analysis
python story_creator.py security_event.yaml

# Step 3: Review results in generated video
# Videos will be in output directory with person identities displayed
```

## 🛠️ Tool Reference

### Standalone Tools

| Tool | Purpose | Command |
|------|---------|---------|
| **Demo Tool** | Test person recognition on single video | `python person_recognition_demo.py` |
| **Labeling GUI** | Manual face labeling interface | `python launch_labeling_gui.py` |
| **Quick Launcher** | Easy GUI launch with demo data | `python launch_labeling_gui.py` |
| **Integration Test** | Validate system setup | `python test_person_recognition_integration.py` |

### GUI Operations

| Operation | Description | GUI Location |
|-----------|-------------|--------------|
| **Load Data** | Import face crops and database | Sidebar → "Load Demo Data" or "Load Data" |
| **Browse Faces** | View face thumbnails | Main area → Grid view |
| **Label Individual** | Add name to single face | Click face → Enter name → "Apply" |
| **Batch Label** | Add name to multiple faces | Select faces → Enter name → "Apply to Selected" |
| **Generate Clusters** | Group similar faces | Sidebar → "Generate Clusters" |
| **View Clusters** | Browse clustered faces | Sidebar → View: "clusters" |
| **Save Work** | Persist labels to database | Sidebar → "Save Database" |
| **Export Labels** | Download labeled data | Sidebar → "Export Labels" |

### Configuration Templates

| Use Case | Key Settings | Template File |
|----------|--------------|---------------|
| **Family Home** | `person_min_confidence: 0.6`, `auto_label_confidence: 0.8` | See "Family Home Surveillance" in CONFIG_REFERENCE.md |
| **Security** | `person_min_confidence: 0.7`, `num_frames_per_second: 3.0` | See "Security Monitoring" in CONFIG_REFERENCE.md |
| **Testing** | `num_frames_per_second: 0.5`, `batch_size: 16` | See "Performance Testing" in CONFIG_REFERENCE.md |

## 🔍 Troubleshooting Quick Fixes

### Common Issues

| Problem | Quick Fix |
|---------|-----------|
| **"MPS device not available"** | Change `embedding_device: "cpu"` in config |
| **GUI won't start** | Run `pip install streamlit` |
| **No faces detected** | Lower `person_min_confidence` to 0.4 |
| **Too many false faces** | Raise `person_min_confidence` to 0.7 |
| **Slow processing** | Reduce `num_frames_per_second` to 1.0 |
| **Out of memory** | Reduce `batch_size` to 4 |
| **Database corrupted** | Restore from backup: `cp backups/persons_*.json persons.json` |

### Health Check Commands

```bash
# Check system components
python test_person_recognition_integration.py

# Validate configuration
python -c "
from person_recognition_processor import PersonRecognitionConfig
config = PersonRecognitionConfig.from_dict({
    'model': 'Yolo11x_Optimized',
    'enable_person_recognition': True
})
print('✅ Configuration valid')
"

# Check database
python -c "
from person_database import PersonDatabase
db = PersonDatabase('persons.json')
print(f'Database: {len(db.list_persons())} persons, {len(db.get_all_track_labels())} labels')
"
```

## 📊 Expected Results

### After Demo Run
- **20 face crops** in `person_recognition_demo_output/person_crops/`
- **Person database** with 3 family members
- **Embeddings file** with face vectors
- **Processing time**: ~2-3 minutes on Mac with MPS

### After Production Analysis  
- **Face crops** organized by video/analysis name
- **Auto-labeled tracks** for known family members
- **Enhanced videos** with person names displayed
- **Updated database** with new detections

### Labeling GUI Session
- **Visual interface** at http://localhost:8501
- **Clustering capability** grouping similar faces
- **Batch labeling** for efficiency
- **Export functionality** for training data

## 📚 Documentation Links

- **Complete Guide**: `PERSON_RECOGNITION_GUIDE.md`
- **Configuration Reference**: `CONFIG_REFERENCE.md`
- **Example Configuration**: `analysis_config_with_person_recognition.yaml`
- **Test Scripts**: `test_person_recognition_integration.py`, `test_labeling_gui.py`