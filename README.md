# EufyVideos

Python library for managing, analyzing, and processing home security video collections from Eufy cameras.

## Requirements

| Dependency | Version |
|------------|---------|
| Python | >=3.12, <3.14 (3.12 recommended) |
| Plotly | ==5.24.1 (Plotly 6.x breaks graph orientations) |
| Pandas | >=2.2.0, <3.0.0 |
| ffmpeg | any recent version |

## 1. System Setup

Install [Homebrew](https://brew.sh/):
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Install ffmpeg and Python 3.12:
```bash
brew install ffmpeg
brew install python@3.12
```

## 2. Project Setup

Clone the repository and enter the project directory:
```bash
git clone https://github.com/jbouguet/EufyVideos.git
cd EufyVideos
```

Create and activate a virtual environment:
```bash
/opt/homebrew/bin/python3.12 -m venv myenv
source myenv/bin/activate
```

Install dependencies:
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Verify the environment:
```bash
python -c "from version_config import verify_environment; verify_environment(raise_on_error=True)"
```

## 3. VS Code Setup

1. Install [VS Code](https://code.visualstudio.com/), then add `code` to PATH: open the Command Palette (`Cmd+Shift+P`), type `shell command`, and select **Shell Command: Install 'code' command in PATH**.

2. Install extensions:
   ```bash
   chmod +x VSCode/install-extensions.sh && ./VSCode/install-extensions.sh
   ```

3. Copy the workspace settings:
   ```bash
   cp "VSCode/settings.json" "/Users/${USER}/Library/Application Support/Code/User/settings.json"
   ```

4. Select the Python interpreter: `Cmd+Shift+P` → **Python: Select Interpreter** → choose `./myenv/bin/python`.

## 4. Git Setup

```bash
brew install gh
gh auth login
git config --global user.email "your@email.com"
git config --global user.name "Your Name"
```

## Quick Start

All commands run from the `EufyVideos/` directory with the virtual environment active (`source myenv/bin/activate`).

```bash
# Main analysis pipeline
python video_analyzer.py --config analysis_config.yaml

# Interactive web dashboard (http://localhost:8050)
python dashboard_interactive.py

# Static HTML dashboard
python dashboard.py

# Occupancy model training/inference
python occupancy.py
```

## Troubleshooting

- **`ffprobe` not found**: `brew install ffmpeg`
- **Wrong Plotly version**: `pip install plotly==5.24.1`
- **Environment check fails**: re-run the verify command above and fix each reported mismatch
