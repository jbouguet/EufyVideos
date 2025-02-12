# EufyVideos

Python library providing tools for managing, visualizing, processing and running
analytics on collections of home security videos captured by a network of Eufy
security cameras.

## Version Requirements

This project has strict version requirements due to visualization dependencies:

- Python: >=3.12,<3.14
- Plotly: ==5.18.0 (strict requirement)
- Pandas: >=2.2.0,<3.0.0

### Important Note on Plotly Version

The visualization components are specifically designed for Plotly 5.18.0. Using Plotly 6.x or other versions may result in incorrect graph orientations and styling issues.

## General Setup Instructions

1. Clone the repository
   ```bash
   git clone https://github.com/jbouguet/EufyVideos.git
   ```
2. Create virtual environment: `python -m venv myenv`
3. Activate virtual environment: `source myenv/bin/activate`
4. Install required libraries: `pip install -r requirements.txt`
5. Verify installation:
   ```python
   from version_config import verify_environment
   verify_environment(raise_on_error=True)
   ```

## Specific Instructions for VS Code Installation and Setup

### 1. Install Python using Homebrew

1. Check if python is already installed:
   ```bash
   which -a python3
   ```
2. Install Python if not already installed using [Homebrew](https://mac.install.guide/python/brew) or other [options](https://mac.install.guide/python/install)
   ```bash
   brew install python
   ```
3. If Homebrew needs to be installed, follow the instructions in https://brew.sh/:
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```
4. Python should now be installed as /opt/homebrew/bin/python3

### 2. Install VS Code and Set Up Command Line Tool

1. Install VS Code from https://code.visualstudio.com/
2. Add VS Code to PATH by either:
   - Opening VS Code
   - Opening Command Palette (Cmd+Shift+P)
   - Typing "shell command" and selecting "Shell Command: Install 'code' command in PATH"

   Or manually by adding to your shell configuration file (`~/.zshrc` or `~/.bash_profile`):
   ```bash
   export PATH="$PATH:/Applications/Visual Studio Code.app/Contents/Resources/app/bin"
   ```
   Then reload your configuration:
   ```bash
   source ~/.zshrc  # or source ~/.bash_profile
   ```

3. Verify the installation:
   ```bash
   which code
   ```

### 3. Clone the Repository
1. Open VS Code
2. Press Cmd+Shift+P to open the Command Palette
3. Type "Git: Clone" and select it
4. Enter repository URL: https://github.com/jbouguet/EufyVideos.git
5. Choose a local folder to clone into (e.g. a folder called "python")
6. Click "Open" when prompted to open the cloned repository
7. A new subfolder "EufyVideos" should now contain the code
8. Set VS Code your default Git editor
   ```bash
   git config --global core.editor "code --wait"
   ```

### 4. Set Up Python Environment
1. Open VS Code's integrated terminal (View > Terminal)
2. Create a new virtual environment (/opt/homebrew/bin/python3 may have to be replaced by /usr/bin/python3 or /usr/local/bin/python3 depending on how python was initially installed on the system):
   ```bash
   /opt/homebrew/bin/python3 -m venv myenv
   ```
3. Activate the virtual environment:
   ```bash
   source myenv/bin/activate
   ```
4. Upgrade pip:
   ```bash
   python -m pip install --upgrade pip
   ```
5. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```
6. Select Python Interpreter:
   - Press Cmd+Shift+P
   - Type "Python: Select Interpreter"
   - Choose the interpreter from myenv (should be './myenv/bin/python')

### 5. Install VS Code Extensions and Settings

1. Make the extension installation script executable:
   ```bash
   chmod +x VSCode/install-extensions.sh
   ```

2. Run the installation script:
   ```bash
   ./VSCode/install-extensions.sh
   ```
   This will install essential extensions including:
   - Python language support and formatting
   - AI assistants (Claude, Copilot, Continue)
   - Jupyter support
   - Useful utilities and themes

3. Set up VS Code settings by copying the provided configuration:
   ```bash
   cp "VSCode/settings.json" "/Users/${USER}/Library/Application Support/Code/User/settings.json"
   ```
   These settings enable:
   - Format on save with Black
   - Python linting
   - Automatic import organization
   - Other Python-specific optimizations

4. Restart VS Code to apply all settings

### 6. Setup the Git workflow

1. Install and configure the git-credential-manager:
   ```bash
   brew install --cask git-credential-manager
   ```
2. Most simple git workflow:
   ```bash
   git add .
   git commit -m "message"
   git push origin main
   ```
3. [cheat-sheet](https://education.github.com/git-cheat-sheet-education.pdf) for the most commonly used git commands.

## Version Compatibility and Common Issues

1. Plotly 6.x Compatibility:
   - The visualization code is not compatible with Plotly 6.x
   - If you see horizontal instead of vertical bars, check your Plotly version
   - Solution: `pip install plotly==5.18.0`

2. Python Version:
   - Python 3.13+ may have type handling differences
   - Recommended: Use Python 3.12.x
   
3. Pandas Version:
   - Pandas 2.2.x or higher is required for proper datetime handling
   - Earlier versions may have date processing issues

4. VS Code Extension Issues:
   - If extensions fail to install, try installing them manually through VS Code's Extensions panel
   - For auto-formatting issues, ensure Black is installed in your Python environment: `pip install black`
   - Check that the Python extension can find your interpreter by running "Python: Select Interpreter" from the Command Palette