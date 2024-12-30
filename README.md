# EufyVideos

Python library providing tools for managing, visualizing, processing and running
analytics on collections of home security videos captured by a network of Eufy
security cameras.

## General Setup Instructions

1. Clone the repository
   ```bash
   git clone https://github.com/jbouguet/EufyVideos.git
   ```
2. Create virtual environment: `python -m venv myenv`
3. Activate virtual environment: `source myenv/bin/activate`
4. Install required libraries: `pip install -r requirements.txt`

## Specific Instructions for VS Code Installation and Setup

### 1. Install Python using Homebew

1. Check if python is already insalled:
   ```bash
   which -a python3
   ```
2. Install Python if not already installed using [Homebrew](https://mac.install.guide/python/brew) or other [options](https://mac.install.guide/python/install)
   ```bash
   brew install python
   ```
3. If Homebrew needs to be installed, follow the instructions in https://brew.sh/.
4. Python should now be installed as /opt/homebrew/bin/python3

### 2. Install VS Code and Python Extensions
1. Install VS Code from https://code.visualstudio.com/
2. Install the Python extension in VS Code:
   - Open VS Code
   - Click the Extensions icon in the left sidebar (or press Cmd+Shift+X)
   - Search for "Python"
   - Install the Microsoft Python extension

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
1. Open VS Code's integrated terminal (View > Terminal)`
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

### 5. Import VS Code settings [optional]

1. Install the suggested VS Code extensions:
   ```bash
   cat VSCode/vscode-extensions.txt | xargs -n 1 code --install-extension
   ```
2. Import general settings:
   ```bash
   cp "VSCode/settings.json" "/User/{user_name}/Library/Application Support/Code/User/settings.json"
   ```
3. Restart VS Code to apply all settings.

### 6. Git workflow

1. Most simple git workflow:
   ```bash
   git add .
   git commit -m "message"
   git push origin main
   ```
2. [cheat-sheet](https://education.github.com/git-cheat-sheet-education.pdf) for the most commonly used git commands.