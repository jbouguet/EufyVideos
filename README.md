# EufyVideos

Support management, visualization, processing and analytics of collections of home security videos cpatured by a network of Eufy security cameras.

## Setup

1. Clone the repository
   ```bash
   git clone https://github.com/jbouguet/EufyVideos.git
   ```
2. Create virtual environment: `python -m venv myenv`
3. Activate: `source myenv/bin/activate`
4. Install: `pip install -r requirements.txt`

## VS Code setup instructions

### 1. Install VS Code and Python
1. Install VS Code from https://code.visualstudio.com/
2. Install the Python extension in VS Code
   - Open VS Code
   - Click the Extensions icon in the left sidebar (or press Cmd+Shift+X)
   - Search for "Python"
   - Install the Microsoft Python extension

### 2. Clone the Repository
1. Open VS Code
2. Press Cmd+Shift+P to open the Command Palette
3. Type "Git: Clone" and select it
4. Enter repository URL: https://github.com/jbouguet/EufyVideos.git
5. Choose a local folder to clone into
6. Click "Open" when prompted to open the cloned repository

### 3. Set Up Python Environment
1. Open VS Code's integrated terminal (View > Terminal or Cmd+`)
2. Create a new virtual environment:
   ```bash
   python -m venv myenv
   ```
3. Activate the virtual environment:
   ```bash
   source myenv/bin/activate
   ```
4. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```
5. Select Python Interpreter:
   - Press Cmd+Shift+P
   - Type "Python: Select Interpreter"
   - Choose the interpreter from myenv (should end with '/myenv/bin/python')

