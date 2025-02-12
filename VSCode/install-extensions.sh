#!/bin/bash

# Array of extension IDs
extensions=(
    # Python essentials
    "ms-python.python"
    "ms-python.black-formatter"
    "ms-python.pylint"
    "ms-python.vscode-pylance"
    "charliermarsh.ruff"
    
    # AI assistants
    "github.copilot"
    "github.copilot-chat"
    "aiqubit.claude"
    "continue.continue"
    
    # Jupyter
    "ms-toolsai.jupyter"
    "ms-toolsai.jupyter-keymap"
    "ms-toolsai.jupyter-renderers"
    
    # Python utilities
    "kevinrose.vsc-python-indent"
    "njpwerner.autodocstring"
    "donjayamanne.python-environment-manager"
    
    # General utilities
    "christian-kohler.path-intellisense"
    "mechatroner.rainbow-csv"
    "redhat.vscode-yaml"
    
    # Theme
    "akamud.vscode-theme-onedark"
)

# Install each extension
for extension in "${extensions[@]}"
do
    echo "Installing $extension..."
    code --install-extension "$extension"
done

echo "Installation complete!"
