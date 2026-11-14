#!/usr/bin/env python3
"""
Simple launcher for the person labeling GUI.

This script provides an easy way to launch the Streamlit-based person labeling tool
with pre-configured paths from your demo output.

Usage:
    python launch_labeling_gui.py
"""

import os
import subprocess
import sys
from pathlib import Path

def main():
    """Launch the person labeling GUI with demo data."""
    
    # Get the current directory
    current_dir = Path(__file__).parent
    
    # Check for demo output directory
    demo_dir = current_dir / "person_recognition_demo_output"
    
    if demo_dir.exists():
        print("🎯 Found demo output directory!")
        print(f"📁 Crops: {demo_dir / 'person_crops'}")
        print(f"💾 Database: {demo_dir / 'persons.json'}")
        print(f"🧠 Embeddings: {demo_dir / 'person_embeddings.json'}")
        print()
        
        # Set environment variables for the Streamlit app
        env = os.environ.copy()
        env['DEMO_CROPS_DIR'] = str(demo_dir / 'person_crops')
        env['DEMO_DATABASE_FILE'] = str(demo_dir / 'persons.json')
        env['DEMO_EMBEDDINGS_FILE'] = str(demo_dir / 'person_embeddings.json')
        
        print("🚀 Launching Person Labeling GUI...")
        print("📝 The web interface will open in your browser")
        print("🔗 URL: http://localhost:8501")
        print()
        print("💡 Usage Tips:")
        print("  1. Use the sidebar to load your data")
        print("  2. Browse crops in the main area")
        print("  3. Select crops and apply batch labels")
        print("  4. Generate clusters for similar faces")
        print("  5. Save your work with the 'Save Database' button")
        print()
        
        # Launch Streamlit
        try:
            subprocess.run([
                sys.executable, "-m", "streamlit", "run", 
                str(current_dir / "person_labeling_streamlit.py"),
                "--browser.gatherUsageStats", "false"
            ], env=env)
        except KeyboardInterrupt:
            print("\n👋 GUI closed. Your labels have been saved!")
    else:
        print("❌ Demo output directory not found!")
        print("💡 Run the person recognition demo first:")
        print("   python person_recognition_demo.py")
        print()
        print("🔄 Or launch the GUI manually:")
        print("   streamlit run person_labeling_streamlit.py")


if __name__ == "__main__":
    main()