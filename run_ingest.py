#!/usr/bin/env python3
"""
Simple ingestion runner - can be executed directly if virtual environment is already active.
"""
import sys
from pathlib import Path

def main():
    """Run ingestion with proper environment setup."""
    
    # Get the script directory
    script_dir = Path(__file__).parent.absolute()
    
    # Change to the project directory
    import os
    os.chdir(script_dir)
    
    # Add project to Python path
    sys.path.insert(0, str(script_dir))
    
    print("🚀 Running NVIDIA Documentation Ingestion...")
    print(f"📂 Project directory: {script_dir}")
    
    try:
        # Import and run the ingestion
        exec(open('run_ingestion.py').read())
    except Exception as e:
        print(f"❌ Ingestion failed: {e}")
        print("\n💡 Tip: Make sure your virtual environment is activated:")
        print("   source .venv/bin/activate")
        print("   pip install -r requirements.txt")
        sys.exit(1)

if __name__ == "__main__":
    main()