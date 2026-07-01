"""
Main entry point for building models.
This script now delegates to the modularized pipeline in the src/ directory.
"""
import sys
import os

# Add current directory to path so src modules can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.pipeline import run_pipeline

if __name__ == "__main__":
    run_pipeline()
