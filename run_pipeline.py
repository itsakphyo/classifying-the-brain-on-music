#!/usr/bin/env python3
"""
Main training script for the brain-on-music classification project.
Run this script to execute the complete ML pipeline.
"""

import sys
import os
from pathlib import Path
import argparse
import logging

# Add src directory to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def setup_logging(log_level="INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def check_requirements():
    """Check if required packages are installed."""
    required_packages = [
        'pandas', 'numpy', 'scikit-learn', 'matplotlib', 
        'seaborn', 'plotly', 'jupyter'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing required packages: {missing_packages}")
        print("Please install them using:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description='Train brain-music classification model')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    parser.add_argument('--skip-eda', action='store_true',
                       help='Skip exploratory data analysis')
    parser.add_argument('--skip-model-selection', action='store_true',
                       help='Skip model selection step')
    parser.add_argument('--model-name', type=str, default=None,
                       help='Specific model to train (if skipping selection)')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    logger.info("Starting brain-music classification training pipeline")
    
    # Check requirements
    if not check_requirements():
        logger.error("Missing required packages. Please install them first.")
        return 1
    
    # Check if data files exist
    data_files = [
        'data/raw/train_data.csv',
        'data/raw/train_labels.csv', 
        'data/raw/test_data.csv'
    ]
    
    missing_files = [f for f in data_files if not Path(f).exists()]
    if missing_files:
        logger.error(f"Missing data files: {missing_files}")
        logger.error("Please ensure your data files are in the data/raw/ directory")
        return 1
    
    try:
        # Import required modules
        import pandas as pd
        import numpy as np
        
        logger.info("Loading data...")
        train_data = pd.read_csv('data/raw/train_data.csv')
        train_labels = pd.read_csv('data/raw/train_labels.csv')
        test_data = pd.read_csv('data/raw/test_data.csv')
        
        logger.info(f"Data loaded: train_data={train_data.shape}, "
                   f"train_labels={train_labels.shape}, test_data={test_data.shape}")
        
        # Create output directories
        os.makedirs('data/processed', exist_ok=True)
        os.makedirs('models', exist_ok=True)
        os.makedirs('reports/figures', exist_ok=True)
        
        logger.info("Pipeline setup complete!")
        logger.info("Next steps:")
        logger.info("1. Run notebooks/01-eda.ipynb for exploratory data analysis")
        logger.info("2. Run notebooks/02-model-selection.ipynb for model comparison")
        logger.info("3. Run notebooks/03-model-training.ipynb for final training")
        logger.info("4. Check the models/ directory for saved artifacts")
        
        # Display project structure
        logger.info("\\nProject structure created:")
        logger.info("├── data/")
        logger.info("│   ├── raw/           # Original data files")
        logger.info("│   └── processed/     # Processed data and artifacts")
        logger.info("├── notebooks/         # Jupyter notebooks for analysis")
        logger.info("├── src/              # Source code modules")
        logger.info("├── models/           # Trained models and metadata")
        logger.info("├── reports/          # Generated reports and figures")
        logger.info("├── config/           # Configuration files")
        logger.info("└── tests/            # Unit tests")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error in training pipeline: {e}")
        return 1

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
