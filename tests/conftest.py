"""
Test configuration and utilities.
"""

import pytest
import sys
from pathlib import Path

# Add src directory to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    import pandas as pd
    import numpy as np
    
    # Create sample feature data
    n_samples = 100
    n_features = 20
    
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    # Create sample target data
    y = pd.Series(np.random.randint(0, 3, n_samples), name='target')
    
    return X, y

@pytest.fixture
def temp_directory(tmp_path):
    """Create temporary directory for testing."""
    return tmp_path

@pytest.fixture
def config_data():
    """Sample configuration data."""
    return {
        'data': {
            'raw_dir': 'data/raw',
            'processed_dir': 'data/processed'
        },
        'models': {
            'random_state': 42,
            'cv_folds': 3
        },
        'features': {
            'k_best_features': 10,
            'variance_threshold': 0.01
        }
    }
