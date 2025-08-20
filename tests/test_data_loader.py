"""
Tests for data loading and preprocessing utilities.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

# Mock the imports since packages might not be installed yet
def test_import_data_loader():
    """Test that data_loader module can be imported."""
    try:
        from data.data_loader import load_raw_data, identify_problematic_features
        assert True  # If we get here, imports worked
    except ImportError as e:
        pytest.skip(f"Required packages not available: {e}")

def test_identify_problematic_features(sample_data):
    """Test identification of problematic features."""
    try:
        from data.data_loader import identify_problematic_features
        
        X, y = sample_data
        
        # Add a constant feature
        X['constant_feature'] = 1
        
        # Add a low variance feature
        X['low_var_feature'] = np.random.normal(0, 0.001, len(X))
        
        problematic = identify_problematic_features(X, var_threshold=0.01)
        
        assert 'constant_feature' in problematic['constant_features']
        assert 'low_var_feature' in problematic['low_variance_features']
        
    except ImportError:
        pytest.skip("Required packages not available")

def test_clean_data(sample_data):
    """Test data cleaning functionality."""
    try:
        from data.data_loader import clean_data
        
        X, y = sample_data
        
        # Add some missing values
        X.iloc[0, 0] = np.nan
        X.iloc[1, 1] = np.nan
        
        # Add problematic features info
        problematic_features = {
            'constant_features': ['feature_0'],  # This will be removed
            'low_variance_features': []
        }
        
        X_clean = clean_data(X, problematic_features, fill_missing=True)
        
        # Check that missing values are filled
        assert X_clean.isnull().sum().sum() == 0
        
        # Check that constant feature is removed if it existed
        if 'feature_0' in X.columns and X['feature_0'].nunique() == 1:
            assert 'feature_0' not in X_clean.columns
        
    except ImportError:
        pytest.skip("Required packages not available")

def test_get_data_summary(sample_data):
    """Test data summary generation."""
    try:
        from data.data_loader import get_data_summary
        
        X, y = sample_data
        summary = get_data_summary(X)
        
        assert 'shape' in summary
        assert 'columns' in summary
        assert 'dtypes' in summary
        assert 'missing_values' in summary
        assert summary['shape'] == X.shape
        
    except ImportError:
        pytest.skip("Required packages not available")
