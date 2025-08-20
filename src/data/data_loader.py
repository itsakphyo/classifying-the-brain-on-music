"""
Data loading and preprocessing utilities for the brain-on-music classification project.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Tuple, Dict, List, Optional


def load_raw_data(data_dir: str = '../data/raw') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load raw training and test data.
    
    Args:
        data_dir (str): Path to the raw data directory
        
    Returns:
        Tuple of (train_data, train_labels, test_data)
    """
    data_path = Path(data_dir)
    
    train_data = pd.read_csv(data_path / 'train_data.csv')
    train_labels = pd.read_csv(data_path / 'train_labels.csv')
    test_data = pd.read_csv(data_path / 'test_data.csv')
    
    return train_data, train_labels, test_data


def load_processed_data(data_dir: str = '../data/processed') -> Dict:
    """
    Load processed data and metadata.
    
    Args:
        data_dir (str): Path to the processed data directory
        
    Returns:
        Dictionary containing processed data and metadata
    """
    data_path = Path(data_dir)
    result = {}
    
    # Load feature importance if available
    feature_importance_file = data_path / 'feature_importance_anova.csv'
    if feature_importance_file.exists():
        result['feature_importance'] = pd.read_csv(feature_importance_file)
    
    # Load problematic features if available
    problematic_features_file = data_path / 'problematic_features.json'
    if problematic_features_file.exists():
        with open(problematic_features_file, 'r') as f:
            result['problematic_features'] = json.load(f)
    
    return result


def identify_problematic_features(df: pd.DataFrame, 
                                var_threshold: float = 0.01) -> Dict[str, List[str]]:
    """
    Identify constant and low variance features.
    
    Args:
        df (pd.DataFrame): Input dataframe
        var_threshold (float): Variance threshold for low variance features
        
    Returns:
        Dictionary with lists of problematic features
    """
    constant_features = []
    low_variance_features = []
    
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            if df[column].nunique() == 1:
                constant_features.append(column)
            else:
                variance = df[column].var()
                if isinstance(variance, (float, int)) and not pd.isnull(variance) and variance < var_threshold:
                    low_variance_features.append(column)
    
    return {
        'constant_features': constant_features,
        'low_variance_features': low_variance_features
    }


def get_feature_correlations(df: pd.DataFrame, 
                           threshold: float = 0.8) -> List[Tuple[str, str, float]]:
    """
    Find highly correlated feature pairs.
    
    Args:
        df (pd.DataFrame): Input dataframe
        threshold (float): Correlation threshold
        
    Returns:
        List of tuples (feature1, feature2, correlation)
    """
    corr_matrix = df.corr().abs()
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    highly_correlated = []
    for column in upper_triangle.columns:
        for row in upper_triangle.index:
            value = upper_triangle.loc[row, column]
            if isinstance(value, (float, int)) and value > threshold:
                highly_correlated.append((column, row, value))
    
    return highly_correlated


def clean_data(df: pd.DataFrame, 
               problematic_features: Optional[Dict] = None,
               fill_missing: bool = True) -> pd.DataFrame:
    """
    Clean the dataset by removing problematic features and handling missing values.
    
    Args:
        df (pd.DataFrame): Input dataframe
        problematic_features (Dict): Dictionary of problematic features to remove
        fill_missing (bool): Whether to fill missing values
        
    Returns:
        Cleaned dataframe
    """
    df_clean = df.copy()
    
    # Remove problematic features if provided
    if problematic_features:
        features_to_remove = set(
            problematic_features.get('constant_features', []) +
            problematic_features.get('low_variance_features', [])
        )
        df_clean = df_clean.drop(columns=[col for col in features_to_remove if col in df_clean.columns])
    
    # Handle missing values
    if fill_missing and df_clean.isnull().sum().sum() > 0:
        # Fill numeric columns with median
        numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
        df_clean[numeric_columns] = df_clean[numeric_columns].fillna(df_clean[numeric_columns].median())
        
        # Fill categorical columns with mode
        categorical_columns = df_clean.select_dtypes(exclude=[np.number]).columns
        for col in categorical_columns:
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode().iloc[0])
    
    return df_clean


def save_processed_data(data: Dict, output_dir: str = '../data/processed') -> None:
    """
    Save processed data and metadata.
    
    Args:
        data (Dict): Dictionary containing data to save
        output_dir (str): Output directory path
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save DataFrames as CSV
    for key, value in data.items():
        if isinstance(value, pd.DataFrame):
            value.to_csv(output_path / f'{key}.csv', index=False)
        elif isinstance(value, dict):
            with open(output_path / f'{key}.json', 'w') as f:
                json.dump(value, f, indent=2)


def get_data_summary(df: pd.DataFrame) -> Dict:
    """
    Get a comprehensive summary of the dataset.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        Dictionary with data summary statistics
    """
    summary = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.value_counts().to_dict(),
        'missing_values': df.isnull().sum().sum(),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
        'numeric_features': len(df.select_dtypes(include=[np.number]).columns),
        'categorical_features': len(df.select_dtypes(exclude=[np.number]).columns)
    }
    
    # Add statistics for numeric columns
    if summary['numeric_features'] > 0:
        numeric_stats = df.select_dtypes(include=[np.number]).describe()
        summary['numeric_stats'] = {
            'mean_of_means': numeric_stats.loc['mean'].mean(),
            'mean_of_stds': numeric_stats.loc['std'].mean(),
            'global_min': numeric_stats.loc['min'].min(),
            'global_max': numeric_stats.loc['max'].max()
        }
    
    return summary
