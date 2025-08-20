"""
Feature engineering and selection utilities.
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from typing import Tuple, List, Optional, Dict, Any


class FeatureEngineer:
    """Feature engineering utilities for brain activity data."""
    
    def __init__(self):
        self.feature_stats = {}
        self.selected_features = None
        
    def create_statistical_features(self, df: pd.DataFrame, 
                                  window_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Create statistical features from existing features.
        
        Args:
            df (pd.DataFrame): Input dataframe
            window_cols (List[str]): Columns to use for windowed statistics
            
        Returns:
            DataFrame with additional statistical features
        """
        if df.empty:
            return df.copy()
            
        df_features = df.copy()
        
        if window_cols is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            window_cols = list(numeric_cols[:min(10, len(numeric_cols))])
        
        # Create rolling statistics if we have sequential data
        for col in window_cols:
            if col in df.columns:
                # Basic statistics with safety checks
                df_features[f'{col}_squared'] = df[col] ** 2
                df_features[f'{col}_sqrt'] = np.sqrt(np.abs(df[col]))
                df_features[f'{col}_log'] = np.log1p(np.abs(df[col]))
        
        return df_features
    
    def create_interaction_features(self, df: pd.DataFrame, 
                                  max_features: int = 50) -> pd.DataFrame:
        """
        Create polynomial interaction features.
        
        Args:
            df (pd.DataFrame): Input dataframe
            max_features (int): Maximum number of base features to use
            
        Returns:
            DataFrame with interaction features
        """
        if df.empty:
            return df.copy()
            
        # Select top features for interactions to avoid explosion
        numeric_cols = df.select_dtypes(include=[np.number]).columns[:max_features]
        
        if len(numeric_cols) == 0:
            return df.copy()
            
        poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
        interaction_features = poly.fit_transform(df[numeric_cols])
        
        # Create feature names
        feature_names = poly.get_feature_names_out(numeric_cols)
        
        # Create DataFrame
        interaction_df = pd.DataFrame(
            interaction_features, 
            columns=feature_names,
            index=df.index
        )
        
        # Remove original features (already in df)
        original_features = set(numeric_cols)
        new_features = [col for col in interaction_df.columns if col not in original_features]
        
        return pd.concat([df, interaction_df[new_features]], axis=1)
    
    def create_aggregation_features(self, df: pd.DataFrame, 
                                  group_size: int = 10) -> pd.DataFrame:
        """
        Create aggregation features across groups of columns.
        
        Args:
            df (pd.DataFrame): Input dataframe
            group_size (int): Size of feature groups for aggregation
            
        Returns:
            DataFrame with aggregation features
        """
        df_agg = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Create groups of features
        for i in range(0, len(numeric_cols), group_size):
            group_cols = numeric_cols[i:i+group_size]
            group_name = f'group_{i//group_size}'
            
            # Calculate aggregations
            df_agg[f'{group_name}_mean'] = df[group_cols].mean(axis=1)
            df_agg[f'{group_name}_std'] = df[group_cols].std(axis=1)
            df_agg[f'{group_name}_max'] = df[group_cols].max(axis=1)
            df_agg[f'{group_name}_min'] = df[group_cols].min(axis=1)
            df_agg[f'{group_name}_median'] = df[group_cols].median(axis=1)
            df_agg[f'{group_name}_range'] = df[group_cols].max(axis=1) - df[group_cols].min(axis=1)
        
        return df_agg


class FeatureSelector:
    """Feature selection utilities."""
    
    def __init__(self):
        self.selector = None
        self.selected_features = None
        self.feature_scores = None
    
    def select_k_best(self, X: pd.DataFrame, y: pd.Series, 
                     k: int = 1000) -> Tuple[pd.DataFrame, List[str]]:
        """
        Select k best features using ANOVA F-test.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable
            k (int): Number of features to select
            
        Returns:
            Tuple of (selected_features_df, selected_feature_names)
        """
        self.selector = SelectKBest(score_func=f_classif, k=min(k, X.shape[1]))
        X_selected = self.selector.fit_transform(X, y)
        
        self.selected_features = X.columns[self.selector.get_support()].tolist()
        self.feature_scores = self.selector.scores_
        
        return pd.DataFrame(X_selected, columns=self.selected_features, index=X.index), self.selected_features
    
    def recursive_feature_elimination(self, X: pd.DataFrame, y: pd.Series,
                                    estimator, n_features: int = 100) -> Tuple[pd.DataFrame, List[str]]:
        """
        Select features using Recursive Feature Elimination.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable
            estimator: Sklearn estimator with feature_importances_ or coef_
            n_features (int): Number of features to select
            
        Returns:
            Tuple of (selected_features_df, selected_feature_names)
        """
        self.selector = RFE(estimator, n_features_to_select=n_features)
        X_selected = self.selector.fit_transform(X, y)
        
        self.selected_features = X.columns[self.selector.support_].tolist()
        
        return pd.DataFrame(X_selected, columns=self.selected_features, index=X.index), self.selected_features
    
    def get_feature_importance_ranking(self) -> pd.DataFrame:
        """
        Get feature importance ranking if available.
        
        Returns:
            DataFrame with features and their scores/rankings
        """
        if (
            self.feature_scores is not None
            and self.selected_features is not None
            and self.selector is not None
        ):
            selected_mask = self.selector.get_support()
            selected_scores = np.array(self.feature_scores)[selected_mask]
            importance_df = pd.DataFrame({
                'feature': self.selected_features,
                'score': selected_scores
            }).sort_values('score', ascending=False)
            
            return importance_df
        
        return pd.DataFrame()


class DimensionalityReducer:
    """Dimensionality reduction utilities."""
    
    def __init__(self):
        self.reducer = None
        self.explained_variance_ratio = None
    
    def apply_pca(self, X: pd.DataFrame, n_components: int = 50, 
                 variance_threshold: float = 0.95) -> Tuple[pd.DataFrame, int]:
        """
        Apply PCA for dimensionality reduction.
        
        Args:
            X (pd.DataFrame): Feature matrix
            n_components (int): Number of components (if variance_threshold not reached)
            variance_threshold (float): Minimum variance to preserve
            
        Returns:
            Tuple of (transformed_data, actual_components_used)
        """
        # Try with specified components first
        pca_temp = PCA(n_components=min(n_components, X.shape[1]))
        pca_temp.fit(X)
        
        # Check if we need fewer components to reach variance threshold
        cumsum_variance = np.cumsum(pca_temp.explained_variance_ratio_)
        optimal_components = np.argmax(cumsum_variance >= variance_threshold) + 1
        
        # Use the minimum of specified and optimal components
        final_components = min(n_components, int(optimal_components), X.shape[1])
        
        self.reducer = PCA(n_components=final_components)
        X_transformed = self.reducer.fit_transform(X)
        self.explained_variance_ratio = self.reducer.explained_variance_ratio_
        
        # Create column names
        columns = [f'PC{i+1}' for i in range(final_components)]
        
        return pd.DataFrame(X_transformed, columns=columns, index=X.index), final_components
    
    def get_component_loadings(self, feature_names: List[str], 
                             top_n: int = 10) -> Dict[str, pd.DataFrame]:
        """
        Get top feature loadings for each principal component.
        
        Args:
            feature_names (List[str]): Original feature names
            top_n (int): Number of top features per component
            
        Returns:
            Dictionary with component loadings
        """
        if self.reducer is None:
            return {}
        
        loadings = {}
        components = self.reducer.components_
        
        for i, component in enumerate(components):
            loading_df = pd.DataFrame({
                'feature': feature_names,
                'loading': np.abs(component)
            }).sort_values('loading', ascending=False).head(top_n)
            
            loadings[f'PC{i+1}'] = loading_df
        
        return loadings


def create_feature_pipeline(
    X: pd.DataFrame,
    y: pd.Series,
    engineer_features: bool = True,
    select_features: bool = True,
    n_features: int = 1000,
    apply_pca: bool = False,
    pca_components: int = 50
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Complete feature engineering and selection pipeline.
    
    Args:
        X (pd.DataFrame): Input features
        y (pd.Series): Target variable
        engineer_features (bool): Whether to engineer new features
        select_features (bool): Whether to apply feature selection
        n_features (int): Number of features to select
        apply_pca (bool): Whether to apply PCA
        pca_components (int): Number of PCA components
        
    Returns:
        Tuple of (processed_features, pipeline_info)
    """
    pipeline_info: Dict[str, Any] = {'steps': []}
    X_processed = X.copy()
    
    # Feature Engineering
    if engineer_features:
        engineer = FeatureEngineer()
        X_processed = engineer.create_statistical_features(X_processed)
        X_processed = engineer.create_aggregation_features(X_processed)
        pipeline_info['steps'].append(f'Feature Engineering: {X_processed.shape[1]} features')
    
    # Feature Selection
    if select_features:
        selector = FeatureSelector()
        X_processed, selected_features = selector.select_k_best(X_processed, y, n_features)
        pipeline_info['steps'].append(f'Feature Selection: {len(selected_features)} features')
        pipeline_info['selected_features'] = selected_features
        pipeline_info['feature_selector'] = selector
    
    # Dimensionality Reduction
    if apply_pca:
        reducer = DimensionalityReducer()
        X_processed, n_components = reducer.apply_pca(X_processed, pca_components)
        pipeline_info['steps'].append(f'PCA: {n_components} components')
        pipeline_info['pca_reducer'] = reducer
        pipeline_info['explained_variance'] = reducer.explained_variance_ratio
    
    pipeline_info['final_shape'] = X_processed.shape
    
    return X_processed, pipeline_info
