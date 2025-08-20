#!/usr/bin/env python3
"""Test script for feature engineering module."""

import pandas as pd
import numpy as np
from src.features.feature_engineering import FeatureEngineer, FeatureSelector, DimensionalityReducer, create_feature_pipeline

def main():
    # Create test data
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(100, 20), columns=[f'feature_{i}' for i in range(20)])
    y = pd.Series(np.random.randint(0, 3, 100))

    print("Testing Feature Engineering Module...")
    print("=" * 50)

    # Test FeatureEngineer
    print("1. Testing FeatureEngineer...")
    engineer = FeatureEngineer()
    X_engineered = engineer.create_statistical_features(X)
    print(f'   Original features: {X.shape[1]}, After engineering: {X_engineered.shape[1]}')

    X_interaction = engineer.create_interaction_features(X.iloc[:, :5])  # Use fewer features for interactions
    print(f'   After interactions: {X_interaction.shape[1]} features')

    X_agg = engineer.create_aggregation_features(X)
    print(f'   After aggregations: {X_agg.shape[1]} features')

    # Test FeatureSelector
    print("\n2. Testing FeatureSelector...")
    selector = FeatureSelector()
    X_selected, selected_features = selector.select_k_best(X_engineered, y, k=15)
    print(f'   After selection: {X_selected.shape[1]} features')
    
    importance_df = selector.get_feature_importance_ranking()
    print(f'   Top 3 features: {importance_df.head(3)["feature"].tolist()}')

    # Test DimensionalityReducer
    print("\n3. Testing DimensionalityReducer...")
    reducer = DimensionalityReducer()
    X_pca, n_components = reducer.apply_pca(X_selected, n_components=10)
    print(f'   After PCA: {X_pca.shape[1]} components')
    if reducer.explained_variance_ratio is not None:
        print(f'   Explained variance ratio sum: {reducer.explained_variance_ratio.sum():.3f}')
    else:
        print('   Explained variance ratio: Not available')

    # Test complete pipeline
    print("\n4. Testing Complete Pipeline...")
    X_final, pipeline_info = create_feature_pipeline(
        X, y, 
        engineer_features=True, 
        select_features=True, 
        n_features=15, 
        apply_pca=True, 
        pca_components=8
    )
    print(f'   Pipeline final shape: {X_final.shape}')
    print(f'   Pipeline steps: {len(pipeline_info["steps"])}')
    for step in pipeline_info["steps"]:
        print(f'   - {step}')

    print("\n" + "=" * 50)
    print("All tests passed successfully! ✅")

if __name__ == "__main__":
    main()
