# Classifying the Brain on Music

## Overview

This project provides a complete end-to-end machine learning workflow for classifying brain activity data using a series of well-structured Jupyter notebooks. Each notebook is self-contained and represents a crucial step in the data science pipeline.

## Notebooks

The analysis is organized into three sequential notebooks that take you from raw data exploration to final model training:

### 📊 `01-eda.ipynb` - Exploratory Data Analysis
**Purpose**: Deep dive into the brain activity dataset to understand patterns, distributions, and data quality.

**What you'll find**:
- Complete data loading and inspection with proper feature naming
- Statistical analysis of 22,000+ brain activity features
- Target variable distribution analysis (5 different brain states)
- Feature correlation analysis and visualization
- PCA visualization for dimensionality reduction insights
- Data quality assessment and problematic feature identification
- Feature importance ranking using ANOVA F-tests

**Key outputs**: Feature importance rankings, data quality reports, and visualizations saved to `data/processed/`

### 🔍 `02-model-selection.ipynb` - Model Comparison & Selection
**Purpose**: Systematic comparison of multiple machine learning algorithms to find the best performer.

**What you'll find**:
- Data preprocessing pipeline with feature scaling and selection
- Comparison of 9 different ML algorithms (Random Forest, XGBoost, LightGBM, SVM, etc.)
- Cross-validation analysis with statistical significance testing
- Hyperparameter tuning for top-performing models
- Detailed performance metrics and confusion matrices
- Model persistence and metadata tracking

**Key outputs**: Best model selection, hyperparameter configurations, and trained model artifacts

### 🎯 `03-model-training.ipynb` - Final Model Training & Evaluation
**Purpose**: Train the final model on the complete dataset and generate predictions.

**What you'll find**:
- Final model training with optimized hyperparameters
- Learning curves analysis to assess model performance
- Feature importance analysis for model interpretability
- Test set predictions with confidence scoring
- Comprehensive model performance reporting
- Model artifacts and prediction results

**Key outputs**: Final trained model, test predictions, and detailed performance reports

## Quick Start

1. **Setup Environment**:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   pip install -r requirements.txt
   ```

2. **Run the Analysis**:
   - Open Jupyter: `jupyter notebook`
   - Execute notebooks in sequence: `01-eda.ipynb` → `02-model-selection.ipynb` → `03-model-training.ipynb`
   - Each notebook is fully self-contained with proper data loading and feature naming

3. **View Results**:
   - Trained models saved in `models/` directory
   - Predictions available in `data/processed/test_predictions.csv`
   - Analysis artifacts and visualizations generated throughout

## Dataset

The project works with brain activity measurements:
- **Training data**: 160 samples × 22,036 features (brain activity measurements)
- **Training labels**: 5 different brain state classifications (0-4)
- **Test data**: 39 samples × 22,036 features for final evaluation

**Note**: The notebooks include proper data loading procedures that handle the CSV files without headers and assign meaningful feature names (`feature_0000`, `feature_0001`, etc.).

## Key Features

✨ **Complete Workflow**: From raw data exploration to final predictions  
🔧 **Proper Data Handling**: Fixes column naming issues and ensures data consistency  
📈 **Comprehensive Analysis**: Statistical tests, visualizations, and model interpretability  
🎯 **Production Ready**: Trained models with metadata and confidence scoring  
📊 **Rich Visualizations**: PCA plots, learning curves, feature importance charts  
🔍 **Model Comparison**: Systematic evaluation of 9+ ML algorithms  

## Results Summary

- **Dataset**: 22,036 brain activity features across 5 classification categories
- **Best Model**: Selected through cross-validation and hyperparameter tuning
- **Performance**: Detailed metrics available in the final notebook
- **Predictions**: Available for 39 test samples with confidence scores

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

itsakphyo@gmail.com