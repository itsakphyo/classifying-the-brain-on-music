# Classifying the Brain on Music

**COSC 74/274 Winter 2024 Final Project (Competition 01)**

*Note: This is a learning project. I am not affiliated with the original university course.*

## Overview

This project provides a complete end-to-end machine learning workflow for classifying fMRI brain images taken while listening to music in five different genres. The objective is to classify brain activity data using machine learning techniques to predict musical genres based on neural responses. This project is structured as a series of well-structured Jupyter notebooks, with each notebook being self-contained and representing a crucial step in the data science pipeline.

## Notebooks

The analysis is organized into three sequential notebooks that take you from raw data exploration to final model training:

### `01-eda.ipynb` - Exploratory Data Analysis
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

### `02-model-selection.ipynb` - Model Comparison & Selection
**Purpose**: Systematic comparison of multiple machine learning algorithms to find the best performer.

**What you'll find**:
- Data preprocessing pipeline with feature scaling and selection
- Comparison of 9 different ML algorithms (Random Forest, XGBoost, LightGBM, SVM, etc.)
- Cross-validation analysis with statistical significance testing
- Hyperparameter tuning for top-performing models
- Detailed performance metrics and confusion matrices
- Model persistence and metadata tracking

**Key outputs**: Best model selection, hyperparameter configurations, and trained model artifacts

### `03-model-training.ipynb` - Final Model Training & Evaluation
**Purpose**: Train the final model on the complete dataset and generate predictions.

**What you'll find**:
- Final model training with optimized hyperparameters
- Learning curves analysis to assess model performance
- Feature importance analysis for model interpretability
- Test set predictions with confidence scoring
- Comprehensive model performance reporting
- Model artifacts and prediction results

**Key outputs**: Final trained model, test predictions, and detailed performance reports


## Dataset

This project works with fMRI brain activity measurements from the "Classifying The Brain on Music" competition. The dataset consists of brain images taken while listening to music in five different genres:

- **Label 0**: Ambient Music
- **Label 1**: Country Music  
- **Label 2**: Heavy Metal
- **Label 3**: Rock 'n Roll
- **Label 4**: Classical Symphonic

### Data Structure

- **Training data (`train_data.csv`)**: 160 event-related brain images (trials) corresponding to twenty 6-second music clips, with four clips in each of the five genres, repeated in-order eight times (runs)
- **Training labels (`train_labels.csv`)**: Correct musical genre labels (0-4) for each of the 160 trials
- **Test data (`test_data.csv`)**: 40 event-related brain images corresponding to novel 6-second music clips in randomized order

### Features

Each brain image contains **22,036 features** corresponding to blood-oxygenation levels at each 2mm-cubed 3D location within a section of the auditory cortex. This represents a challenging multiway classification problem where there are many more features (brain sites) than samples (trials), which is typical in human brain imaging studies.

The data comes from a one-person subset of a larger 20-subject study, making this a particularly interesting case study in neuroimaging-based music genre classification.

**Note**: The notebooks include proper data loading procedures that handle the CSV files without headers and assign meaningful feature names (`feature_0000`, `feature_0001`, etc.).


## Results Summary

- **Dataset**: 22,036 brain activity features across 5 music genre classification categories
- **Best Model**: Selected through cross-validation and hyperparameter tuning
- **Performance**: Detailed metrics available in the final notebook
- **Predictions**: Available for 40 test samples with confidence scores

## Citation

Koulogeorge, A., Kunwar, A., Agarwal, A., Chip N., Wang, C., dartmouth, Nirogi, G.R., Veeramachaneni, G., Wang, J., Kumaran, K., MACasey, Ming, SouYoung, Jain, S., TaiyuanZhang0805, barrios, w. and Diao, X. (2024) *Classifying The Brain on Music*. Kaggle. Available at: https://www.kaggle.com/competitions/classifying-the-brain-on-music/overview (Accessed: 26 August 2025).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

itsakphyo@gmail.com