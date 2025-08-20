# Classifying the Brain on Music

A machine learning project focused on classifying brain activity patterns in response to music stimuli.

## Project Overview

This project analyzes brain activity data to classify different responses to musical stimuli. The goal is to develop machine learning models that can accurately predict brain state patterns based on neural activity features.

## Project Structure

```
classifying-the-brain-on-music/
├── data/
│   ├── raw/                 # Original, immutable data dump
│   └── processed/           # The final, canonical data sets for modeling
├── notebooks/               # Jupyter notebooks for exploration and analysis
│   ├── 01-eda.ipynb        # Exploratory Data Analysis
│   ├── 02-model-selection.ipynb  # Model selection and comparison
│   └── 03-model-training.ipynb   # Final model training
├── src/                     # Source code for use in this project
│   ├── data/               # Scripts to download or generate data
│   ├── features/           # Scripts to turn raw data into features for modeling
│   ├── models/             # Scripts to train models and make predictions
│   └── visualization/      # Scripts to create exploratory and results oriented visualizations
├── models/                  # Trained and serialized models, model predictions, model summaries
├── reports/                 # Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures/            # Generated graphics and figures to be used in reporting
├── config/                  # Configuration files
├── tests/                   # Unit tests
├── requirements.txt         # Python package dependencies
├── .gitignore              # Git ignore file
├── LICENSE                 # MIT License
└── README.md               # This file
```

## Dataset

The dataset contains brain activity measurements with the following files:
- `train_data.csv`: Training features (brain activity data)
- `train_labels.csv`: Training labels (classification targets)
- `test_data.csv`: Test features for model evaluation

## Getting Started

### Prerequisites

- Python 3.8+
- pip or conda for package management

### Installation

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd classifying-the-brain-on-music
   ```

2. Create a virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # On Windows
   # source .venv/bin/activate  # On macOS/Linux
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

1. **Exploratory Data Analysis**: Start with `notebooks/01-eda.ipynb` to understand the dataset
2. **Model Selection**: Use `notebooks/02-model-selection.ipynb` to compare different algorithms
3. **Model Training**: Train the final model using `notebooks/03-model-training.ipynb`

## Results

[To be updated with model performance metrics and key findings]

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

itsakphyo@gmail.com
