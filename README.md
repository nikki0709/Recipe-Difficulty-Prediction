# Recipe Difficulty Prediction Project

**Predicting Recipe Difficulty from Text Using Machine Learning**

Ruiyang Wu, Nikki Yuan | SI 670 Applied Machine Learning | University of Michigan | Fall 2025

## Overview

This project predicts recipe difficulty (easy, medium, hard) from recipe text using machine learning. The system achieves 86% accuracy using TF-IDF features and ensemble methods.

## Setup

### Install Dependencies

```bash
pip install pandas numpy scikit-learn nltk matplotlib seaborn
```

### Download NLTK Data

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')
```

## Project Structure

```
SI670final_project/
├── data/
│   ├── raw/
│   │   └── full_dataset.csv         # Input: Full recipe dataset
│   ├── processed/                    # Intermediate: Split datasets with labels
│   └── features/                     # Output: Extracted features
├── models/                           # Output: Trained models and vectorizer
├── results/                          # Output: Performance metrics and visualizations
├── split_dataset.py                  # Step 1: Split dataset
├── add_difficulty_labels.py         # Step 2: Add difficulty labels
├── preprocess_and_feature_engineering.py # Step 3: Extract features
├── train_and_evaluate_models.py     # Step 4: Train and evaluate models
└── create_per_class_chart.py        # Optional: Generate per-class chart
```

## How to Run

### Prerequisites

1. Place your recipe dataset in `data/raw/full_dataset.csv`
   - Required columns: `ingredients`, `directions`
   - Format: CSV file with recipe data

### Step 1: Split Dataset

```bash
python3 split_dataset.py
```

**What it does:**
- Samples 30,000 recipes from `data/raw/full_dataset.csv`
- Splits into train (70%), validation (15%), test (15%)
- Outputs: `data/processed/train_dataset.csv`, `val_dataset.csv`, `test_dataset.csv`

### Step 2: Add Difficulty Labels

```bash
python3 add_difficulty_labels.py
```

**What it does:**
- Creates difficulty labels (easy, medium, hard) using rule-based heuristics
- Outputs: `data/processed/*_dataset_labeled.csv` files

**Labeling Rules:**
- **Easy**: ≤5 steps, ≤8 ingredients, ≤15 avg words/step, no advanced techniques
- **Hard**: (steps > 12 AND words/step > 18 AND cooking_verbs > 3) OR ingredients > 17 OR advanced techniques
- **Medium**: Everything else

### Step 3: Preprocess and Extract Features

```bash
python3 preprocess_and_feature_engineering.py
```

**What it does:**
- Preprocesses text (cleaning, tokenization, lemmatization, stopword removal)
- Extracts TF-IDF features (1000 features from instructions)
- Extracts numeric features (text length, cooking verb count, etc.)
- Outputs: 
  - `data/features/train_features.csv`, `val_features.csv`, `test_features.csv`
  - `models/tfidf_vectorizer.pkl`

**Features:**
- 1000 TF-IDF features (unigrams + bigrams from instructions)
- 6 numeric features (text length, cooking verb count, etc.)
- Note: Labeling features (num_steps, num_ingredients, avg_words_per_step) are excluded to prevent data leakage

### Step 4: Train and Evaluate Models

```bash
python3 train_and_evaluate_models.py
```

**What it does:**
- Trains 4 models: Naive Bayes, Random Forest, Logistic Regression, Gradient Boosting
- Evaluates on test set
- Outputs:
  - `models/*_model.pkl` (trained models)
  - `results/model_comparison.png` (performance comparison chart)
  - `results/*_confusion_matrix.png` (confusion matrices)
  - `results/model_performance_summary.csv` (performance metrics)

**Expected Results:**
- Random Forest: ~86% accuracy, 0.82 F1-score (best performer)
- Gradient Boosting: ~85% accuracy, 0.82 F1-score
- Logistic Regression: ~83% accuracy, 0.77 F1-score
- Naive Bayes: ~48% accuracy, 0.44 F1-score (baseline)

### Step 5: Generate Per-Class Performance Chart (Optional)

```bash
python3 create_per_class_chart.py
```

**What it does:**
- Generates per-class performance visualization for Random Forest
- Shows Precision, Recall, and F1-Score for each difficulty class
- Outputs: `results/random_forest_per_class_performance.png`

## Expected Outputs

After running all steps, you should have:

**Models:**
- `models/tfidf_vectorizer.pkl`
- `models/random_forest_model.pkl`
- `models/gradient_boosting_model.pkl`
- `models/logistic_regression_model.pkl`
- `models/naive_bayes_model.pkl`

**Results:**
- `results/model_performance_summary.csv`
- `results/model_comparison.png`
- `results/random_forest_per_class_performance.png`
- `results/*_confusion_matrix.png`


## Notes

- All scripts use `RANDOM_SEED = 42` for reproducibility
- TF-IDF vectorizer is fitted only on training data to avoid data leakage
- Models are saved as pickle files for later use
- Performance metrics are saved to CSV for easy comparison


