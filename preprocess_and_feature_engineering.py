"""
Preprocess recipe data and extract features for difficulty prediction.

Features extracted:
1. TF-IDF features from recipe instructions
2. Cooking verbs extraction
3. Numeric features (text-based metrics, cooking verb counts, etc.)
"""

import pandas as pd
import numpy as np
import ast
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pickle

# Download required NLTK data (run once)
try:
    import nltk
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
except Exception as e:
    print(f"Note: Some NLTK downloads may have failed: {e}")
    pass

def parse_list_string(list_str):
    """Parse string representation of list into actual list."""
    try:
        return ast.literal_eval(list_str)
    except (ValueError, SyntaxError):
        items = re.findall(r'"([^"]*)"', list_str)
        return items if items else []

def clean_text(text):
    """Clean text: lowercase, remove punctuation, tokenize."""
    if pd.isna(text) or text == '':
        return ''
    # Convert to lowercase
    text = str(text).lower()
    # Remove special characters but keep spaces
    text = re.sub(r'[^a-z\s]', ' ', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

def extract_cooking_verbs(directions_str):
    """Extract cooking verbs from directions."""
    directions = parse_list_string(directions_str)
    directions_text = ' '.join(directions).lower()
    
    # Common cooking verbs
    cooking_verbs = [
        'bake', 'boil', 'braise', 'broil', 'brown', 'caramelize',
        'chop', 'cook', 'crisp', 'crush', 'cut', 'deglaze',
        'dice', 'drain', 'fry', 'grate', 'grill', 'heat',
        'knead', 'marinate', 'mash', 'melt', 'mix', 'peel',
        'pour', 'roast', 'saute', 'simmer', 'slice',
        'steam', 'stir', 'toss', 'whisk', 'zest', 'blend',
        'puree', 'reduce', 'season', 'tenderize', 'whip'
    ]
    
    found_verbs = []
    for verb in cooking_verbs:
        if verb in directions_text:
            found_verbs.append(verb)
    
    return ' '.join(found_verbs)

def preprocess_dataset(input_file, output_file, vectorizer=None, fit_vectorizer=True):
    """
    Preprocess dataset and extract features.
    
    Args:
        input_file: Input CSV file path
        output_file: Output CSV file path
        vectorizer: Existing TfidfVectorizer (for validation/test sets)
        fit_vectorizer: Whether to fit the vectorizer (True for train, False for val/test)
    """
    print(f"Reading {input_file}...")
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df):,} recipes")
    
    # Initialize lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # Parse directions
    print("Parsing and cleaning directions...")
    df['directions_parsed'] = df['directions'].apply(parse_list_string)
    df['directions_text'] = df['directions_parsed'].apply(lambda x: ' '.join(x) if x else '')
    
    # Clean directions text
    df['directions_cleaned'] = df['directions_text'].apply(clean_text)
    
    # Remove stopwords and lemmatize
    print("Removing stopwords and lemmatizing...")
    stop_words = set(stopwords.words('english'))
    
    def process_text(text):
        if not text:
            return ''
        tokens = word_tokenize(text)
        # Remove stopwords and lemmatize
        processed = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
        return ' '.join(processed)
    
    df['directions_processed'] = df['directions_cleaned'].apply(process_text)
    
    # Extract cooking verbs
    print("Extracting cooking verbs...")
    df['cooking_verbs'] = df['directions'].apply(extract_cooking_verbs)
    
    # Derived features
    print("Calculating derived features...")
    
    # Text-based features
    df['total_chars_directions'] = df['directions_text'].apply(len)
    df['total_words_directions'] = df['directions_text'].apply(lambda x: len(x.split()))
    df['avg_chars_per_word'] = df.apply(
        lambda row: row['total_chars_directions'] / row['total_words_directions'] 
        if row['total_words_directions'] > 0 else 0, axis=1
    )
    
    # Cooking verb counts
    df['num_cooking_verbs'] = df['cooking_verbs'].apply(lambda x: len(x.split()) if x else 0)
    
    # Ingredient list features
    df['ingredients_parsed'] = df['ingredients'].apply(parse_list_string)
    df['ingredients_text'] = df['ingredients_parsed'].apply(lambda x: ' '.join(x) if x else '')
    df['total_chars_ingredients'] = df['ingredients_text'].apply(len)
    df['total_words_ingredients'] = df['ingredients_text'].apply(lambda x: len(x.split()))
    
    # TF-IDF vectorization
    print("Extracting TF-IDF features...")
    if fit_vectorizer:
        # Fit vectorizer on training data
        vectorizer = TfidfVectorizer(
            max_features=1000,  # Top 1000 features
            ngram_range=(1, 2),  # Unigrams and bigrams
            min_df=2,  # Minimum document frequency
            max_df=0.95  # Maximum document frequency
        )
        tfidf_matrix = vectorizer.fit_transform(df['directions_processed'])
        print(f"Fitted vectorizer with {len(vectorizer.get_feature_names_out())} features")
        
        # Save vectorizer for later use
        with open('models/tfidf_vectorizer.pkl', 'wb') as f:
            pickle.dump(vectorizer, f)
        print("Saved vectorizer to models/tfidf_vectorizer.pkl")
    else:
        # Use existing vectorizer for validation/test
        if vectorizer is None:
            with open('models/tfidf_vectorizer.pkl', 'rb') as f:
                vectorizer = pickle.load(f)
        tfidf_matrix = vectorizer.transform(df['directions_processed'])
        print(f"Transformed using existing vectorizer with {len(vectorizer.get_feature_names_out())} features")
    
    # Convert TF-IDF matrix to DataFrame
    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=[f'tfidf_{i}' for i in range(tfidf_matrix.shape[1])]
    )
    
    # Combine all features
    print("Combining features...")
    feature_columns = [
        'total_chars_directions', 'total_words_directions', 'avg_chars_per_word',
        'num_cooking_verbs',
        'total_chars_ingredients', 'total_words_ingredients'
    ]
    
    # Create final feature DataFrame
    features_df = pd.concat([
        df[['difficulty'] + feature_columns],  # Target + features
        tfidf_df  # TF-IDF features
    ], axis=1)
    
    # Save processed data
    print(f"Saving processed data to {output_file}...")
    features_df.to_csv(output_file, index=False)
    print(f"Saved {features_df.shape[0]:,} samples with {features_df.shape[1]-1} features")
    
    return vectorizer, features_df

if __name__ == "__main__":
    # Process train set (fit vectorizer)
    print("="*60)
    print("Processing TRAIN set")
    print("="*60)
    vectorizer, _ = preprocess_dataset(
        'data/processed/train_dataset_labeled.csv',
        'data/features/train_features.csv',
        fit_vectorizer=True
    )
    
    # Process validation set (use existing vectorizer)
    print("\n" + "="*60)
    print("Processing VALIDATION set")
    print("="*60)
    preprocess_dataset(
        'data/processed/val_dataset_labeled.csv',
        'data/features/val_features.csv',
        vectorizer=vectorizer,
        fit_vectorizer=False
    )
    
    # Process test set (use existing vectorizer)
    print("\n" + "="*60)
    print("Processing TEST set")
    print("="*60)
    preprocess_dataset(
        'data/processed/test_dataset_labeled.csv',
        'data/features/test_features.csv',
        vectorizer=vectorizer,
        fit_vectorizer=False
    )
    
    print("\n" + "="*60)
    print("Preprocessing complete!")
    print("="*60)

