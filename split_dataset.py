"""
Script to sample 30,000 recipes from full_dataset.csv and split into train/val/test sets.

Split proportions:
- Train: 70% (21,000 recipes)
- Validation: 15% (4,500 recipes)
- Test: 15% (4,500 recipes)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# File paths
INPUT_FILE = 'data/raw/full_dataset.csv'
TRAIN_FILE = 'data/processed/train_dataset.csv'
VAL_FILE = 'data/processed/val_dataset.csv'
TEST_FILE = 'data/processed/test_dataset.csv'

# Target sample size and split proportions
TOTAL_SAMPLES = 30000
TRAIN_SIZE = 21000
VAL_SIZE = 4500
TEST_SIZE = 4500

print(f"Reading dataset from {INPUT_FILE}...")
print("This may take a while for large files...")

# Read the full dataset
# Using chunks to handle large files more efficiently
chunk_list = []
chunk_size = 100000  # Process in chunks of 100k rows

for chunk in pd.read_csv(INPUT_FILE, chunksize=chunk_size):
    chunk_list.append(chunk)
    print(f"Read {len(chunk_list) * chunk_size:,} rows so far...")

# Combine all chunks
df_full = pd.concat(chunk_list, ignore_index=True)
print(f"\nTotal recipes in dataset: {len(df_full):,}")

# Sample 30,000 recipes randomly
if len(df_full) < TOTAL_SAMPLES:
    print(f"Warning: Dataset has only {len(df_full):,} recipes, using all available recipes.")
    df_sampled = df_full.copy()
else:
    df_sampled = df_full.sample(n=TOTAL_SAMPLES, random_state=RANDOM_SEED).reset_index(drop=True)
    print(f"Sampled {len(df_sampled):,} recipes randomly.")

# Split into train/val/test
# First split: separate train (70%) from temp (30%)
df_train, df_temp = train_test_split(
    df_sampled,
    test_size=(VAL_SIZE + TEST_SIZE) / TOTAL_SAMPLES,
    random_state=RANDOM_SEED
)

# Second split: separate val (15%) and test (15%) from temp
df_val, df_test = train_test_split(
    df_temp,
    test_size=TEST_SIZE / (VAL_SIZE + TEST_SIZE),
    random_state=RANDOM_SEED
)

print(f"\nSplit complete:")
print(f"  Train: {len(df_train):,} recipes ({len(df_train)/TOTAL_SAMPLES*100:.1f}%)")
print(f"  Validation: {len(df_val):,} recipes ({len(df_val)/TOTAL_SAMPLES*100:.1f}%)")
print(f"  Test: {len(df_test):,} recipes ({len(df_test)/TOTAL_SAMPLES*100:.1f}%)")

# Save to CSV files
print(f"\nSaving splits to CSV files...")
df_train.to_csv(TRAIN_FILE, index=False)
print(f"  Saved train set to {TRAIN_FILE}")

df_val.to_csv(VAL_FILE, index=False)
print(f"  Saved validation set to {VAL_FILE}")

df_test.to_csv(TEST_FILE, index=False)
print(f"  Saved test set to {TEST_FILE}")

print("\nDone! Dataset split complete.")

