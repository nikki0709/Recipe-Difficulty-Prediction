"""
Add difficulty labels to recipe datasets using rule-based heuristics.

Heuristics based on:
- Number of steps in directions
- Number of ingredients
- Average word length per step (complexity indicator)
- Number of distinct cooking verbs (to distinguish verbose from complex)
- Presence of advanced cooking techniques
"""

import pandas as pd
import ast
import re

def parse_list_string(list_str):
    """Parse string representation of list into actual list."""
    try:
        # Try to evaluate as Python literal
        return ast.literal_eval(list_str)
    except (ValueError, SyntaxError):
        # Fallback: try to extract items manually
        # Remove brackets and split by quotes
        items = re.findall(r'"([^"]*)"', list_str)
        return items if items else []

def count_ingredients(ingredients_str):
    """Count number of ingredients."""
    ingredients = parse_list_string(ingredients_str)
    return len(ingredients) if ingredients else 0

def count_steps(directions_str):
    """Count number of steps in directions."""
    directions = parse_list_string(directions_str)
    return len(directions) if directions else 0

def avg_words_per_step(directions_str):
    """Calculate average number of words per step."""
    directions = parse_list_string(directions_str)
    if not directions or len(directions) == 0:
        return 0
    total_words = sum(len(step.split()) for step in directions)
    return total_words / len(directions)

def count_cooking_verbs(directions_str):
    """Count number of distinct cooking verbs in directions.
    
    This helps distinguish between verbose descriptions and actual complexity.
    Recipes with many different cooking techniques are more complex.
    """
    directions = parse_list_string(directions_str)
    directions_text = ' '.join(directions).lower()
    
    # Common cooking verbs (not advanced, but indicate actual cooking steps)
    cooking_verbs = [
        'bake', 'boil', 'braise', 'broil', 'brown', 'caramelize',
        'chop', 'cook', 'crisp', 'crush', 'cut', 'deglaze',
        'dice', 'drain', 'fry', 'grate', 'grill', 'heat',
        'knead', 'marinate', 'mash', 'melt', 'mix', 'peel',
        'pour', 'roast', 'saute', 'simmer', 'slice',
        'steam', 'stir', 'toss', 'whisk', 'zest', 'blend',
        'puree', 'reduce', 'season', 'tenderize', 'whip',
        'fold', 'beat', 'whisk', 'combine', 'add', 'remove',
        'flip', 'turn', 'cover', 'uncover', 'strain', 'sieve'
    ]
    
    found_verbs = set()
    for verb in cooking_verbs:
        # Check for verb as whole word (not substring)
        if re.search(r'\b' + re.escape(verb) + r'\b', directions_text):
            found_verbs.add(verb)
    
    return len(found_verbs)

def has_advanced_techniques(directions_str):
    """Check for advanced cooking techniques.
    
    Includes both English and French culinary terms that indicate advanced skill.
    Focuses on rare, truly advanced techniques (not common ones like 'whisk', 'fold', 'reduce').
    """
    directions = parse_list_string(directions_str)
    directions_text = ' '.join(directions).lower()
    
    # Advanced techniques - rare terms that indicate professional/advanced cooking
    advanced_keywords = [
        # French techniques (used in English cooking) - rare and advanced
        'sous vide', 'flambé', 'flambe', 'julienne', 'brunoise', 
        'mise en place', 'confit',
        # Advanced English techniques - rare and indicate skill
        'emulsify', 'emulsifying',  # Creating stable emulsions
        'clarify', 'clarifying',    # Clarifying butter/broth
        'deglaze', 'deglazing',     # Advanced pan technique
        'temper', 'tempering'       # Tempering chocolate/eggs (context-dependent but often advanced)
    ]
    
    return any(keyword in directions_text for keyword in advanced_keywords)

def assign_difficulty(row):
    """
    Assign difficulty level based on heuristics.
    
    Rules:
    - Easy: <= 5 steps, <= 8 ingredients, avg words/step <= 15, no advanced techniques
    - Medium: Everything that doesn't meet easy or hard criteria
    - Hard: Requires multiple complexity signals OR truly advanced techniques OR very high ingredient count
            Uses cooking verb count to avoid marking verbose-but-simple recipes as hard
    """
    num_steps = row['num_steps']
    num_ingredients = row['num_ingredients']
    avg_words = row['avg_words_per_step']
    num_cooking_verbs = row['num_cooking_verbs']
    has_advanced = row['has_advanced_techniques']
    
    # Hard criteria: More conservative to avoid over-labeling
    # Requires multiple signals to avoid false positives from verbose descriptions
    # A recipe with many steps AND high word count AND multiple techniques is truly complex
    hard_conditions = (
        (num_steps > 12 and avg_words > 18 and num_cooking_verbs > 3) or  # Multiple complexity signals
        (num_ingredients > 17) or                                          # Very high ingredient count
        has_advanced                                                        # Truly advanced techniques
    )
    
    if hard_conditions:
        return 'hard'
    
    # Easy criteria: Simple and short recipes
    elif (num_steps <= 5) and (num_ingredients <= 8) and (avg_words <= 15) and not has_advanced:
        return 'easy'
    
    # Medium (everything else)
    else:
        return 'medium'

def add_difficulty_labels(input_file, output_file):
    """Add difficulty labels to a dataset file."""
    print(f"Reading {input_file}...")
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df):,} recipes")
    
    # Calculate features
    print("Calculating features...")
    df['num_ingredients'] = df['ingredients'].apply(count_ingredients)
    df['num_steps'] = df['directions'].apply(count_steps)
    df['avg_words_per_step'] = df['directions'].apply(avg_words_per_step)
    df['num_cooking_verbs'] = df['directions'].apply(count_cooking_verbs)
    df['has_advanced_techniques'] = df['directions'].apply(has_advanced_techniques)
    
    # Assign difficulty labels
    print("Assigning difficulty labels...")
    df['difficulty'] = df.apply(assign_difficulty, axis=1)
    
    # Print distribution
    print("\nDifficulty distribution:")
    print(df['difficulty'].value_counts())
    print("\nDifficulty percentages:")
    print(df['difficulty'].value_counts(normalize=True) * 100)
    
    # Save to new file
    print(f"\nSaving to {output_file}...")
    df.to_csv(output_file, index=False)
    print("Done!")

if __name__ == "__main__":
    # Process train, validation, and test sets
    files = [
        ('data/processed/train_dataset.csv', 'data/processed/train_dataset_labeled.csv'),
        ('data/processed/val_dataset.csv', 'data/processed/val_dataset_labeled.csv'),
        ('data/processed/test_dataset.csv', 'data/processed/test_dataset_labeled.csv')
    ]
    
    for input_file, output_file in files:
        print(f"\n{'='*60}")
        print(f"Processing {input_file} -> {output_file}")
        print('='*60)
        add_difficulty_labels(input_file, output_file)

