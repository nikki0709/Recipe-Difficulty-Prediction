"""
Create per-class performance chart for Random Forest model.

Shows Precision, Recall, and F1-Score for Easy, Medium, and Hard classes.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import pickle

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# Load the Random Forest model and test data
print("Loading model and test data...")
with open('models/random_forest_model.pkl', 'rb') as f:
    model = pickle.load(f)

test_df = pd.read_csv('data/features/test_features.csv')
X_test = test_df.drop('difficulty', axis=1)
y_test = test_df['difficulty']

# Get predictions
print("Generating predictions...")
y_pred = model.predict(X_test)

# Calculate per-class metrics
classes = sorted(y_test.unique())
precision_per_class = precision_score(y_test, y_pred, labels=classes, average=None, zero_division=0)
recall_per_class = recall_score(y_test, y_pred, labels=classes, average=None, zero_division=0)
f1_per_class = f1_score(y_test, y_pred, labels=classes, average=None, zero_division=0)

# Convert to percentages
precision_per_class = precision_per_class * 100
recall_per_class = recall_per_class * 100
f1_per_class = f1_per_class * 100

# Create the chart
fig, ax = plt.subplots(figsize=(10, 6))

# Set up the data
x = np.arange(len(classes))  # Easy, Medium, Hard positions
width = 0.25  # Width of bars

# Create bars
bars1 = ax.bar(x - width, precision_per_class, width, label='Precision (%)', 
               color='#2E86AB', alpha=0.9, edgecolor='white', linewidth=1.5)
bars2 = ax.bar(x, recall_per_class, width, label='Recall (%)', 
               color='#06A77D', alpha=0.9, edgecolor='white', linewidth=1.5)
bars3 = ax.bar(x + width, f1_per_class, width, label='F1-Score (%)', 
               color='#F18F01', alpha=0.9, edgecolor='white', linewidth=1.5)

# Customize the chart
ax.set_xlabel('Difficulty Class', fontsize=20, fontweight='bold', labelpad=15)
ax.set_ylabel('Score (%)', fontsize=20, fontweight='bold', labelpad=15)
ax.set_title('Per-Class Performance: Random Forest', 
             fontsize=24, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(classes, fontsize=18, fontweight='bold')
ax.set_ylim([0, 115])  # Increased to accommodate labels
ax.tick_params(axis='y', labelsize=16)
ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)

# Add value labels on bars with better positioning
def add_value_labels(bars, offset=0):
    """Add value labels on top of bars with vertical offset."""
    for bar in bars:
        height = bar.get_height()
        # Position label above the bar with offset to prevent overlap
        y_pos = height + offset + 2
        ax.text(bar.get_x() + bar.get_width()/2., y_pos,
               f'{height:.1f}%',
               ha='center', va='bottom', fontsize=12, fontweight='bold')

# Add labels with different offsets to prevent overlap
add_value_labels(bars1, offset=0)    # Precision labels at base level
add_value_labels(bars2, offset=3)    # Recall labels slightly higher
add_value_labels(bars3, offset=6)    # F1-Score labels highest

# Add legend
legend = ax.legend(loc='upper left', fontsize=16, framealpha=0.95, 
                   edgecolor='gray', frameon=True)
legend.get_frame().set_linewidth(1.5)

# Add a subtle background color
ax.set_facecolor('#fafafa')

# Tight layout
plt.tight_layout()

# Save the figure
output_path = 'results/random_forest_per_class_performance.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\nChart saved to {output_path}")

# Print the metrics for reference
print("\nPer-Class Performance Metrics:")
print("=" * 60)
print(f"{'Class':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
print("-" * 60)
for i, cls in enumerate(classes):
    print(f"{cls:<10} {precision_per_class[i]:>10.2f}% {recall_per_class[i]:>10.2f}% {f1_per_class[i]:>10.2f}%")
print("=" * 60)

# Show the plot
plt.show()

