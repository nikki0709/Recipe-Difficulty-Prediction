"""
Train and evaluate multiple models for recipe difficulty prediction.

Models implemented:
1. Naive Bayes
2. Random Forest
3. Logistic Regression
4. Gradient Boosting (XGBoost)

Evaluation metrics:
- Accuracy
- F1-score (macro and per-class)
- Confusion matrix
"""

import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def load_data():
    """Load train, validation, and test feature sets."""
    print("Loading datasets...")
    train_df = pd.read_csv('data/features/train_features.csv')
    val_df = pd.read_csv('data/features/val_features.csv')
    test_df = pd.read_csv('data/features/test_features.csv')
    
    # Separate features and target
    X_train = train_df.drop('difficulty', axis=1)
    y_train = train_df['difficulty']
    
    X_val = val_df.drop('difficulty', axis=1)
    y_val = val_df['difficulty']
    
    X_test = test_df.drop('difficulty', axis=1)
    y_test = test_df['difficulty']
    
    print(f"Train: {X_train.shape[0]:,} samples, {X_train.shape[1]} features")
    print(f"Validation: {X_val.shape[0]:,} samples")
    print(f"Test: {X_test.shape[0]:,} samples")
    print(f"Class distribution (train):\n{y_train.value_counts()}\n")
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def train_model(model, model_name, X_train, y_train, X_val, y_val):
    """Train a model and evaluate on validation set."""
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print('='*60)
    
    # Train model
    print("Training...")
    model.fit(X_train, y_train)
    
    # Predict on validation set
    y_val_pred = model.predict(X_val)
    
    # Calculate metrics
    accuracy = accuracy_score(y_val, y_val_pred)
    f1_macro = f1_score(y_val, y_val_pred, average='macro')
    f1_per_class = f1_score(y_val, y_val_pred, average=None)
    
    print(f"\nValidation Results:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  F1-score (macro): {f1_macro:.4f}")
    print(f"  F1-score per class:")
    classes = sorted(y_val.unique())
    for cls, f1 in zip(classes, f1_per_class):
        print(f"    {cls}: {f1:.4f}")
    
    return model, accuracy, f1_macro, y_val_pred

def evaluate_on_test(model, model_name, X_test, y_test):
    """Evaluate model on test set."""
    print(f"\n{'='*60}")
    print(f"Test Set Evaluation: {model_name}")
    print('='*60)
    
    # Predict on test set
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_test_pred)
    f1_macro = f1_score(y_test, y_test_pred, average='macro')
    f1_per_class = f1_score(y_test, y_test_pred, average=None)
    
    print(f"\nTest Results:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  F1-score (macro): {f1_macro:.4f}")
    print(f"  F1-score per class:")
    classes = sorted(y_test.unique())
    for cls, f1 in zip(classes, f1_per_class):
        print(f"    {cls}: {f1:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred, labels=classes)
    print(f"\nConfusion Matrix:")
    print(pd.DataFrame(cm, index=classes, columns=classes))
    
    # Classification report
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_test_pred, target_names=classes))
    
    return accuracy, f1_macro, cm, y_test_pred

def plot_confusion_matrix(cm, classes, model_name, save_path):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title(f'Confusion Matrix: {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix to {save_path}")

def plot_model_comparison(results):
    """Plot comparison of model performance."""
    models = list(results.keys())
    accuracies = [results[m]['accuracy'] for m in models]
    f1_scores = [results[m]['f1_macro'] for m in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, accuracies, width, label='Accuracy', alpha=0.8)
    bars2 = ax.bar(x + width/2, f1_scores, width, label='F1-score (macro)', alpha=0.8)
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim([0, 1])
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('results/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved model comparison plot to results/model_comparison.png")

def main():
    """Main training and evaluation pipeline."""
    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    
    # Initialize models
    models = {
        'Naive Bayes': MultinomialNB(alpha=1.0),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=RANDOM_SEED, n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_SEED)
    }
    
    # Train and evaluate on validation set
    trained_models = {}
    val_results = {}
    
    for model_name, model in models.items():
        trained_model, accuracy, f1_macro, _ = train_model(
            model, model_name, X_train, y_train, X_val, y_val
        )
        trained_models[model_name] = trained_model
        val_results[model_name] = {'accuracy': accuracy, 'f1_macro': f1_macro}
        
        # Save model
        model_file = f'models/{model_name.lower().replace(" ", "_")}_model.pkl'
        with open(model_file, 'wb') as f:
            pickle.dump(trained_model, f)
        print(f"Saved model to {model_file}")
    
    # Evaluate on test set
    print(f"\n{'='*60}")
    print("FINAL TEST SET EVALUATION")
    print('='*60)
    
    test_results = {}
    classes = sorted(y_test.unique())
    
    for model_name, model in trained_models.items():
        accuracy, f1_macro, cm, _ = evaluate_on_test(
            model, model_name, X_test, y_test
        )
        test_results[model_name] = {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'confusion_matrix': cm
        }
        
        # Plot confusion matrix
        cm_file = f'results/{model_name.lower().replace(" ", "_")}_confusion_matrix.png'
        plot_confusion_matrix(cm, classes, model_name, cm_file)
    
    # Plot model comparison
    plot_model_comparison(test_results)
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)
    print("\nTest Set Performance:")
    summary_df = pd.DataFrame({
        'Model': list(test_results.keys()),
        'Accuracy': [test_results[m]['accuracy'] for m in test_results.keys()],
        'F1-score (macro)': [test_results[m]['f1_macro'] for m in test_results.keys()]
    })
    summary_df = summary_df.sort_values('F1-score (macro)', ascending=False)
    print(summary_df.to_string(index=False))
    
    # Save summary
    summary_df.to_csv('results/model_performance_summary.csv', index=False)
    print("\nSaved performance summary to results/model_performance_summary.csv")
    
    print("\nDone!")

if __name__ == "__main__":
    main()

