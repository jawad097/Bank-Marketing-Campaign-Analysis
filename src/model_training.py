"""
Model Training Module for Bank Marketing Campaign Analysis

This module handles:
1. Loading processed data
2. Training multiple models
3. Model evaluation
4. Hyperparameter tuning
5. Saving the best model
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
import xgboost as xgb
import joblib
import time

# Set random seed for reproducibility
np.random.seed(42)

# Constants
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, 'processed_data.csv')

def load_data():
    """Load the processed data"""
    if not os.path.exists(PROCESSED_DATA_PATH):
        raise FileNotFoundError(f"Processed data not found at {PROCESSED_DATA_PATH}. Run data_preparation.py first.")
    
    df = pd.read_csv(PROCESSED_DATA_PATH)
    print(f"Processed data loaded with shape: {df.shape}")
    return df

def split_data(df):
    """Split the data into training and testing sets"""
    # Separate features and target
    X = df.drop('y', axis=1)
    y = df['y']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test

def train_logistic_regression(X_train, y_train):
    """Train a Logistic Regression model"""
    print("\n--- Training Logistic Regression ---")
    
    # Create and train the model
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    
    return model

def train_random_forest(X_train, y_train):
    """Train a Random Forest model"""
    print("\n--- Training Random Forest ---")
    
    # Create and train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    return model

def train_xgboost(X_train, y_train):
    """Train an XGBoost model"""
    print("\n--- Training XGBoost ---")
    
    # Create and train the model
    model = xgb.XGBClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    return model

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate a model using various metrics"""
    print(f"\n--- Evaluating {model_name} ---")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Print metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(os.path.join(MODELS_DIR, f'{model_name.lower().replace(" ", "_")}_confusion_matrix.png'))
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend()
    plt.savefig(os.path.join(MODELS_DIR, f'{model_name.lower().replace(" ", "_")}_roc_curve.png'))
    
    # Return metrics as a dictionary
    metrics = {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc
    }
    
    return metrics

def tune_hyperparameters(X_train, y_train, X_test, y_test, best_model_name):
    """Tune hyperparameters for the best model"""
    print(f"\n--- Hyperparameter Tuning for {best_model_name} ---")
    
    if best_model_name == 'Logistic Regression':
        # Define parameter grid
        param_grid = {
            'C': [0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga']
        }
        
        # Create base model
        model = LogisticRegression(random_state=42, max_iter=1000)
        
    elif best_model_name == 'Random Forest':
        # Define parameter grid
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        # Create base model
        model = RandomForestClassifier(random_state=42)
        
    elif best_model_name == 'XGBoost':
        # Define parameter grid
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }
        
        # Create base model
        model = xgb.XGBClassifier(random_state=42)
        
    else:
        raise ValueError(f"Unsupported model: {best_model_name}")
    
    # Create grid search
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )
    
    # Fit grid search
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    end_time = time.time()
    
    print(f"Grid search completed in {end_time - start_time:.2f} seconds")
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    # Evaluate best model
    best_model = grid_search.best_estimator_
    metrics = evaluate_model(best_model, X_test, y_test, f"{best_model_name} (Tuned)")
    
    return best_model, metrics

def save_model(model, model_name):
    """Save the model to disk"""
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
    
    model_path = os.path.join(MODELS_DIR, f'{model_name.lower().replace(" ", "_")}.joblib')
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

def main():
    """Main function to execute the model training pipeline"""
    print("Starting model training...")
    
    # Load data
    df = load_data()
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(df)
    
    # Train models
    models = {
        'Logistic Regression': train_logistic_regression(X_train, y_train),
        'Random Forest': train_random_forest(X_train, y_train),
        'XGBoost': train_xgboost(X_train, y_train)
    }
    
    # Evaluate models
    metrics = {}
    for model_name, model in models.items():
        metrics[model_name] = evaluate_model(model, X_test, y_test, model_name)
    
    # Find the best model based on ROC AUC
    best_model_name = max(metrics, key=lambda k: metrics[k]['roc_auc'])
    print(f"\nBest model: {best_model_name} with ROC AUC: {metrics[best_model_name]['roc_auc']:.4f}")
    
    # Tune hyperparameters for the best model
    tuned_model, tuned_metrics = tune_hyperparameters(X_train, y_train, X_test, y_test, best_model_name)
    
    # Save the best model
    save_model(tuned_model, f"{best_model_name} (Tuned)")
    
    # Save the original models as well
    for model_name, model in models.items():
        save_model(model, model_name)
    
    print("Model training completed successfully!")

if __name__ == "__main__":
    main()
