"""
Model Utilities Module for Bank Marketing Campaign Analysis

This module provides utility functions for:
1. Loading trained models
2. Making predictions
3. Batch processing
"""

import os
import pandas as pd
import numpy as np
import joblib

# Constants
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
PREPROCESSOR_PATH = os.path.join(DATA_DIR, 'preprocessor.joblib')

def load_model(model_name):
    """Load a trained model from disk"""
    model_path = os.path.join(MODELS_DIR, f'{model_name.lower().replace(" ", "_")}.joblib')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    model = joblib.load(model_path)
    print(f"Model loaded from {model_path}")
    
    return model

def load_preprocessor():
    """Load the preprocessor from disk"""
    if not os.path.exists(PREPROCESSOR_PATH):
        raise FileNotFoundError(f"Preprocessor not found at {PREPROCESSOR_PATH}")
    
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    print(f"Preprocessor loaded from {PREPROCESSOR_PATH}")
    
    return preprocessor

def preprocess_data(df, preprocessor=None):
    """Preprocess data using the saved preprocessor"""
    if preprocessor is None:
        preprocessor = load_preprocessor()
    
    # Check if target variable exists in the dataframe
    has_target = 'y' in df.columns
    
    # Separate features and target if target exists
    if has_target:
        X = df.drop('y', axis=1)
        y = df['y']
    else:
        X = df
    
    # Preprocess the features
    X_processed = preprocessor.transform(X)
    
    # Create a DataFrame with processed features
    # Instead of trying to reconstruct feature names, we'll just use numeric column names
    # This avoids issues with mismatched feature dimensions
    X_processed_df = pd.DataFrame(X_processed)
    
    # Return processed features and target if target exists
    if has_target:
        return X_processed_df, y
    else:
        return X_processed_df

def predict(model, X, threshold=0.5):
    """Make predictions using a trained model"""
    # Get probability predictions
    y_pred_proba = model.predict_proba(X)[:, 1]
    
    # Convert probabilities to binary predictions based on threshold
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    return y_pred, y_pred_proba

def batch_predict(model, input_file, output_file, preprocessor=None):
    """Process a batch of data and save predictions to a file"""
    # Load the input data
    df = pd.read_csv(input_file)
    print(f"Input data loaded with shape: {df.shape}")
    
    # Preprocess the data
    X_processed = preprocess_data(df, preprocessor)
    
    # Make predictions
    y_pred, y_pred_proba = predict(model, X_processed)
    
    # Add predictions to the original dataframe
    df['prediction'] = y_pred
    df['probability'] = y_pred_proba
    
    # Save predictions to output file
    df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")
    
    return df

def get_feature_importance(model, X):
    """Get feature importance from the model"""
    # Check if the model has feature_importances_ attribute (tree-based models)
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        feature_names = [f"Feature {i}" for i in range(len(importance))]
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
    # Check if the model has coef_ attribute (linear models)
    elif hasattr(model, 'coef_'):
        importance = np.abs(model.coef_[0])
        feature_names = [f"Feature {i}" for i in range(len(importance))]
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
    else:
        raise ValueError("Model does not have feature_importances_ or coef_ attribute")
    
    return feature_importance

def get_available_models():
    """Get a list of available trained models"""
    if not os.path.exists(MODELS_DIR):
        return []
    
    model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith('.joblib')]
    model_names = [os.path.splitext(f)[0].replace('_', ' ').title() for f in model_files]
    
    return model_names

if __name__ == "__main__":
    # Example usage
    print("Available models:")
    models = get_available_models()
    for model_name in models:
        print(f"- {model_name}")
