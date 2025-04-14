"""
Data Preparation Module for Bank Marketing Campaign Analysis

This module handles:
1. Data loading and downloading
2. Exploratory data analysis
3. Data cleaning
4. Feature engineering
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import urllib.request
import zipfile
import io

# Set random seed for reproducibility
np.random.seed(42)

# Constants
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip"
RAW_DATA_PATH = os.path.join(DATA_DIR, 'bank-additional', 'bank-additional-full.csv')
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, 'processed_data.csv')

def download_dataset():
    """Download the UCI Bank Marketing dataset if it doesn't exist"""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    
    if not os.path.exists(RAW_DATA_PATH):
        print("Downloading dataset...")
        response = urllib.request.urlopen(DATASET_URL)
        zip_file = zipfile.ZipFile(io.BytesIO(response.read()))
        zip_file.extractall(DATA_DIR)
        print(f"Dataset downloaded and extracted to {DATA_DIR}")
    else:
        print(f"Dataset already exists at {RAW_DATA_PATH}")

def load_data():
    """Load the dataset and return a pandas DataFrame"""
    # Download dataset if it doesn't exist
    download_dataset()
    
    # Load the dataset
    df = pd.read_csv(RAW_DATA_PATH, sep=';')
    print(f"Dataset loaded with shape: {df.shape}")
    return df

def explore_data(df):
    """Perform exploratory data analysis"""
    print("\n--- Exploratory Data Analysis ---")
    
    # Display basic information
    print("\nBasic Information:")
    print(df.info())
    
    # Display summary statistics
    print("\nSummary Statistics:")
    print(df.describe())
    
    # Check for missing values
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    # Check target variable distribution
    print("\nTarget Variable Distribution:")
    print(df['y'].value_counts(normalize=True) * 100)
    
    # Save distribution plot
    plt.figure(figsize=(10, 6))
    sns.countplot(x='y', data=df)
    plt.title('Target Variable Distribution')
    plt.savefig(os.path.join(DATA_DIR, 'target_distribution.png'))
    
    # Check correlations between numerical features
    numerical_features = df.select_dtypes(include=['int64', 'float64']).columns
    plt.figure(figsize=(12, 10))
    correlation_matrix = df[numerical_features].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Matrix of Numerical Features')
    plt.savefig(os.path.join(DATA_DIR, 'correlation_matrix.png'))
    
    return df

def clean_data(df):
    """Clean the dataset by handling missing values and encoding categorical variables"""
    print("\n--- Data Cleaning ---")
    
    # Make a copy of the dataframe
    df_cleaned = df.copy()
    
    # Replace 'unknown' values with NaN
    for col in df_cleaned.columns:
        if df_cleaned[col].dtype == object:
            df_cleaned[col] = df_cleaned[col].replace('unknown', np.nan)
    
    # Check for missing values after replacement
    missing_values = df_cleaned.isnull().sum()
    print("\nMissing Values after replacing 'unknown':")
    print(missing_values[missing_values > 0])
    
    # Identify numerical and categorical columns
    numerical_features = df_cleaned.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = df_cleaned.select_dtypes(include=['object']).columns.tolist()
    
    # Remove target variable from features
    if 'y' in numerical_features:
        numerical_features.remove('y')
    if 'y' in categorical_features:
        categorical_features.remove('y')
    
    # Create preprocessing pipelines
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Convert target variable to binary (0/1)
    df_cleaned['y'] = df_cleaned['y'].map({'no': 0, 'yes': 1})
    
    # Fit and transform the data
    X = df_cleaned.drop('y', axis=1)
    y = df_cleaned['y']
    
    X_processed = preprocessor.fit_transform(X)
    
    # Get feature names after one-hot encoding
    onehot_features = []
    for i, feature in enumerate(categorical_features):
        categories = preprocessor.named_transformers_['cat'].named_steps['onehot'].categories_[i]
        for category in categories:
            onehot_features.append(f"{feature}_{category}")
    
    # Create a new DataFrame with processed features
    processed_feature_names = numerical_features + onehot_features
    X_processed_df = pd.DataFrame(X_processed, columns=processed_feature_names)
    
    # Add target variable back to the processed data
    processed_df = X_processed_df.copy()
    processed_df['y'] = y.values
    
    print(f"\nProcessed data shape: {processed_df.shape}")
    return processed_df, preprocessor

def engineer_features(df):
    """Perform feature engineering to create new features"""
    print("\n--- Feature Engineering ---")
    
    # Make a copy of the dataframe
    df_engineered = df.copy()
    
    # Add any additional feature engineering steps here
    # For example, you could create interaction terms or polynomial features
    
    print(f"Engineered data shape: {df_engineered.shape}")
    return df_engineered

def save_processed_data(df, preprocessor):
    """Save the processed data and preprocessor"""
    # Save processed data
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Processed data saved to {PROCESSED_DATA_PATH}")
    
    # Save preprocessor
    import joblib
    preprocessor_path = os.path.join(DATA_DIR, 'preprocessor.joblib')
    joblib.dump(preprocessor, preprocessor_path)
    print(f"Preprocessor saved to {preprocessor_path}")

def main():
    """Main function to execute the data preparation pipeline"""
    print("Starting data preparation...")
    
    # Load data
    df = load_data()
    
    # Explore data
    df = explore_data(df)
    
    # Clean data
    df_cleaned, preprocessor = clean_data(df)
    
    # Engineer features
    df_engineered = engineer_features(df_cleaned)
    
    # Save processed data
    save_processed_data(df_engineered, preprocessor)
    
    print("Data preparation completed successfully!")

if __name__ == "__main__":
    main()
