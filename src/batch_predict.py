"""
Batch Prediction Script for Bank Marketing Campaign Analysis

This script processes a CSV file with customer data and generates predictions.
Usage: python batch_predict.py input.csv output.csv [model_name]
"""

import os
import sys
import pandas as pd
from model_utils import load_model, load_preprocessor, batch_predict, get_available_models

def print_usage():
    """Print usage information"""
    print("Usage: python batch_predict.py input.csv output.csv [model_name]")
    print("  input.csv: Path to the input CSV file with customer data")
    print("  output.csv: Path to save the predictions")
    print("  model_name: (Optional) Name of the model to use for predictions")
    print("              If not provided, the script will use the best available model")
    print("\nAvailable models:")
    models = get_available_models()
    if models:
        for model_name in models:
            print(f"  - {model_name}")
    else:
        print("  No models available. Please run model_training.py first.")

def main():
    """Main function to execute the batch prediction"""
    # Check command line arguments
    if len(sys.argv) < 3:
        print_usage()
        return
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found")
        return
    
    # Get available models
    models = get_available_models()
    if not models:
        print("Error: No models available. Please run model_training.py first.")
        return
    
    # Determine which model to use
    if len(sys.argv) >= 4:
        model_name = sys.argv[3]
        if model_name not in models:
            print(f"Error: Model '{model_name}' not found")
            print("Available models:")
            for m in models:
                print(f"  - {m}")
            return
    else:
        # Use the first model that has "tuned" in its name, or the first available model
        tuned_models = [m for m in models if "tuned" in m.lower()]
        if tuned_models:
            model_name = tuned_models[0]
        else:
            model_name = models[0]
    
    print(f"Using model: {model_name}")
    
    try:
        # Load model and preprocessor
        model = load_model(model_name)
        preprocessor = load_preprocessor()
        
        # Process the batch
        result_df = batch_predict(model, input_file, output_file, preprocessor)
        
        # Print summary
        total_customers = len(result_df)
        predicted_subscribers = result_df['prediction'].sum()
        subscription_rate = predicted_subscribers / total_customers * 100
        
        print("\nPrediction Summary:")
        print(f"Total customers: {total_customers}")
        print(f"Predicted subscribers: {predicted_subscribers}")
        print(f"Predicted subscription rate: {subscription_rate:.2f}%")
        
        print(f"\nPredictions saved to {output_file}")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
