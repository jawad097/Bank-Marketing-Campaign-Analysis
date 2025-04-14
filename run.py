"""
Run Script for Bank Marketing Campaign Analysis

This script provides a convenient entry point for running different components of the project.
"""

import os
import sys
import argparse
import subprocess

def run_data_preparation():
    """Run the data preparation script"""
    print("Running data preparation...")
    subprocess.run([sys.executable, "src/data_preparation.py"])

def run_model_training():
    """Run the model training script"""
    print("Running model training...")
    subprocess.run([sys.executable, "src/model_training.py"])

def run_batch_prediction(input_file, output_file, model_name=None):
    """Run the batch prediction script"""
    print("Running batch prediction...")
    cmd = [sys.executable, "src/batch_predict.py", input_file, output_file]
    if model_name:
        cmd.append(model_name)
    subprocess.run(cmd)

def run_streamlit_app():
    """Run the Streamlit app"""
    print("Running Streamlit app...")
    subprocess.run(["streamlit", "run", "app/app.py"])

def main():
    """Main function to parse arguments and run the appropriate script"""
    parser = argparse.ArgumentParser(description="Bank Marketing Campaign Analysis")
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Data preparation command
    data_parser = subparsers.add_parser("prepare", help="Run data preparation")
    
    # Model training command
    train_parser = subparsers.add_parser("train", help="Run model training")
    
    # Batch prediction command
    predict_parser = subparsers.add_parser("predict", help="Run batch prediction")
    predict_parser.add_argument("input", help="Input CSV file with customer data")
    predict_parser.add_argument("output", help="Output CSV file for predictions")
    predict_parser.add_argument("--model", help="Model name to use for predictions")
    
    # Streamlit app command
    app_parser = subparsers.add_parser("app", help="Run Streamlit app")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run the appropriate command
    if args.command == "prepare":
        run_data_preparation()
    elif args.command == "train":
        run_model_training()
    elif args.command == "predict":
        run_batch_prediction(args.input, args.output, args.model)
    elif args.command == "app":
        run_streamlit_app()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
