#!/bin/bash
# Setup script for Bank Marketing Campaign Analysis project

# Print colored text
print_color() {
    case $1 in
        "green") echo -e "\033[0;32m$2\033[0m" ;;
        "red") echo -e "\033[0;31m$2\033[0m" ;;
        "yellow") echo -e "\033[0;33m$2\033[0m" ;;
        "blue") echo -e "\033[0;34m$2\033[0m" ;;
        *) echo "$2" ;;
    esac
}

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    print_color "red" "Python 3 is not installed. Please install Python 3 and try again."
    exit 1
fi

print_color "blue" "Setting up Bank Marketing Campaign Analysis project..."

# Create virtual environment
print_color "yellow" "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
print_color "yellow" "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
print_color "yellow" "Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
print_color "yellow" "Creating necessary directories..."
mkdir -p data models

# Run data preparation
print_color "yellow" "Running data preparation..."
python run.py prepare

# Run model training
print_color "yellow" "Running model training..."
python run.py train

print_color "green" "Setup completed successfully!"
print_color "blue" "To activate the virtual environment in the future, run: source venv/bin/activate"
print_color "blue" "To run the Streamlit app, run: python run.py app"
