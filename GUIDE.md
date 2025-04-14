# Bank Marketing Campaign Analysis - Installation and Usage Guide

This guide provides detailed instructions for setting up and using the Bank Marketing Campaign Analysis project.

## Table of Contents
- [Installation](#installation)
- [Project Components](#project-components)
- [Usage Examples](#usage-examples)
- [Example Workflow](#example-workflow)
- [Troubleshooting](#troubleshooting)

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Setup Steps

1. **Clone the repository**
   ```
   git clone <repository-url>
   cd bank-marketing-campaign
   ```

2. **Create a virtual environment (optional but recommended)**
   ```
   python -m venv venv
   ```

   Activate the virtual environment:
   - On Windows:
     ```
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```
     source venv/bin/activate
     ```

3. **Install dependencies**
   ```
   pip install -r requirements.txt
   ```

## Project Components

The project consists of the following components:

1. **Data Preparation**: Loads, cleans, and processes the bank marketing dataset.
2. **Model Training**: Trains multiple machine learning models and selects the best one.
3. **Batch Prediction**: Processes a CSV file with customer data and generates predictions.
4. **Streamlit Web App**: Provides a user-friendly interface for uploading data and visualizing predictions.

## Usage Examples

The project includes a convenient `run.py` script that serves as an entry point for all components.

### Data Preparation

To prepare the data:

```
python run.py prepare
```

This will:
- Download the UCI Bank Marketing dataset (if not already downloaded)
- Clean and preprocess the data
- Save the processed data to `data/processed_data.csv`
- Save the preprocessor to `data/preprocessor.joblib`

### Model Training

To train the models:

```
python run.py train
```

This will:
- Load the processed data
- Train multiple models (Logistic Regression, Random Forest, XGBoost)
- Evaluate the models and select the best one
- Tune the hyperparameters of the best model
- Save all models to the `models/` directory

### Batch Prediction

To make predictions on a batch of customer data:

```
python run.py predict data/sample_customers.csv predictions.csv
```

To specify a particular model:

```
python run.py predict data/sample_customers.csv predictions.csv --model "XGBoost (Tuned)"
```

### Streamlit Web App

To run the Streamlit web application:

```
python run.py app
```

This will start the web app at http://localhost:8501, where you can:
- Upload customer data
- Make predictions
- Visualize results
- Analyze feature importance

## Example Workflow

Here's a complete workflow example:

1. **Prepare the data**
   ```
   python run.py prepare
   ```

2. **Train the models**
   ```
   python run.py train
   ```

3. **Make batch predictions**
   ```
   python run.py predict data/sample_customers.csv predictions.csv
   ```

4. **Run the web app for interactive analysis**
   ```
   python run.py app
   ```

## Troubleshooting

### Common Issues

1. **Missing dependencies**
   
   If you encounter errors about missing packages, ensure you've installed all dependencies:
   ```
   pip install -r requirements.txt
   ```

2. **Data not found**
   
   If you get errors about missing data files, make sure you've run the data preparation step first:
   ```
   python run.py prepare
   ```

3. **Models not found**
   
   If you get errors about missing model files, make sure you've run the model training step:
   ```
   python run.py train
   ```

4. **Streamlit not starting**
   
   If Streamlit fails to start, try installing it separately:
   ```
   pip install streamlit
   streamlit run app/app.py
   ```

### Getting Help

If you encounter any issues not covered in this guide, please:
1. Check the project documentation
2. Look for error messages in the console output
3. Contact the project maintainers

## Additional Resources

- [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [UCI Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/bank+marketing)
