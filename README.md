# Bank Marketing Campaign Analysis and Prediction

This project aims to predict whether a customer will subscribe to a term deposit based on features from a bank marketing dataset. It helps banks optimize marketing strategies by targeting the right customers.

## Project Structure

- `data/`: Contains the dataset
- `notebooks/`: Jupyter notebooks for exploratory data analysis
- `src/`: Source code for data preparation, model training, and utilities
- `models/`: Saved trained models
- `app/`: Streamlit web application

## Setup and Installation

1. Clone this repository
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Data Preparation and Model Training

1. Run the data preparation script:
   ```
   python src/data_preparation.py
   ```

2. Train the model:
   ```
   python src/model_training.py
   ```

### Using the Web Application

1. Start the Streamlit app:
   ```
   streamlit run app/app.py
   ```

2. Upload a CSV file with customer data
3. View predictions and visualizations

## Features

- Data cleaning and preprocessing
- Feature engineering
- Model training with various algorithms (Logistic Regression, Random Forest, XGBoost)
- Model evaluation and hyperparameter tuning
- Batch prediction capability
- Interactive web UI for non-technical users

## Dataset

The project uses the UCI Bank Marketing dataset, which includes various customer attributes and whether they subscribed to a term deposit.

## License

This project is open source and available under the MIT License.
