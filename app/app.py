"""
Streamlit Web Application for Bank Marketing Campaign Analysis

This application provides a user interface for:
1. Uploading customer data
2. Making predictions
3. Visualizing results
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
from model_utils import (
    load_model, load_preprocessor, preprocess_data, predict,
    get_feature_importance, get_available_models
)

# Set page configuration
st.set_page_config(
    page_title="Bank Marketing Campaign Predictor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')

# Cache the model loading to improve performance
@st.cache_resource
def get_model_and_preprocessor(model_name):
    """Load the model and preprocessor"""
    model = load_model(model_name)
    preprocessor = load_preprocessor()
    return model, preprocessor

def load_example_data():
    """Load example data for demonstration"""
    try:
        # Try to load the original dataset
        df = pd.read_csv(os.path.join(DATA_DIR, 'bank-additional-full.csv'), sep=';')
        # Take a small sample for demonstration
        return df.sample(n=min(100, len(df)), random_state=42)
    except:
        # If the original dataset is not available, create a synthetic dataset
        data = {
            'age': np.random.randint(18, 90, 100),
            'job': np.random.choice(['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed', 'unknown'], 100),
            'marital': np.random.choice(['divorced', 'married', 'single', 'unknown'], 100),
            'education': np.random.choice(['basic.4y', 'basic.6y', 'basic.9y', 'high.school', 'illiterate', 'professional.course', 'university.degree', 'unknown'], 100),
            'default': np.random.choice(['no', 'yes', 'unknown'], 100),
            'housing': np.random.choice(['no', 'yes', 'unknown'], 100),
            'loan': np.random.choice(['no', 'yes', 'unknown'], 100),
            'contact': np.random.choice(['cellular', 'telephone'], 100),
            'month': np.random.choice(['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'], 100),
            'day_of_week': np.random.choice(['mon', 'tue', 'wed', 'thu', 'fri'], 100),
            'duration': np.random.randint(0, 1000, 100),
            'campaign': np.random.randint(1, 10, 100),
            'pdays': np.random.randint(-1, 100, 100),
            'previous': np.random.randint(0, 10, 100),
            'poutcome': np.random.choice(['failure', 'nonexistent', 'success'], 100),
            'emp.var.rate': np.random.uniform(-3, 3, 100),
            'cons.price.idx': np.random.uniform(90, 95, 100),
            'cons.conf.idx': np.random.uniform(-50, -30, 100),
            'euribor3m': np.random.uniform(0, 5, 100),
            'nr.employed': np.random.uniform(4500, 5500, 100)
        }
        return pd.DataFrame(data)

def make_predictions(df, model, preprocessor, threshold=0.5):
    """Process the data and make predictions"""
    # Preprocess the data
    X_processed = preprocess_data(df, preprocessor)
    
    # Make predictions
    y_pred, y_pred_proba = predict(model, X_processed, threshold)
    
    # Add predictions to the dataframe
    df_with_predictions = df.copy()
    df_with_predictions['prediction'] = y_pred
    df_with_predictions['probability'] = y_pred_proba
    
    return df_with_predictions, X_processed

def plot_feature_importance(model, X_processed):
    """Plot feature importance"""
    # For models with feature_importances_ attribute (tree-based models)
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        feature_names = [f"Feature {i}" for i in range(len(importance))]
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
    
    # For models with coef_ attribute (linear models)
    elif hasattr(model, 'coef_'):
        importance = np.abs(model.coef_[0])
        feature_names = [f"Feature {i}" for i in range(len(importance))]
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
    
    else:
        # If model doesn't have feature importance, return empty plot
        return go.Figure().update_layout(title="Feature importance not available for this model")
    
    # Take top 15 features
    top_features = feature_importance.head(15)
    
    # Create a bar chart
    fig = px.bar(
        top_features,
        x='importance',
        y='feature',
        orientation='h',
        title='Top 15 Feature Importance',
        labels={'importance': 'Importance', 'feature': 'Feature'},
        color='importance',
        color_continuous_scale='Viridis'
    )
    
    return fig

def plot_prediction_distribution(df_with_predictions):
    """Plot prediction distribution"""
    # Count predictions
    prediction_counts = df_with_predictions['prediction'].value_counts().reset_index()
    prediction_counts.columns = ['Prediction', 'Count']
    prediction_counts['Prediction'] = prediction_counts['Prediction'].map({0: 'No Subscription', 1: 'Subscription'})
    
    # Create a pie chart
    fig = px.pie(
        prediction_counts,
        values='Count',
        names='Prediction',
        title='Prediction Distribution',
        color_discrete_sequence=['#FF6B6B', '#4ECDC4']
    )
    
    return fig

def plot_probability_distribution(df_with_predictions):
    """Plot probability distribution"""
    # Create a histogram
    fig = px.histogram(
        df_with_predictions,
        x='probability',
        nbins=20,
        title='Subscription Probability Distribution',
        labels={'probability': 'Probability of Subscription'},
        color_discrete_sequence=['#6C5B7B']
    )
    
    # Add a vertical line at the threshold
    fig.add_vline(x=0.5, line_dash="dash", line_color="red")
    
    return fig

def plot_feature_analysis(df_with_predictions, feature, categorical=True):
    """Plot feature analysis"""
    if categorical:
        # Group by feature and calculate mean probability
        feature_analysis = df_with_predictions.groupby(feature)['probability'].mean().reset_index()
        feature_analysis.columns = [feature, 'Average Probability']
        
        # Sort by average probability
        feature_analysis = feature_analysis.sort_values('Average Probability', ascending=False)
        
        # Create a bar chart
        fig = px.bar(
            feature_analysis,
            x=feature,
            y='Average Probability',
            title=f'Average Subscription Probability by {feature}',
            labels={feature: feature, 'Average Probability': 'Average Probability'},
            color='Average Probability',
            color_continuous_scale='Viridis'
        )
    else:
        # Create a scatter plot
        fig = px.scatter(
            df_with_predictions,
            x=feature,
            y='probability',
            title=f'Subscription Probability vs {feature}',
            labels={feature: feature, 'probability': 'Probability'},
            color='prediction',
            color_discrete_map={0: '#FF6B6B', 1: '#4ECDC4'}
        )
        
        # Add a trend line
        fig.add_trace(
            go.Scatter(
                x=df_with_predictions[feature],
                y=df_with_predictions['probability'].rolling(window=10).mean(),
                mode='lines',
                name='Trend',
                line=dict(color='black', width=2)
            )
        )
    
    return fig

def main():
    """Main function for the Streamlit app"""
    # Add a title and description
    st.title("Bank Marketing Campaign Predictor")
    st.markdown("""
    This application predicts whether a customer will subscribe to a term deposit based on various features.
    Upload a CSV file with customer data to get predictions.
    """)
    
    # Sidebar
    st.sidebar.title("Settings")
    
    # Get available models
    models = get_available_models()
    if not models:
        st.error("No models available. Please run model_training.py first.")
        return
    
    # Model selection
    model_name = st.sidebar.selectbox(
        "Select Model",
        models,
        index=0
    )
    
    # Prediction threshold
    threshold = st.sidebar.slider(
        "Prediction Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Probability threshold for classifying as a subscriber"
    )
    
    # Load model and preprocessor
    try:
        model, preprocessor = get_model_and_preprocessor(model_name)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return
    
    # File upload
    uploaded_file = st.file_uploader("Upload a CSV file with customer data", type=["csv"])
    
    # Use example data if no file is uploaded
    use_example_data = st.checkbox("Use example data", value=not uploaded_file)
    
    if uploaded_file is not None:
        # Load the uploaded file
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"File uploaded successfully with {df.shape[0]} rows and {df.shape[1]} columns.")
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            return
    elif use_example_data:
        # Load example data
        df = load_example_data()
        st.info(f"Using example data with {df.shape[0]} rows and {df.shape[1]} columns.")
    else:
        st.warning("Please upload a CSV file or use example data.")
        return
    
    # Show the data
    with st.expander("View Data"):
        st.dataframe(df)
    
    # Make predictions
    df_with_predictions, X_processed = make_predictions(df, model, preprocessor, threshold)
    
    # Show predictions
    st.subheader("Predictions")
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Customers", df_with_predictions.shape[0])
    with col2:
        st.metric("Predicted Subscribers", df_with_predictions['prediction'].sum())
    with col3:
        subscription_rate = df_with_predictions['prediction'].mean() * 100
        st.metric("Subscription Rate", f"{subscription_rate:.2f}%")
    
    # Show the predictions
    with st.expander("View Predictions"):
        st.dataframe(df_with_predictions[['prediction', 'probability']])
    
    # Download predictions
    csv = df_with_predictions.to_csv(index=False)
    st.download_button(
        label="Download Predictions",
        data=csv,
        file_name="predictions.csv",
        mime="text/csv"
    )
    
    # Visualizations
    st.subheader("Visualizations")
    
    # Feature importance
    st.plotly_chart(plot_feature_importance(model, X_processed), use_container_width=True)
    
    # Prediction distribution
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_prediction_distribution(df_with_predictions), use_container_width=True)
    with col2:
        st.plotly_chart(plot_probability_distribution(df_with_predictions), use_container_width=True)
    
    # Feature analysis
    st.subheader("Feature Analysis")
    
    # Select features for analysis
    categorical_features = [col for col in df.columns if df[col].dtype == 'object']
    numerical_features = [col for col in df.columns if df[col].dtype in ['int64', 'float64']]
    
    # Categorical feature analysis
    if categorical_features:
        selected_categorical = st.selectbox(
            "Select Categorical Feature",
            categorical_features,
            index=0 if categorical_features else None
        )
        st.plotly_chart(plot_feature_analysis(df_with_predictions, selected_categorical, categorical=True), use_container_width=True)
    
    # Numerical feature analysis
    if numerical_features:
        selected_numerical = st.selectbox(
            "Select Numerical Feature",
            numerical_features,
            index=0 if numerical_features else None
        )
        st.plotly_chart(plot_feature_analysis(df_with_predictions, selected_numerical, categorical=False), use_container_width=True)

if __name__ == "__main__":
    main()
