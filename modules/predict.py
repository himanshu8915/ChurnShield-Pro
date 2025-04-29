import pandas as pd
import joblib
import os
import numpy as np

MODEL_PATH = "models/churn_model.pkl"
PREPROCESSOR_PATH = "models/preprocessor.pkl"

def load_artifacts():
    """Load the trained model and preprocessor."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(PREPROCESSOR_PATH):
        raise FileNotFoundError("Model or preprocessor not found. Please train the model first.")
    
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    return model, preprocessor

def preprocess_data(df):
    """Prepare data for prediction."""
    df = df.copy()
    
    # Ensure numeric columns are properly formatted
    for col in ['tenure', 'MonthlyCharges', 'TotalCharges']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Fill missing values
    df['tenure'] = df['tenure'].fillna(0)
    df['MonthlyCharges'] = df['MonthlyCharges'].fillna(df['MonthlyCharges'].median() if not df['MonthlyCharges'].empty else 0)
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median() if not df['TotalCharges'].empty else 0)
    
    # Create value ratio feature
    tenure_safe = np.maximum(df['tenure'], 1)  # Avoid division by zero
    df['value_ratio'] = df['TotalCharges'] / (df['MonthlyCharges'] * tenure_safe + 1e-6)
    
    # Ensure Contract column exists (use default if missing)
    if 'Contract' not in df.columns:
        df['Contract'] = 'Month-to-month'  # Default value
    
    return df

def predict_churn(df):
    """Generate churn predictions for the given dataframe."""
    # Preprocess the data
    df = preprocess_data(df)
    
    # Load model and preprocessor
    model, preprocessor = load_artifacts()
    
    # Extract features used for prediction
    try:
        features = df[['tenure', 'MonthlyCharges', 'TotalCharges', 'value_ratio', 'Contract']]
    except KeyError as e:
        # Handle case where some features might be missing
        missing_cols = set(['tenure', 'MonthlyCharges', 'TotalCharges', 'value_ratio', 'Contract']) - set(df.columns)
        raise ValueError(f"Missing required columns for prediction: {missing_cols}")
    
    # Transform features using preprocessor
    processed_data = preprocessor.transform(features)
    
    # Make predictions
    churn_probs = model.predict_proba(processed_data)[:, 1]
    churn_preds = (churn_probs >= 0.5).astype(int)
    
    # Add predictions to the dataframe
    result_df = df.copy()
    result_df['churn_probability'] = churn_probs
    result_df['churn_prediction'] = churn_preds
    
    # Set plan_type based on Contract for reporting
    result_df['plan_type'] = result_df['Contract']
    
    # Select columns for output
    output_cols = ['customerID', 'churn_probability', 'churn_prediction', 'plan_type', 'MonthlyCharges']
    available_cols = [col for col in output_cols if col in result_df.columns]
    
    # Ensure customerID exists (create if missing)
    if 'customerID' not in result_df.columns:
        result_df['customerID'] = range(1, len(result_df) + 1)
        available_cols = ['customerID'] + [col for col in available_cols if col != 'customerID']
    
    return result_df[available_cols]