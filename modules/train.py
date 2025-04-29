import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib
import os

REQUIRED_COLUMNS = [
    'customerID', 'tenure', 'MonthlyCharges', 'TotalCharges', 'Contract', 'Churn'
]

def validate_data(df):
    """Validate that the dataframe has the required columns and proper data."""
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    if df['Churn'].nunique() < 2:
        raise ValueError("Need both churned and non-churned examples in the 'Churn' column.")
    
    # Ensure numeric columns are numeric
    for col in ['tenure', 'MonthlyCharges', 'TotalCharges']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Fill missing values with median for numeric columns
    df['tenure'].fillna(df['tenure'].median(), inplace=True)
    df['MonthlyCharges'].fillna(df['MonthlyCharges'].median(), inplace=True)
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    
    # Ensure 'Churn' is properly formatted
    valid_churn_values = ['Yes', 'No', 1, 0]
    invalid_churn = ~df['Churn'].isin(valid_churn_values)
    if invalid_churn.any():
        raise ValueError(f"'Churn' column must contain 'Yes'/'No' or 1/0 values. Found invalid values: {df.loc[invalid_churn, 'Churn'].unique()}")

def preprocess_data(df):
    """Prepare data for model training."""
    # Validate data first
    validate_data(df)
    
    df = df.copy()
    
    # Convert Churn to binary
    if df['Churn'].dtype == 'object':
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    # Ensure TotalCharges and MonthlyCharges are numeric
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['MonthlyCharges'] = pd.to_numeric(df['MonthlyCharges'], errors='coerce')
    
    # Fill missing values
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    df['tenure'].fillna(df['tenure'].median(), inplace=True)
    
    # Create value ratio feature
    df['value_ratio'] = df['TotalCharges'] / (df['MonthlyCharges'] * np.maximum(df['tenure'], 1) + 1e-6)
    
    # Select features for model training
    features = df[['tenure', 'MonthlyCharges', 'TotalCharges', 'value_ratio', 'Contract']]
    target = df['Churn']
    
    # Create preprocessor for numerical and categorical features
    numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'value_ratio']
    categorical_features = ['Contract']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
        ]
    )
    
    # Fit and transform the data
    X_processed = preprocessor.fit_transform(features)
    
    return X_processed, target, preprocessor

def train_model(X, y):
    """Train a model using the processed features."""
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    
    # Initialize and train the model
    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        min_samples_leaf=50,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    print("\n--- Model Evaluation ---")
    print(f"ROC AUC Score: {roc_auc_score(y_test, y_prob):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return model

def save_artifacts(model, preprocessor):
    """Save the trained model and preprocessor."""
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/churn_model.pkl")
    joblib.dump(preprocessor, "models/preprocessor.pkl")
    print("\nModel and preprocessor saved successfully.")