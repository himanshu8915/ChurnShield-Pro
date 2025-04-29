import streamlit as st
import pandas as pd
import os
from modules.train import train_model, preprocess_data, save_artifacts
from modules.predict import predict_churn, load_artifacts
from modules.chatbot import generate_summary, ask_churn_bot
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up the page layout
st.set_page_config(page_title="ChurnShield Pro", layout="wide")

# Initialize session state variables if they don't exist
if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'churn_summary' not in st.session_state:
    st.session_state.churn_summary = None

# Sidebar for Navigation
st.sidebar.title("ChurnShield Pro")
st.sidebar.markdown("#### Customer Churn Management Dashboard")
option = st.sidebar.selectbox(
    "Navigation",
    ("Upload Data", "Train Model", "Churn Predictions", "Churn Insights", "Chat with ChurnBot")
)

# Upload CSV
def upload_csv():
    uploaded_file = st.file_uploader("Upload your customer dataset (CSV)", type="csv")
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.uploaded_data = df
            st.write("Preview of your dataset:")
            st.dataframe(df.head())
            return df
        except Exception as e:
            st.error(f"Error reading the file: {e}")
            return None
    return st.session_state.uploaded_data

# Main Section for App
if option == "Upload Data":
    st.title("📊 Upload Your Customer Data")
    st.markdown("Upload your customer dataset to begin churn analysis.")
    
    df = upload_csv()
    
    if df is not None:
        cols = df.columns.tolist()
        st.success(f"✅ Data uploaded successfully! {len(df)} records found.")
        
        # Display schema information
        st.markdown("### Dataset Information")
        st.markdown(f"**Number of records:** {len(df)}")
        st.markdown(f"**Number of features:** {len(cols)}")
        
        with st.expander("View Column Details"):
            for col in cols:
                st.write(f"**{col}**: {df[col].dtype}")

elif option == "Train Model":
    st.title("🔄 Train the Churn Prediction Model")
    
    # Get data
    if st.session_state.uploaded_data is None:
        df = upload_csv()
    else:
        df = st.session_state.uploaded_data
        st.info("Using previously uploaded data. Upload new data if needed.")
    
    if df is not None:
        # Check if the required column 'Churn' exists
        if 'Churn' not in df.columns:
            st.error("The uploaded dataset must contain a 'Churn' column with 'Yes' or 'No' values.")
        else:
            st.write("Dataset ready for training:")
            st.dataframe(df.head())
            
            # Training section
            if st.button("Train Model", key="train_button"):
                try:
                    with st.spinner("Training in progress..."):
                        # Preprocess and train
                        X_processed, y, preprocessor = preprocess_data(df)
                        model = train_model(X_processed, y)
                        save_artifacts(model, preprocessor)
                        
                    st.success("✅ Model trained successfully!")
                    
                    # Display model information
                    st.markdown("### Model Information")
                    st.markdown("- **Model Type**: Gradient Boosting Classifier")
                    
                    # Reset predictions since we've trained a new model
                    st.session_state.predictions = None
                    st.session_state.churn_summary = None
                    
                except Exception as e:
                    st.error(f"❌ Error in training model: {e}")

elif option == "Churn Predictions":
    st.title("🔮 Churn Predictions")
    
    # Get data
    if st.session_state.uploaded_data is None:
        df = upload_csv()
    else:
        df = st.session_state.uploaded_data
        st.info("Using previously uploaded data. Upload new data if needed.")
    
    if df is not None:
        try:
            if st.button("Generate Predictions", key="predict_button"):
                with st.spinner("Calculating churn predictions..."):
                    # Load trained model and preprocessor
                    try:
                        predictions = predict_churn(df)
                        st.session_state.predictions = predictions
                        
                        # Display predictions
                        st.markdown("### Customer Churn Predictions")
                        st.dataframe(predictions)
                        
                        # Visualization
                        churn_counts = predictions['churn_prediction'].value_counts()
                        st.markdown("### Churn Distribution")
                        st.bar_chart(churn_counts)
                        
                        # Summary metrics
                        churn_rate = predictions['churn_prediction'].mean() * 100
                        st.metric("Overall Churn Rate", f"{churn_rate:.2f}%")
                        
                        # Generate and save the summary for chatbot
                        summary = generate_summary(predictions)
                        st.session_state.churn_summary = summary
                        
                    except FileNotFoundError:
                        st.error("Model not found. Please train the model first.")
                    
            # Display existing predictions if available
            elif st.session_state.predictions is not None:
                st.markdown("### Customer Churn Predictions")
                st.dataframe(st.session_state.predictions)
                
                # Visualization of existing predictions
                churn_counts = st.session_state.predictions['churn_prediction'].value_counts()
                st.markdown("### Churn Distribution")
                st.bar_chart(churn_counts)
                
                # Summary metrics
                churn_rate = st.session_state.predictions['churn_prediction'].mean() * 100
                st.metric("Overall Churn Rate", f"{churn_rate:.2f}%")
            
        except Exception as e:
            st.error(f"Error generating predictions: {e}")

elif option == "Churn Insights":
    st.title("📈 Churn Insights")
    
    if st.session_state.predictions is None:
        # Get data
        if st.session_state.uploaded_data is None:
            df = upload_csv()
        else:
            df = st.session_state.uploaded_data
            st.info("Using previously uploaded data. Upload new data if needed.")
        
        if df is not None:
            try:
                with st.spinner("Generating insights..."):
                    predictions = predict_churn(df)
                    st.session_state.predictions = predictions
                    summary = generate_summary(predictions)
                    st.session_state.churn_summary = summary
            except FileNotFoundError:
                st.error("Model not found. Please train the model first.")
                st.info("Please go to the 'Train Model' tab to train a model first.")
    
    if st.session_state.predictions is not None:
        predictions = st.session_state.predictions
        
        # Display the summary
        if st.session_state.churn_summary:
            st.markdown("### Churn Summary")
            st.markdown(st.session_state.churn_summary)
        
        # Create more detailed insights
        st.markdown("### Detailed Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # High-risk customer segments
            st.markdown("#### High-Risk Customers")
            high_risk = predictions[predictions['churn_probability'] > 0.7]
            if not high_risk.empty:
                st.dataframe(high_risk)
                st.markdown(f"**{len(high_risk)}** customers have a churn risk over 70%")
            else:
                st.info("No high-risk customers found")
        
        with col2:
            # Correlation with fees
            st.markdown("#### Churn vs. Monthly Charges")
            if 'MonthlyCharges' in predictions.columns:
                charges_by_churn = predictions.groupby('churn_prediction')['MonthlyCharges'].mean()
                st.bar_chart(charges_by_churn)
            
            # Plan type distribution for churners
            if 'plan_type' in predictions.columns:
                st.markdown("#### Plan Types of Churners")
                churner_plans = predictions[predictions['churn_prediction'] == 1]['plan_type'].value_counts()
                st.bar_chart(churner_plans)

elif option == "Chat with ChurnBot":
    st.title("🤖 Chat with ChurnBot")
    st.markdown("Ask questions about your churn data and get recommendations to reduce churn.")
    
    if st.session_state.uploaded_data is None and st.session_state.predictions is None:
        st.warning("Please upload data and generate predictions before using the chatbot.")
        df = upload_csv()
        
        if df is not None:
            try:
                with st.spinner("Analyzing your data..."):
                    predictions = predict_churn(df)
                    st.session_state.predictions = predictions
                    summary = generate_summary(predictions)
                    st.session_state.churn_summary = summary
                st.success("Analysis complete! You can now chat with ChurnBot.")
            except FileNotFoundError:
                st.error("Model not found. Please train the model first.")
    
    if st.session_state.churn_summary:
        # Display the summary for context
        with st.expander("View Churn Summary"):
            st.markdown(st.session_state.churn_summary)
        
        # Chat interface
        st.markdown("### Ask ChurnBot")
        user_query = st.text_input("What would you like to know about reducing churn?", 
                                  placeholder="E.g., How can I reduce churn for premium customers?")
        
        if user_query:
            with st.spinner("ChurnBot is thinking..."):
                try:
                    # Call the chatbot function
                    response = ask_churn_bot(user_query, st.session_state.churn_summary)
                    
                    # Display the response in a chat-like interface
                    st.markdown("### ChurnBot Response")
                    st.markdown(f"{response}")
                    
                except Exception as e:
                    st.error(f"Error with ChurnBot: {e}")
                    st.info("Make sure your GROQ_API_KEY is properly set in the .env file.")
    else:
        st.info("Generate predictions first to enable the chatbot.")