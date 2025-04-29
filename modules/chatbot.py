import os
import pandas as pd
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.schema import SystemMessage, HumanMessage

# Load environment variables
load_dotenv()

# Initialize Groq LLM
def initialize_llm():
    """Initialize the LLM chat model."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables. Please add it to your .env file.")
    
    return ChatGroq(
        groq_api_key=api_key,
        model_name="llama3-70b-8192"
    )

def generate_summary(df):
    """Generate a summary of churn predictions for the chatbot."""
    # Calculate churn rate
    churn_rate = df['churn_prediction'].mean() if 'churn_prediction' in df.columns else 0.0
    
    # Calculate average charges by churn status if available
    try:
        avg_charges = df.groupby('churn_prediction')['MonthlyCharges'].mean().to_dict()
    except (KeyError, ValueError):
        avg_charges = {1: 0, 0: 0}
    
    # Count plan types for churners if available
    try:
        churners = df[df['churn_prediction'] == 1]
        plan_counts = churners['plan_type'].value_counts().to_dict() if not churners.empty else {}
    except (KeyError, ValueError):
        plan_counts = {}
    
    # Format plan counts nicely
    plan_str = ", ".join([f"{plan}: {count}" for plan, count in plan_counts.items()]) if plan_counts else "No data available"
    
    # Create summary text
    summary = f"""
📊 Churn Dashboard Summary:
- Overall churn rate: {churn_rate:.2%}
- Average monthly charges:
  • Churners: ${avg_charges.get(1, 0):.2f}
  • Non-churners: ${avg_charges.get(0, 0):.2f}
- Most affected plans: {plan_str}
"""
    
    # Add high-risk customer information if available
    try:
        high_risk = df[df['churn_probability'] > 0.7]
        high_risk_count = len(high_risk)
        summary += f"- High-risk customers: {high_risk_count} ({high_risk_count/len(df):.1%} of total)\n"
    except (KeyError, ValueError):
        pass
    
    return summary.strip()

def ask_churn_bot(user_query, summary):
    """Send a query to the churn chatbot and get a response."""
    try:
        # Initialize the LLM
        llm = initialize_llm()
        
        # Create the system message with churn summary
        system_content = f"""
You are ChurnBot, an expert advisor on customer churn reduction strategies. 
Your role is to help businesses reduce customer churn by analyzing data and providing actionable insights.

Here are the latest churn insights from the customer database:
{summary}

As ChurnBot, you should:
1. Provide practical, easy-to-implement strategies based on the data
2. Focus on actionable recommendations rather than generic advice
3. Be concise yet thorough in your response
4. Use bullet points for clarity when appropriate
5. Reference specific metrics from the summary when relevant
6. Maintain a professional, helpful tone
"""
        
        # Create the messages
        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=user_query)
        ]
        
        # Get response from LLM
        response = llm(messages)
        return response.content.strip()
        
    except Exception as e:
        # Handle exceptions more gracefully
        if "GROQ_API_KEY" in str(e):
            return "Error: GROQ API key not found or invalid. Please check your .env file."
        else:
            return f"Sorry, I encountered an error while processing your query: {str(e)}"