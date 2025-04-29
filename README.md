# ChurnShield Pro

A comprehensive customer churn prediction and management dashboard built with Streamlit.

## Features

- **Data Upload**: Upload your customer data in CSV format
- **Model Training**: Train a machine learning model to predict customer churn
- **Churn Predictions**: Generate churn predictions for your customers
- **Insights Dashboard**: Visualize churn patterns and risk factors
- **ChurnBot**: AI-powered chatbot for churn reduction strategies

## Project Structure

```
churnshield_pro/
│
├── app.py                       # Main Streamlit app
├── requirements.txt             # Required Python packages
├── .env                         # Environment variables (API keys)
│
├── models/                      # Folder to save trained models
│   ├── churn_model.pkl          # Trained ML model
│   └── preprocessor.pkl         # Data preprocessor
│
├── modules/                     # Application modules
│   ├── train.py                 # Model training functions
│   ├── predict.py               # Prediction functions
│   └── chatbot.py               # ChurnBot implementation
│
└── .venv/                       # Virtual environment (not included in repo)
```

## Setup Instructions

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/churnshield-pro.git
cd churnshield-pro
```

2. **Create and activate a virtual environment**

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Set up environment variables**

Create a `.env` file with your API keys:

```
GROQ_API_KEY=your_groq_api_key_here
```

5. **Run the application**

```bash
streamlit run app.py
```

## Using the Dashboard

1. **Upload Data**: Start by uploading your customer data CSV
2. **Train Model**: Go to the "Train Model" tab to train the churn prediction model
3. **Generate Predictions**: Use the "Churn Predictions" tab to predict churn for your customers
4. **Review Insights**: Explore churn patterns in the "Churn Insights" tab
5. **Get Recommendations**: Chat with ChurnBot for personalized churn reduction strategies

## Data Format

Your CSV file should contain the following columns:

- `customerID`: Unique identifier for each customer
- `tenure`: Number of months the customer has been with the company
- `MonthlyCharges`: Monthly charges in dollars
- `TotalCharges`: Total charges in dollars
- `Contract`: Contract type (e.g., "Month-to-month", "One year", "Two year")
- `Churn`: "Yes" or "No" indicating whether the customer has churned

## ChurnBot API

The ChurnBot feature requires a valid GROQ API key. You can get one by signing up at [Groq.com](https://groq.com).

## License

This project is licensed under the MIT License - see the LICENSE file for details.