# Streamlit Deployment Guide

## Project: Telecom Churn Prediction System

### Prerequisites
- GitHub account with the repository pushed
- Streamlit Cloud account (https://streamlit.io/cloud)

### Deployment Steps

#### 1. Prepare Your Repository
Ensure all required files are committed to GitHub:
- `streamlit_app.py` - Main Streamlit application
- `requirements.txt` - Python dependencies
- `logistic_churn_model.pkl` - Trained model
- `scaler.pkl` - Feature scaler
- `encoders.pkl` - Label encoders
- `feature_columns.pkl` - Feature column names
- `.streamlit/config.toml` - Streamlit configuration

#### 2. Push to GitHub
```bash
git add .
git commit -m "Add Streamlit deployment files"
git push origin main
```

#### 3. Deploy on Streamlit Cloud
1. Go to https://streamlit.io/cloud
2. Click **"New app"**
3. Select your GitHub repository
4. Choose the branch (main)
5. Set main file path to: `streamlit_app.py`
6. Click **"Deploy"**

#### 4. Configuration
The app will automatically:
- Install dependencies from `requirements.txt`
- Load the pre-trained model files
- Start the web server

### File Structure
```
churn_predction/
├── streamlit_app.py              # Main Streamlit app
├── test_model.py                 # Local testing script
├── requirements.txt              # Dependencies
├── .streamlit/
│   └── config.toml              # Streamlit theme config
├── logistic_churn_model.pkl     # Trained model
├── scaler.pkl                   # Feature scaler
├── encoders.pkl                 # Label encoders
├── feature_columns.pkl          # Feature names
└── DEPLOYMENT.md                # This file
```

### Features
- **Interactive UI** - User-friendly form for customer data input
- **Real-time Predictions** - Get churn prediction instantly
- **Sample Customers** - Pre-loaded examples for quick testing
- **Visual Insights** - Color-coded risk indicators and insights
- **Model Info** - Display model performance metrics

### App URL Format
```
https://share.streamlit.io/[USERNAME]/[REPO-NAME]/main/streamlit_app.py
```

### Troubleshooting

#### Model Files Not Found
- Ensure pickle files are in the root directory
- Verify file names match exactly:
  - `logistic_churn_model.pkl`
  - `scaler.pkl`
  - `encoders.pkl`
  - `feature_columns.pkl`

#### Import Errors
- Check `requirements.txt` contains all dependencies
- Verify versions match your local environment

#### Timeout Issues
- The app caches model resources with `@st.cache_resource`
- First load may take 30-60 seconds

### Performance Metrics
- **Model Accuracy**: 74.5%
- **ROC-AUC Score**: 0.821
- **Training Samples**: ~7,000 (SMOTE balanced)
- **Features**: 19 customer attributes

### Support
For issues with Streamlit deployment, visit: https://docs.streamlit.io/en/stable/deploy_intro.html
