# 📊 Telecom Churn Prediction System

A machine learning application that predicts customer churn for telecom companies using Logistic Regression. Features an interactive Streamlit web interface for real-time predictions.

## 🎯 Project Overview

This system analyzes customer data to identify which customers are likely to leave (churn) based on their usage patterns, contract details, and service preferences. The model helps companies proactively retain at-risk customers.

### Key Metrics
- **Model Accuracy**: 74.5%
- **ROC-AUC Score**: 0.821
- **Training Data**: ~7,000 customers (SMOTE balanced)
- **Target Variable**: Customer Churn (Yes/No)

## 📁 Project Structure

```
churn_predction/
├── streamlit_app.py               # Interactive web UI
├── test_model.py                  # Local testing script
├── data_prep.py                   # Data preprocessing pipeline
├── model_saver.py                 # Model training & saving
├── model1_logistic_regression.py  # Logistic regression implementation
├── model2_random_forest.py        # Random forest implementation
├── model3_xgboost.py              # XGBoost implementation
├── main.py                        # Comparative analysis
├── requirements.txt               # Python dependencies
├── .streamlit/config.toml         # Streamlit configuration
├── .gitignore                     # Git ignore rules
├── DEPLOYMENT.md                  # Streamlit deployment guide
└── README.md                      # This file
```

## 🚀 Quick Start

### Local Testing
```bash
python test_model.py
```

### Streamlit Deployment
```bash
streamlit run streamlit_app.py
```

## 📦 Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

### Core Dependencies
- **pandas** - Data manipulation
- **scikit-learn** - ML models & preprocessing
- **xgboost** - Gradient boosting
- **streamlit** - Web interface
- **imbalanced-learn** - SMOTE balancing
- **matplotlib/seaborn** - Visualization

## 🔄 Data Pipeline

### 1. Data Preparation (`data_prep.py`)
- Loads Telco customer churn data
- Removes leakage columns (ID, geographic info)
- Handles missing values
- Label encodes categorical variables
- Applies SMOTE for class balancing
- Scales features with StandardScaler

### 2. Model Training
Three models trained and compared:
- **Logistic Regression** - Interpretability focus (74.5% accuracy)
- **Random Forest** - Ensemble approach
- **XGBoost** - Gradient boosting method

### 3. Prediction
- Accepts customer data in dictionary format
- Applies same preprocessing pipeline
- Returns churn probability (0-1)

## 💡 Key Features

### Web Interface (`streamlit_app.py`)
- **Interactive Form** - Input customer details across 19 attributes
- **Real-time Predictions** - Instant churn risk predictions
- **Sample Customers** - Pre-loaded examples:
  - 🔴 High Churn Risk (91.58%)
  - 🟢 Loyal Customer (96.66% retention)
  - 🟡 Moderate Risk
- **Visual Insights** - AI-generated recommendations
- **Model Metrics** - Performance statistics

### Customer Attributes (19 Features)
**Demographics**
- Gender, Age, Partner, Dependents

**Services**
- Phone Service, Multiple Lines, Internet Service Type

**Internet Add-ons**
- Online Security, Online Backup, Device Protection, Tech Support
- Streaming TV, Streaming Movies

**Account**
- Tenure (months), Monthly Charges, Total Charges
- Contract Type, Paperless Billing, Payment Method

## 🎨 Streamlit Features

### UI Components
- Clean, intuitive layout with 3-column sections
- Color-coded predictions (Red for churn, Green for retention)
- Real-time metric displays
- Automatic insight generation

### Sample Predictions
```
High Churn Risk Customer: 91.58% churn probability
Loyal Customer: 3.34% churn probability (96.66% retention)
```

## 🔍 Model Insights

### Top Churn Drivers (Odds Ratios)
1. **Total Charges** (3.52x) - Higher charges = Higher churn
2. **Monthly Charges** (2.03x) - Price sensitivity
3. **Internet Service Type** (1.25x) - Service selection matters
4. **Streaming Services** (0.95x) - Engagement reduces churn
5. **Multiple Lines** (0.94x) - Service bundling helps

### Risk Factors
- ⚠️ New customers (tenure < 6 months)
- ⚠️ Month-to-month contracts
- ⚠️ No additional services
- ⚠️ High monthly charges

### Retention Factors
- ✅ Long tenure (> 24 months)
- ✅ Multi-year contracts (1-2 year)
- ✅ Multiple services subscribed
- ✅ Family responsibilities (partner/dependents)

## 📊 Data Sample

### Input Format
```python
new_customer = {
    'Gender': 'Female',
    'Senior Citizen': 0,
    'Partner': 'Yes',
    'Dependents': 'Yes',
    'Tenure Months': 48,
    'Phone Service': 'Yes',
    'Multiple Lines': 'Yes',
    'Internet Service': 'Fiber optic',
    'Online Security': 'Yes',
    'Online Backup': 'Yes',
    'Device Protection': 'Yes',
    'Tech Support': 'Yes',
    'Streaming TV': 'Yes',
    'Streaming Movies': 'Yes',
    'Contract': '2 year',
    'Paperless Billing': 'No',
    'Payment Method': 'Bank transfer',
    'Monthly Charges': 89.95,
    'Total Charges': 3997.50
}
```

## 🌐 Deployment

### Streamlit Cloud
1. Push repository to GitHub
2. Visit https://streamlit.io/cloud
3. Connect your GitHub account
4. Select this repository
5. Deploy `streamlit_app.py`

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed instructions.

## 📈 Performance

### Model Evaluation
```
Accuracy:  74.48%
ROC-AUC:   0.8211
Precision: 51% (Churn class)
Recall:    72% (Churn class)
```

### Data Balancing
- Original churn rate: 26.5%
- After SMOTE: 50-50 balanced training set

## 🛠️ Development

### Test Models
```bash
python test_model.py           # Quick prediction test
python model1_logistic_regression.py  # LR details
python model2_random_forest.py        # RF analysis
python model3_xgboost.py              # XGBoost results
python main.py                 # Comparative analysis
```

### Retrain Model
```bash
python model_saver.py  # Retrain and save pickles
```

## 📝 Notes

- All models use SMOTE-balanced training data
- Features are standardized using StandardScaler
- Categorical variables use LabelEncoder
- Prediction pipeline mirrors training preprocessing

## 🤝 Contributing

To improve the model:
1. Experiment with feature engineering in `data_prep.py`
2. Add new models in `model*.py` files
3. Adjust hyperparameters in model training
4. Test locally before deployment

## 📄 License

This project is part of a telecom churn prediction demonstration.

## 📞 Support

For Streamlit issues: https://docs.streamlit.io/
For scikit-learn: https://scikit-learn.org/
For model questions: Review the model training files

---

**Last Updated**: March 2025  
**Model Version**: 1.0  
**Status**: Production Ready ✅
