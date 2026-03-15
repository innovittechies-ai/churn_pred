import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime

# Page Configuration
st.set_page_config(
    page_title="Churn Prediction System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        margin-bottom: 30px;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .churn-high {
        background-color: #ffcccc;
        padding: 15px;
        border-left: 5px solid #ff0000;
        border-radius: 5px;
    }
    .churn-low {
        background-color: #ccffcc;
        padding: 15px;
        border-left: 5px solid #00cc00;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# LOAD MODEL & RESOURCES
# ==========================================
@st.cache_resource
def load_model_resources():
    """Load pickled model, scaler, and encoders"""
    try:
        model = pickle.load(open('logistic_churn_model.pkl', 'rb'))
        scaler = pickle.load(open('scaler.pkl', 'rb'))
        encoders = pickle.load(open('encoders.pkl', 'rb'))
        features = pickle.load(open('feature_columns.pkl', 'rb'))
        return model, scaler, encoders, features
    except FileNotFoundError as e:
        st.error(f"❌ Missing file: {e.filename}")
        st.info("Please run model_saver.py to generate the model files.")
        return None, None, None, None

# ==========================================
# PREDICTION FUNCTION
# ==========================================
def predict_new_customer(customer_data, model, scaler, label_encoders, feature_columns):
    """Predict churn for a customer"""
    input_df = pd.DataFrame([customer_data])
    
    # Pre-process numeric columns
    for col in ['Monthly Charges', 'Total Charges']:
        if col in input_df.columns:
            input_df[col] = pd.to_numeric(input_df[col], errors='coerce').fillna(0)
    
    # Apply label encoding
    for col, le in label_encoders.items():
        if col in input_df.columns:
            input_df[col] = input_df[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
            input_df[col] = le.transform(input_df[col])

    # Select available features
    available_features = [col for col in feature_columns if col in input_df.columns]
    input_df = input_df[available_features]

    # Scale and predict
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    return prediction, probability

# ==========================================
# SAMPLE CUSTOMERS
# ==========================================
SAMPLE_CUSTOMERS = {
    "🔴 High Churn Risk": {
        'Gender': 'Female',
        'Senior Citizen': 0,
        'Partner': 'Yes',
        'Dependents': 'No',
        'Tenure Months': 1,
        'Phone Service': 'No',
        'Multiple Lines': 'No',
        'Internet Service': 'DSL',
        'Online Security': 'No',
        'Online Backup': 'No',
        'Device Protection': 'No',
        'Tech Support': 'No',
        'Streaming TV': 'No',
        'Streaming Movies': 'No',
        'Contract': 'Month-to-month',
        'Paperless Billing': 'Yes',
        'Payment Method': 'Electronic check',
        'Monthly Charges': 29.85,
        'Total Charges': 29.85
    },
    "🟢 Loyal Customer": {
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
    },
    "🟡 Moderate Risk": {
        'Gender': 'Male',
        'Senior Citizen': 0,
        'Partner': 'No',
        'Dependents': 'No',
        'Tenure Months': 12,
        'Phone Service': 'Yes',
        'Multiple Lines': 'No',
        'Internet Service': 'Fiber optic',
        'Online Security': 'No',
        'Online Backup': 'Yes',
        'Device Protection': 'No',
        'Tech Support': 'No',
        'Streaming TV': 'Yes',
        'Streaming Movies': 'No',
        'Contract': '1 year',
        'Paperless Billing': 'Yes',
        'Payment Method': 'Credit card',
        'Monthly Charges': 65.50,
        'Total Charges': 786.00
    }
}

# ==========================================
# MAIN APP
# ==========================================
def main():
    st.markdown("<h1 class='main-header'>📊 Telecom Churn Prediction System</h1>", unsafe_allow_html=True)
    
    # Load model resources
    model, scaler, encoders, features = load_model_resources()
    
    if model is None:
        st.stop()
    
    # Sidebar - Demo selector
    st.sidebar.title("🚀 Quick Start")
    demo_choice = st.sidebar.selectbox(
        "Load Sample Customer",
        ["Custom Input", "🔴 High Churn Risk", "🟢 Loyal Customer", "🟡 Moderate Risk"]
    )
    
    # ==========================================
    # INPUT SECTION
    # ==========================================
    st.subheader("👤 Customer Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Demographics**")
        gender = st.selectbox("Gender", ["Male", "Female"], key="gender")
        senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No", key="senior")
        partner = st.selectbox("Partner", ["Yes", "No"], key="partner")
        dependents = st.selectbox("Dependents", ["Yes", "No"], key="dependents")
    
    with col2:
        st.write("**Billing Information**")
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=50.0, step=0.01, key="monthly")
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=500.0, step=1.0, key="total")
        tenure_months = st.slider("Tenure (Months)", min_value=0, max_value=72, value=12, key="tenure")
    
    st.divider()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Services**")
        phone_service = st.selectbox("Phone Service", ["Yes", "No"], key="phone")
        multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"], key="lines")
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"], key="internet")
    
    with col2:
        st.write("**Internet Add-ons**")
        online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"], key="security")
        online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"], key="backup")
        device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"], key="device")
    
    with col3:
        st.write("**Additional Services**")
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"], key="tech")
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"], key="tv")
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"], key="movies")
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Account Details**")
        contract = st.selectbox("Contract", ["Month-to-month", "1 year", "2 year"], key="contract")
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"], key="paperless")
    
    with col2:
        st.write("**Payment**")
        payment_method = st.selectbox(
            "Payment Method", 
            ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
            key="payment"
        )
    
    st.divider()
    
    # ==========================================
    # PREDICTION
    # ==========================================
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button("🔮 Predict Churn Risk", use_container_width=True, type="primary"):
            customer_data = {
                'Gender': gender,
                'Senior Citizen': senior_citizen,
                'Partner': partner,
                'Dependents': dependents,
                'Tenure Months': tenure_months,
                'Phone Service': phone_service,
                'Multiple Lines': multiple_lines,
                'Internet Service': internet_service,
                'Online Security': online_security,
                'Online Backup': online_backup,
                'Device Protection': device_protection,
                'Tech Support': tech_support,
                'Streaming TV': streaming_tv,
                'Streaming Movies': streaming_movies,
                'Contract': contract,
                'Paperless Billing': paperless_billing,
                'Payment Method': payment_method,
                'Monthly Charges': monthly_charges,
                'Total Charges': total_charges
            }
            
            prediction, probability = predict_new_customer(customer_data, model, scaler, encoders, features)
            
            # Display prediction
            st.success("✅ Prediction Complete")
            
            col1, col2 = st.columns(2)
            with col1:
                if prediction == 1:
                    st.markdown(f"""
                    <div class='churn-high'>
                        <h3>⚠️ CHURN RISK: HIGH</h3>
                        <p style='font-size: 24px; font-weight: bold;'>{probability:.1%}</p>
                        <p>This customer is likely to churn.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class='churn-low'>
                        <h3>✅ CHURN RISK: LOW</h3>
                        <p style='font-size: 24px; font-weight: bold;'>{(1-probability):.1%} Retention</p>
                        <p>This customer is likely to stay.</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                # Metrics display
                st.metric("Churn Probability", f"{probability:.2%}")
                st.metric("Retention Probability", f"{(1-probability):.2%}")
            
            # Insights
            st.write("---")
            st.write("**📈 Key Insights:**")
            
            insights = []
            if tenure_months < 6:
                insights.append("🔴 **New Customer**: Low tenure significantly increases churn risk")
            elif tenure_months > 24:
                insights.append("🟢 **Loyal Customer**: Long tenure indicates high retention")
            
            if contract == "Month-to-month":
                insights.append("🔴 **Flexible Contract**: Month-to-month customers churn more frequently")
            elif contract == "2 year":
                insights.append("🟢 **Committed Contract**: 2-year contracts reduce churn risk")
            
            if monthly_charges > 80:
                insights.append("🔴 **High Charges**: Customers pay more might be more price-sensitive")
            
            if phone_service == "Yes" and (online_security == "Yes" or tech_support == "Yes"):
                insights.append("🟢 **High Engagement**: Customers using multiple services tend to stay")
            
            if len(insights) == 0:
                insights.append("📊 Customer profile shows average churn indicators")
            
            for insight in insights:
                st.write(f"• {insight}")
    
    with col2:
        # Load sample
        if demo_choice != "Custom Input":
            st.info("📋 Sample customer pre-loaded in sidebar")
            sample = SAMPLE_CUSTOMERS[demo_choice]
            
            st.write("**Sample Data:**")
            st.json({
                'Contract': sample['Contract'],
                'Tenure Months': sample['Tenure Months'],
                'Monthly Charges': sample['Monthly Charges'],
                'Services': f"{sum([v == 'Yes' for v in [sample[k] for k in ['Phone Service', 'Online Security', 'Online Backup', 'Device Protection', 'Tech Support', 'Streaming TV', 'Streaming Movies']]])} services"
            })
    
    # Footer
    st.write("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("**Model Info**")
        st.caption("• Logistic Regression\n• Accuracy: 74.5%\n• ROC-AUC: 0.821")
    with col2:
        st.write("**Data Features**")
        st.caption(f"• Total Features: {len(features)}\n• Training Samples: ~7,000\n• Balanced with SMOTE")
    with col3:
        st.write("**Last Updated**")
        st.caption(f"• Date: {datetime.now().strftime('%Y-%m-%d')}\n• Version: 1.0\n• Status: Active")

if __name__ == "__main__":
    main()
