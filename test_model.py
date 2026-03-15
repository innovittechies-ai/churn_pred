import pandas as pd
import numpy as np
import pickle

def predict_new_customer(customer_data, model, scaler, label_encoders, feature_columns):
    """
    Takes a dictionary of customer data and returns a Churn prediction.
    """
    # 1. Convert input to DataFrame
    input_df = pd.DataFrame([customer_data])
    
    # 2. Pre-process Numeric Columns (Crucial for the Scaler)
    # Ensure charges are floats, not strings
    for col in ['Monthly Charges', 'Total Charges']:
        if col in input_df.columns:
            input_df[col] = pd.to_numeric(input_df[col], errors='coerce').fillna(0)
    
    # 3. Apply Label Encoding (using the encoders from training)
    # Note: Ensure your encoders dictionary was saved during data_prep
    for col, le in label_encoders.items():
        if col in input_df.columns:
            # If the user input is a new category, default to the first known class
            input_df[col] = input_df[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
            input_df[col] = le.transform(input_df[col])

    # 4. Ensure column order matches training — only select columns that exist
    available_features = [col for col in feature_columns if col in input_df.columns]
    if len(available_features) != len(feature_columns):
        missing = set(feature_columns) - set(available_features)
        print(f"Warning: Missing columns {missing}. Using available: {available_features}")
    input_df = input_df[available_features]

    # 5. Scale the data
    input_scaled = scaler.transform(input_df)

    # 6. Predict
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    return prediction, probability

# ==========================================
# TEST WITH NEW DATA
# ==========================================
if __name__ == "__main__":
    try:
        # Loading the names as defined in your training script
        model    = pickle.load(open('logistic_churn_model.pkl', 'rb'))
        features = pickle.load(open('feature_columns.pkl', 'rb'))
        
        # Note: Ensure you saved 'scaler.pkl' and 'encoders.pkl' in your data_prep.py!
        scaler   = pickle.load(open('scaler.pkl', 'rb'))
        encoders = pickle.load(open('encoders.pkl', 'rb'))
        
        # Example customer data - HIGH CHURN RISK
        new_customer_churn = {
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
        }

        # Example customer data - NO CHURN (Loyal customer)
        new_customer_loyal = {
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

        print("\n" + "="*50)
        print("TEST 1: HIGH CHURN RISK CUSTOMER")
        print("="*50)
        result, prob = predict_new_customer(new_customer_churn, model, scaler, encoders, features)
        status = "CHURN" if result == 1 else "NOT CHURN"
        print(f"PREDICTION : {status}")
        print(f"CONFIDENCE : {prob:.2%}")
        
        print("\n" + "="*50)
        print("TEST 2: LOYAL CUSTOMER (NO CHURN)")
        print("="*50)
        result, prob = predict_new_customer(new_customer_loyal, model, scaler, encoders, features)
        status = "CHURN" if result == 1 else "NOT CHURN"
        print(f"PREDICTION : {status}")
        print(f"CONFIDENCE : {prob:.2%}")
        print("="*50)
        
    except FileNotFoundError as e:
        print(f"Error: Missing file {e.filename}. Make sure to save all .pkl files during training.")
