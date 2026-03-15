import pandas as pd
import numpy as np
import pickle  # Added to save the scaler and encoders
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE

def load_and_prepare():
    # 1. Load
    df = pd.read_excel('Telco_customer_churn.xlsx')

    # 2. Drop Leakage/ID columns and unnecessary features
    leakage_cols = ['Churn Label', 'Churn Score', 'Churn Reason', 'Customer ID', 
                    'Count', 'Country', 'State', 'City', 'Zip Code', 'Lat Long',
                    'CustomerID', 'Latitude', 'Longitude', 'CLTV']
    df.drop(columns=[c for c in leakage_cols if c in df.columns], inplace=True)

    # 3. Fix Total Charges
    if 'Total Charges' in df.columns:
        df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce')
    df.dropna(inplace=True)

    # 4. Find & Encode Target
    target_col = 'Churn Value'
    
    # 5. Categorical Encoding (Crucial: Save the encoders!)
    encoders = {}
    categorical_cols = df.select_dtypes(include='object').columns
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le  # Store the encoder for this column

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 6. Stratified Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 7. SMOTE
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

    # 8. Scaled sets
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_bal)
    X_test_scaled  = scaler.transform(X_test)

    # ==========================================
    # SAVE THE PIECES (Fixes your FileNotFoundError)
    # ==========================================
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    with open('encoders.pkl', 'wb') as f:
        pickle.dump(encoders, f)

    print(f"--- Files Saved: scaler.pkl, encoders.pkl ---")
    
    return (X, y, X_train, X_test, y_train, y_test,
            X_train_bal, y_train_bal,
            X_train_scaled, X_test_scaled)
