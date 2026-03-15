import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, accuracy_score)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# =============================================================================
# STEP 1 — LOAD DATA
# =============================================================================
print("=" * 60)
print("STEP 1: LOADING DATA")
print("=" * 60)

df = pd.read_excel('Telco_customer_churn.xlsx')
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nMissing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")

# =============================================================================
# STEP 2 — PREPROCESSING
# =============================================================================
print("\n" + "=" * 60)
print("STEP 2: PREPROCESSING")
print("=" * 60)

# Drop customer ID if present
id_cols = [c for c in df.columns if 'id' in c.lower() or 'customerid' in c.lower()]
if id_cols:
    df.drop(columns=id_cols, inplace=True)
    print(f"Dropped ID columns: {id_cols}")

# Convert TotalCharges to numeric (may contain spaces)
if 'TotalCharges' in df.columns:
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Drop rows with nulls
before = len(df)
df.dropna(inplace=True)
print(f"Dropped {before - len(df)} rows with nulls. Remaining: {len(df)}")

# Encode target — find churn column
churn_col = [c for c in df.columns if 'churn' in c.lower()][0]
print(f"Target column: '{churn_col}'")

# Encode all object columns
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

print(f"Target distribution:\n{df[churn_col].value_counts()}")
churn_rate = df[churn_col].mean() * 100
print(f"Churn rate: {churn_rate:.1f}%")

# =============================================================================
# STEP 3 — SPLIT & BALANCE
# =============================================================================
print("\n" + "=" * 60)
print("STEP 3: SPLIT & SMOTE BALANCING")
print("=" * 60)

X = df.drop(columns=[churn_col])
y = df[churn_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# SMOTE — handle class imbalance (churners are minority)
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
print(f"Before SMOTE: {y_train.value_counts().to_dict()}")
print(f"After  SMOTE: {pd.Series(y_train_bal).value_counts().to_dict()}")

# Scale features (required for Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_bal)
X_test_scaled  = scaler.transform(X_test)

# =============================================================================
# STEP 4 — MODEL 1: LOGISTIC REGRESSION (Baseline / Interpretability)
# =============================================================================
print("\n" + "=" * 60)
print("MODEL 1: LOGISTIC REGRESSION — Baseline (Interpretability)")
print("=" * 60)
print("""
Purpose : Understand WHICH factors drive churn and by HOW MUCH.
          Odds ratios tell stakeholders the direct impact of each
          variable (e.g. month-to-month contract = 3x churn risk).
Steps   : 1. Train on scaled + SMOTE-balanced data
          2. Evaluate with classification report + ROC-AUC
          3. Extract coefficients as odds ratios for business insight
""")

lr = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
lr.fit(X_train_scaled, y_train_bal)

lr_pred     = lr.predict(X_test_scaled)
lr_proba    = lr.predict_proba(X_test_scaled)[:, 1]
lr_auc      = roc_auc_score(y_test, lr_proba)
lr_accuracy = accuracy_score(y_test, lr_pred)

print(f"Accuracy : {lr_accuracy:.4f}")
print(f"ROC-AUC  : {lr_auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, lr_pred, target_names=['No Churn', 'Churn']))

# Odds ratios — business interpretability
odds_ratios = pd.Series(np.exp(lr.coef_[0]), index=X.columns)
print("Top 5 Churn Drivers (Odds Ratios):")
print(odds_ratios.sort_values(ascending=False).head(5).to_string())

# =============================================================================
# STEP 5 — MODEL 2: RANDOM FOREST (Robust Ensemble)
# =============================================================================
print("\n" + "=" * 60)
print("MODEL 2: RANDOM FOREST — Robust Ensemble")
print("=" * 60)
print("""
Purpose : Capture non-linear interactions between features.
          Provides stable Feature Importance ranking across
          hundreds of trees — less prone to overfitting than
          a single decision tree.
Steps   : 1. Train on SMOTE-balanced data (no scaling needed)
          2. Evaluate with classification report + ROC-AUC
          3. Extract feature importances for segment analysis
""")

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_leaf=5,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train_bal, y_train_bal)

rf_pred     = rf.predict(X_test)
rf_proba    = rf.predict_proba(X_test)[:, 1]
rf_auc      = roc_auc_score(y_test, rf_proba)
rf_accuracy = accuracy_score(y_test, rf_pred)

print(f"Accuracy : {rf_accuracy:.4f}")
print(f"ROC-AUC  : {rf_auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, rf_pred, target_names=['No Churn', 'Churn']))

rf_importances = pd.Series(rf.feature_importances_, index=X.columns)
print("Top 5 Feature Importances:")
print(rf_importances.sort_values(ascending=False).head(5).to_string())

# =============================================================================
# STEP 6 — MODEL 3: XGBOOST (Performance Leader)
# =============================================================================
print("\n" + "=" * 60)
print("MODEL 3: XGBOOST — Performance Leader")
print("=" * 60)
print("""
Purpose : Maximize predictive accuracy using gradient boosting.
          Each tree corrects the errors of the previous one.
          scale_pos_weight handles class imbalance natively
          without needing SMOTE.
Steps   : 1. Compute scale_pos_weight from original class ratio
          2. Train on original (unbalanced) train split
          3. Evaluate with classification report + ROC-AUC
          4. Extract feature importances
""")

neg  = (y_train == 0).sum()
pos  = (y_train == 1).sum()
spw  = neg / pos
print(f"scale_pos_weight = {spw:.2f}  (neg:{neg} / pos:{pos})")

xgb = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=spw,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1
)
xgb.fit(X_train, y_train)

xgb_pred     = xgb.predict(X_test)
xgb_proba    = xgb.predict_proba(X_test)[:, 1]
xgb_auc      = roc_auc_score(y_test, xgb_proba)
xgb_accuracy = accuracy_score(y_test, xgb_pred)

print(f"Accuracy : {xgb_accuracy:.4f}")
print(f"ROC-AUC  : {xgb_auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, xgb_pred, target_names=['No Churn', 'Churn']))

xgb_importances = pd.Series(xgb.feature_importances_, index=X.columns)
print("Top 5 Feature Importances:")
print(xgb_importances.sort_values(ascending=False).head(5).to_string())

# =============================================================================
# STEP 7 — MODEL COMPARISON
# =============================================================================
print("\n" + "=" * 60)
print("STEP 7: MODEL COMPARISON SUMMARY")
print("=" * 60)

results = pd.DataFrame({
    'Model'   : ['Logistic Regression', 'Random Forest', 'XGBoost'],
    'Accuracy': [lr_accuracy, rf_accuracy, xgb_accuracy],
    'ROC-AUC' : [lr_auc,      rf_auc,      xgb_auc]
})
results['Accuracy'] = results['Accuracy'].map('{:.4f}'.format)
results['ROC-AUC']  = results['ROC-AUC'].map('{:.4f}'.format)
print(results.to_string(index=False))

# =============================================================================
# STEP 8 — VISUALIZATIONS
# =============================================================================
print("\n" + "=" * 60)
print("STEP 8: GENERATING VISUALIZATIONS")
print("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle('Multi-Model Churn Prediction System', fontsize=16, fontweight='bold')

models_info = [
    ('Logistic Regression', lr_pred,  lr_proba,  'Blues'),
    ('Random Forest',       rf_pred,  rf_proba,  'Greens'),
    ('XGBoost',             xgb_pred, xgb_proba, 'Oranges'),
]

# Row 1 — Confusion Matrices
for i, (name, pred, _, cmap) in enumerate(models_info):
    cm = confusion_matrix(y_test, pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, ax=axes[0, i],
                xticklabels=['No Churn', 'Churn'],
                yticklabels=['No Churn', 'Churn'])
    axes[0, i].set_title(f'{name}\nConfusion Matrix')
    axes[0, i].set_ylabel('Actual')
    axes[0, i].set_xlabel('Predicted')

# Row 2 — ROC Curves (all on one plot) + Feature Importances (RF & XGB)
ax_roc = axes[1, 0]
for name, _, proba, _ in models_info:
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc_val = roc_auc_score(y_test, proba)
    ax_roc.plot(fpr, tpr, label=f'{name} (AUC={auc_val:.3f})')
ax_roc.plot([0, 1], [0, 1], 'k--')
ax_roc.set_title('ROC Curves — All Models')
ax_roc.set_xlabel('False Positive Rate')
ax_roc.set_ylabel('True Positive Rate')
ax_roc.legend(fontsize=8)

# RF Feature Importance
top_rf = rf_importances.sort_values(ascending=False).head(10)
axes[1, 1].barh(top_rf.index[::-1], top_rf.values[::-1], color='steelblue')
axes[1, 1].set_title('Random Forest — Top 10 Feature Importances')
axes[1, 1].set_xlabel('Importance')

# XGBoost Feature Importance
top_xgb = xgb_importances.sort_values(ascending=False).head(10)
axes[1, 2].barh(top_xgb.index[::-1], top_xgb.values[::-1], color='darkorange')
axes[1, 2].set_title('XGBoost — Top 10 Feature Importances')
axes[1, 2].set_xlabel('Importance')

plt.tight_layout()
plt.savefig('churn_model_results.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: churn_model_results.png")

# =============================================================================
# STEP 9 — HIGH-RISK SEGMENT IDENTIFICATION
# =============================================================================
print("\n" + "=" * 60)
print("STEP 9: HIGH-RISK SEGMENT IDENTIFICATION")
print("=" * 60)

X_test_df = X_test.copy()
X_test_df['Churn_Probability'] = xgb_proba
X_test_df['Actual_Churn']      = y_test.values
X_test_df['Risk_Segment'] = pd.cut(
    X_test_df['Churn_Probability'],
    bins=[0, 0.3, 0.6, 1.0],
    labels=['Low Risk', 'Medium Risk', 'High Risk']
)

segment_summary = X_test_df.groupby('Risk_Segment', observed=True).agg(
    Count=('Churn_Probability', 'count'),
    Avg_Churn_Prob=('Churn_Probability', 'mean'),
    Actual_Churn_Rate=('Actual_Churn', 'mean')
).reset_index()

print(segment_summary.to_string(index=False))
high_risk_count = (X_test_df['Risk_Segment'] == 'High Risk').sum()
print(f"\nHigh-risk customers flagged for retention: {high_risk_count}")
print("Targeting these customers enables the projected 15-25% revenue loss reduction.")
