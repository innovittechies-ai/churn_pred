import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pickle  # Added for saving the model
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, accuracy_score)
from data_prep import load_and_prepare

# =============================================================================
# MODEL 1: LOGISTIC REGRESSION — Baseline (Interpretability)
# =============================================================================
print("=" * 60)
print("MODEL 1: LOGISTIC REGRESSION")
print("=" * 60)

# Note: Ensure your load_and_prepare() returns the scaler object now!
# If your data_prep.py doesn't return the scaler, you'll need to modify it.
(X, y, X_train, X_test, y_train, y_test,
 X_train_bal, y_train_bal,
 X_train_scaled, X_test_scaled) = load_and_prepare()

# 1. Train
lr = LogisticRegression(max_iter=500, random_state=42, class_weight='balanced')
lr.fit(X_train_scaled, y_train_bal)

# 2. Evaluate
lr_pred  = lr.predict(X_test_scaled)
lr_proba = lr.predict_proba(X_test_scaled)[:, 1]
acc      = accuracy_score(y_test, lr_pred)
auc      = roc_auc_score(y_test, lr_proba)

print(f"\nAccuracy : {acc:.4f}")
print(f"ROC-AUC  : {auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, lr_pred, target_names=['No Churn', 'Churn']))

# 3. Odds Ratios — business insight
odds_ratios = pd.Series(np.exp(lr.coef_[0]), index=X.columns)
print("\nTop 5 Churn Drivers (Odds Ratios):")
print(odds_ratios.sort_values(ascending=False).head(5).to_string())

# =============================================================================
# SAVE MODEL COMPONENTS (Using Pickle)
# =============================================================================
# We save the model and the feature names to ensure inference matches training
with open('logistic_churn_model.pkl', 'wb') as f:
    pickle.dump(lr, f)

with open('feature_columns.pkl', 'wb') as f:
    pickle.dump(X.columns.tolist(), f)

print("\n" + "-"*30)
print("SUCCESS: Model and Features saved as .pkl files")
print("-"*30)

# =============================================================================
# Visualize Results
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Logistic Regression — Churn Prediction', fontweight='bold')

# Confusion Matrix
cm = confusion_matrix(y_test, lr_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['No Churn', 'Churn'],
            yticklabels=['No Churn', 'Churn'])
axes[0].set_title('Confusion Matrix')
axes[0].set_ylabel('Actual')
axes[0].set_xlabel('Predicted')

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, lr_proba)
axes[1].plot(fpr, tpr, color='steelblue', label=f'AUC = {auc:.3f}')
axes[1].plot([0, 1], [0, 1], 'k--')
axes[1].set_title('ROC Curve')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].legend()

plt.tight_layout()
plt.savefig('lr_results.png', dpi=150, bbox_inches='tight')