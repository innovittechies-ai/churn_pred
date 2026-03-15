import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, accuracy_score)
from data_prep import load_and_prepare

# =============================================================================
# MODEL 2: RANDOM FOREST — Robust Ensemble
# =============================================================================
# Purpose : Capture non-linear interactions between features that Logistic
#           Regression cannot detect. Aggregates 300 decision trees to produce
#           a stable Feature Importance ranking — far less prone to overfitting
#           than any single tree. Best for understanding which features matter.
#
# Steps   : 1. Load & preprocess data via data_prep
#           2. Train on SMOTE-balanced data (no scaling needed for trees)
#           3. Evaluate — Accuracy, ROC-AUC, Classification Report
#           4. Extract & rank feature importances
#           5. Plot Confusion Matrix + ROC Curve + Feature Importance bar chart
# =============================================================================

print("=" * 60)
print("MODEL 2: RANDOM FOREST")
print("=" * 60)

(X, y, X_train, X_test, y_train, y_test,
 X_train_bal, y_train_bal,
 X_train_scaled, X_test_scaled) = load_and_prepare()

# Train
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_leaf=5,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train_bal, y_train_bal)

# Evaluate
rf_pred  = rf.predict(X_test)
rf_proba = rf.predict_proba(X_test)[:, 1]
acc      = accuracy_score(y_test, rf_pred)
auc      = roc_auc_score(y_test, rf_proba)

print(f"\nAccuracy : {acc:.4f}")
print(f"ROC-AUC  : {auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, rf_pred, target_names=['No Churn', 'Churn']))

# Feature Importances
importances = pd.Series(rf.feature_importances_, index=X.columns)
print("Top 5 Feature Importances:")
print(importances.sort_values(ascending=False).head(5).to_string())

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Random Forest — Churn Prediction', fontweight='bold')

# Confusion Matrix
cm = confusion_matrix(y_test, rf_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=axes[0],
            xticklabels=['No Churn', 'Churn'],
            yticklabels=['No Churn', 'Churn'])
axes[0].set_title('Confusion Matrix')
axes[0].set_ylabel('Actual')
axes[0].set_xlabel('Predicted')

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, rf_proba)
axes[1].plot(fpr, tpr, color='green', label=f'AUC = {auc:.3f}')
axes[1].plot([0, 1], [0, 1], 'k--')
axes[1].set_title('ROC Curve')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].legend()

# Feature Importance
top10 = importances.sort_values(ascending=False).head(10)
axes[2].barh(top10.index[::-1], top10.values[::-1], color='steelblue')
axes[2].set_title('Top 10 Feature Importances')
axes[2].set_xlabel('Importance')

plt.tight_layout()
plt.savefig('rf_results.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: rf_results.png")
