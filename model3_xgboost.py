import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from xgboost import XGBClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, accuracy_score)
from data_prep import load_and_prepare

# =============================================================================
# MODEL 3: XGBOOST — Performance Leader
# =============================================================================
# Purpose : Maximize predictive accuracy using gradient boosting — each new
#           tree corrects the residual errors of all previous trees.
#           Uses scale_pos_weight to handle class imbalance natively,
#           making SMOTE unnecessary for this model.
#           Best for real-time churn scoring & immediate marketing interventions.
#
# Steps   : 1. Load & preprocess data via data_prep
#           2. Compute scale_pos_weight = negatives / positives
#           3. Train on original (unbalanced) train split
#           4. Evaluate — Accuracy, ROC-AUC, Classification Report
#           5. Extract feature importances
#           6. Identify high-risk customer segments by churn probability
#           7. Plot Confusion Matrix + ROC Curve + Feature Importance bar chart
# =============================================================================

print("=" * 60)
print("MODEL 3: XGBOOST")
print("=" * 60)

(X, y, X_train, X_test, y_train, y_test,
 X_train_bal, y_train_bal,
 X_train_scaled, X_test_scaled) = load_and_prepare()

# Class imbalance weight
neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
spw = neg / pos
print(f"\nscale_pos_weight = {spw:.2f}  (neg:{neg} / pos:{pos})")

# Train
xgb = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=spw,
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1
)
xgb.fit(X_train, y_train)

# Evaluate
xgb_pred  = xgb.predict(X_test)
xgb_proba = xgb.predict_proba(X_test)[:, 1]
acc        = accuracy_score(y_test, xgb_pred)
auc        = roc_auc_score(y_test, xgb_proba)

print(f"\nAccuracy : {acc:.4f}")
print(f"ROC-AUC  : {auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, xgb_pred, target_names=['No Churn', 'Churn']))

# Feature Importances
importances = pd.Series(xgb.feature_importances_, index=X.columns)
print("Top 5 Feature Importances:")
print(importances.sort_values(ascending=False).head(5).to_string())

# High-Risk Segment Identification
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

print("\nHigh-Risk Segment Summary:")
print(segment_summary.to_string(index=False))
high_risk = (X_test_df['Risk_Segment'] == 'High Risk').sum()
print(f"\nHigh-risk customers flagged for retention: {high_risk}")
print("Targeting these enables the projected 15-25% revenue loss reduction.")

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('XGBoost — Churn Prediction', fontweight='bold')

# Confusion Matrix
cm = confusion_matrix(y_test, xgb_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', ax=axes[0],
            xticklabels=['No Churn', 'Churn'],
            yticklabels=['No Churn', 'Churn'])
axes[0].set_title('Confusion Matrix')
axes[0].set_ylabel('Actual')
axes[0].set_xlabel('Predicted')

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, xgb_proba)
axes[1].plot(fpr, tpr, color='darkorange', label=f'AUC = {auc:.3f}')
axes[1].plot([0, 1], [0, 1], 'k--')
axes[1].set_title('ROC Curve')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].legend()

# Feature Importance
top10 = importances.sort_values(ascending=False).head(10)
axes[2].barh(top10.index[::-1], top10.values[::-1], color='darkorange')
axes[2].set_title('Top 10 Feature Importances')
axes[2].set_xlabel('Importance')

plt.tight_layout()
plt.savefig('xgb_results.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: xgb_results.png")
