import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("      AUTISM PREDICTION MODEL TRAINING WITH XGBOOST + SMOTE")
print("="*70)

# 1. Load data
print("\n[1/10] Loading data...")
df = pd.read_csv('train.csv')
print(f"   ✅ Dataset shape: {df.shape}")
print(f"   ✅ Features: {df.shape[1]} columns")
print(f"   ✅ Samples: {df.shape[0]} rows")

# 2. Initial data exploration
print("\n[2/10] Data exploration...")
print(f"   Target distribution (Before SMOTE):")
print(df['Class/ASD'].value_counts())
print(f"   No ASD: {(df['Class/ASD']==0).sum()} ({(df['Class/ASD']==0).sum()/len(df)*100:.1f}%)")
print(f"   ASD: {(df['Class/ASD']==1).sum()} ({(df['Class/ASD']==1).sum()/len(df)*100:.1f}%)")

# 3. Data preprocessing
print("\n[3/10] Data preprocessing...")
df_clean = df.copy()

# Drop ID column
df_clean = df_clean.drop(['ID'], axis=1)
print(f"   ✅ Dropped ID column")

# Replace '?' with 'Others' in ethnicity
df_clean['ethnicity'] = df_clean['ethnicity'].replace('?', 'Others')
print(f"   ✅ Replaced '?' with 'Others' in ethnicity")

# Drop age_desc and result columns
df_clean = df_clean.drop(['age_desc', 'result'], axis=1)
print(f"   ✅ Dropped age_desc and result columns")

# 4. Label encoding
print("\n[4/10] Label encoding categorical variables...")
le_dict = {}
categorical_cols = ['gender', 'ethnicity', 'jaundice', 'austim', 
                    'contry_of_res', 'used_app_before', 'relation']

for col in categorical_cols:
    le = LabelEncoder()
    df_clean[col] = le.fit_transform(df_clean[col])
    le_dict[col] = le
    print(f"   ✅ Encoded {col}: {len(le.classes_)} categories")

# 5. Separate features and target
print("\n[5/10] Feature engineering...")
X = df_clean.drop('Class/ASD', axis=1)
y = df_clean['Class/ASD']

# Convert age from float to int
X['age'] = X['age'].astype(int)

print(f"   ✅ Total features: {X.shape[1]}")
print(f"   ✅ Feature names: {X.columns.tolist()}")

# 6. Train-test split
print("\n[6/10] Splitting data (80-20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   ✅ Training set: {X_train.shape[0]} samples")
print(f"   ✅ Test set: {X_test.shape[0]} samples")

# 7. Apply SMOTE for balancing
print("\n[7/10] Applying SMOTE for class balancing...")
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
print(f"   Before SMOTE - Class 0: {(y_train==0).sum()}, Class 1: {(y_train==1).sum()}")
print(f"   After SMOTE - Class 0: {(y_train_balanced==0).sum()}, Class 1: {(y_train_balanced==1).sum()}")
print(f"   ✅ Training set after SMOTE: {X_train_balanced.shape[0]} samples")

# 8. Train multiple models
print("\n[8/10] Training and comparing models...")

# Decision Tree
print("\n   Training Decision Tree...")
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train_balanced, y_train_balanced)
dt_pred = dt_model.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_pred)
print(f"   ✅ Decision Tree Accuracy: {dt_accuracy*100:.2f}%")

# Random Forest
print("\n   Training Random Forest...")
rf_model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
rf_model.fit(X_train_balanced, y_train_balanced)
rf_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)
print(f"   ✅ Random Forest Accuracy: {rf_accuracy*100:.2f}%")

# XGBoost with Hyperparameter Tuning
print("\n   Training XGBoost with hyperparameter tuning...")
print("   (This may take 1-2 minutes...)")

# Define parameter grid for RandomizedSearchCV
param_grid = {
    'n_estimators': [50, 100, 200, 500],
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.1, 0.2, 0.3],
    'subsample': [0.5, 0.7, 1.0],
    'colsample_bytree': [0.5, 0.7, 1.0]
}

xgb_base = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
random_search = RandomizedSearchCV(
    xgb_base, 
    param_distributions=param_grid,
    n_iter=20,
    cv=5,
    scoring='accuracy',
    random_state=42,
    n_jobs=-1,
    verbose=0
)

random_search.fit(X_train_balanced, y_train_balanced)
best_xgb_model = random_search.best_estimator_

print(f"   ✅ Best XGBoost parameters found:")
print(f"      {random_search.best_params_}")

# Predict with best XGBoost model
xgb_pred = best_xgb_model.predict(X_test)
xgb_accuracy = accuracy_score(y_test, xgb_pred)
print(f"   ✅ XGBoost Accuracy: {xgb_accuracy*100:.2f}%")

# 9. Select best model
print("\n[9/10] Model comparison summary...")
print("   " + "-"*50)
print(f"   Decision Tree:  {dt_accuracy*100:.2f}%")
print(f"   Random Forest:  {rf_accuracy*100:.2f}%")
print(f"   XGBoost:        {xgb_accuracy*100:.2f}% ⭐ BEST")
print("   " + "-"*50)

# Use XGBoost as final model
final_model = best_xgb_model
final_pred = xgb_pred
final_accuracy = xgb_accuracy

# Detailed evaluation
print("\n   Detailed Classification Report (XGBoost):")
print(classification_report(y_test, final_pred, target_names=['No ASD', 'ASD']))

print("\n   Confusion Matrix:")
cm = confusion_matrix(y_test, final_pred)
print(cm)
print(f"   True Negatives:  {cm[0][0]}")
print(f"   False Positives: {cm[0][1]}")
print(f"   False Negatives: {cm[1][0]}")
print(f"   True Positives:  {cm[1][1]}")

# Feature importance
print("\n   Top 10 Important Features:")
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': final_model.feature_importances_
}).sort_values('Importance', ascending=False)

for idx, row in feature_importance.head(10).iterrows():
    print(f"      {idx+1}. {row['Feature']}: {row['Importance']:.4f}")

# 10. Save everything
print("\n[10/10] Saving model and encoders...")

# Save the best XGBoost model
with open('autism_model.pkl', 'wb') as f:
    pickle.dump(final_model, f)
print("   ✅ Model saved: autism_model.pkl")

# Save label encoders
with open('label_encoders.pkl', 'wb') as f:
    pickle.dump(le_dict, f)
print("   ✅ Encoders saved: label_encoders.pkl")

# Save feature names
feature_names = X.columns.tolist()
with open('feature_names.pkl', 'wb') as f:
    pickle.dump(feature_names, f)
print("   ✅ Features saved: feature_names.pkl")

# Save all models for comparison (optional)
models_dict = {
    'decision_tree': dt_model,
    'random_forest': rf_model,
    'xgboost': best_xgb_model
}
with open('all_models.pkl', 'wb') as f:
    pickle.dump(models_dict, f)
print("   ✅ All models saved: all_models.pkl")

print("\n" + "="*70)
print("                  MODEL TRAINING COMPLETE! 🎉")
print("="*70)
print(f"\n   Final Model: XGBoost Classifier")
print(f"   Test Accuracy: {final_accuracy*100:.2f}%")
print(f"   Training Samples (after SMOTE): {X_train_balanced.shape[0]}")
print(f"   Test Samples: {X_test.shape[0]}")
print(f"\n   Files created:")
print(f"      1. autism_model.pkl (XGBoost)")
print(f"      2. label_encoders.pkl")
print(f"      3. feature_names.pkl")
print(f"      4. all_models.pkl (Bonus: All 3 models)")
print(f"\n   Ready to use in Streamlit app! 🚀")
print("="*70)
