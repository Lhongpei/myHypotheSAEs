
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json
USE_TUNING = False
USE_BALANCED_TEST = True
X_train = pd.read_csv('com_data/X_train.csv')
X_test = pd.read_csv('com_data/X_test.csv')
y_train = pd.read_csv('com_data/y_train.csv').values.ravel()  # Flatten to 1D array
y_test = pd.read_csv('com_data/y_test.csv').values.ravel()  # Flatten to 1D array
if USE_BALANCED_TEST:
    test_all_info_balanced_file = 'result_cache/concurrent_one_vs_rest/test_all_info_even.jsonl'
    with open(test_all_info_balanced_file, 'r', encoding='utf-8') as f:
        test_all_info_balanced = [json.loads(line.strip()) for line in f if line.strip()]
        
    selected_idx = [item['idx'] for item in test_all_info_balanced]
    X_test = X_test.iloc[selected_idx].reset_index(drop=True)
    y_test = y_test[selected_idx]

def preprocess_data(df):
    """Feature engineering function with proper missing value handling"""
    df = df.copy()
    
    # Handle missing values
    for col in ['height', 'weight', 'bmi', 'language']:
        df[col] = df[col].replace(['unknown', '?', 'unkown', 'UNKNOWN'], np.nan)
    
    # Convert numerical columns
    for col in ['height', 'weight', 'bmi']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Split blood pressure
    df[['systolic_bp', 'diastolic_bp']] = (
        df['blood_pressure'].str.split('/', expand=True)
        .replace('unknown', np.nan)
        .astype(float)
    )
    
    # Extract hour from admit_time
    df['admit_time'] = pd.to_datetime(df['admit_time'], errors='coerce')
    df['admit_hour'] = df['admit_time'].dt.hour
    
    # Create age groups
    bins = [0, 50, 65, 75, 85, 120]
    labels = ['<50', '50-64', '65-74', '75-84', '85+']
    df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels)
    
    # Convert length_of_stay to numerical
    df['length_of_stay'] = (
        df['length_of_stay'].astype(str)
        .str.extract('(\d+)', expand=False)
        .astype(float)
    )
    
    # Drop original columns we've transformed
    df = df.drop(['blood_pressure', 'admit_time'], axis=1)
    
    return df

# Load and preprocess data
X_train_processed = preprocess_data(X_train)
X_test_processed = preprocess_data(X_test)

# Define preprocessing pipelines
numeric_features = ['age', 'height', 'weight', 'bmi', 'number_of_records', 
                   'systolic_bp', 'diastolic_bp', 'admit_hour', 'length_of_stay']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_features = ['gender', 'race', 'marital_status', 'insurance', 
                       'language', 'admit_type', 'admit_location', 'age_group']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# Define the parameter grid for tuning
param_grid = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [None, 10, 20, 30],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4],
    'classifier__max_features': ['sqrt', 'log2', None],
    'classifier__bootstrap': [True, False]
}

# Create base pipeline
base_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        random_state=42,
        n_estimators=200,  # Default value, will be tuned
        max_depth=10,  # Default value, will be tuned
        class_weight='balanced',  # Handle class imbalance
        n_jobs=-1))
])

# Create GridSearchCV object
grid_search = GridSearchCV(
    estimator=base_pipeline,
    param_grid=param_grid,
    cv=5,  # 5-fold cross-validation
    scoring='accuracy',  # or 'f1_weighted' for imbalanced classes
    verbose=2,
    n_jobs=-1
)
if USE_TUNING:
    # Fit the grid search to the data
    print("Starting grid search...")
    grid_search.fit(X_train_processed, y_train)
    print("Grid search complete!")

    # Get the best model
    best_rf = grid_search.best_estimator_

    # Evaluate on test set
    y_pred = best_rf.predict(X_test_processed)

    # Print best parameters
    print("\nBest Parameters:")
    print(grid_search.best_params_)
else:
    # Use a pre-defined model without tuning
    best_rf = base_pipeline
    best_rf.fit(X_train_processed, y_train)
    y_pred = best_rf.predict(X_test_processed)
# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix (Raw Counts):")
print(cm)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=best_rf.named_steps['classifier'].classes_,
            yticklabels=best_rf.named_steps['classifier'].classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# Feature importance
feature_names = numeric_features + list(
    best_rf.named_steps['preprocessor'].named_transformers_['cat']
    .named_steps['onehot'].get_feature_names_out(categorical_features)
)

importances = best_rf.named_steps['classifier'].feature_importances_
sorted_idx = importances.argsort()[::-1]

print("\nTop 20 Feature Importances:")
for i in sorted_idx[:20]:
    print(f"{feature_names[i]:<40}: {importances[i]:.4f}")