import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
USE_TUNING = False
USE_BALANCED_TEST = False

# Load data
X_train = pd.read_csv('com_data/X_train.csv')

X_test = pd.read_csv('com_data/X_test.csv')
y_train = pd.read_csv('com_data/y_train.csv').values.ravel()  # Flatten to 1D array
y_test = pd.read_csv('com_data/y_test.csv').values.ravel()  # Flatten to 1D array
un_labeled_data = pd.read_csv('synthetic_X.csv')
toy_dir = './toy_data/'
if os.path.exists(toy_dir) is False:
    os.makedirs(toy_dir)
if USE_BALANCED_TEST:
    test_all_info_balanced_file = 'result_cache/concurrent_one_vs_rest/test_all_info_even.jsonl'
    with open(test_all_info_balanced_file, 'r', encoding='utf-8') as f:
        test_all_info_balanced = [json.loads(line.strip()) for line in f if line.strip()]
        
    selected_idx = [item['idx'] for item in test_all_info_balanced]
    X_test = X_test.iloc[selected_idx].reset_index(drop=True)
    y_test = y_test[selected_idx]

def preprocess_data_replace(df):
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
def preprocess_data(df, label=None):
    """Feature engineering function with proper missing value handling"""
    df = df.copy()
    
    # If label is provided, ensure it is a Series or numpy array with the same length as df
    if label is not None:
        if len(label) != len(df):
            raise ValueError("The length of the label must match the number of rows in the DataFrame.")
        label = pd.Series(label, index=df.index)  # Ensure label has the same index as df
    
    # Drop rows containing 'unknown' in any form (case-insensitive)
    mask = ~df.apply(lambda row: row.astype(str).str.contains('unknown', case=False, na=False).any(), axis=1)
    mask &= ~df.apply(lambda row: row.astype(str).str.contains('\?', case=False, na=False).any(), axis=1)
    df = df[mask]
    
    # If label is provided, also drop the corresponding labels
    if label is not None:
        label = label[mask]
    
    # Convert numerical columns
    for col in ['height', 'weight', 'bmi']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Split blood pressure
    if 'blood_pressure' in df.columns:
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
    #select_column gender,race,age,height,weight,bmi,marital_status,number_of_records,insurance,language,admit_type,admit_location,length_of_stay,systolic_bp,diastolic_bp
    select_columns = ['gender', 'race', 'age', 'height', 'weight', 'bmi',
                     'marital_status', 'number_of_records', 'insurance',
                     'language', 'admit_type', 'admit_location',
                     'length_of_stay', 'systolic_bp', 'diastolic_bp']
    # Return the processed DataFrame and the label (if provided)
    if label is not None:
        return df, label.values  # Return label as a numpy array
    else:
        return df
# Load and preprocess data
X_train_processed, y_train  = preprocess_data(X_train, y_train)
X_test_processed, y_test = preprocess_data(X_test, y_test)

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

# Create base pipeline with Logistic Regression
base_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(
        random_state=42,
        max_iter=10,  # Increase max_iter to ensure convergence
        # class_weight='balanced',  # Handle class imbalance/
        solver='lbfgs'))  # Use 'liblinear' for small datasets
])

y_process_func = lambda x: 'Home' if x =='Home' else 'Other'
y_train = [y_process_func(y) for y in y_train]
y_test = [y_process_func(y) for y in y_test]
# # Fit the model
base_pipeline.fit(X_train_processed, y_train)

# Predict on test set
y_pred = base_pipeline.predict(X_test_processed)

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
            xticklabels=base_pipeline.named_steps['classifier'].classes_,
            yticklabels=base_pipeline.named_steps['classifier'].classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# Feature importance (coefficients for Logistic Regression)
feature_names = numeric_features + list(
    base_pipeline.named_steps['preprocessor'].named_transformers_['cat']
    .named_steps['onehot'].get_feature_names_out(categorical_features)
)
print(base_pipeline.named_steps['classifier'].classes_)
importances_0 = base_pipeline.named_steps['classifier'].coef_[0]
sorted_idx = importances_0.argsort()[::-1]

print("\nFeature Importances (Coefficients) for Label 'Home':")
coef_dict = {}
for i in sorted_idx:
    print(f"{feature_names[i]:<40}: {importances_0[i]:.4f}")
    coef_dict[feature_names[i]] = importances_0[i]
# importances_1 = base_pipeline.named_steps['classifier'].coef_[1]
# sorted_idx = importances_1.argsort()[::-1]
# 
# print("\nFeature Importances (Coefficients) for Label 'Other':")
# for i in sorted_idx:
#     print(f"{feature_names[i]:<40}: {importances_1[i]:.4f}")
#     
with open(os.path.join(toy_dir, 'coef.json'), 'w') as f:
    json.dump(coef_dict, f, indent=4)
# # Save all data that can be accurately predicted in training
# correct_indices_test = np.where(y_pred == y_test)[0].astype(int)
# correct_indices_train = np.where(base_pipeline.predict(X_train_processed) == y_train)[0].astype(int)

# # Check the types and contents of correct_indices_train and y_train
# print("Type of correct_indices_train:", type(correct_indices_train))
# print("Type of y_train:", type(y_train))
# print("correct_indices_train:", correct_indices_train)
# print("y_train:", y_train)

# # Ensure correct_indices_train is an integer array
# if not isinstance(correct_indices_train, np.ndarray) or correct_indices_train.dtype != int:
#     raise ValueError("correct_indices_train must be an integer array")

# # Ensure y_train is a NumPy array
# if not isinstance(y_train, np.ndarray):
#     y_train = np.array(y_train)

# # Create new training dataset
# new_training_dataset = pd.concat([
#     X_train_processed.iloc[correct_indices_train],
#     X_test_processed.iloc[correct_indices_test]
# ]).reset_index(drop=True)

# selected_y_train = [y_train[i] for i in correct_indices_train]
# selected_y_test = [y_test[i] for i in correct_indices_test]
# # Check the lengths

# # Concatenate the labels
# correct_labels = np.concatenate([selected_y_train, selected_y_test])

# # Check the lengths again
# print("Length of correct_labels:", len(correct_labels))

# # Final check
# try:
#     print("All test predictions are correct:", np.all(base_pipeline.predict(new_training_dataset) == correct_labels))
# except Exception as e:
#     print("Error in final check:", e)
# #change to original feature names (t)
# print(new_training_dataset.columns)
# import sklearn
# # divid to train and test
# X_train_final, X_test_final, y_train_final, y_test_final = train_test_split(
#     new_training_dataset, correct_labels, test_size=0.13, random_state=42
# )
# X_train_final = X_train_final.drop(columns=['id'])
# X_test_final = X_test_final.drop(columns=['id'])
# # Save the final datasets
# X_train_final.to_csv(os.path.join(toy_dir, 'X_train.csv'), index= False)
# X_test_final.to_csv(os.path.join(toy_dir, 'X_test.csv'), index= False)
# pd.DataFrame(y_train_final, columns=['discharge_category']).to_csv(os.path.join(toy_dir, 'y_train.csv'), index=False)
# pd.DataFrame(y_test_final, columns=['discharge_category']).to_csv(os.path.join(toy_dir, 'y_test.csv'), index=False)